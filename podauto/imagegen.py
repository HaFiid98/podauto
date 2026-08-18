"""Image generation stage: prompt -> provider call -> PNG on disk.

Rotation, quota and licence gating all live in `providers.Rotator`; this module
is only the part that knows how to talk to an endpoint. It calls
`Rotator.generate()` with a closure and raises `QuotaExhausted` / `RateLimited`
out of that closure to drive key-then-provider fallback.

Three things about this module are deliberate.

* **Endpoints and models are data, and unverified data is refused, not guessed.**
  The endpoint paths and response shapes in config could not be checked against
  provider docs while this was written. So `require_ready()` refuses to call a
  provider whose `endpoint` or `model` is still empty or marked UNSET, and
  `extract_image()` raises `UnexpectedResponse` naming the keys it actually saw
  rather than returning something wrong. A guess that fails loudly on the first
  call is recoverable; a guess that silently saves 200 bytes of JSON as a .png
  is not.

* **Response handling is keyed on `api_shape`, not on provider name.** Four
  shapes cover the five configured providers, so adding a sixth that speaks an
  existing dialect is a config edit.

* **5xx retries happen here, not in the Rotator.** providers.json says
  `on_http_5xx: retry_same_key_twice_then_advance`, and the Rotator has no
  retry loop -- it advances on any exception. Retrying inside the closure is
  what makes that config line true instead of decorative.
"""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

from .models import PipelineStatus, Record
from .providers import (
    KeyRef,
    NoProviderAvailable,
    QuotaExhausted,
    RateLimited,
    Rotator,
)

# 5:6 is the Merch aspect. Generate small and native; the vectorize stage is
# what reaches 4500x5400, because asking an endpoint for that directly either
# errors or returns an upscaled blur.
ASPECT_W, ASPECT_H = 5, 6

# The fallback request size, used when a provider states no ceiling. 848x1024 and
# not the more conventional 832x1024: 832/1024 is 0.8125, which is 2.5% off 5:6
# and gets rejected by imageqa's aspect gate, so a provider with no declared
# max_resolution would have produced nothing but QA failures. 848 is what
# native_size() computes for a 1024 ceiling, so the fallback and the formula
# agree instead of differing by a hardcoded constant.
DEFAULT_NATIVE = (848, 1024)

# Hard ceiling on what we will ask an endpoint for, independent of what
# max_resolution claims. output_spec says generate native and upscale in the
# vectorize stage; without this ceiling, setting max_resolution to the print
# size would make native_size return 4500x5400 and request exactly what the spec
# forbids -- which either errors or bills for an upscaled blur.
MAX_NATIVE_H = 1536

# Body text that means "your allowance is gone for today" rather than "slow
# down". The distinction matters: one retires the key until midnight, the other
# cools it for 15 minutes.
QUOTA_MARKERS = ("quota", "exceeded your", "insufficient_quota", "billing",
                 "credit", "limit reached", "out of credits")

_PLACEHOLDER = re.compile(r"^\s*$|UNSET|TODO|FIXME|PLACEHOLDER", re.IGNORECASE)


class ProviderNotConfigured(RuntimeError):
    """endpoint or model is still a placeholder -- nothing was called."""


class UnexpectedResponse(RuntimeError):
    """The response did not contain an image where the adapter looked."""


class ProviderHttpError(RuntimeError):
    """A 4xx/5xx that is neither a rate limit nor a quota wall.

    Its own class rather than httpx.HTTPStatusError so the message survives
    without carrying request/response objects into the Rotator's error list.
    """


def is_placeholder(value: Any) -> bool:
    """True for a config field that was never filled in."""
    return value is None or bool(_PLACEHOLDER.search(str(value)))


def require_ready(name: str, cfg: dict[str, Any]) -> None:
    """Refuse to call a provider whose endpoint or model is still a placeholder.

    Without this the first live run POSTs at a bare hostname and the failure
    reads as a network problem rather than an unfinished config.
    """
    missing = [f for f in ("endpoint", "api_shape") if is_placeholder(cfg.get(f))]
    # A model is required unless the shape bakes it into the endpoint.
    if cfg.get("api_shape") != "url_prompt" and is_placeholder(cfg.get("model")):
        missing.append("model")
    if missing:
        raise ProviderNotConfigured(
            f"{name}: {', '.join(missing)} not set in config/providers.json. "
            f"Fill these in from the provider's current docs -- they were never "
            f"verified. Nothing was called."
        )


def native_size(cfg: dict[str, Any]) -> tuple[int, int]:
    """Largest 5:6 box fitting the provider's ceiling, on an 8px grid.

    Diffusion endpoints reject or silently round dimensions that are not
    multiples of 8, so the rounding is done here rather than discovered per
    provider.
    """
    cap = cfg.get("max_resolution")
    if not cap or is_placeholder(cap):
        return DEFAULT_NATIVE
    m = re.match(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*$", str(cap))
    if not m:
        return DEFAULT_NATIVE
    cap_w, cap_h = int(m.group(1)), int(m.group(2))
    h = min(cap_h, cap_w * ASPECT_H // ASPECT_W, MAX_NATIVE_H)
    w = h * ASPECT_W // ASPECT_H
    floor8 = lambda n: max(8, n - n % 8)          # noqa: E731
    return floor8(w), floor8(h)


# --- response decoding --------------------------------------------------

# Magic bytes, used as the acceptance test for anything we decode. This is what
# makes tolerant searching safe: a field is only accepted as the image if its
# decoded bytes actually start like an image, so a wrong guess about the
# response shape cannot write JSON or an error string to a .png file.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF8", "gif"),
    (b"BM", "bmp"),
)

# Keys observed to carry base64 image payloads across the configured providers.
# Not exhaustive and not verified -- the magic-byte check is what makes an
# incomplete list safe rather than wrong.
_B64_KEYS = {"data", "image", "b64_json", "imagebytes", "inlinedata", "content",
             "b64", "base64", "png"}


def image_kind(blob: bytes) -> str | None:
    """Identify `blob` by magic bytes, or None if it is not an image."""
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 12:
        return None
    blob = bytes(blob)
    for magic, kind in _MAGIC:
        if blob.startswith(magic):
            return kind
    if blob.startswith(b"RIFF") and blob[8:12] == b"WEBP":
        return "webp"
    return None


def _decode_b64(text: str) -> bytes | None:
    """Decode a base64 string only if the result is an image."""
    if not isinstance(text, str) or len(text) < 64:
        return None
    try:
        blob = base64.b64decode(text, validate=False)
    except Exception:                                  # noqa: BLE001
        return None
    return blob if image_kind(blob) else None


def find_image_in_json(node: Any, depth: int = 0) -> bytes | None:
    """Walk a decoded JSON body for a base64 image payload.

    Tolerant by design: the exact response paths for these providers could not
    be verified, so rather than hardcode a path that may be wrong, this searches
    and accepts only what proves to be an image. Preferred keys are tried first
    so a well-shaped response is not decoded field by field.
    """
    if depth > 8:
        return None
    if isinstance(node, str):
        return _decode_b64(node)
    if isinstance(node, dict):
        for key, value in node.items():
            if key.lower().replace("_", "") in _B64_KEYS:
                found = find_image_in_json(value, depth + 1)
                if found:
                    return found
        for value in node.values():
            found = find_image_in_json(value, depth + 1)
            if found:
                return found
        return None
    if isinstance(node, list):
        for value in node:
            found = find_image_in_json(value, depth + 1)
            if found:
                return found
    return None


def extract_image(response: httpx.Response) -> bytes:
    """Pull image bytes out of a response, or say precisely what arrived instead."""
    if image_kind(response.content):
        return response.content

    body: Any = None
    try:
        body = response.json()
    except Exception:                                  # noqa: BLE001
        raise UnexpectedResponse(
            f"content-type={response.headers.get('content-type')!r}, "
            f"{len(response.content)} bytes, not an image and not JSON: "
            f"{response.content[:120]!r}"
        ) from None

    found = find_image_in_json(body)
    if found:
        return found

    shape = sorted(body.keys()) if isinstance(body, dict) else type(body).__name__
    raise UnexpectedResponse(
        f"no image found in JSON response. top-level shape: {shape}. "
        f"first 200 chars: {str(body)[:200]!r}"
    )


# --- request building ---------------------------------------------------

@dataclass
class Request:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    json_body: dict[str, Any] | None = None


def apply_auth(req: Request, cfg: dict[str, Any], key: KeyRef) -> None:
    """Attach the credential the way this provider expects it.

    Kept separate from api_shape: how a provider wants to be authenticated is
    independent of how it wants a prompt, and pairing the two would mean a new
    shape for every combination.
    """
    secret = key.secret
    kind = cfg.get("auth", "none")
    if kind == "none" or not secret:
        return
    if kind == "bearer":
        req.headers["Authorization"] = f"Bearer {secret}"
    elif kind == "api_key":
        req.headers[cfg.get("auth_header", "x-goog-api-key")] = secret
    elif kind == "header":
        req.headers[cfg.get("auth_header", "Authorization")] = secret
    elif kind == "query":
        req.params[cfg.get("auth_param", "key")] = secret


def build_request(name: str, cfg: dict[str, Any], key: KeyRef, prompt: str) -> Request:
    """Translate (provider config, key, prompt) into one HTTP request.

    api_shape names the dialect. The four that build a request cover six of the
    seven configured providers, so another that speaks an existing dialect is a
    config edit. The fifth, gemini_generate_content, is recognised and refused --
    measured, it cannot serve this pipeline.
    """
    require_ready(name, cfg)
    shape = cfg["api_shape"]
    width, height = native_size(cfg)
    endpoint = str(cfg["endpoint"])
    model = str(cfg.get("model", ""))

    if shape == "url_prompt":
        # Prompt travels in the path; the response body is the image itself.
        url = endpoint.replace("{prompt}", quote(prompt, safe=""))
        req = Request("GET", url, params={
            "width": width, "height": height, "nologo": "true", "model": model or "flux",
        })
    elif shape == "gemini_generate_content":
        # Unimplemented on purpose, and refusing rather than left half-built.
        # Measured 2026-08-16 against the live API with a valid key (ListModels
        # returned 53 models, so this is not an auth problem):
        #   * image generation is quota 0 without billing -- 429 carrying
        #     "limit: 0, model: gemini-2.5-flash-preview-image";
        #   * the aspect-ratio enum has no 5:6. Nearest is 4:5, 4% off, and
        #     imageqa.ASPECT_TOLERANCE is 0.02 -- so even a billed account would
        #     return files this pipeline rejects until something pads 4:5 to 5:6,
        #     a step that does not exist.
        # A body that could actually return an image needs responseModalities
        # and imageConfig. Adding them would buy a billed call whose output fails
        # QA by construction, so this raises instead: Rotator.generate records
        # the reason verbatim and advances to the next provider, and nobody has
        # to debug a request that was never going to work. The evidence is also
        # in the gemini licence_notes in config/providers.json.
        raise ProviderNotConfigured(
            f"{name}: gemini image generation is not implemented, deliberately. "
            f"The API offers no 5:6 aspect ratio (nearest 4:5 is 4% off, image "
            f"QA allows 2%) and image quota is 0 without billing, so even a "
            f"correct request would produce a file this pipeline rejects. "
            f"Nothing was called."
        )
    elif shape == "hf_text_to_image":
        req = Request("POST", endpoint.replace("{model}", model), json_body={
            "inputs": prompt,
            "parameters": {"width": width, "height": height},
        })
    elif shape == "openai_images":
        # size AND width/height, which looks redundant and is not: verified
        # 2026-08-16 against the HF router, nscale honours `size` and ignores
        # width/height, together does the reverse and returns 1024x768 -- a
        # landscape image that fails the 5:6 aspect gate -- when only `size` is
        # sent. Both routes accept the other's field without complaint, so one
        # body serves both. Dropping either one silently breaks one provider.
        req = Request("POST", endpoint, json_body={
            "model": model, "prompt": prompt, "n": 1,
            "size": f"{width}x{height}", "width": width, "height": height,
            "response_format": "b64_json",
        })
    elif shape == "cloudflare_ai_run":
        url = (endpoint.replace("{account_id}", str(cfg.get("account_id", "")))
                       .replace("{model}", model))
        req = Request("POST", url, json_body={
            "prompt": prompt, "width": width, "height": height,
        })
    else:
        raise ProviderNotConfigured(
            f"{name}: unknown api_shape {shape!r}. Shapes that produce an "
            f"image: url_prompt, hf_text_to_image, openai_images, "
            f"cloudflare_ai_run. gemini_generate_content is recognised and "
            f"refused -- see the branch above for why."
        )

    req.headers.setdefault("Accept", "image/png, application/json")
    apply_auth(req, cfg, key)
    return req


# --- the call ------------------------------------------------------------

def classify_failure(status: int, body: bytes) -> Exception | None:
    """Map an HTTP status onto the exception that drives the right rotation.

    The 429-vs-quota split is the one that matters: RateLimited cools a key for
    fifteen minutes, QuotaExhausted retires it until midnight. Guessing wrong in
    the retiring direction throws away a key that would have worked minutes
    later, so a 429 is only read as exhaustion when the body says so.
    """
    if status < 400:
        return None
    text = body[:2000].decode("utf-8", "replace").lower()
    quota_ish = any(m in text for m in QUOTA_MARKERS)

    if status == 429:
        return QuotaExhausted() if quota_ish else RateLimited()
    if status in (402, 403) and quota_ish:
        return QuotaExhausted()
    return ProviderHttpError(f"HTTP {status}: {text[:200]}")


def call_provider(client: httpx.Client, name: str, cfg: dict[str, Any],
                  key: KeyRef, prompt: str, retries_5xx: int = 2,
                  sleep: Callable[[float], None] = time.sleep) -> bytes:
    """One provider call, returning image bytes.

    Retries 5xx on the same key before letting the exception escape, because
    providers.json promises `on_http_5xx: retry_same_key_twice_then_advance` and
    the Rotator has no retry loop -- it advances on any exception. Doing it here
    is what makes that config line true rather than decorative.
    """
    req = build_request(name, cfg, key, prompt)

    for attempt in range(retries_5xx + 1):
        response = client.request(
            req.method, req.url, headers=req.headers,
            params=req.params or None, json=req.json_body,
        )
        if 500 <= response.status_code < 600 and attempt < retries_5xx:
            sleep(2 ** attempt)
            continue
        failure = classify_failure(response.status_code, response.content)
        if failure:
            raise failure
        return extract_image(response)

    raise UnexpectedResponse(f"{name}: exhausted 5xx retries")       # unreachable


def image_path_for(out_dir: Path, record: Record, index: int,
                   variation, kind: str) -> Path:
    """Deterministic path, so a re-run overwrites rather than accumulating.

    Keyed on the record id and variation index instead of a timestamp: the
    pipeline is re-runnable by design, and a timestamped name would leave one
    orphan file per attempt for a human reviewer to disambiguate.
    """
    stem = f"{record.id}_{index}_{variation.style_id}"
    return Path(out_dir) / f"{stem}.{kind}"


# --- record level --------------------------------------------------------

@dataclass
class GenSummary:
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def generate_for_record(record: Record, rotator: Rotator, client: httpx.Client,
                        out_dir: str | Path = "data/images",
                        force: bool = False) -> GenSummary:
    """Fill in image_path for each variation, one provider call at a time.

    Every image costs quota, so an existing file is left alone unless `force`.
    A variation that cannot be generated is recorded in qa_notes and skipped --
    one dead provider must not discard the other variations of a record that
    already passed four gates to get here.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = GenSummary()

    for index, variation in enumerate(record.variations):
        if not force and variation.image_path and Path(variation.image_path).is_file():
            summary.skipped += 1
            continue

        def call(key: KeyRef, cfg: dict[str, Any], _v=variation) -> bytes:
            return call_provider(client, key.provider, cfg, key, _v.graphic_prompt)

        try:
            blob, key = rotator.generate(call)
        except NoProviderAvailable as exc:
            summary.failed += 1
            note = f"generation failed: {exc}"
            variation.qa_notes.append(note)
            summary.errors.append(f"{record.niche}/{variation.style_id}: {exc}")
            continue

        path = image_path_for(out, record, index, variation, image_kind(blob) or "png")
        path.write_bytes(blob)
        variation.image_path = str(path)
        variation.image_provider = key.ident
        summary.generated += 1

    if summary.generated:
        record.pipeline_status = PipelineStatus.IMAGES_GENERATED
    return summary





