"""Image generation stage. No network: every test drives httpx.MockTransport.

The module's own docstring says endpoints and response shapes could not be
verified against provider docs, so these tests concentrate on the properties
that hold regardless of the exact shape: a placeholder config is refused rather
than called, a non-image response is never written to disk as an image, and the
HTTP status maps onto the rotation the config promises.
"""

import base64
import io
from datetime import date
from pathlib import Path

import httpx
import pytest
from PIL import Image

from podauto.imagegen import (
    MAX_NATIVE_H,
    ProviderHttpError,
    ProviderNotConfigured,
    UnexpectedResponse,
    build_request,
    call_provider,
    classify_failure,
    extract_image,
    generate_for_record,
    image_kind,
    native_size,
    require_ready,
)
from podauto.models import IdeaType, PipelineStatus, Record, SourcePlatform, StyleVariation
from podauto.providers import KeyRef, QuotaExhausted, RateLimited, Rotator, UsageLedger


def png_bytes(size=(16, 19), colour=(200, 40, 40, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, colour).save(buf, format="PNG")
    return buf.getvalue()


READY = {
    "endpoint": "https://example.test/v1/models/{model}:run",
    "api_shape": "hf_text_to_image",
    "model": "some/model",
    "auth": "bearer",
    "max_resolution": "1024x1024",
}


def key(provider="alpha", label="k1", env="ALPHA_KEY") -> KeyRef:
    return KeyRef(provider=provider, label=label, env_var=env)


def client_returning(*responses, record=None) -> httpx.Client:
    """A client that replies with the given responses in order, then repeats the last."""
    seq = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- unverified config is refused, not called ---------------------------

def test_placeholder_model_is_refused_by_name():
    """The endpoints and models in config were never verified against provider
    docs, so an unfinished field has to stop the call rather than produce a POST
    at a bare hostname that reads as a network fault."""
    cfg = dict(READY, model="UNSET -- pick the model and confirm the route")
    with pytest.raises(ProviderNotConfigured) as exc:
        require_ready("gemini", cfg)
    assert "model" in str(exc.value)
    assert "Nothing was called" in str(exc.value)


def test_placeholder_endpoint_is_refused():
    with pytest.raises(ProviderNotConfigured) as exc:
        require_ready("gemini", dict(READY, endpoint=""))
    assert "endpoint" in str(exc.value)


def test_url_prompt_shape_needs_no_model():
    """Pollinations bakes the model into the URL, so requiring one would block a
    provider that is correctly configured."""
    require_ready("pollinations", {
        "endpoint": "https://x.test/prompt/{prompt}", "api_shape": "url_prompt",
        "model": "",
    })


def test_unknown_api_shape_lists_the_known_ones():
    with pytest.raises(ProviderNotConfigured) as exc:
        build_request("x", dict(READY, api_shape="telepathy"), key(), "a cat")
    assert "url_prompt" in str(exc.value)


# --- sizing --------------------------------------------------------------

def test_native_size_fits_5_6_inside_the_cap_on_an_8px_grid():
    w, h = native_size({"max_resolution": "1024x1024"})
    assert h == 1024 and w % 8 == 0 and w < h
    assert abs((w / h) - (5 / 6)) < 0.01


def test_native_size_falls_back_when_the_cap_is_absent_or_junk():
    assert native_size({}) == (848, 1024)
    assert native_size({"max_resolution": "UNSET"}) == (848, 1024)
    assert native_size({"max_resolution": "big"}) == (848, 1024)


def test_the_fallback_size_is_what_the_formula_would_produce():
    """832x1024 was the original fallback and is 2.5% off 5:6 -- outside the
    tolerance imageqa allows, so a provider declaring no max_resolution would have
    generated nothing but QA failures. Fallback and formula have to agree."""
    assert native_size({}) == native_size({"max_resolution": "1024x1024"})


def test_never_requests_the_final_print_size():
    """4500x5400 is reached by vectorizing, not by asking an endpoint for it.
    A max_resolution set to the print size must not become the request size."""
    w, h = native_size({"max_resolution": "4500x5400"})
    assert (w, h) != (4500, 5400)
    assert h <= MAX_NATIVE_H and w % 8 == 0
    assert abs((w / h) - (5 / 6)) < 0.01


# --- request building ----------------------------------------------------

def test_prompt_is_percent_encoded_into_the_path_for_url_prompt():
    """Pollinations carries the prompt in the URL, and prompts contain commas,
    spaces and quotes -- all of which have to survive as data, not structure."""
    req = build_request("pollinations", {
        "endpoint": "https://x.test/prompt/{prompt}", "api_shape": "url_prompt",
        "model": "flux", "auth": "none",
    }, key(), 'a cat, text "MEOW"/2')
    assert " " not in req.url and req.url.count("?") == 0
    assert "%20" in req.url and "%22" in req.url and "%2F" in req.url
    assert req.method == "GET"
    assert req.params["width"] and req.params["height"]


def test_model_is_substituted_into_the_endpoint_path():
    req = build_request("hf", READY, key(), "a cat")
    assert req.url == "https://example.test/v1/models/some/model:run"
    assert req.json_body["inputs"] == "a cat"


def test_bearer_auth_goes_in_the_header(monkeypatch):
    monkeypatch.setenv("ALPHA_KEY", "s3cret")
    req = build_request("hf", READY, key(), "a cat")
    assert req.headers["Authorization"] == "Bearer s3cret"


def test_api_key_auth_uses_a_header_not_the_query_string(monkeypatch):
    """A key in the query string lands in server logs and browser history.

    Deliberately not pinned to an api_shape: apply_auth is separate from the
    shape dispatch precisely so how a provider wants to be authenticated stays
    independent of how it wants a prompt.
    """
    monkeypatch.setenv("ALPHA_KEY", "s3cret")
    req = build_request("some_provider", dict(READY, auth="api_key"), key(), "a cat")
    assert "s3cret" not in req.url
    assert req.headers["x-goog-api-key"] == "s3cret"
    assert req.params == {}


def test_keyless_provider_sends_no_credential():
    req = build_request("pollinations", {
        "endpoint": "https://x.test/prompt/{prompt}", "api_shape": "url_prompt",
        "auth": "none", "model": "flux",
    }, KeyRef("pollinations", "anonymous", None), "a cat")
    assert "Authorization" not in req.headers


def test_the_gemini_shape_refuses_instead_of_building_a_request():
    """Measured 2026-08-16, not guessed: the Gemini API has no 5:6 aspect ratio
    (nearest 4:5 is 4% off against imageqa's 2% gate) and image generation is
    quota 0 without billing. A request that could return an image would still
    return one this pipeline rejects, so the branch refuses before the call and
    says why -- the alternative is a plausible-looking POST whose failure reads
    as a network fault. Both facts belong in the message: someone who enables
    billing would otherwise fix only half the problem.
    """
    with pytest.raises(ProviderNotConfigured) as exc:
        build_request("gemini", dict(
            READY, auth="api_key", api_shape="gemini_generate_content"),
            key(), "a cat")
    assert "5:6" in str(exc.value)
    assert "billing" in str(exc.value)
    assert "Nothing was called" in str(exc.value)


def test_an_unknown_shape_is_not_offered_gemini_as_an_option():
    """The shape list in that error is what someone reads after a typo, so it
    must not point at the one shape that cannot generate."""
    with pytest.raises(ProviderNotConfigured) as exc:
        build_request("x", dict(READY, api_shape="hf_text_to_imag"), key(), "a cat")
    msg = str(exc.value)
    assert "hf_text_to_image" in msg
    assert "recognised and refused" in msg


def test_openai_images_shape_asks_for_base64():
    req = build_request("together", dict(
        READY, api_shape="openai_images"), key(), "a cat")
    assert req.json_body["response_format"] == "b64_json"
    assert "x" in req.json_body["size"]


def test_openai_images_states_the_size_both_ways_the_routes_accept():
    """Not redundancy. On the HF router nscale honours `size` and together
    honours width/height; sending only `size` gets 1024x768 back from together,
    which the aspect gate rejects. One body has to satisfy both."""
    req = build_request("together", dict(
        READY, api_shape="openai_images", max_resolution="1024x1024"), key(), "a cat")
    w, h = native_size({"max_resolution": "1024x1024"})
    assert req.json_body["size"] == f"{w}x{h}"
    assert (req.json_body["width"], req.json_body["height"]) == (w, h)


# --- response decoding ---------------------------------------------------

def test_a_raw_image_body_is_returned_untouched():
    """Pollinations replies with the PNG itself, no JSON envelope."""
    blob = png_bytes()
    got = extract_image(httpx.Response(200, content=blob,
                                       headers={"content-type": "image/png"}))
    assert got == blob


def test_base64_is_found_wherever_the_response_puts_it():
    """The response paths could not be verified, so the search is tolerant. Each
    of these is a plausible envelope for the same payload."""
    b64 = base64.b64encode(png_bytes()).decode()
    bodies = [
        {"data": [{"b64_json": b64}]},                       # openai_images
        {"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": b64}}]}}]},
        {"result": {"image": b64}},                          # cloudflare
        {"images": [b64]},
    ]
    for body in bodies:
        assert image_kind(extract_image(httpx.Response(200, json=body))) == "png"


def test_a_base64_field_that_is_not_an_image_is_refused():
    """The magic-byte check is what makes tolerant searching safe: without it a
    long string in a plausibly-named field gets written to disk as a .png."""
    not_image = base64.b64encode(b"upstream connect error, retry later " * 3).decode()
    with pytest.raises(UnexpectedResponse):
        extract_image(httpx.Response(200, json={"data": not_image}))


def test_a_json_error_body_names_the_keys_it_actually_saw():
    """One live call has to be enough to tell the operator what to correct, so the
    message carries the real shape rather than 'no image found'."""
    with pytest.raises(UnexpectedResponse) as exc:
        extract_image(httpx.Response(200, json={"error": "bad model", "hint": "x"}))
    assert "error" in str(exc.value) and "hint" in str(exc.value)


def test_an_html_error_page_is_reported_as_neither_image_nor_json():
    with pytest.raises(UnexpectedResponse) as exc:
        extract_image(httpx.Response(200, content=b"<html><body>502 Bad Gateway",
                                     headers={"content-type": "text/html"}))
    assert "text/html" in str(exc.value)
    assert "502 Bad Gateway" in str(exc.value)


def test_image_kind_reads_magic_bytes_not_extensions():
    assert image_kind(png_bytes()) == "png"
    assert image_kind(b"\xff\xd8\xff" + b"\0" * 20) == "jpg"
    assert image_kind(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert image_kind(b'{"error": "no such model"}') is None
    assert image_kind(b"") is None
    assert image_kind("\x89PNG\r\n\x1a\n" + "x" * 20) is None      # str, not bytes


# --- status -> rotation --------------------------------------------------

def test_a_plain_429_only_cools_the_key():
    """RateLimited cools for fifteen minutes, QuotaExhausted retires until
    midnight. A bare 429 with no quota language is the transient one, and reading
    it as exhaustion discards a key that would work again in a minute."""
    assert isinstance(classify_failure(429, b"Too Many Requests"), RateLimited)


def test_a_429_that_says_quota_retires_the_key():
    body = b'{"error": {"message": "You exceeded your current quota"}}'
    assert isinstance(classify_failure(429, body), QuotaExhausted)


def test_402_and_403_are_quota_only_when_the_body_says_so():
    assert isinstance(classify_failure(402, b"out of credits"), QuotaExhausted)
    assert isinstance(classify_failure(403, b"billing required"), QuotaExhausted)
    # A plain 403 is a wrong or unauthorised key, not an empty wallet.
    assert isinstance(classify_failure(403, b"invalid api key"), ProviderHttpError)


def test_other_failures_keep_the_status_and_body_in_the_message():
    err = classify_failure(400, b"model not found: some/model")
    assert isinstance(err, ProviderHttpError)
    assert "400" in str(err) and "model not found" in str(err)


def test_success_classifies_as_no_failure():
    assert classify_failure(200, b"") is None
    assert classify_failure(204, b"") is None


# --- 5xx retry lives here, not in the Rotator ---------------------------

def test_5xx_is_retried_on_the_same_key_twice_then_raised():
    """providers.json promises retry_same_key_twice_then_advance and the Rotator
    has no retry loop, so if this does not happen here the config line is
    decorative."""
    seen: list[httpx.Request] = []
    slept: list[float] = []
    client = client_returning(httpx.Response(503, content=b"upstream down"),
                              record=seen)
    with pytest.raises(ProviderHttpError):
        call_provider(client, "hf", READY, key(), "a cat", sleep=slept.append)
    assert len(seen) == 3                      # first attempt plus two retries
    assert slept == [1, 2]                     # backoff, and no sleep after the last


def test_a_5xx_that_clears_on_retry_returns_the_image():
    seen: list[httpx.Request] = []
    client = client_returning(
        httpx.Response(500, content=b"oops"),
        httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"}),
        record=seen)
    blob = call_provider(client, "hf", READY, key(), "a cat", sleep=lambda _s: None)
    assert image_kind(blob) == "png"
    assert len(seen) == 2


def test_a_429_is_not_retried_because_retrying_is_what_it_forbids():
    seen: list[httpx.Request] = []
    client = client_returning(httpx.Response(429, content=b"slow down"), record=seen)
    with pytest.raises(RateLimited):
        call_provider(client, "hf", READY, key(), "a cat", sleep=lambda _s: None)
    assert len(seen) == 1


# --- record level --------------------------------------------------------

def provider_cfg(**over) -> dict:
    cfg = dict(READY, enabled=True, commercial_use_confirmed=True,
               keys=["ALPHA_KEY"])
    cfg.update(over)
    return cfg


def rotator_for(tmp_path, providers: dict) -> Rotator:
    config = {
        "rotation": {"order": list(providers), "cooldown_seconds": 900,
                     "max_attempts_per_image": 6},
        "providers": providers,
    }
    return Rotator(config, UsageLedger.load(tmp_path / "ledger.json", date.today()))


def make_record(n: int = 2) -> Record:
    return Record(
        id="rec1", timestamp="2026-08-16T00:00:00Z",
        source_url="https://example.test/1",
        source_platform=SourcePlatform.AMAZON, idea_type=IdeaType.READY_SHIRT,
        niche="Gardening Grandma", raw_title_or_text="Garden Nan",
        variations=[StyleVariation(style_id=f"style{i}", style_name="S",
                                   graphic_prompt=f"a cat {i}") for i in range(n)],
    )


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALPHA_KEY", "s3cret")
    monkeypatch.setenv("BETA_KEY", "s3cret2")


def ok_png() -> httpx.Response:
    return httpx.Response(200, content=png_bytes(),
                          headers={"content-type": "image/png"})


def test_each_variation_gets_a_file_and_a_recorded_provider(tmp_path, env):
    record = make_record(2)
    rot = rotator_for(tmp_path, {"alpha": provider_cfg()})
    summary = generate_for_record(record, rot, client_returning(ok_png()),
                                  out_dir=tmp_path / "images")

    assert summary.generated == 2 and summary.failed == 0
    for variation in record.variations:
        assert Path(variation.image_path).is_file()
        assert variation.image_provider == "alpha:ALPHA_KEY"
    assert record.pipeline_status == PipelineStatus.IMAGES_GENERATED


def test_the_filename_is_deterministic_so_a_re_run_overwrites(tmp_path, env):
    """A timestamped name would leave one orphan per attempt for the human
    reviewer to disambiguate."""
    rot = rotator_for(tmp_path, {"alpha": provider_cfg()})
    first = make_record(1)
    generate_for_record(first, rot, client_returning(ok_png()),
                        out_dir=tmp_path / "images")
    second = make_record(1)
    generate_for_record(second, rot, client_returning(ok_png()),
                        out_dir=tmp_path / "images", force=True)

    assert first.variations[0].image_path == second.variations[0].image_path
    assert "rec1" in first.variations[0].image_path
    assert "style0" in first.variations[0].image_path
    assert len(list((tmp_path / "images").iterdir())) == 1


def test_an_existing_image_is_not_paid_for_twice(tmp_path, env):
    """Every image costs quota, so a re-run must not re-request what is on disk."""
    record = make_record(1)
    rot = rotator_for(tmp_path, {"alpha": provider_cfg()})
    generate_for_record(record, rot, client_returning(ok_png()),
                        out_dir=tmp_path / "images")

    seen: list[httpx.Request] = []
    again = generate_for_record(record, rot, client_returning(ok_png(), record=seen),
                               out_dir=tmp_path / "images")
    assert (again.skipped, again.generated) == (1, 0)
    assert seen == []


def test_force_regenerates_over_an_existing_file(tmp_path, env):
    record = make_record(1)
    rot = rotator_for(tmp_path, {"alpha": provider_cfg()})
    generate_for_record(record, rot, client_returning(ok_png()),
                        out_dir=tmp_path / "images")

    seen: list[httpx.Request] = []
    again = generate_for_record(record, rot, client_returning(ok_png(), record=seen),
                               out_dir=tmp_path / "images", force=True)
    assert again.generated == 1 and len(seen) == 1


def test_a_stale_path_pointing_at_a_deleted_file_is_regenerated(tmp_path, env):
    """The skip is keyed on the file existing, not on the field being set --
    otherwise clearing data/images would leave records that can never be refilled."""
    record = make_record(1)
    rot = rotator_for(tmp_path, {"alpha": provider_cfg()})
    generate_for_record(record, rot, client_returning(ok_png()),
                        out_dir=tmp_path / "images")
    Path(record.variations[0].image_path).unlink()

    again = generate_for_record(record, rot, client_returning(ok_png()),
                                out_dir=tmp_path / "images")
    assert again.generated == 1 and again.skipped == 0


def test_one_dead_variation_does_not_discard_the_others(tmp_path, env):
    """The record passed four gates to get here. A provider failure on variation
    two must leave variation one's image and say what happened in qa_notes."""
    record = make_record(2)
    rot = rotator_for(tmp_path, {"alpha": provider_cfg()})
    client = client_returning(ok_png(), httpx.Response(400, content=b"bad model"))

    summary = generate_for_record(record, rot, client, out_dir=tmp_path / "images")

    assert (summary.generated, summary.failed) == (1, 1)
    assert Path(record.variations[0].image_path).is_file()
    assert record.variations[1].image_path is None
    assert record.variations[1].qa_notes and "400" in record.variations[1].qa_notes[0]
    assert summary.errors and "style1" in summary.errors[0]
    # One image did land, so the record still advances rather than stalling.
    assert record.pipeline_status == PipelineStatus.IMAGES_GENERATED


def test_a_quota_wall_falls_through_to_the_next_provider(tmp_path, env):
    """The whole point of raising QuotaExhausted out of the call closure: the
    Rotator retires the key and the second provider produces the image."""
    record = make_record(1)
    rot = rotator_for(tmp_path, {
        "alpha": provider_cfg(),
        "beta": provider_cfg(keys=["BETA_KEY"]),
    })
    client = client_returning(
        httpx.Response(429, content=b'{"message": "quota exceeded for today"}'),
        ok_png())

    summary = generate_for_record(record, rot, client, out_dir=tmp_path / "images")

    assert summary.generated == 1
    assert record.variations[0].image_provider == "beta:BETA_KEY"
    assert "alpha:ALPHA_KEY" in rot.ledger.exhausted


def test_a_provider_with_an_unset_model_fails_without_a_network_call(tmp_path, env):
    """require_ready runs inside the closure, so an unfinished config spends no
    quota and the reason reaches qa_notes instead of reading as a network fault."""
    record = make_record(1)
    rot = rotator_for(tmp_path, {"alpha": provider_cfg(model="UNSET -- pick one")})
    seen: list[httpx.Request] = []

    summary = generate_for_record(record, rot, client_returning(ok_png(), record=seen),
                                 out_dir=tmp_path / "images")

    assert seen == []
    assert summary.failed == 1 and summary.generated == 0
    assert "ProviderNotConfigured" in record.variations[0].qa_notes[0]
    # Nothing was generated, so the record must not claim IMAGES_GENERATED.
    assert record.pipeline_status != PipelineStatus.IMAGES_GENERATED


def test_no_candidates_at_all_reports_the_skip_reasons(tmp_path, env):
    """A fresh clone has every provider unconfirmed, and the failure has to say so
    rather than looking like there was no work to do."""
    record = make_record(1)
    rot = rotator_for(tmp_path, {"alpha": provider_cfg(commercial_use_confirmed=False)})

    summary = generate_for_record(record, rot, client_returning(ok_png()),
                                 out_dir=tmp_path / "images")

    assert (summary.generated, summary.skipped, summary.failed) == (0, 0, 1)
    assert len(summary.errors) == 1
    assert "commercial_use_confirmed" in record.variations[0].qa_notes[0]

def test_the_extension_follows_the_bytes_not_the_request(tmp_path, env):
    """We ask for PNG; a provider that answers with JPEG must not be saved under
    a .png name, because the vectorize stage opens these by path."""
    record = make_record(1)
    rot = rotator_for(tmp_path, {"alpha": provider_cfg()})
    jpeg = io.BytesIO()
    Image.new("RGB", (16, 19), (10, 20, 30)).save(jpeg, format="JPEG")
    client = client_returning(httpx.Response(200, content=jpeg.getvalue(),
                                             headers={"content-type": "image/jpeg"}))

    generate_for_record(record, rot, client, out_dir=tmp_path / "images")
    assert record.variations[0].image_path.endswith(".jpg")







