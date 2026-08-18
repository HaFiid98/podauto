"""Image provider rotation with multi-key fallback.

Rotation order is: next key within a provider, then next provider. A key that
hits its daily quota is retired until local midnight; a key that returns 429 is
cooled down for a configured window.

Two hard rules:

* Keys are read from ENVIRONMENT VARIABLES, never stored in providers.json.
  The config holds only the env var name, so the config file stays safe to
  commit and keys never land in the repo.

* A provider with commercial_use_confirmed=false is skipped entirely. Selling
  generated art on merchandise needs both the model licence and the service
  terms to permit it, and that is a determination you make and record -- not
  something this code guesses.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


class NoProviderAvailable(RuntimeError):
    """Every enabled key is exhausted, cooling down, or unconfirmed."""


class ConfigError(ValueError):
    """providers.json is malformed -- raised loudly rather than worked around.

    Specifically covers the two ways the keys-live-in-env rule gets broken by
    hand-editing: pasting the secret itself where the variable NAME belongs, and
    embedding it in the `auth` field.
    """


# An environment variable NAME, which is what providers.json is allowed to hold.
# Upper snake case is a convention strong enough to make a rule out of, and it
# is the cheapest reliable way to tell "GEMINI_API_KEY" from the key itself:
# every provider token format in the wild carries lowercase, a dot or a dash.
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Recognised token prefixes, used only to make the error message specific.
_SECRET_PREFIXES = ("sk_", "sk-", "hf_", "AQ.", "AIza", "ghp_", "gho_", "xox",
                    "pat_", "Bearer ", "bearer ")

_AUTH_KINDS = {"none", "bearer", "api_key", "header", "query"}



def looks_like_secret(value: str) -> bool:
    """True if `value` is a credential rather than an environment variable name.

    Deliberately biased toward false positives: the cost of rejecting an oddly
    named variable is one clear error message, and the cost of accepting a
    pasted token is a credential sitting in a file designed to be committed.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    v = value.strip()
    return v.startswith(_SECRET_PREFIXES) or not _ENV_NAME.match(v)


def validate_config(config: dict[str, Any]) -> None:
    """Reject a config that carries credentials instead of variable names.

    The module rule -- keys live in the environment, providers.json holds only
    the variable name -- was documented but unenforced, so a hand-edit that
    pasted three live tokens into the file produced an AttributeError deep in
    rotation rather than a message saying what was wrong. Every problem is
    collected before raising, so one run fixes the whole file.
    """
    problems: list[str] = []

    for name, cfg in (config.get("providers") or {}).items():
        if not isinstance(cfg, dict):
            problems.append(f"{name}: provider entry must be an object")
            continue

        auth = cfg.get("auth")
        if auth is not None and auth not in _AUTH_KINDS:
            hint = (" -- looks like a credential; put the token in a variable "
                    "and name it under 'keys'"
                    if looks_like_secret(str(auth)) else "")
            problems.append(
                f"{name}.auth: {auth!r} is not one of "
                f"{sorted(_AUTH_KINDS)}{hint}")

        for i, entry in enumerate(cfg.get("keys") or []):
            where = f"{name}.keys[{i}]"
            if isinstance(entry, str):
                if looks_like_secret(entry):
                    problems.append(
                        f"{where}: expected an environment variable NAME in "
                        f"upper snake case, got what looks like the key itself. "
                        f"Put the value in .env as e.g. "
                        f"{name.upper()}_API_KEY=... and write "
                        f'"{name.upper()}_API_KEY" here.')
            elif isinstance(entry, dict):
                env = entry.get("env")
                if env is not None and looks_like_secret(str(env)):
                    problems.append(
                        f"{where}.env: expected an environment variable NAME, "
                        f"got what looks like the key itself.")
            else:
                problems.append(
                    f"{where}: must be a variable name string or an object "
                    f'like {{"label": "primary", "env": "NAME"}}')

    if problems:
        raise ConfigError(
            "providers.json holds credentials or malformed entries:\n  - "
            + "\n  - ".join(problems)
            + "\n\nKeys are read from environment variables so this file stays "
              "safe to commit. See config/providers.example.json."
        )


@dataclass
class KeyRef:
    provider: str
    label: str
    env_var: str | None

    @property
    def secret(self) -> str | None:
        if self.env_var is None:
            return None          # keyless provider, e.g. an open endpoint
        return os.environ.get(self.env_var)

    @property
    def ident(self) -> str:
        return f"{self.provider}:{self.label}"


@dataclass
class UsageLedger:
    """Per-key usage, persisted so quota state survives a restart."""

    path: Path
    day: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    cooldowns: dict[str, str] = field(default_factory=dict)
    exhausted: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path, today: date) -> UsageLedger:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            led = cls(
                path=p,
                day=data.get("day", ""),
                counts=data.get("counts", {}),
                cooldowns=data.get("cooldowns", {}),
                exhausted=data.get("exhausted", []),
            )
        else:
            led = cls(path=p)
        if led.day != today.isoformat():
            led.day = today.isoformat()
            led.counts = {}
            led.exhausted = []      # daily quotas reset; cooldowns do not
        return led

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "day": self.day,
            "counts": self.counts,
            "cooldowns": self.cooldowns,
            "exhausted": self.exhausted,
        }, indent=2))

    def used(self, ident: str) -> int:
        return self.counts.get(ident, 0)

    def is_cooling(self, ident: str, now: datetime) -> bool:
        until = self.cooldowns.get(ident)
        if not until:
            return False
        return now < datetime.fromisoformat(until)

    def record_use(self, ident: str) -> None:
        self.counts[ident] = self.counts.get(ident, 0) + 1

    def cool_down(self, ident: str, now: datetime, seconds: int) -> None:
        self.cooldowns[ident] = (now + timedelta(seconds=seconds)).isoformat()

    def mark_exhausted(self, ident: str) -> None:
        if ident not in self.exhausted:
            self.exhausted.append(ident)


class Rotator:
    def __init__(self, config: dict[str, Any], ledger: UsageLedger):
        self.config = config
        self.ledger = ledger
        self.rotation = config.get("rotation", {})

    @classmethod
    def from_files(cls, config_path: str | Path, ledger_path: str | Path,
                   today: date | None = None) -> Rotator:
        cfg = json.loads(Path(config_path).read_text())
        validate_config(cfg)
        return cls(cfg, UsageLedger.load(ledger_path, today or date.today()))

    def _keys_for(self, name: str, cfg: dict) -> list[KeyRef]:
        """Normalize the two accepted key spellings into KeyRefs.

        Both `"keys": ["GEMINI_API_KEY"]` and
        `"keys": [{"label": "primary", "env": "GEMINI_API_KEY"}]` are valid; the
        shorthand exists because the object form invites pasting the secret into
        a second field. Either way the string is a variable NAME -- validate_config
        rejects a value that looks like the credential itself.
        """
        entries = cfg.get("keys") or []
        if not entries and cfg.get("auth") == "none":
            return [KeyRef(provider=name, label="anonymous", env_var=None)]

        refs: list[KeyRef] = []
        for i, entry in enumerate(entries):
            if isinstance(entry, str):
                refs.append(KeyRef(provider=name, label=entry, env_var=entry))
            else:
                refs.append(KeyRef(
                    provider=name,
                    label=entry.get("label", f"key{i}"),
                    env_var=entry.get("env"),
                ))
        return refs

    def skip_reasons(self) -> dict[str, str]:
        """Why each configured provider is unavailable. Surfaced in logs so a
        silent no-op is never mistaken for 'no work to do'.

        The unset-variable case is reported by NAME, because an enabled provider
        with a confirmed licence and an empty variable is the failure most likely
        to look like a bug in the rotator rather than a missing export.
        """
        out: dict[str, str] = {}
        for name in self.rotation.get("order", []):
            cfg = self.config.get("providers", {}).get(name)
            if cfg is None:
                out[name] = "not present in providers.json"
            elif not cfg.get("enabled"):
                out[name] = "disabled"
            elif not cfg.get("commercial_use_confirmed"):
                out[name] = "commercial_use_confirmed is false -- verify licence and set it"
            else:
                keys = self._keys_for(name, cfg)
                if not keys:
                    out[name] = "no keys configured"
                elif not any(k.env_var is None or k.secret for k in keys):
                    unset = ", ".join(k.env_var for k in keys if k.env_var)
                    out[name] = f"no key value in the environment (unset: {unset})"
        return out

    def candidates(self, now: datetime | None = None) -> Iterator[KeyRef]:
        now = now or datetime.now(timezone.utc)
        providers = self.config.get("providers", {})
        for name in self.rotation.get("order", []):
            cfg = providers.get(name)
            if not cfg or not cfg.get("enabled"):
                continue
            if not cfg.get("commercial_use_confirmed"):
                continue
            quota = cfg.get("daily_quota")
            for key in self._keys_for(name, cfg):
                if key.env_var and not key.secret:
                    continue                                  # env var unset
                if key.ident in self.ledger.exhausted:
                    continue
                if self.ledger.is_cooling(key.ident, now):
                    continue
                if quota is not None and self.ledger.used(key.ident) >= quota:
                    continue
                yield key

    def generate(
        self,
        call: Callable[[KeyRef, dict], Any],
        now: datetime | None = None,
    ) -> tuple[Any, KeyRef]:
        """Try candidates in order until one succeeds.

        `call(key, provider_cfg)` should raise QuotaExhausted or RateLimited to
        drive rotation; any other exception advances to the next candidate too,
        but is recorded verbatim so real bugs stay visible.
        """
        now = now or datetime.now(timezone.utc)
        max_attempts = self.rotation.get("max_attempts_per_image", 6)
        cooldown = self.rotation.get("cooldown_seconds", 900)
        errors: list[str] = []
        attempts = 0

        for key in self.candidates(now):
            if attempts >= max_attempts:
                break
            attempts += 1
            cfg = self.config["providers"][key.provider]
            try:
                result = call(key, cfg)
            except QuotaExhausted:
                self.ledger.mark_exhausted(key.ident)
                errors.append(f"{key.ident}: quota exhausted")
            except RateLimited:
                self.ledger.cool_down(key.ident, now, cooldown)
                errors.append(f"{key.ident}: rate limited, cooling {cooldown}s")
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"{key.ident}: {type(exc).__name__}: {exc}")
            else:
                self.ledger.record_use(key.ident)
                self.ledger.save()
                return result, key
            self.ledger.save()

        raise NoProviderAvailable(
            "no provider produced an image. attempts=%d; %s; skipped=%s"
            % (attempts, "; ".join(errors) or "no candidates", self.skip_reasons())
        )


class QuotaExhausted(Exception):
    """Key's daily allowance is gone -- retire it until tomorrow."""


class RateLimited(Exception):
    """Transient 429 -- cool this key down but keep it for later."""
