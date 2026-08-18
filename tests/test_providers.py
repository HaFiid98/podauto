import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from podauto.providers import (
    ConfigError,
    NoProviderAvailable,
    QuotaExhausted,
    RateLimited,
    Rotator,
    UsageLedger,
    looks_like_secret,
    validate_config,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
TODAY = date(2026, 8, 10)


def cfg(**overrides):
    base = {
        "rotation": {
            "order": ["alpha", "beta"],
            "cooldown_seconds": 900,
            "max_attempts_per_image": 6,
        },
        "providers": {
            "alpha": {
                "enabled": True,
                "commercial_use_confirmed": True,
                "auth": "bearer",
                "daily_quota": 2,
                "keys": [{"label": "a1", "env": "A1"}, {"label": "a2", "env": "A2"}],
            },
            "beta": {
                "enabled": True,
                "commercial_use_confirmed": True,
                "auth": "bearer",
                "daily_quota": 10,
                "keys": [{"label": "b1", "env": "B1"}],
            },
        },
    }
    base.update(overrides)
    return base


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("A1", "secret-a1")
    monkeypatch.setenv("A2", "secret-a2")
    monkeypatch.setenv("B1", "secret-b1")


@pytest.fixture
def ledger(tmp_path):
    return UsageLedger.load(tmp_path / "usage.json", TODAY)


def test_unconfirmed_commercial_use_is_skipped(env, ledger):
    c = cfg()
    c["providers"]["alpha"]["commercial_use_confirmed"] = False
    rot = Rotator(c, ledger)
    assert [k.provider for k in rot.candidates(NOW)] == ["beta"]
    assert "commercial_use_confirmed" in rot.skip_reasons()["alpha"]


def test_keys_missing_from_env_are_skipped(monkeypatch, ledger):
    monkeypatch.delenv("A1", raising=False)
    monkeypatch.delenv("A2", raising=False)
    monkeypatch.setenv("B1", "secret-b1")
    rot = Rotator(cfg(), ledger)
    assert [k.label for k in rot.candidates(NOW)] == ["b1"]


def test_quota_exhaustion_advances_to_next_key_then_next_provider(env, ledger):
    rot = Rotator(cfg(), ledger)
    tried = []

    def call(key, _cfg):
        tried.append(key.ident)
        if key.provider == "alpha":
            raise QuotaExhausted()
        return "image-bytes"

    result, key = rot.generate(call, now=NOW)
    assert result == "image-bytes"
    assert key.ident == "beta:b1"
    assert tried == ["alpha:a1", "alpha:a2", "beta:b1"]


def test_rate_limited_key_cools_down_and_is_skipped_next_pass(env, ledger):
    rot = Rotator(cfg(), ledger)

    def limited_then_ok(key, _cfg):
        if key.ident == "alpha:a1":
            raise RateLimited()
        return "ok"

    rot.generate(limited_then_ok, now=NOW)
    assert ledger.is_cooling("alpha:a1", NOW)
    assert "alpha:a1" not in [k.ident for k in rot.candidates(NOW)]


def test_daily_quota_retires_a_key(env, ledger):
    rot = Rotator(cfg(), ledger)
    ledger.counts["alpha:a1"] = 2          # quota is 2
    assert "alpha:a1" not in [k.ident for k in rot.candidates(NOW)]


def test_successful_use_is_counted_and_persisted(env, tmp_path):
    path = tmp_path / "usage.json"
    led = UsageLedger.load(path, TODAY)
    rot = Rotator(cfg(), led)
    rot.generate(lambda k, c: "ok", now=NOW)

    reloaded = UsageLedger.load(path, TODAY)
    assert reloaded.used("alpha:a1") == 1


def test_ledger_resets_counts_on_a_new_day(env, tmp_path):
    path = tmp_path / "usage.json"
    led = UsageLedger.load(path, TODAY)
    led.counts["alpha:a1"] = 5
    led.mark_exhausted("alpha:a1")
    led.save()

    tomorrow = UsageLedger.load(path, date(2026, 8, 11))
    assert tomorrow.used("alpha:a1") == 0
    assert tomorrow.exhausted == []


def test_keyless_provider_yields_one_anonymous_candidate(ledger):
    c = {
        "rotation": {"order": ["open"]},
        "providers": {
            "open": {"enabled": True, "commercial_use_confirmed": True, "auth": "none", "keys": []},
        },
    }
    rot = Rotator(c, ledger)
    keys = list(rot.candidates(NOW))
    assert len(keys) == 1 and keys[0].label == "anonymous"


def test_total_failure_reports_why_each_provider_was_skipped(env, ledger):
    c = cfg()
    c["providers"]["beta"]["enabled"] = False
    rot = Rotator(c, ledger)

    with pytest.raises(NoProviderAvailable) as exc:
        rot.generate(lambda k, _c: (_ for _ in ()).throw(QuotaExhausted()), now=NOW)

    msg = str(exc.value)
    assert "quota exhausted" in msg
    assert "disabled" in msg


def test_unexpected_errors_advance_but_stay_visible(env, ledger):
    rot = Rotator(cfg(), ledger)

    def flaky(key, _cfg):
        if key.provider == "alpha":
            raise ValueError("malformed response")
        return "ok"

    result, key = rot.generate(flaky, now=NOW)
    assert result == "ok" and key.provider == "beta"


def test_a_refusal_reason_survives_into_the_final_error(env, ledger):
    """The second half of "stay visible", and the reason build_request can refuse
    a provider by raising: if nothing succeeds, the message has to reach the
    human. It ends up in the record's qa_notes via generate_for_record, which is
    the only place anyone would read why no image was produced.
    """
    rot = Rotator(cfg(), ledger)

    def refuse(key, _cfg):
        raise RuntimeError(f"{key.provider}: no 5:6 aspect ratio, nothing was called")

    with pytest.raises(NoProviderAvailable) as exc:
        rot.generate(refuse, now=NOW)
    assert "no 5:6 aspect ratio" in str(exc.value)
    assert "RuntimeError" in str(exc.value), "the exception type is part of the clue"


def test_shipped_config_is_inert_until_verified(env, ledger):
    """The committable template must not generate anything -- a fresh clone is
    safe by default. Asserted against providers.example.json, not providers.json:
    the live copy is the operator's, and once they confirm a licence it is
    supposed to stop being inert."""
    rot = Rotator.from_files("config/providers.example.json", ledger.path, today=TODAY)
    assert list(rot.candidates(NOW)) == []
    assert all(rot.skip_reasons().values())


# --- keys stay out of the config file -----------------------------------

def test_key_shorthand_names_an_env_var(monkeypatch, ledger):
    """`"keys": ["NAME"]` is the short spelling. The object form invites pasting
    the secret into a second field, so the plain string exists to make the safe
    thing the easy thing."""
    monkeypatch.setenv("SHORTHAND_KEY", "secret")
    c = {
        "rotation": {"order": ["alpha"]},
        "providers": {"alpha": {
            "enabled": True, "commercial_use_confirmed": True,
            "auth": "bearer", "keys": ["SHORTHAND_KEY"],
        }},
    }
    keys = list(Rotator(c, ledger).candidates(NOW))
    assert [k.env_var for k in keys] == ["SHORTHAND_KEY"]
    assert keys[0].secret == "secret"


def test_a_pasted_credential_is_rejected_not_dereferenced():
    """This is the failure that motivated validate_config: a hand-edit put three
    live tokens where the variable names belong, and the rotator raised
    AttributeError from inside key iteration instead of saying what was wrong."""
    with pytest.raises(ConfigError) as exc:
        validate_config({"providers": {"gemini": {"keys": ["AQ.Ab8RN6I56L9eu"]}}})
    msg = str(exc.value)
    assert "GEMINI_API_KEY" in msg          # tells you what to write instead
    assert "environment variable" in msg


def test_a_credential_in_the_auth_field_is_rejected():
    """The token had been pasted as auth: 'Bearer hf_...', which no key check
    would have looked at."""
    with pytest.raises(ConfigError) as exc:
        validate_config({"providers": {"hf": {"auth": "Bearer hf_zQHYOLbc"}}})
    assert "auth" in str(exc.value)


def test_every_malformed_entry_is_reported_at_once():
    """One run should fix the whole file, not surface the next problem after
    each edit."""
    with pytest.raises(ConfigError) as exc:
        validate_config({"providers": {
            "a": {"keys": ["sk_live_aaaa"]},
            "b": {"keys": [{"label": "x", "env": "hf_bbbb"}]},
            "c": {"keys": [42]},
        }})
    msg = str(exc.value)
    assert "a.keys[0]" in msg and "b.keys[0].env" in msg and "c.keys[0]" in msg


def test_valid_env_names_and_object_form_both_pass():
    validate_config({"providers": {
        "a": {"auth": "bearer", "keys": ["GEMINI_API_KEY", "GEMINI_API_KEY_2"]},
        "b": {"auth": "none", "keys": []},
        "c": {"auth": "api_key", "keys": [{"label": "primary", "env": "X_KEY"}]},
    }})


def test_secret_detector_separates_names_from_tokens():
    for name in ["GEMINI_API_KEY", "A", "HF_TOKEN_2"]:
        assert not looks_like_secret(name), name
    for token in ["sk_ANU2JOCS", "hf_zQHYOLbc", "AQ.Ab8RN6I5", "AIzaSyAbc",
                  "Bearer abc", "lowercase_key", "has-a-dash"]:
        assert looks_like_secret(token), token


def test_live_config_carries_no_credentials():
    """Guards the recurrence of the mistake, on the actual file. Skipped when
    absent -- providers.json is gitignored, so a fresh clone has only the
    template."""
    path = Path("config/providers.json")
    if not path.is_file():
        pytest.skip("no operator-local providers.json")
    validate_config(json.loads(path.read_text()))


def test_enabled_provider_with_an_unset_variable_says_which_one(monkeypatch, ledger):
    """An enabled provider with a confirmed licence and an empty variable yields
    no candidates, which looks like a rotator bug rather than a missing export."""
    monkeypatch.delenv("MISSING_KEY", raising=False)
    c = {
        "rotation": {"order": ["alpha"]},
        "providers": {"alpha": {
            "enabled": True, "commercial_use_confirmed": True,
            "auth": "bearer", "keys": ["MISSING_KEY"],
        }},
    }
    rot = Rotator(c, ledger)
    assert list(rot.candidates(NOW)) == []
    assert "MISSING_KEY" in rot.skip_reasons()["alpha"]
