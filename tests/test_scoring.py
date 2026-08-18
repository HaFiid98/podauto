from datetime import datetime, timedelta, timezone

import pytest

from podauto.models import Gate, IdeaType, Record, ScoreConfidence, SourcePlatform
from podauto.scoring import score, variations_allowed

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


def make(idea_type=IdeaType.READY_SHIRT, **kw) -> Record:
    defaults = dict(
        id="x",
        timestamp=NOW.isoformat(),
        source_url="https://example.com/x",
        source_platform=SourcePlatform.AMAZON,
        idea_type=idea_type,
        niche="Pickleball Dad",
        raw_title_or_text="Pickleball Dad Retro Sunset",
    )
    defaults.update(kw)
    return Record(**defaults)


def test_full_data_ready_shirt_scores_high_confidence():
    r = score(make(
        trend_velocity=80,
        bsr_rank=40_000,
        bsr_category="Clothing",
        bsr_captured_at=NOW.isoformat(),
        listings_count=30,
    ), now=NOW)
    assert r.score_confidence is ScoreConfidence.HIGH
    # 80*.40 + 100*.35 + 100*.25 = 92
    assert r.hitter_score == pytest.approx(92.0)
    assert r.gate is Gate.PASS


def test_trend_idea_is_not_penalised_for_absent_bsr():
    """The original model scored a missing BSR as 25/100, capping trend ideas.

    Absent is not weak: the trend_idea profile has no bsr term at all.
    """
    r = score(make(
        idea_type=IdeaType.TREND_IDEA,
        trend_velocity=90,
        listings_count=30,
    ), now=NOW)
    assert "bsr" not in r.score_components["components"]
    assert r.score_confidence is ScoreConfidence.HIGH
    # 90*.60 + 100*.40 = 94
    assert r.hitter_score == pytest.approx(94.0)
    assert r.gate is Gate.PASS


def test_bsr_only_never_auto_passes():
    """One signal out of three cannot clear the gate, however good it looks."""
    r = score(make(bsr_rank=10_000, bsr_category="Clothing",
                   bsr_captured_at=NOW.isoformat()), now=NOW)
    assert r.hitter_score == pytest.approx(100.0)
    assert r.score_confidence is ScoreConfidence.LOW
    assert r.gate is Gate.HOLD
    assert variations_allowed(r, configured=4) == 0


def test_partial_confidence_passes_at_higher_bar_on_cheap_path():
    r = score(make(idea_type=IdeaType.TREND_IDEA, trend_velocity=90), now=NOW)
    assert r.score_confidence is ScoreConfidence.PARTIAL
    assert r.gate is Gate.PASS
    # Under-evidenced pass gets one variation, not the configured X.
    assert variations_allowed(r, configured=4) == 1


def test_partial_confidence_below_bar_is_held_not_rejected():
    r = score(make(idea_type=IdeaType.TREND_IDEA, trend_velocity=60), now=NOW)
    assert r.score_confidence is ScoreConfidence.PARTIAL
    assert r.gate is Gate.HOLD


def test_weak_signal_is_rejected():
    r = score(make(idea_type=IdeaType.TREND_IDEA, trend_velocity=20,
                   listings_count=900), now=NOW)
    # 20*.60 + 10*.40 = 16
    assert r.hitter_score == pytest.approx(16.0)
    assert r.gate is Gate.REJECT


def test_stale_bsr_is_dropped_with_a_note():
    old = (NOW - timedelta(days=90)).isoformat()
    r = score(make(trend_velocity=80, bsr_rank=10_000,
                   bsr_category="Clothing", bsr_captured_at=old,
                   listings_count=30), now=NOW)
    assert "bsr" not in r.score_components["components"]
    assert any("90d old" in n for n in r.score_components["missing"])
    assert r.score_confidence is ScoreConfidence.PARTIAL


def test_missing_bsr_category_degrades_but_keeps_component():
    r = score(make(trend_velocity=80, bsr_rank=40_000,
                   bsr_captured_at=NOW.isoformat(), listings_count=30), now=NOW)
    assert "bsr" in r.score_components["degraded"]
    assert r.score_confidence is ScoreConfidence.HIGH


def test_no_signals_at_all_holds_and_reports_everything_missing():
    r = score(make(idea_type=IdeaType.TREND_IDEA), now=NOW)
    assert r.hitter_score == 0.0
    assert r.gate is Gate.HOLD
    assert r.score_components["weight_present"] == 0.0
    assert len(r.score_components["missing"]) == 2


def test_missing_inputs_are_always_named():
    r = score(make(bsr_rank=40_000, bsr_category="Clothing",
                   bsr_captured_at=NOW.isoformat()), now=NOW)
    missing = " ".join(r.score_components["missing"])
    assert "trend_velocity" in missing
    assert "listings_count" in missing
