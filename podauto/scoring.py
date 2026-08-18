"""Hitter score.

Two corrections to the original model, both load-bearing:

1. Weights renormalize over components that actually fired. The original spec
   needs three inputs but the collector row only carries BSR. Defaulting the
   missing 65% to zero means nothing ever clears the gate; defaulting to 50
   drags everything to mid-range and the gate stops discriminating. Neither is
   fixable by tuning, so instead we score on what we have and report what we
   did not have.

2. Separate profiles per idea_type. A trend_idea has no BSR because it is not
   a product yet -- that is an ABSENT signal, not a weak one. Scoring it 25/100
   put a hard ceiling on 35% of the score for exactly the category with the
   earliest market entry, which works against the point of tracking trends.

Pure functions, no I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .models import Gate, IdeaType, Record, ScoreConfidence

# Fraction of total weight that must be present for each confidence tier.
HIGH_CONFIDENCE_WEIGHT = 0.95
PARTIAL_CONFIDENCE_WEIGHT = 0.50

# A BSR reading older than this is a stale claim about current demand.
BSR_MAX_AGE_DAYS = 30

PROFILES: dict[IdeaType, dict[str, float]] = {
    IdeaType.READY_SHIRT: {"trend": 0.40, "bsr": 0.35, "competition": 0.25},
    # No bsr term at all -- not a zero, simply not part of the model here.
    IdeaType.TREND_IDEA: {"trend": 0.60, "competition": 0.40},
}

# Gate thresholds by confidence. A full-data 65 and a BSR-only 65 are not the
# same claim, so they do not clear the same bar.
THRESHOLDS: dict[ScoreConfidence, tuple[float, float]] = {
    # (pass_at, reject_below)
    ScoreConfidence.HIGH: (65.0, 65.0),
    ScoreConfidence.PARTIAL: (75.0, 50.0),
    ScoreConfidence.LOW: (101.0, 35.0),   # never auto-passes; holds or rejects
}


def bsr_points(rank: int) -> float:
    if rank < 50_000:
        return 100.0
    if rank < 150_000:
        return 75.0
    if rank < 500_000:
        return 50.0
    return 25.0


def competition_points(listings: int) -> float:
    if listings < 50:
        return 100.0
    if listings < 200:
        return 70.0
    if listings < 500:
        return 40.0
    return 10.0


def _age_days(iso_ts: str, now: datetime) -> float | None:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds() / 86400.0


def score(record: Record, now: datetime | None = None) -> Record:
    """Populate hitter_score, score_confidence, score_components, gate."""
    now = now or datetime.now(timezone.utc)
    profile = PROFILES[record.idea_type]
    components: dict[str, dict] = {}
    notes: list[str] = []

    if "trend" in profile and record.trend_velocity is not None:
        components["trend"] = {
            "points": max(0.0, min(100.0, float(record.trend_velocity))),
            "weight": profile["trend"],
        }
    elif "trend" in profile:
        notes.append("trend_velocity absent: no Google Trends / social signal collected")

    if "bsr" in profile:
        if record.bsr_rank is None:
            notes.append("bsr_rank absent")
        else:
            age = _age_days(record.bsr_captured_at, now) if record.bsr_captured_at else None
            if age is not None and age > BSR_MAX_AGE_DAYS:
                notes.append(f"bsr_rank dropped: reading is {age:.0f}d old (max {BSR_MAX_AGE_DAYS}d)")
            else:
                comp = {"points": bsr_points(record.bsr_rank), "weight": profile["bsr"]}
                if not record.bsr_category:
                    # Rank is category-relative; without the category the band
                    # is a guess. Kept, but flagged rather than trusted.
                    comp["degraded"] = "bsr_category missing -- band unreliable"
                if age is None and record.bsr_captured_at:
                    comp["degraded"] = "bsr_captured_at unparseable"
                elif record.bsr_captured_at is None:
                    comp["degraded"] = "bsr_captured_at missing -- staleness unknown"
                components["bsr"] = comp

    if "competition" in profile and record.listings_count is not None:
        components["competition"] = {
            "points": competition_points(record.listings_count),
            "weight": profile["competition"],
        }
    elif "competition" in profile:
        notes.append("listings_count absent: no Amazon saturation scrape")

    present_weight = sum(c["weight"] for c in components.values())
    total_weight = sum(profile.values())

    if present_weight == 0:
        record.hitter_score = 0.0
        record.score_confidence = ScoreConfidence.LOW
        record.score_components = {
            "profile": record.idea_type.value,
            "components": {},
            "missing": notes,
            "weight_present": 0.0,
        }
        record.gate = Gate.HOLD
        return record

    # Renormalize: each present component's weight is rescaled so the ones we
    # actually have sum to 1.0.
    raw = sum(c["points"] * c["weight"] for c in components.values())
    record.hitter_score = round(raw / present_weight, 1)

    ratio = present_weight / total_weight
    if ratio >= HIGH_CONFIDENCE_WEIGHT:
        confidence = ScoreConfidence.HIGH
    elif ratio >= PARTIAL_CONFIDENCE_WEIGHT:
        confidence = ScoreConfidence.PARTIAL
    else:
        confidence = ScoreConfidence.LOW
    record.score_confidence = confidence

    record.score_components = {
        "profile": record.idea_type.value,
        "components": components,
        "missing": notes,
        "weight_present": round(ratio, 3),
        "degraded": [k for k, v in components.items() if "degraded" in v],
    }

    pass_at, reject_below = THRESHOLDS[confidence]
    if record.hitter_score >= pass_at:
        record.gate = Gate.PASS
    elif record.hitter_score < reject_below:
        record.gate = Gate.REJECT
    else:
        # Plausible but under-evidenced. Parked for the missing signal rather
        # than thrown away.
        record.gate = Gate.HOLD

    return record


def variations_allowed(record: Record, configured: int) -> int:
    """Cheap path for under-evidenced passes: one variation, not X.

    Keeps image quota pointed at the ideas we have real evidence for.
    """
    if record.gate is not Gate.PASS:
        return 0
    if record.score_confidence is ScoreConfidence.HIGH:
        return configured
    return 1
