from datetime import date

import pytest

from podauto.models import IdeaType, Record, SourcePlatform, TmStatus
from podauto.triage import Denylist, triage_record


@pytest.fixture(scope="module")
def dl() -> Denylist:
    return Denylist.load("config/denylist.json")


def make(text: str, niche: str = "Generic") -> Record:
    return Record(
        id="x", timestamp="2026-08-10T00:00:00+00:00",
        source_url="https://example.com", source_platform=SourcePlatform.AMAZON,
        idea_type=IdeaType.READY_SHIRT, niche=niche, raw_title_or_text=text,
    )


# --- tier 1 -------------------------------------------------------------

def test_franchise_is_hard_blocked(dl):
    r = dl.check("Vintage Star Wars Dad Shirt")
    assert r.status is TmStatus.BLOCKED
    assert "star wars" in r.matched_terms


def test_brand_is_hard_blocked(dl):
    assert dl.check("Just Nike Vibes").status is TmStatus.BLOCKED


def test_slogan_is_hard_blocked(dl):
    assert dl.check("May The Force Be With You Fishing").status is TmStatus.BLOCKED


def test_leetspeak_near_miss_still_blocks(dl):
    """normalize_text folds D1sn3y -> disney before matching."""
    r = dl.check("D1sn3y Dad Life")
    assert r.status is TmStatus.BLOCKED
    assert "disney" in r.matched_terms


def test_longest_match_wins(dl):
    """'star wars' should be reported, not a shorter substring of it."""
    r = dl.check("Star Wars Marathon")
    assert r.matched_terms == ["star wars"]


def test_blocked_never_becomes_needs_review(dl):
    """Tier 1 short-circuits: a hard block must not be softened by later tiers."""
    r = dl.check("Property of Disney Athletics Est. 1985")
    assert r.status is TmStatus.BLOCKED


# --- tier 2 -------------------------------------------------------------

def test_ambiguous_word_alone_passes(dl):
    """The whole point of tier 2: 'frozen' by itself is ordinary English."""
    assert dl.check("Frozen Pipes Plumber Life").status is TmStatus.NO_FLAGS_FOUND


def test_ambiguous_word_with_companion_needs_review(dl):
    r = dl.check("Frozen Let It Go Snowflake Design")
    assert r.status is TmStatus.NEEDS_REVIEW
    assert "frozen" in r.matched_terms


def test_apple_pie_is_fine_but_apple_iphone_is_not(dl):
    assert dl.check("Apple Pie Baking Season").status is TmStatus.NO_FLAGS_FOUND
    assert dl.check("Apple iPhone Repair Tech").status is TmStatus.NEEDS_REVIEW


def test_high_false_positive_words_pass_alone(dl):
    """These are common English; blocking them outright would kill real designs."""
    for text in ["Take A Guess Trivia Night", "Mind The Gap Traveler",
                 "Life Coach In Training", "Champion Of Naps"]:
        assert dl.check(text).status is TmStatus.NO_FLAGS_FOUND, text


def test_tier_2_is_review_not_block(dl):
    """Ambiguity is exactly what a human should judge, so it must not auto-block."""
    assert dl.check("Supreme Box Logo Style").status is TmStatus.NEEDS_REVIEW


# --- tier 3 -------------------------------------------------------------

def test_property_of_pattern(dl):
    r = dl.check("Property Of Riverside Wrestling")
    assert r.status is TmStatus.NEEDS_REVIEW
    assert any("property of" in t.lower() for t in r.matched_terms)


def test_athletic_dept_pattern(dl):
    assert dl.check("Pickleball Athletic Dept").status is TmStatus.NEEDS_REVIEW


def test_est_year_pattern(dl):
    assert dl.check("Fishing Club Est. 1974").status is TmStatus.NEEDS_REVIEW


def test_trademark_symbol_pattern(dl):
    assert dl.check("Best Dad Ever®").status is TmStatus.NEEDS_REVIEW


def test_pattern_runs_on_raw_text_not_normalized(dl):
    """Normalization strips the symbols tier 3 looks for, so patterns must see raw."""
    assert dl.check("Coffee Lover™").status is TmStatus.NEEDS_REVIEW


# --- clean path ---------------------------------------------------------

def test_clean_concept_finds_no_flags(dl):
    r = dl.check("Pickleball Dad Retro Sunset")
    assert r.status is TmStatus.NO_FLAGS_FOUND
    assert r.matched_terms == []
    assert r.reason == ""


def test_there_is_no_cleared_status():
    """A keyword screen cannot clear a mark. The value must not exist."""
    assert not hasattr(TmStatus, "CLEARED")
    assert "cleared" not in {s.value for s in TmStatus}


# --- record integration -------------------------------------------------

def test_triage_record_populates_reason_for_fast_review(dl):
    rec = triage_record(make("Frozen Let It Go Winter"), dl)
    assert rec.tm_status is TmStatus.NEEDS_REVIEW
    # Review needs the matched term AND why, or each item costs five minutes.
    assert "frozen" in rec.tm_flag_reason
    assert "tier2" in rec.tm_flag_reason
    assert rec.matched_terms


def test_triage_record_screens_niche_too(dl):
    rec = triage_record(make("Cute Character Tee", niche="Pokemon Fan Art"), dl)
    assert rec.tm_status is TmStatus.BLOCKED


def test_triage_record_screens_generated_listing_fields(dl):
    rec = make("Generic Sports Design")
    rec.listing.title = "Yankees Baseball Fan Retro T-Shirt"
    assert triage_record(rec, dl).tm_status is TmStatus.BLOCKED


# --- list hygiene -------------------------------------------------------

def test_staleness_is_measurable(dl):
    assert dl.staleness_days(date(2026, 9, 10)) == 31
    assert dl.is_stale(date(2026, 9, 10)) is True
    assert dl.is_stale(date(2026, 8, 20)) is False


def test_shipped_list_declares_it_is_not_exhaustive(dl):
    """Guards against the list being mistaken for complete coverage."""
    assert "not exhaustive" in dl.meta["note"].lower()
    assert "does not clear" in dl.meta["what_this_is"].lower() or \
           "not" in dl.meta["what_this_is"].lower()
