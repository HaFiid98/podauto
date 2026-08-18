from datetime import datetime, timezone

import pytest

from podauto.models import Gate, IdeaType, PipelineStatus, Record, ScoreConfidence, SourcePlatform, TmStatus
from podauto.prompts import (
    GROUND_CANDIDATES,
    MAX_TEXT_WORDS,
    StyleDatabase,
    affinity,
    build_prompt,
    derive_text,
    ground_for_style,
    output_suffix,
    select_styles,
    synthesize,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def db() -> StyleDatabase:
    return StyleDatabase.load("config/styles.json")


def make(idea_type=IdeaType.READY_SHIRT, text="Pickleball Dad Retro Sunset",
         niche="Pickleball Dad", **kw) -> Record:
    r = Record(
        id="x", timestamp=NOW.isoformat(), source_url="https://example.com",
        source_platform=SourcePlatform.AMAZON, idea_type=idea_type,
        niche=niche, raw_title_or_text=text,
    )
    # Default to a high-confidence pass so variation count is not the thing
    # under test unless a test says so.
    r.gate = kw.get("gate", Gate.PASS)
    r.score_confidence = kw.get("confidence", ScoreConfidence.HIGH)
    return r


# --- style database -----------------------------------------------------

def test_all_twelve_styles_load(db):
    assert len(db.styles) == 12


def test_every_style_has_a_lettering_fragment(db):
    """Model renders the text now, so each style must describe how it looks."""
    missing = [s["id"] for s in db.styles.values() if not s.get("lettering")]
    assert missing == []


def test_meta_declares_model_rendered_strategy(db):
    assert "model_rendered" in db.meta["note"]


def test_meta_carries_the_spelling_caveat(db):
    """The tradeoff should be documented where someone editing styles sees it."""
    assert "text_rendering_caveat" in db.meta
    assert "not guaranteed" in db.meta["text_rendering_caveat"]


# --- prompt hygiene -----------------------------------------------------

def test_no_midjourney_negative_syntax(db):
    """'--no' is Midjourney-only. Flux/SDXL may read it as positive prompt text,
    which would ask for exactly what it means to exclude."""
    assert "--no" not in output_suffix("#FF00FF")
    prompt = build_prompt(db.get("retro_70s_sunset"), make(), derive_text(make()))
    assert "--" not in prompt


def test_no_black_background_contradiction(db):
    """Original spec said solid black here and transparent PNG elsewhere."""
    suffix = output_suffix("#FF00FF")
    assert "black background" not in suffix.lower()


def test_the_suffix_names_the_ground_as_both_a_fill_and_an_exclusion(db):
    """Naming the ground colour is only half of it. "isolated on a plain uniform
    background" -- the previous wording -- let the model reuse a palette colour as
    the ground, which is what makes removal cut into the design."""
    suffix = output_suffix("#FF00FF")
    assert "#FF00FF" in suffix
    assert "solid magenta background" in suffix
    assert "no magenta anywhere in the artwork" in suffix


def test_every_style_gets_a_ground_its_own_palette_does_not_name(db):
    """The collision this prevents was measured, not hypothesized: 2 of the first
    3 real generations reused a palette colour as the ground."""
    for style in db.styles.values():
        ground = ground_for_style(style)
        words = dict(GROUND_CANDIDATES)[ground]
        text = " ".join(str(style.get(k, "")) for k in
                        ("palette_hint", "graphic", "lettering")).lower()
        clashes = [w for w in words if w in text]
        assert not clashes, f"{style['id']} asks for {ground} but mentions {clashes}"


def test_a_magenta_palette_does_not_get_a_magenta_ground(db):
    """y2k_cyber_vector's own palette_hint says magenta, so the default chroma
    would be indistinguishable from the artwork."""
    assert ground_for_style(db.get("y2k_cyber_vector")) != "#FF00FF"


def test_a_style_naming_two_chroma_families_still_gets_a_usable_ground(db):
    """cottagecore names both a pink (dusty rose) and a green (sage)."""
    assert ground_for_style(db.get("cottagecore_botanical")) == "#00FFFF"


def test_no_style_writes_its_own_background_instruction(db):
    """The ground is appended once, by output_suffix, and keyed out by printready.
    A style that also names a background puts two contradictory ground
    instructions in one prompt, and the model resolves that by painting a
    full-bleed rectangle no removal can save. Three styles used to do this
    ("one colour on transparent", "on flat ground")."""
    banned = ("background", "on transparent", "flat ground", "backdrop colour")
    offenders = {
        s["id"]: [b for b in banned
                  if b in f"{s.get('graphic', '')} {s.get('lettering', '')} "
                           f"{s.get('palette_hint', '')}".lower()]
        for s in db.styles.values()
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_the_ground_hint_is_recorded_on_every_variation(db):
    """printready reads it to report that the model ignored the instruction. A
    hint the prompt used but the record did not keep would make that undetectable."""
    rec = synthesize(make(), db, 3)
    for var in rec.variations:
        assert var.ground_hint in dict(GROUND_CANDIDATES)
        assert var.ground_hint in var.graphic_prompt


def test_prompt_excludes_garment_and_mockup(db):
    prompt = build_prompt(db.get("flat_mascot"), make(), derive_text(make()))
    assert "no garment" in prompt and "no mockup" in prompt


# --- text handling ------------------------------------------------------

def test_text_is_quoted_and_stated_exactly(db):
    prompt = build_prompt(db.get("bold_typographic"), make(), derive_text(make()))
    assert '"PICKLEBALL DAD RETRO SUNSET"' in prompt
    assert "spelled exactly as written" in prompt


def test_long_text_is_truncated_and_flagged():
    r = make(text="This Is A Very Long Shirt Slogan That Will Not Render Well")
    spec = derive_text(r)
    assert len(spec.line_1.split()) == MAX_TEXT_WORDS
    assert spec.truncated is True


def test_short_text_is_not_flagged():
    assert derive_text(make(text="Pickleball Dad")).truncated is False


def test_case_follows_the_style(db):
    r = make()
    # minimal_line_art is configured lowercase, distressed_collegiate uppercase
    lower = build_prompt(db.get("minimal_line_art"), r, derive_text(r))
    upper = build_prompt(db.get("distressed_collegiate"), r, derive_text(r))
    assert '"pickleball dad retro sunset"' in lower
    assert '"PICKLEBALL DAD RETRO SUNSET"' in upper


# --- reversibility ------------------------------------------------------

def test_text_spec_is_populated_even_though_nothing_reads_it(db):
    """The hook that makes switching strategies a config change, not a migration."""
    r = synthesize(make(), db, configured_variations=2)
    spec = r.variations[0].text_spec
    assert spec["line_1"] == "Pickleball Dad Retro Sunset"
    assert spec["strategy"] == "model_rendered"
    assert spec["arrangement"] and spec["case"] and spec["position"]


# --- variation selection ------------------------------------------------

def test_ready_shirt_leads_with_an_aesthetic_match(db):
    r = make(idea_type=IdeaType.READY_SHIRT, niche="Vintage Fishing Outdoors")
    styles = select_styles(r, db, count=3)
    assert len(styles) == 3
    # woodcut_engraving and retro_70s_sunset both tag 'outdoors'
    assert styles[0] in {"woodcut_engraving", "retro_70s_sunset"}


def test_trend_idea_uses_the_selected_pool_in_order(db):
    r = make(idea_type=IdeaType.TREND_IDEA, niche="Cat Meme")
    styles = select_styles(r, db, count=2, selected=["y2k_cyber_vector", "kawaii_pastel"])
    assert styles == ["y2k_cyber_vector", "kawaii_pastel"]


def test_selected_pool_is_respected_for_ready_shirt_fill(db):
    r = make(niche="Pickleball Dad")
    styles = select_styles(r, db, count=3, selected=["kawaii_pastel", "grunge_punk"])
    assert len(styles) == 3
    assert "kawaii_pastel" in styles and "grunge_punk" in styles


def test_no_duplicate_styles(db):
    r = make()
    styles = select_styles(r, db, count=5)
    assert len(styles) == len(set(styles))


# --- style affinity -----------------------------------------------------

def test_niche_match_is_prefix_tolerant(db):
    """Exact tag equality missed the obvious fits: 'gardening' is not 'garden',
    so Gardening Grandma drew a distressed collegiate athletic look while
    cottagecore_botanical -- tagged garden/plant/farmhouse/mom/tea, written for
    precisely that niche -- scored zero."""
    assert db.match_for_niche("Gardening Grandma") == "cottagecore_botanical"


def test_affinity_matches_a_longer_token_against_a_shorter_tag(db):
    """The direction that was silently failing in production."""
    assert affinity({"gardening"}, {"good_for": ["garden"]}) == 1
    assert affinity({"grandmas"}, {"good_for": ["grandma"]}) == 1


def test_affinity_matches_a_shorter_token_against_a_longer_tag():
    """Prefix tolerance runs both ways -- the concept text is not privileged."""
    assert affinity({"garden"}, {"good_for": ["gardening"]}) == 1


def test_affinity_floor_blocks_short_stems():
    """The 4-character floor is the part carrying false-positive risk: without
    it 'cat' prefix-matches into 'category' and every three-letter token starts
    scoring against unrelated tags."""
    assert affinity({"cat"}, {"good_for": ["category"]}) == 0
    assert affinity({"category"}, {"good_for": ["cat"]}) == 0
    assert affinity({"art"}, {"good_for": ["artisan"]}) == 0
    # Equality is exempt from the floor -- a short tag still matches itself.
    assert affinity({"cat"}, {"good_for": ["cat"]}) == 1


def test_affinity_counts_each_tag_at_most_once():
    """Two tokens hitting one tag is one point, not two -- otherwise a repeated
    stem would outscore a style that genuinely fits on several tags."""
    assert affinity({"garden", "gardening"}, {"good_for": ["garden"]}) == 1


def test_affinity_ignores_unrelated_tags():
    assert affinity({"pickleball", "dad"}, {"good_for": ["occult", "nautical"]}) == 0


def test_match_for_niche_honours_exclusions(db):
    """select_styles passes its caution ban through to the lead pick, so the
    exclusion has to be respected here or the ban leaks at position 0."""
    assert db.match_for_niche("Gardening Grandma",
                              exclude={"cottagecore_botanical"}) != "cottagecore_botanical"


# --- caution styles and pool coverage -----------------------------------

def test_flagged_record_avoids_caution_styles(db):
    """distressed_collegiate's own caution says never to pair it with a
    'Property of' construction. The first smoke run handed it exactly that
    record as the lead style -- a warning no code reads is a comment."""
    r = make(niche="School Spirit", text="Property Of Riverside Wrestling")
    r.tm_status = TmStatus.NEEDS_REVIEW
    styles = select_styles(r, db, count=3)
    assert "distressed_collegiate" not in styles


def test_unflagged_record_may_still_use_caution_styles(db):
    """The filter keys on the triage flag, not on the style being risky per se.

    Asserted as a contrast on one record rather than as reachability: taking the
    whole pool proves only that the style exists in it. Flipping tm_status is
    what isolates the filter.
    """
    r = make(niche="Wrestling Team Practice", text="Wrestling Practice Squad")

    r.tm_status = TmStatus.NEEDS_REVIEW
    flagged = set(select_styles(r, db, count=12))
    r.tm_status = TmStatus.NO_FLAGS_FOUND
    clean = set(select_styles(r, db, count=12))

    assert "distressed_collegiate" not in flagged
    assert "distressed_collegiate" in clean


def test_style_selection_varies_across_records(db):
    """21 variations drew from 3 of 12 styles before rotation: every record
    started at the top of the pool. That is visibly repetitive on a storefront
    and wastes the library."""
    used = set()
    for i in range(8):
        r = make(niche=f"Niche {i}", text=f"Concept Number {i}")
        r.dedupe_key = f"key{i}"
        used |= set(select_styles(r, db, count=3))
    assert len(used) > 3, f"only {len(used)} distinct styles across 8 records"


def test_style_selection_is_stable_for_the_same_record(db):
    """Re-running a stage must not reshuffle styles -- the pipeline is
    re-runnable by design, so selection is keyed on dedupe_key, not hash()."""
    r = make()
    r.dedupe_key = "stable-key"
    assert select_styles(r, db, count=3) == select_styles(r, db, count=3)


def test_explicit_style_list_is_honoured_verbatim(db):
    """An operator-supplied list is an instruction: no rotation, no filtering."""
    r = make(niche="School Spirit", text="Property Of Riverside")
    r.tm_status = TmStatus.NEEDS_REVIEW
    styles = select_styles(r, db, count=2,
                           selected=["distressed_collegiate", "kawaii_pastel"])
    assert styles == ["distressed_collegiate", "kawaii_pastel"]


# --- integration with the scorer ---------------------------------------

def test_high_confidence_pass_gets_all_configured_variations(db):
    r = synthesize(make(), db, configured_variations=4)
    assert len(r.variations) == 4
    assert r.pipeline_status is PipelineStatus.PROMPTED


def test_partial_confidence_pass_gets_one_variation(db):
    """Under-evidenced ideas take the cheap path -- quota follows the evidence."""
    r = make(confidence=ScoreConfidence.PARTIAL)
    assert len(synthesize(r, db, configured_variations=4).variations) == 1


def test_held_record_generates_nothing(db):
    r = make(gate=Gate.HOLD, confidence=ScoreConfidence.LOW)
    out = synthesize(r, db, configured_variations=4)
    assert out.variations == []
    assert out.pipeline_status is not PipelineStatus.PROMPTED


def test_rejected_record_generates_nothing(db):
    r = make(gate=Gate.REJECT, confidence=ScoreConfidence.HIGH)
    assert synthesize(r, db, configured_variations=4).variations == []


def test_unknown_style_raises(db):
    with pytest.raises(KeyError):
        db.get("no_such_style")
