import pytest

from podauto.content_policy import ContentPolicy
from podauto.listing import (
    BULLET_MAX,
    TITLE_MAX,
    TITLE_MIN,
    build_brand,
    build_bullets,
    build_title,
    check_variation_similarity,
    generate,
    validate,
)
from podauto.models import ContentPolicyStatus, IdeaType, Listing, PipelineStatus, Record, SourcePlatform, StyleVariation


def make(niche="Pickleball Dad", text="Pickleball Dad Retro Sunset") -> Record:
    return Record(
        id="x", timestamp="2026-08-10T00:00:00+00:00",
        source_url="https://example.com", source_platform=SourcePlatform.AMAZON,
        idea_type=IdeaType.READY_SHIRT, niche=niche, raw_title_or_text=text,
    )


# --- title bounds -------------------------------------------------------

@pytest.mark.parametrize("niche,text", [
    ("Pickleball Dad", "Pickleball Dad Retro Sunset"),
    ("Cat", "Nap"),                                        # very short input
    ("Vintage Fishing", "Vintage Fishing Legend Angler Fisherman Retro Classic Design"),
    ("Gardening Grandma", "Plant Lady Botanical Wildflower Cottage Garden Dreams"),
    ("BBQ", "Grill Master"),
])
def test_title_always_lands_in_bounds(niche, text):
    title = build_title(make(niche, text), "Vintage 70s Retro Sunset")
    assert TITLE_MIN <= len(title) <= TITLE_MAX, f"{len(title)}: {title!r}"


def test_title_ends_with_tshirt():
    assert build_title(make()).endswith("T-Shirt")


def test_title_does_not_stutter_the_niche():
    """Niche and concept overlap heavily; the word should appear once."""
    title = build_title(make("Pickleball Dad", "Pickleball Dad Retro"))
    assert title.lower().split().count("pickleball") == 1


def test_validate_rejects_a_title_one_char_short():
    """The original spec's own example was 59 chars against its 60 minimum --
    exactly the off-by-one an assertion catches and a spec reader does not."""
    short = "A" * 59
    issues = validate(Listing(title=short, brand="X Apparel Co",
                              bullet_1="b1", bullet_2="b2"))
    assert any(i.field == "title" for i in issues)


def test_validate_rejects_a_title_one_char_long():
    issues = validate(Listing(title="A" * 81, brand="X Apparel Co",
                              bullet_1="b1", bullet_2="b2"))
    assert any(i.field == "title" for i in issues)


def test_validate_accepts_titles_at_both_edges():
    for length in (TITLE_MIN, TITLE_MAX):
        issues = validate(Listing(title="A" * length, brand="X Apparel Co",
                                  bullet_1="b1", bullet_2="b2"))
        assert not any(i.field == "title" for i in issues), length


# --- brand --------------------------------------------------------------

def test_brand_follows_the_formula():
    assert build_brand(make()).endswith("Apparel Co")


def test_brand_is_not_a_generic_store_name():
    brand = build_brand(make())
    assert brand.lower() not in {"t-shirt store", "apparel co", "shirt shop"}


def test_brand_survives_an_empty_niche():
    assert build_brand(make(niche="", text="Funny Cat Nap")).endswith("Apparel Co")


# --- bullets ------------------------------------------------------------

def test_exactly_two_bullets_within_limit():
    b1, b2 = build_bullets(make())
    assert 0 < len(b1) <= BULLET_MAX
    assert 0 < len(b2) <= BULLET_MAX


def test_bullet_1_targets_audience_and_bullet_2_targets_gifting():
    b1, b2 = build_bullets(make())
    assert "loves" in b1.lower()
    assert "gift" in b2.lower()


def test_bullets_avoid_restricted_claims():
    """Material, print, shipping and quality claims are Amazon's to make."""
    b1, b2 = build_bullets(make())
    joined = f"{b1} {b2}".lower()
    for banned in ("cotton", "shipping", "quality print", "size chart",
                   "machine washable", "sweatshirt"):
        assert banned not in joined, banned


def test_generated_listing_passes_content_policy():
    """The generator must not produce copy its own policy gate would reject."""
    policy = ContentPolicy.load("config/content_policy.json")
    rec, _ = generate(make())
    for field in (rec.listing.title, rec.listing.brand,
                  rec.listing.bullet_1, rec.listing.bullet_2):
        assert policy.check_listing_text(field).status is ContentPolicyStatus.PASS, field


# --- duplicate risk -----------------------------------------------------

def test_near_identical_titles_are_flagged():
    titles = [
        "Pickleball Dad Retro Sunset Vintage Graphic T-Shirt",
        "Pickleball Dad Retro Sunset Vintage Novelty T-Shirt",
    ]
    assert check_variation_similarity(titles)


def test_distinct_titles_are_not_flagged():
    titles = [
        "Pickleball Dad Retro Sunset Classic Graphic T-Shirt",
        "Cottage Garden Wildflower Botanical Grandma Gift T-Shirt",
    ]
    assert check_variation_similarity(titles) == []


# --- integration --------------------------------------------------------

def test_generate_populates_every_field_and_advances_status():
    rec, issues = generate(make())
    assert issues == []
    assert rec.listing.title and rec.listing.brand
    assert rec.listing.bullet_1 and rec.listing.bullet_2
    assert rec.listing.description
    assert rec.pipeline_status is PipelineStatus.LISTED


def test_issues_are_returned_not_raised():
    """A one-char-short listing should reach review flagged, not vanish."""
    rec, issues = generate(make(niche="", text=""))
    assert isinstance(issues, list)
    assert rec.listing is not None


def test_status_does_not_advance_when_issues_exist():
    rec, issues = generate(make(niche="", text=""))
    if issues:
        assert rec.pipeline_status is not PipelineStatus.LISTED


def test_similarity_check_runs_across_variations():
    rec = make()
    rec.variations = [
        StyleVariation(style_id="a", style_name="Retro", graphic_prompt="p"),
        StyleVariation(style_id="b", style_name="Retro", graphic_prompt="p"),
    ]
    _, issues = generate(rec)
    assert any("similar" in str(i) for i in issues)
