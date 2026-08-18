import pytest

from podauto.content_policy import ContentPolicy, check_record
from podauto.models import ContentPolicyStatus, IdeaType, Record, SourcePlatform


@pytest.fixture(scope="module")
def cp() -> ContentPolicy:
    return ContentPolicy.load("config/content_policy.json")


def make(text: str = "Pickleball Dad Retro", niche: str = "Pickleball") -> Record:
    return Record(
        id="x", timestamp="2026-08-10T00:00:00+00:00",
        source_url="https://example.com", source_platform=SourcePlatform.AMAZON,
        idea_type=IdeaType.READY_SHIRT, niche=niche, raw_title_or_text=text,
    )


# --- prohibited categories ---------------------------------------------

def test_profanity_fails(cp):
    r = cp.check_concept("This Shit Is Bananas Design")
    assert r.status is ContentPolicyStatus.FAIL
    assert "profanity" in r.reason


def test_drug_reference_fails(cp):
    assert cp.check_concept("420 Friendly Weed Leaf").status is ContentPolicyStatus.FAIL


def test_hate_content_fails(cp):
    assert cp.check_concept("White Power Eagle").status is ContentPolicyStatus.FAIL


def test_medical_claim_fails(cp):
    assert cp.check_concept("This Tea Cures Cancer").status is ContentPolicyStatus.FAIL


def test_political_slogan_fails(cp):
    assert cp.check_concept("MAGA Rally Shirt").status is ContentPolicyStatus.FAIL


def test_weapon_model_fails(cp):
    assert cp.check_concept("AR 15 Enthusiast").status is ContentPolicyStatus.FAIL


def test_clean_concept_passes(cp):
    r = cp.check_concept("Pickleball Dad Retro Sunset")
    assert r.status is ContentPolicyStatus.PASS
    assert r.violations == []


def test_substring_does_not_false_positive(cp):
    """Word-boundary padding: 'grass' must not trip on a drug term, and
    'assassin' must not trip on profanity."""
    assert cp.check_concept("Grass Fed Farm Life").status is ContentPolicyStatus.PASS
    assert cp.check_concept("Classic Bass Fishing").status is ContentPolicyStatus.PASS


# --- listing field rules ------------------------------------------------

def test_material_claim_is_forbidden(cp):
    r = cp.check_listing_text("Soft Cotton Comfortable Tee", "title")
    assert r.status is ContentPolicyStatus.FAIL
    assert "soft cotton" in r.reason


def test_shipping_claim_is_forbidden(cp):
    assert cp.check_listing_text("Fast Shipping Guaranteed", "bullet_1").status is ContentPolicyStatus.FAIL


def test_print_quality_claim_is_forbidden(cp):
    assert cp.check_listing_text("High Quality Print Design", "bullet_1").status is ContentPolicyStatus.FAIL


def test_price_claim_is_forbidden(cp):
    assert cp.check_listing_text("Only $19.99 Today", "title").status is ContentPolicyStatus.FAIL
    assert cp.check_listing_text("Save 20% off now", "title").status is ContentPolicyStatus.FAIL


def test_contact_info_is_forbidden(cp):
    for text in ["Email us at shop@example.com", "Visit www.mystore.com",
                 "Call 555-123-4567"]:
        assert cp.check_listing_text(text, "bullet_1").status is ContentPolicyStatus.FAIL, text


def test_html_is_forbidden(cp):
    assert cp.check_listing_text("Great <b>design</b> here", "bullet_1").status is ContentPolicyStatus.FAIL
    assert cp.check_listing_text("Tea &amp; Coffee", "title").status is ContentPolicyStatus.FAIL


def test_emoji_is_forbidden(cp):
    assert cp.check_listing_text("Pickleball Dad 🎾 Shirt", "title").status is ContentPolicyStatus.FAIL


def test_excess_caps_is_forbidden(cp):
    assert cp.check_listing_text("BEST DAD EVER GIFT IDEA", "title").status is ContentPolicyStatus.FAIL


def test_limited_caps_is_allowed(cp):
    """Two all-caps words is within the configured limit -- acronyms are normal."""
    assert cp.check_listing_text("Retro BBQ Dad Grilling Tee", "title").status is ContentPolicyStatus.PASS


def test_clean_listing_passes(cp):
    r = cp.check_listing_text("Vintage Fishing Legend Funny Angler Retro T-Shirt", "title")
    assert r.status is ContentPolicyStatus.PASS


# --- record integration -------------------------------------------------

def test_check_record_passes_clean_record(cp):
    rec = check_record(make(), cp)
    assert rec.content_policy_status is ContentPolicyStatus.PASS
    assert rec.content_policy_reason == ""


def test_check_record_fails_on_concept(cp):
    rec = check_record(make("Smoke Weed Every Day"), cp)
    assert rec.content_policy_status is ContentPolicyStatus.FAIL
    assert "drugs" in rec.content_policy_reason


def test_check_record_screens_generated_listing_fields(cp):
    rec = make()
    rec.listing.title = "Pickleball Dad Tee"
    rec.listing.bullet_1 = "Made with soft cotton for all day comfort"
    rec = check_record(rec, cp)
    assert rec.content_policy_status is ContentPolicyStatus.FAIL
    assert "bullet_1" in rec.content_policy_reason


def test_reason_names_the_field_and_the_rule(cp):
    """Review needs to know which field broke which rule, not just that it failed."""
    rec = make()
    rec.listing.title = "Only $9.99 Fast Shipping"
    rec = check_record(rec, cp)
    assert "title" in rec.content_policy_reason
    assert "price_claim" in rec.content_policy_reason


# --- epistemics ---------------------------------------------------------

def test_policy_can_pass_unlike_trademark_triage(cp):
    """Content policy checks OUR text against published rules, so PASS is a
    claim that holds -- unlike a trademark screen, which can only fail to find."""
    assert ContentPolicyStatus.PASS.value == "pass"
    assert cp.check_concept("Cat Nap Enthusiast").status is ContentPolicyStatus.PASS


def test_config_declares_it_is_a_scaffold(cp):
    assert "scaffold" in cp.meta["note"].lower()


def test_reason_reports_the_original_wording_not_the_normalized_form(cp):
    """normalize_text folds leetspeak, so '420 friendly' stores as 'a2o
    friendly'. Matching is unaffected (both sides fold), but the review queue
    printed 'a2o friendly' -- which a human cannot act on."""
    r = cp.check_concept("420 Friendly Weed Leaf Design")
    assert r.status is ContentPolicyStatus.FAIL
    assert "420 friendly" in r.reason
    assert "a2o" not in r.reason
