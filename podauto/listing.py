"""Amazon Merch listing generator.

Pure text, no external calls, every constraint assertable.

Character bounds are enforced by construction and then re-checked, not trusted.
The original spec's own worked example was 59 characters against its stated
60-character minimum -- which is exactly the kind of off-by-one that a runtime
assertion catches and a human reading a spec does not.

Redbubble is out of v1. Listing has room to grow tags when it comes back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .dedupe import concept_tokens, normalize_text
from .models import IdeaType, Listing, PipelineStatus, Record

TITLE_MIN = 60
TITLE_MAX = 80
BULLET_MAX = 256
BULLET_COUNT = 2          # Merch gives you 2. The original spec said 2-3 in one
                          # place and 2 in another; this resolves it.

# Similarity above this between two titles on the same record is a
# duplicate-content risk against your own account.
#
# Calibrated to the metric, not picked round: _similarity is Jaccard over token
# SETS, which penalises a swapped word twice -- once out of the intersection and
# once into the union. Two 8-token titles differing by a single word score
# 7/9 = 0.78, not the ~0.9 the character-level intuition suggests. A 0.85 limit
# therefore only ever fires on titles that are exactly identical, making the
# check dead code precisely where it matters: variations of one record differ by
# a single style word by construction. 0.75 catches the one-word swap and still
# leaves a wide margin -- two genuinely different titles share only the
# boilerplate "t"/"shirt" tokens and score around 0.14.
TITLE_SIMILARITY_LIMIT = 0.75

_FILLER = ("Retro", "Classic", "Vintage Style", "Graphic", "Novelty", "Design")

# Reached only when _FILLER runs dry -- a very short concept ("Cat" + "Nap")
# exhausts the pool around 46 chars, well under the 60 minimum. These are
# gift/audience words rather than more style adjectives, so a padded title still
# reads as a real listing instead of a keyword pile.
_FILLER_EXTRA = (
    "Gift Idea", "For Men", "For Women", "Funny Saying", "Cool Novelty",
    "Birthday Present", "Casual Everyday", "Humor Lover",
)


@dataclass
class ListingIssue:
    field: str
    problem: str

    def __str__(self) -> str:
        return f"{self.field}: {self.problem}"


def _titlecase(text: str) -> str:
    return " ".join(w if w.isupper() else w.capitalize() for w in text.split())


def build_title(record: Record, style_name: str = "") -> str:
    """[Core Keyword] [Niche/Recipient] [Style/Theme] T-Shirt, 60-80 chars.

    Builds up from the concept, pads with neutral filler when short, and trims
    from the least meaningful end when long.
    """
    core = _titlecase(record.niche)
    concept = _titlecase(" ".join(record.raw_title_or_text.split()[:6]))
    style_words = _titlecase(style_name.replace("Vintage ", "")) if style_name else ""

    parts = [p for p in (core, concept, style_words) if p]
    # Drop words already present so the title does not stutter the niche back.
    seen: set[str] = set()
    words: list[str] = []
    for part in parts:
        for w in part.split():
            key = normalize_text(w)
            if key and key not in seen:
                seen.add(key)
                words.append(w)

    title = " ".join(words)
    if not title.lower().endswith("t-shirt"):
        title = f"{title} T-Shirt"

    # Too long: drop words before the trailing "T-Shirt" until it fits.
    while len(title) > TITLE_MAX and len(words) > 1:
        words.pop()
        title = " ".join(words) + " T-Shirt"

    # Too short: pad with neutral filler that carries no policy risk.
    # Filler consults `seen` for the same reason the main loop does -- "Retro"
    # is both a common concept word and the first filler, so without this a
    # Pickleball Dad Retro concept pads to "... Retro Retro Classic ...".
    # A filler that would overshoot TITLE_MAX is skipped rather than ending the
    # loop, since a shorter one further down the pool may still fit.
    for filler in _FILLER + _FILLER_EXTRA:
        if len(title) >= TITLE_MIN:
            break
        filler_keys = set(normalize_text(filler).split())
        if filler_keys & seen:
            continue
        candidate = " ".join(words + [filler]) + " T-Shirt"
        if len(candidate) > TITLE_MAX:
            continue
        words.append(filler)
        seen |= filler_keys
        title = candidate

    return title


def build_brand(record: Record) -> str:
    """[Niche Keyword] Apparel Co. Screened by triage before use."""
    tokens = concept_tokens(record.niche) or concept_tokens(record.raw_title_or_text)
    head = _titlecase(" ".join(tokens[:2])) if tokens else "Everyday"
    return f"{head} Apparel Co"


def build_bullets(record: Record) -> tuple[str, str]:
    """Bullet 1: audience and the joke. Bullet 2: gift intent and occasions.

    No material, print, shipping or quality claims -- those are Amazon's to
    make, and content_policy.py will fail the record if they appear.
    """
    niche = _titlecase(record.niche)
    concept = record.raw_title_or_text.strip().rstrip(".")

    b1 = (
        f"Perfect for anyone who loves {niche.lower()}. "
        f"This {concept.lower()} design says what you are thinking without saying a word."
    )
    b2 = (
        f"A thoughtful gift idea for birthdays, Father's Day, Christmas or any "
        f"{niche.lower()} milestone. Great for friends, family and anyone in the hobby."
    )
    return b1[:BULLET_MAX], b2[:BULLET_MAX]


def build_description(record: Record) -> str:
    niche = _titlecase(record.niche)
    return (
        f"{niche} design for enthusiasts and gift givers. "
        f"Wear it to the club, to practice, or anywhere you want to start a conversation."
    )


def _similarity(a: str, b: str) -> float:
    """Jaccard over normalized token sets. Cheap and adequate for near-dupes."""
    ta, tb = set(normalize_text(a).split()), set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def validate(listing: Listing) -> list[ListingIssue]:
    """Re-check every constraint after construction. Belt and braces."""
    issues: list[ListingIssue] = []

    if not TITLE_MIN <= len(listing.title) <= TITLE_MAX:
        issues.append(ListingIssue(
            "title", f"{len(listing.title)} chars, must be {TITLE_MIN}-{TITLE_MAX}"))

    if not listing.brand:
        issues.append(ListingIssue("brand", "empty"))
    elif len(listing.brand) > 50:
        issues.append(ListingIssue("brand", f"{len(listing.brand)} chars, over 50"))

    for name, bullet in (("bullet_1", listing.bullet_1), ("bullet_2", listing.bullet_2)):
        if not bullet:
            issues.append(ListingIssue(name, "empty"))
        elif len(bullet) > BULLET_MAX:
            issues.append(ListingIssue(name, f"{len(bullet)} chars, over {BULLET_MAX}"))

    return issues


def check_variation_similarity(titles: list[str]) -> list[ListingIssue]:
    """Near-identical titles across one record's variations are a
    duplicate-content flag against your own account."""
    issues: list[ListingIssue] = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            sim = _similarity(titles[i], titles[j])
            if sim >= TITLE_SIMILARITY_LIMIT:
                issues.append(ListingIssue(
                    f"title[{i}]/title[{j}]",
                    f"{sim:.0%} similar, duplicate-listing risk",
                ))
    return issues


def generate(record: Record) -> tuple[Record, list[ListingIssue]]:
    """Populate record.listing and return any constraint violations.

    Issues are returned rather than raised: a listing that is one character
    short should reach human review flagged, not vanish.
    """
    style_name = record.variations[0].style_name if record.variations else ""

    record.listing = Listing(
        title=build_title(record, style_name),
        brand=build_brand(record),
        bullet_1=build_bullets(record)[0],
        bullet_2=build_bullets(record)[1],
        description=build_description(record),
    )

    issues = validate(record.listing)

    if len(record.variations) > 1:
        titles = [build_title(record, v.style_name) for v in record.variations]
        issues += check_variation_similarity(titles)

    if not issues:
        record.pipeline_status = PipelineStatus.LISTED
    return record, issues
