"""Dedupe key derivation.

Reddit, TikTok and Amazon will resurface the same idea repeatedly. Without a
stable key we regenerate identical designs and burn image quota on duplicates.

The key is derived from niche + normalized concept text, deliberately NOT from
source_url -- the same idea arriving from three platforms should collapse to
one record, which is the whole point.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Words that carry no concept signal and vary randomly between platforms.
_NOISE = {
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "with",
    "shirt", "tshirt", "t", "tee", "shirts", "tees", "apparel", "design",
    "funny", "gift", "gifts", "vintage", "retro", "cute", "best", "new",
    "womens", "mens", "kids", "unisex", "premium", "classic",
}

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse leetspeak and punctuation.

    Also used by the trademark triage matcher, so near-miss spellings like
    "D1sn3y" normalize onto the same form as the literal term.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().translate(_LEET)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def concept_tokens(text: str) -> list[str]:
    """Content-bearing tokens, deduped and sorted for order independence."""
    tokens = {t for t in normalize_text(text).split() if t not in _NOISE and len(t) > 2}
    return sorted(tokens)


def dedupe_key(niche: str, raw_text: str) -> str:
    """Stable key over niche + concept tokens.

    Word-order changes and platform boilerplate collapse to the same key;
    genuinely different concepts do not.
    """
    basis = f"{normalize_text(niche)}|{' '.join(concept_tokens(raw_text))}"
    return hashlib.sha256(basis.encode()).hexdigest()[:16]
