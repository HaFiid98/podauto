"""Amazon content-policy gate.

Deliberately a separate module from triage, because the epistemics differ.

Trademark triage asks "do we infringe someone else's rights?" -- not answerable
from a word list, which is why it only ever flags. This module asks "does our
own text break a published rule?" -- which IS answerable by inspection, since
both the rule and the text are things we can see. So this one is a hard gate:
FAIL stops the record.

That said, the lists are still scaffolds. A PASS means "no configured rule was
violated", and the configured rules are only as complete as
config/content_policy.json.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dedupe import normalize_text
from .models import ContentPolicyStatus, Record

# Emoji and pictographic ranges. Amazon rejects these in listing copy.
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U00002190-\U000021ff"
    "]"
)

_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")


@dataclass
class Violation:
    rule: str
    matched: str
    detail: str = ""

    def __str__(self) -> str:
        base = f"{self.rule}: {self.matched!r}"
        return f"{base} ({self.detail})" if self.detail else base


@dataclass
class PolicyResult:
    status: ContentPolicyStatus
    violations: list[Violation] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(str(v) for v in self.violations)


class ContentPolicy:
    def __init__(self, data: dict[str, Any]):
        self.meta = data.get("_meta", {})

        # Both sides of every comparison are normalized, so leetspeak folding is
        # symmetric ("420 friendly" and "a2o friendly" both fold to the same
        # form and still match). But the normalized form is what a reviewer
        # would otherwise see in the reason string, and "a2o friendly" is not a
        # phrase anyone can act on -- so keep a map back to the original wording
        # and report that instead.
        self.display: dict[str, str] = {}

        self.categories: dict[str, list[str]] = {}
        for name, terms in data.get("prohibited_categories", {}).items():
            normalized = []
            for term in terms:
                norm = normalize_text(term)
                normalized.append(norm)
                self.display.setdefault(norm, term)
            self.categories[name] = normalized

        rules = data.get("listing_field_rules", {})
        self.forbidden = []
        for term in rules.get("forbidden_phrases", {}).get("terms", []):
            norm = normalize_text(term)
            self.forbidden.append(norm)
            self.display.setdefault(norm, term)
        self.price_re = (
            re.compile(rules["price_pattern"], re.IGNORECASE)
            if rules.get("price_pattern") else None
        )
        self.contact_re = (
            re.compile(rules["contact_pattern"], re.IGNORECASE)
            if rules.get("contact_pattern") else None
        )
        self.html_re = (
            re.compile(rules["html_pattern"], re.IGNORECASE)
            if rules.get("html_pattern") else None
        )
        self.max_caps_words = rules.get("max_consecutive_caps_words", 2)
        self.allow_emoji = rules.get("allow_emoji", False)

    @classmethod
    def load(cls, path: str | Path) -> ContentPolicy:
        return cls(json.loads(Path(path).read_text()))

    def check_concept(self, text: str) -> PolicyResult:
        """Prohibited-category screen on design concept text.

        Applies to the idea itself, so it can run before any expensive stage.
        """
        padded = f" {normalize_text(text)} "
        violations = [
            Violation(rule=f"prohibited:{category}",
                      matched=self.display.get(term, term))
            for category, terms in self.categories.items()
            for term in terms
            if f" {term} " in padded
        ]
        return PolicyResult(
            status=ContentPolicyStatus.FAIL if violations else ContentPolicyStatus.PASS,
            violations=violations,
        )

    def check_listing_text(self, text: str, field_name: str = "listing") -> PolicyResult:
        """Listing-copy rules: forbidden claims, prices, contact info, caps, emoji.

        These govern what a SELLER may write, distinct from the prohibited
        content categories above.
        """
        violations: list[Violation] = []
        padded = f" {normalize_text(text)} "

        for phrase in self.forbidden:
            if f" {phrase} " in padded:
                violations.append(Violation(
                    rule=f"forbidden_phrase:{field_name}",
                    matched=self.display.get(phrase, phrase),
                    detail="print, material, shipping and promotional claims are not the seller's to make",
                ))

        for regex, rule, detail in (
            (self.price_re, "price_claim",
             "prices and discounts belong to Amazon, not listing copy"),
            (self.contact_re, "contact_info",
             "no emails, URLs or phone numbers in listings"),
            (self.html_re, "html_markup",
             "markup is not permitted in listing fields"),
        ):
            if regex:
                m = regex.search(text)
                if m:
                    violations.append(Violation(
                        rule=f"{rule}:{field_name}", matched=m.group(0), detail=detail))

        if not self.allow_emoji:
            m = _EMOJI.search(text)
            if m:
                violations.append(Violation(
                    rule=f"emoji:{field_name}", matched=m.group(0),
                    detail="emoji are not permitted in listing fields"))

        caps = _CAPS_WORD.findall(text)
        if len(caps) > self.max_caps_words:
            violations.append(Violation(
                rule=f"excess_caps:{field_name}",
                matched=" ".join(caps[: self.max_caps_words + 1]),
                detail=f"{len(caps)} all-caps words, limit is {self.max_caps_words}",
            ))

        return PolicyResult(
            status=ContentPolicyStatus.FAIL if violations else ContentPolicyStatus.PASS,
            violations=violations,
        )


def check_record(record: Record, policy: ContentPolicy) -> Record:
    """Screen concept text, plus listing fields if they have been generated."""
    result = policy.check_concept(f"{record.raw_title_or_text} {record.niche}")
    violations = list(result.violations)

    for field_name, value in (
        ("title", record.listing.title),
        ("brand", record.listing.brand),
        ("bullet_1", record.listing.bullet_1),
        ("bullet_2", record.listing.bullet_2),
        ("description", record.listing.description),
    ):
        if value:
            violations += policy.check_concept(value).violations
            violations += policy.check_listing_text(value, field_name).violations

    record.content_policy_status = (
        ContentPolicyStatus.FAIL if violations else ContentPolicyStatus.PASS
    )
    record.content_policy_reason = "; ".join(str(v) for v in violations)
    return record
