"""Trademark triage: a tiered keyword screen.

What this does: finds terms it knows about and routes them.
What this does NOT do: clear anything. There is no CLEARED outcome anywhere in
this module. A clean pass yields NO_FLAGS_FOUND, which is the honest description
of what a keyword screen tells you -- trademark turns on likelihood of
confusion, not string equality, so "Star Warz" passes this screen and still
infringes. The manual review downstream is load-bearing, not decorative.

Matching runs on text normalized by dedupe.normalize_text(), so leetspeak and
accent near-misses ("D1sn3y") fold onto their literal forms before comparison.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .dedupe import normalize_text
from .models import Record, TmStatus


@dataclass
class Flag:
    tier: int
    term: str
    reason: str
    companion: str | None = None


@dataclass
class TriageResult:
    status: TmStatus
    flags: list[Flag] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(
            f"tier{f.tier}: {f.term}"
            + (f" + {f.companion}" if f.companion else "")
            + f" ({f.reason})"
            for f in self.flags
        )

    @property
    def matched_terms(self) -> list[str]:
        return [f.term for f in self.flags]


class Denylist:
    def __init__(self, data: dict[str, Any]):
        self.meta = data.get("_meta", {})

        # Tier 1 is a flat term -> category map so the reason names the category.
        self.hard_block: dict[str, str] = {}
        for category, terms in data.get("tier_1_hard_block", {}).items():
            for term in terms:
                self.hard_block[normalize_text(term)] = category

        self.co_occurrence = [
            {
                "term": normalize_text(entry["term"]),
                "companions": [normalize_text(c) for c in entry["companions"]],
                "note": entry.get("note", ""),
            }
            for entry in data.get("tier_2_co_occurrence", [])
        ]

        # Patterns run against RAW text, not normalized -- normalization strips
        # the punctuation and symbols some of them look for.
        self.patterns = [
            {
                "id": p["id"],
                "regex": re.compile(p["regex"], re.IGNORECASE),
                "reason": p["reason"],
            }
            for p in data.get("tier_3_patterns", [])
        ]

    @classmethod
    def load(cls, path: str | Path) -> Denylist:
        return cls(json.loads(Path(path).read_text()))

    def staleness_days(self, today: date | None = None) -> int | None:
        """Days since last_reviewed. Surfaced so list rot stays visible."""
        raw = self.meta.get("last_reviewed")
        if not raw:
            return None
        try:
            reviewed = date.fromisoformat(raw)
        except ValueError:
            return None
        return ((today or date.today()) - reviewed).days

    def is_stale(self, today: date | None = None) -> bool:
        days = self.staleness_days(today)
        cadence = self.meta.get("refresh_cadence_days")
        if days is None or not cadence:
            return False
        return days > cadence

    def check(self, text: str) -> TriageResult:
        normalized = normalize_text(text)
        padded = f" {normalized} "

        # Tier 1 -- hard block. Longest terms first so a multi-word match wins
        # over a substring of itself.
        for term in sorted(self.hard_block, key=len, reverse=True):
            if f" {term} " in padded:
                return TriageResult(
                    status=TmStatus.BLOCKED,
                    flags=[Flag(tier=1, term=term,
                                reason=f"exact match, {self.hard_block[term]}")],
                )

        flags: list[Flag] = []

        # Tier 2 -- ambiguous term only fires alongside a companion.
        for entry in self.co_occurrence:
            if f" {entry['term']} " not in padded:
                continue
            for companion in entry["companions"]:
                if f" {companion} " in padded:
                    flags.append(Flag(
                        tier=2, term=entry["term"], companion=companion,
                        reason=entry["note"] or "ambiguous term with brand companion",
                    ))
                    break

        # Tier 3 -- patterns, against raw text.
        for pattern in self.patterns:
            match = pattern["regex"].search(text)
            if match:
                flags.append(Flag(tier=3, term=match.group(0).strip(),
                                  reason=pattern["reason"]))

        if flags:
            return TriageResult(status=TmStatus.NEEDS_REVIEW, flags=flags)
        return TriageResult(status=TmStatus.NO_FLAGS_FOUND)


def triage_record(record: Record, denylist: Denylist) -> Record:
    """Screen a record's concept text and any generated listing fields."""
    parts = [record.raw_title_or_text, record.niche]
    if record.listing.title:
        parts += [record.listing.title, record.listing.brand,
                  record.listing.bullet_1, record.listing.bullet_2]

    result = denylist.check(" ".join(p for p in parts if p))
    record.tm_status = result.status
    record.tm_flag_reason = result.reason
    record.matched_terms = result.matched_terms
    return record
