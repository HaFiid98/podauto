"""Ingestion: raw rows -> validated Records, with dedupe applied.

Accepts CSV or JSON. Invalid rows are reported, never silently dropped --
a collector that starts emitting garbage should be visible immediately.
"""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dedupe import dedupe_key
from .models import IdeaType, PipelineStatus, Record, SourcePlatform

REQUIRED = ("source_url", "source_platform", "idea_type", "niche", "raw_title_or_text")


@dataclass
class IngestResult:
    accepted: list[Record]
    duplicates: list[tuple[str, str]]      # (dedupe_key, raw_text)
    invalid: list[tuple[int, str]]         # (row_number, reason)

    def summary(self) -> str:
        return (
            f"accepted={len(self.accepted)} "
            f"duplicates={len(self.duplicates)} "
            f"invalid={len(self.invalid)}"
        )


def _clean_int(value: Any) -> int | None:
    if value in (None, "", "null", "NULL", "None"):
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _clean_float(value: Any) -> float | None:
    if value in (None, "", "null", "NULL", "None"):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_row(row: dict[str, Any], row_num: int) -> tuple[Record | None, str | None]:
    missing = [f for f in REQUIRED if not str(row.get(f, "")).strip()]
    if missing:
        return None, f"missing required field(s): {', '.join(missing)}"

    try:
        platform = SourcePlatform(str(row["source_platform"]).strip().lower())
    except ValueError:
        return None, f"unknown source_platform: {row['source_platform']!r}"

    try:
        idea_type = IdeaType(str(row["idea_type"]).strip().lower())
    except ValueError:
        return None, f"unknown idea_type: {row['idea_type']!r}"

    ts = str(row.get("timestamp", "")).strip() or datetime.now(timezone.utc).isoformat()
    niche = str(row["niche"]).strip()
    raw_text = str(row["raw_title_or_text"]).strip()

    rec = Record(
        id=str(uuid.uuid4()),
        timestamp=ts,
        source_url=str(row["source_url"]).strip(),
        source_platform=platform,
        idea_type=idea_type,
        niche=niche,
        raw_title_or_text=raw_text,
        dedupe_key=dedupe_key(niche, raw_text),
        bsr_rank=_clean_int(row.get("bsr_rank")),
        bsr_category=(str(row.get("bsr_category", "")).strip() or None),
        bsr_captured_at=(str(row.get("bsr_captured_at", "")).strip() or None),
        trend_velocity=_clean_float(row.get("trend_velocity")),
        listings_count=_clean_int(row.get("listings_count")),
        pipeline_status=PipelineStatus.DEDUPED,
    )
    return rec, None


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return list(csv.DictReader(text.splitlines()))


def ingest(rows: list[dict[str, Any]], known_keys: set[str] | None = None) -> IngestResult:
    known = set(known_keys or ())
    accepted: list[Record] = []
    duplicates: list[tuple[str, str]] = []
    invalid: list[tuple[int, str]] = []

    for i, row in enumerate(rows, start=1):
        rec, err = parse_row(row, i)
        if err:
            invalid.append((i, err))
            continue
        if rec.dedupe_key in known:
            duplicates.append((rec.dedupe_key, rec.raw_title_or_text))
            continue
        known.add(rec.dedupe_key)   # also collapses duplicates within one batch
        accepted.append(rec)

    return IngestResult(accepted=accepted, duplicates=duplicates, invalid=invalid)
