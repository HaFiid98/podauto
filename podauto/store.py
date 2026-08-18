"""JSONL-backed record store.

Deliberately not Google Sheets. Sheets has per-minute write quotas and no
transactions -- parallel writes produce lost updates. This is a local file
store with atomic replace; swap the backend for Postgres when volume needs it,
the interface is small on purpose.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterator

from .models import (
    ContentPolicyStatus,
    Gate,
    IdeaType,
    Listing,
    PipelineStatus,
    Record,
    ScoreConfidence,
    SourcePlatform,
    StyleVariation,
    TmStatus,
)


def _record_from_dict(d: dict) -> Record:
    d = dict(d)
    d["source_platform"] = SourcePlatform(d["source_platform"])
    d["idea_type"] = IdeaType(d["idea_type"])
    d["tm_status"] = TmStatus(d.get("tm_status", "unchecked"))
    d["content_policy_status"] = ContentPolicyStatus(d.get("content_policy_status", "unchecked"))
    d["pipeline_status"] = PipelineStatus(d.get("pipeline_status", "ingested"))
    if d.get("score_confidence"):
        d["score_confidence"] = ScoreConfidence(d["score_confidence"])
    if d.get("gate"):
        d["gate"] = Gate(d["gate"])
    d["variations"] = [StyleVariation(**v) for v in d.get("variations", [])]
    d["listing"] = Listing(**d.get("listing", {}))
    return Record(**d)


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Record]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(_record_from_dict(json.loads(line)))
        return out

    def by_status(self, *statuses: PipelineStatus) -> Iterator[Record]:
        wanted = set(statuses)
        for r in self.all():
            if r.pipeline_status in wanted:
                yield r

    def existing_keys(self) -> set[str]:
        return {r.dedupe_key for r in self.all() if r.dedupe_key}

    def write_all(self, records: list[Record]) -> None:
        """Atomic full rewrite: temp file + replace, so an interrupted run
        never leaves a half-written store."""
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                for r in records:
                    f.write(json.dumps(r.to_dict(), default=str) + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def upsert(self, record: Record) -> None:
        records = self.all()
        for i, r in enumerate(records):
            if r.id == record.id:
                records[i] = record
                break
        else:
            records.append(record)
        self.write_all(records)

    def upsert_many(self, incoming: list[Record]) -> None:
        records = self.all()
        index = {r.id: i for i, r in enumerate(records)}
        for rec in incoming:
            if rec.id in index:
                records[index[rec.id]] = rec
            else:
                index[rec.id] = len(records)
                records.append(rec)
        self.write_all(records)
