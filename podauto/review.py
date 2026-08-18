"""Human review decisions and export.

Two jobs that belong together because one is only useful after the other:
recording what a person decided about each design, and getting the approved ones
out of the store in a form you can upload.

Decisions are per **variation**, not per record. A record holds several designs
that share one title and brand, and lettering is model-rendered with no
verification stage -- measured, a third to two thirds of the designs on a record
are usable. A record-level verdict would either ship the misspelled ones or throw
away the good one.

The record-level status is a roll-up of those decisions, and it deliberately
stays at AWAITING_REVIEW while any variation is still undecided. A half-reviewed
record dropping out of the queue is exactly the failure the review step exists to
prevent -- that queue is the only place model-rendered lettering is ever checked.

`export` writes files and touches no status. Writing a CSV is not an upload to
Amazon, so nothing here sets UPLOADED; that value stays unused until something
actually uploads. Same rule the image stages already follow: producing a file is
not a stage transition.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .models import PipelineStatus, Record, StyleVariation
from .printready import IGNORED_GROUND_PHRASE

# What an approval looks like in StyleVariation.review_decision. A rejection
# stores its reason there instead, so this marker is what distinguishes the two.
APPROVED_MARK = "approved"

# qa_notes that do not mean the print file is damaged. An allowlist on purpose:
# anything unclassified BLOCKS an approval, so a warning nobody has thought about
# yet errs toward making a human read it rather than toward being waved through.
#
#   ignored ground -- measured 2026-08-17 on 3 of 3 real generations. Removal keys
#     the colour it actually finds rather than the one asked for, and all three
#     files passed print QA at edge_opaque 0.0012 / 0.0 / 0.0039 against a 0.02
#     limit. The note records a prompt defect, not a file defect. Blocking on it
#     would put --force on every real approval, and a flag typed every time stops
#     carrying information -- which would waste the one gate that catches a
#     ground-colour collision.
#   DPI header -- both DPI warnings (imageqa.py:224, 228) say it themselves: the
#     pixel dimensions are already right and printready writes the header.
INFORMATIONAL_NOTES = (IGNORED_GROUND_PHRASE, "DPI header")

EXPORT_COLUMNS = [
    "record_id", "niche", "style_id", "style_name",
    "title", "brand", "bullet_1", "bullet_2", "description",
    "print_file", "intended_text", "qa_notes",
]


class ReviewError(Exception):
    """A decision that must not be applied silently -- an ambiguous record id, a
    rejection with no reason, an approval of something with no file to upload."""


@dataclass
class Decision:
    """What happened to one variation, so the CLI can report per design.

    The two note lists are kept apart rather than joined into `detail` because a
    forced approval still has to show what it overrode: `blocking` is printed
    whether or not the decision was applied.
    """

    index: int
    style_name: str
    applied: bool
    detail: str = ""
    blocking: list[str] = field(default_factory=list)
    informational: list[str] = field(default_factory=list)


@dataclass
class DecisionSummary:
    record_id: str = ""
    niche: str = ""
    status: PipelineStatus | None = None
    approved: int = 0
    rejected: int = 0
    undecided: int = 0
    decisions: list[Decision] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.approved} approved", f"{self.rejected} rejected"]
        if self.undecided:
            parts.append(f"{self.undecided} still undecided")
        return ", ".join(parts)


@dataclass
class ExportSummary:
    rows: int = 0
    records: int = 0
    copied: int = 0
    skipped: list[str] = field(default_factory=list)
    path: Path | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_approved(variation: StyleVariation) -> bool:
    return variation.review_decision == APPROVED_MARK


def is_decided(variation: StyleVariation) -> bool:
    return bool(variation.review_decision)


def split_notes(variation: StyleVariation) -> tuple[list[str], list[str]]:
    """(blocking, informational) qa_notes -- see INFORMATIONAL_NOTES."""
    blocking, informational = [], []
    for note in variation.qa_notes:
        target = (informational if any(m in note for m in INFORMATIONAL_NOTES)
                  else blocking)
        target.append(note)
    return blocking, informational


# --- finding the record the reviewer meant --------------------------------

def resolve(records: list[Record], prefix: str) -> Record:
    """Find the one record whose id starts with `prefix`.

    `review` prints `rec.id[:8]`, so a prefix is what the reviewer actually has
    in front of them. An ambiguous or unknown prefix raises: applying a decision
    to the wrong record is unrecoverable once the reviewer has moved on, and a
    silent no-op reads as "already done".
    """
    prefix = (prefix or "").strip()
    if not prefix:
        raise ReviewError("no record id given")

    hits = [r for r in records if r.id.startswith(prefix)]
    if not hits:
        raise ReviewError(f"no record id starts with {prefix!r}")
    if len(hits) > 1:
        listed = ", ".join(f"{r.id[:12]} ({r.niche})" for r in hits[:5])
        raise ReviewError(f"{prefix!r} matches {len(hits)} records: {listed}")
    return hits[0]


def pick_variations(record: Record, indices: list[int] | None) -> list[int]:
    """Which variation indices a decision applies to. None means all of them."""
    if not record.variations:
        raise ReviewError(f"{record.id[:8]} has no variations to decide on")
    if indices is None:
        return list(range(len(record.variations)))

    out, bad = [], []
    for i in indices:
        if 0 <= i < len(record.variations):
            if i not in out:
                out.append(i)
        else:
            bad.append(i)
    if bad:
        raise ReviewError(f"{record.id[:8]} has variations 0-"
                          f"{len(record.variations) - 1}; no {bad}")
    return out


# --- recording a decision -------------------------------------------------

def roll_up(record: Record) -> PipelineStatus:
    """The record-level status implied by its variations' decisions.

    Undecided variations keep the record in the review queue. That is the whole
    point: a record must not leave the queue because someone got halfway through
    it, and `review` is the only place the lettering is ever checked.
    """
    if not record.variations:
        return record.pipeline_status
    if any(is_approved(v) for v in record.variations):
        if all(is_decided(v) for v in record.variations):
            return PipelineStatus.APPROVED
        return PipelineStatus.AWAITING_REVIEW
    if all(is_decided(v) for v in record.variations):
        return PipelineStatus.REJECTED_BY_HUMAN
    return PipelineStatus.AWAITING_REVIEW


def decide(record: Record, approve: bool, indices: list[int] | None = None,
           reason: str = "", force: bool = False) -> DecisionSummary:
    """Record one reviewer decision across the chosen variations.

    Three refusals, all of them things that would otherwise produce a bad export
    or lose the only feedback the pipeline ever gets:

    * a rejection with no reason -- the reason is the only signal that reaches
      the styles and prompts, and it is gone once the reviewer moves on;
    * approving a variation with no print file -- there is nothing to upload, and
      it would export as a row pointing at nothing;
    * approving over a *blocking* qa_note without --force -- a ground-colour
      collision means the removal cut into the design, and this is the last point
      at which reading that warning can still stop something. Notes on
      INFORMATIONAL_NOTES are reported and do not gate; the ignored-ground
      warning fires on essentially every generation, and gating on it would make
      --force routine enough to be meaningless.
    """
    reason = (reason or "").strip()
    if not approve and not reason:
        raise ReviewError("reject needs --reason; a rejection with no reason is "
                          "the one piece of feedback that could fix the prompts")

    chosen = pick_variations(record, indices)
    out = DecisionSummary(record_id=record.id, niche=record.niche)

    for i in chosen:
        var = record.variations[i]
        blocking, informational = split_notes(var)
        detail = ""
        if is_decided(var) and not force:
            detail = f"already decided ({var.review_decision!r}) -- use --force"
        elif approve and not var.print_path:
            detail = "no print file -- run printready first, nothing to upload"
        elif approve and blocking and not force:
            detail = (f"{len(blocking)} blocking QA warning(s) -- read them, "
                      f"then --force to approve anyway")

        if detail:
            out.decisions.append(Decision(i, var.style_name, False, detail,
                                          blocking, informational))
            continue

        var.review_decision = APPROVED_MARK if approve else reason
        out.decisions.append(Decision(i, var.style_name, True,
                                      APPROVED_MARK if approve else reason,
                                      blocking, informational))

    if any(d.applied for d in out.decisions):
        record.reviewed_at = now_iso()
        if reason:
            record.review_decision = reason
        elif approve and not record.review_decision:
            record.review_decision = APPROVED_MARK
        record.pipeline_status = roll_up(record)

    out.approved = sum(1 for v in record.variations if is_approved(v))
    out.rejected = sum(1 for v in record.variations
                       if is_decided(v) and not is_approved(v))
    out.undecided = sum(1 for v in record.variations if not is_decided(v))
    out.status = record.pipeline_status
    return out


# --- export ---------------------------------------------------------------

def export_rows(records: list[Record]) -> tuple[list[dict], list[str]]:
    """One row per approved variation, plus the reasons anything was left out.

    A row per variation rather than per record because each design is its own
    Merch product; they only share the title and brand. `intended_text` travels
    with the row so the final read-the-artwork check is possible at the upload
    desk, not only in the review queue -- the lettering is model-rendered and
    nothing else verifies it.
    """
    rows, skipped = [], []
    for rec in records:
        if rec.pipeline_status is not PipelineStatus.APPROVED:
            continue
        for i, var in enumerate(rec.variations):
            if not is_approved(var):
                continue
            if not var.print_path:
                skipped.append(f"{rec.id[:8]}[{i}] {var.style_name}: no print file")
                continue
            if not Path(var.print_path).is_file():
                skipped.append(f"{rec.id[:8]}[{i}] {var.style_name}: "
                               f"print file missing from disk ({var.print_path})")
                continue
            rows.append({
                "record_id": rec.id,
                "niche": rec.niche,
                "style_id": var.style_id,
                "style_name": var.style_name,
                "title": rec.listing.title,
                "brand": rec.listing.brand,
                "bullet_1": rec.listing.bullet_1,
                "bullet_2": rec.listing.bullet_2,
                "description": rec.listing.description,
                "print_file": str(Path(var.print_path).resolve()),
                "intended_text": var.text_spec.get("line_1", ""),
                "qa_notes": " | ".join(var.qa_notes),
            })
    return rows, skipped


def export(records: list[Record], out_dir: str | Path = "data/export",
           copy_files: bool = False) -> ExportSummary:
    """Write listings.csv for every approved variation. Reads the store, never
    writes it -- see the module docstring on why nothing here sets UPLOADED."""
    rows, skipped = export_rows(records)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "listings.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = ExportSummary(rows=len(rows), skipped=skipped, path=path,
                            records=len({r["record_id"] for r in rows}))

    if copy_files and rows:
        files = out_dir / "files"
        files.mkdir(parents=True, exist_ok=True)
        for row in rows:
            src = Path(row["print_file"])
            dst = files / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            shutil.copy2(src, dst)
            summary.copied += 1

    return summary
