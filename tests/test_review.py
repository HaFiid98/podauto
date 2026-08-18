"""Review-decision and export tests.

The review queue is the pipeline's only safety property -- lettering is
model-rendered and verified nowhere else -- so these tests care most about the
ways a decision could quietly go wrong: applied to the wrong record, applied to a
design with no file behind it, or applied in a way that lets a half-reviewed
record slip out of the queue.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from podauto.models import Listing, PipelineStatus, Record, StyleVariation
from podauto.printready import IGNORED_GROUND_PHRASE
from podauto.review import (
    APPROVED_MARK,
    EXPORT_COLUMNS,
    ReviewError,
    decide,
    export,
    export_rows,
    resolve,
    roll_up,
)

# Copied verbatim out of data/quality2.jsonl -- these are notes the store really
# holds, on the three real generations for "Pickleball Dad". Reworded warnings
# should fail the classifier tests loudly rather than quietly reclassifying a
# design, so the strings are pinned here and tied to the phrase constant below.
IGNORED_GROUND = (
    "print ready warning: asked for a #FF00FF ground, got #D53536 -- the model "
    "ignored the ground instruction, so removal is keying whatever it painted "
    "instead")
COLLISION = (
    "print ready warning: 4.5% of the art's bounding box is the ground colour "
    "#000000 (limit 1%) -- the design reuses the ground colour, so edges touching "
    "it will have been cut away; check the file before approving")

assert IGNORED_GROUND_PHRASE in IGNORED_GROUND, "warning text and phrase drifted"


def a_variation(tmp_path, i, with_file=True, notes=()) -> StyleVariation:
    var = StyleVariation(style_id=f"s{i}", style_name=f"Style {i}",
                         graphic_prompt="p", text_spec={"line_1": f"Text {i}"},
                         qa_notes=list(notes))
    var.image_path = str(tmp_path / f"gen_{i}.png")
    if with_file:
        path = tmp_path / f"gen_{i}_print.png"
        path.write_bytes(b"not really a png, nothing here opens it")
        var.print_path = str(path)
    return var


def a_record(tmp_path, rid="a0d829b1-f13d-4984", n=3, niche="Pickleball Dad",
             **kw) -> Record:
    rec = Record(id=rid, timestamp="2026-08-17T00:00:00", source_url="u",
                 source_platform="amazon", idea_type="ready_shirt",
                 niche=niche, raw_title_or_text=niche)
    rec.pipeline_status = PipelineStatus.AWAITING_REVIEW
    rec.listing = Listing(title="A Title", brand="A Brand",
                          bullet_1="One", bullet_2="Two", description="Desc")
    rec.variations = [a_variation(tmp_path, i, **kw) for i in range(n)]
    return rec


# --- finding the record the reviewer meant --------------------------------

def test_a_prefix_resolves_to_the_record_review_printed(tmp_path):
    """`review` prints rec.id[:8], so that is what the reviewer types."""
    recs = [a_record(tmp_path, rid="a0d829b1-aaa"), a_record(tmp_path, rid="b5d6204f-bbb")]
    assert resolve(recs, "a0d829b1").id == "a0d829b1-aaa"
    assert resolve(recs, "b5d6").id == "b5d6204f-bbb"


def test_an_ambiguous_prefix_refuses_and_names_the_candidates(tmp_path):
    """Deciding the wrong record is unrecoverable once the reviewer moves on."""
    recs = [a_record(tmp_path, rid="a0d8-one", niche="Alpha"),
            a_record(tmp_path, rid="a0d8-two", niche="Beta")]
    with pytest.raises(ReviewError) as exc:
        resolve(recs, "a0d8")
    assert "matches 2 records" in str(exc.value)
    assert "Alpha" in str(exc.value) and "Beta" in str(exc.value)


def test_an_unknown_prefix_is_an_error_not_a_silent_no_op(tmp_path):
    """Exiting 0 having done nothing reads as 'already decided'."""
    with pytest.raises(ReviewError):
        resolve([a_record(tmp_path)], "zzzz")


def test_an_empty_prefix_is_refused_rather_than_matching_everything(tmp_path):
    with pytest.raises(ReviewError):
        resolve([a_record(tmp_path)], "")


# --- recording a decision ------------------------------------------------

def test_approving_one_variation_leaves_the_others_undecided(tmp_path):
    """The measured reason decisions are per variation: a third to two thirds of
    the designs on a record are usable, so a record-level verdict would ship the
    misspelled ones alongside the good one."""
    rec = a_record(tmp_path)
    summary = decide(rec, approve=True, indices=[0])

    assert rec.variations[0].review_decision == APPROVED_MARK
    assert rec.variations[1].review_decision == ""
    assert rec.variations[2].review_decision == ""
    assert summary.approved == 1 and summary.undecided == 2


def test_omitting_the_variation_flag_decides_every_design(tmp_path):
    rec = a_record(tmp_path)
    decide(rec, approve=False, reason="all three misspelled")
    assert all(v.review_decision == "all three misspelled" for v in rec.variations)


def test_a_variation_index_that_does_not_exist_is_refused(tmp_path):
    rec = a_record(tmp_path, n=2)
    with pytest.raises(ReviewError) as exc:
        decide(rec, approve=True, indices=[5])
    assert "variations 0-1" in str(exc.value)


def test_rejecting_without_a_reason_is_refused(tmp_path):
    """The reason is the only feedback that ever reaches the styles and prompts."""
    rec = a_record(tmp_path)
    with pytest.raises(ReviewError) as exc:
        decide(rec, approve=False, reason="   ")
    assert "--reason" in str(exc.value)
    assert all(not v.review_decision for v in rec.variations), "store was touched"


def test_the_rejection_reason_is_kept_verbatim_on_the_variation(tmp_path):
    rec = a_record(tmp_path)
    decide(rec, approve=False, indices=[1], reason="lettering doubled: PICKLEBALL twice")
    assert rec.variations[1].review_decision == "lettering doubled: PICKLEBALL twice"


def test_approving_a_variation_with_no_print_file_is_refused(tmp_path):
    """Nothing to upload. Approved anyway it would export as a row pointing at
    nothing, which is worse than an unapproved design."""
    rec = a_record(tmp_path, with_file=False)
    summary = decide(rec, approve=True, indices=[0])

    assert not rec.variations[0].review_decision
    assert summary.approved == 0
    assert "no print file" in summary.decisions[0].detail
    assert rec.pipeline_status is PipelineStatus.AWAITING_REVIEW


def test_approving_over_a_qa_warning_needs_force(tmp_path):
    """A warning nothing makes you read is a comment. This is the last point at
    which the ground-collision measurement can still stop something."""
    rec = a_record(tmp_path, notes=[COLLISION])
    summary = decide(rec, approve=True, indices=[0])
    assert not rec.variations[0].review_decision
    assert "blocking QA warning" in summary.decisions[0].detail
    assert summary.decisions[0].blocking == [COLLISION]

    forced = decide(rec, approve=True, indices=[0], force=True)
    assert rec.variations[0].review_decision == APPROVED_MARK
    assert forced.approved == 1


def test_rejecting_a_warned_variation_does_not_need_force(tmp_path):
    """The gate is on approving past a warning, not on agreeing with it."""
    rec = a_record(tmp_path, notes=[COLLISION])
    decide(rec, approve=False, indices=[0], reason="collision ate the outline")
    assert rec.variations[0].review_decision == "collision ate the outline"


def test_redeciding_needs_force_so_a_rerun_cannot_overwrite_a_verdict(tmp_path):
    rec = a_record(tmp_path)
    decide(rec, approve=False, indices=[0], reason="misspelled")

    summary = decide(rec, approve=True, indices=[0])
    assert rec.variations[0].review_decision == "misspelled"
    assert "already decided" in summary.decisions[0].detail

    decide(rec, approve=True, indices=[0], force=True)
    assert rec.variations[0].review_decision == APPROVED_MARK


def test_a_decision_stamps_reviewed_at(tmp_path):
    rec = a_record(tmp_path)
    assert not rec.reviewed_at
    decide(rec, approve=True, indices=[0])
    assert rec.reviewed_at.startswith("20")


def test_a_refused_decision_does_not_stamp_anything(tmp_path):
    rec = a_record(tmp_path, with_file=False)
    decide(rec, approve=True, indices=[0])
    assert not rec.reviewed_at


# --- which notes are worth stopping for ----------------------------------
#
# Measured 2026-08-17: all 3 real generations carry the ignored-ground warning, so
# a gate on "any note at all" would put --force on every approval. A flag typed
# every time carries no information, which would waste the gate on the one note
# that means the file may be damaged.

def test_the_ignored_ground_warning_does_not_block_an_approval(tmp_path):
    """The 3-of-3 case. printready keys the colour it actually found and the file
    passed print QA; the note records a prompt defect, not a file defect."""
    rec = a_record(tmp_path, notes=[IGNORED_GROUND])
    summary = decide(rec, approve=True, indices=[0])

    assert rec.variations[0].review_decision == APPROVED_MARK
    assert summary.decisions[0].informational == [IGNORED_GROUND]
    assert summary.decisions[0].blocking == []


def test_an_informational_note_is_still_reported_on_a_successful_approval(tmp_path):
    """Not blocking is not the same as not shown."""
    rec = a_record(tmp_path, notes=[IGNORED_GROUND])
    summary = decide(rec, approve=True, indices=[0])
    assert summary.decisions[0].applied
    assert summary.decisions[0].informational, "the note vanished on the way through"


def test_a_note_nobody_classified_blocks_rather_than_being_waved_through(tmp_path):
    """The allowlist is the whole design: an unclassified warning defaults to
    making a human read it, because the alternative default is silent approval of
    a defect nobody has thought about yet."""
    rec = a_record(tmp_path, notes=["print ready warning: something new and unclassified"])
    summary = decide(rec, approve=True, indices=[0])

    assert not rec.variations[0].review_decision
    assert "blocking QA warning" in summary.decisions[0].detail


def test_a_hard_failure_note_blocks(tmp_path):
    """imageqa failures use a different prefix from warnings; neither is on the
    allowlist, so both gate."""
    rec = a_record(tmp_path, notes=["image qa (print_ready): expected 4500x5400"])
    assert not decide(rec, approve=True, indices=[0]).decisions[0].applied


def test_a_mixed_set_of_notes_blocks_on_the_blocking_one(tmp_path):
    """The real y2k variation carries both. The informational note must not
    dilute the collision, and the collision must not suppress the note."""
    rec = a_record(tmp_path, notes=[IGNORED_GROUND, COLLISION])
    summary = decide(rec, approve=True, indices=[0])

    assert not summary.decisions[0].applied
    assert summary.decisions[0].blocking == [COLLISION]
    assert summary.decisions[0].informational == [IGNORED_GROUND]


def test_forcing_past_a_blocking_note_still_reports_what_was_overridden(tmp_path):
    """A forced approval that printed nothing would leave no trace of the
    warning it went around."""
    rec = a_record(tmp_path, notes=[COLLISION])
    summary = decide(rec, approve=True, indices=[0], force=True)

    assert summary.decisions[0].applied
    assert summary.decisions[0].blocking == [COLLISION]


# --- roll-up -------------------------------------------------------------

def test_a_half_reviewed_record_stays_in_the_review_queue(tmp_path):
    """The load-bearing rule. A record must not leave the queue because someone
    got halfway through it -- that queue is the only place the model-rendered
    lettering is ever checked."""
    rec = a_record(tmp_path)
    decide(rec, approve=True, indices=[0])
    assert rec.pipeline_status is PipelineStatus.AWAITING_REVIEW
    assert roll_up(rec) is PipelineStatus.AWAITING_REVIEW


def test_a_fully_reviewed_record_with_one_approval_is_approved(tmp_path):
    rec = a_record(tmp_path)
    decide(rec, approve=True, indices=[0])
    decide(rec, approve=False, indices=[1, 2], reason="misspelled")
    assert rec.pipeline_status is PipelineStatus.APPROVED


def test_a_record_with_every_variation_rejected_is_rejected_by_human(tmp_path):
    rec = a_record(tmp_path)
    decide(rec, approve=False, reason="all three misspelled")
    assert rec.pipeline_status is PipelineStatus.REJECTED_BY_HUMAN


def test_a_record_with_no_variations_keeps_its_status(tmp_path):
    """Gate-rejected records have no variations; roll-up must not invent a
    verdict for them."""
    rec = a_record(tmp_path, n=0)
    rec.pipeline_status = PipelineStatus.GATE_REJECTED
    assert roll_up(rec) is PipelineStatus.GATE_REJECTED
    with pytest.raises(ReviewError):
        decide(rec, approve=True)


# --- export --------------------------------------------------------------

def approved_record(tmp_path, approve=(0,), **kw) -> Record:
    rec = a_record(tmp_path, **kw)
    decide(rec, approve=True, indices=list(approve))
    rest = [i for i in range(len(rec.variations)) if i not in approve]
    if rest:
        decide(rec, approve=False, indices=rest, reason="not usable")
    return rec


def test_one_row_per_approved_variation_not_per_record(tmp_path):
    """Each design is its own Merch product; they only share title and brand."""
    rec = approved_record(tmp_path, approve=(0, 2))
    rows, skipped = export_rows([rec])

    assert len(rows) == 2 and not skipped
    assert {r["style_id"] for r in rows} == {"s0", "s2"}
    assert {r["title"] for r in rows} == {"A Title"}


def test_a_rejected_variation_never_reaches_the_export(tmp_path):
    rec = approved_record(tmp_path, approve=(0,))
    rows, _ = export_rows([rec])
    assert [r["style_id"] for r in rows] == ["s0"]


def test_a_record_still_awaiting_review_exports_nothing(tmp_path):
    rec = a_record(tmp_path)
    decide(rec, approve=True, indices=[0])          # half reviewed
    assert export_rows([rec]) == ([], [])


def test_the_intended_text_travels_with_the_row(tmp_path):
    """Lettering is model-rendered and verified nowhere, so the string it was
    supposed to say has to be readable at the upload desk, not only in `review`."""
    rec = approved_record(tmp_path)
    rows, _ = export_rows([rec])
    assert rows[0]["intended_text"] == "Text 0"


def test_a_print_file_that_vanished_is_reported_not_written_as_a_row(tmp_path):
    rec = approved_record(tmp_path)
    Path(rec.variations[0].print_path).unlink()

    rows, skipped = export_rows([rec])
    assert rows == []
    assert "missing from disk" in skipped[0]


def test_export_writes_every_column_it_declares(tmp_path):
    rec = approved_record(tmp_path)
    summary = export([rec], out_dir=tmp_path / "out")

    with (tmp_path / "out" / "listings.csv").open() as f:
        read = list(csv.DictReader(f))
    assert read[0].keys() == set(EXPORT_COLUMNS) or list(read[0]) == EXPORT_COLUMNS
    assert summary.rows == 1 and summary.records == 1
    assert read[0]["print_file"].endswith("_print.png")
    assert Path(read[0]["print_file"]).is_absolute()


def test_export_leaves_the_pipeline_status_alone(tmp_path):
    """Writing a CSV is not an upload to Amazon. UPLOADED stays unused until
    something actually uploads."""
    rec = approved_record(tmp_path)
    export([rec], out_dir=tmp_path / "out")
    assert rec.pipeline_status is PipelineStatus.APPROVED


def test_rerunning_export_is_idempotent(tmp_path):
    rec = approved_record(tmp_path)
    first = export([rec], out_dir=tmp_path / "out", copy_files=True)
    second = export([rec], out_dir=tmp_path / "out", copy_files=True)

    assert first.rows == second.rows == 1
    assert second.copied == 0, "re-copied a file that was already there"
    with (tmp_path / "out" / "listings.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 1, "rows accumulated across runs"


def test_copy_files_gathers_the_prints_into_one_folder(tmp_path):
    rec = approved_record(tmp_path, approve=(0, 1))
    summary = export([rec], out_dir=tmp_path / "out", copy_files=True)

    gathered = sorted(p.name for p in (tmp_path / "out" / "files").iterdir())
    assert len(gathered) == 2 and summary.copied == 2


def test_both_note_classes_reach_the_export_csv(tmp_path):
    """The gate decides what stops an approval; the CSV is what someone reads at
    the upload desk, and it gets everything."""
    rec = a_record(tmp_path, notes=[IGNORED_GROUND, COLLISION])
    decide(rec, approve=True, indices=[0], force=True)
    decide(rec, approve=False, indices=[1, 2], reason="not usable")

    rows, _ = export_rows([rec])
    assert IGNORED_GROUND in rows[0]["qa_notes"]
    assert COLLISION in rows[0]["qa_notes"]


def test_an_empty_export_still_writes_a_header_only_csv(tmp_path):
    """A missing file and an empty one are different states; the second says
    'nothing was approved', the first says 'export never ran'."""
    summary = export([a_record(tmp_path)], out_dir=tmp_path / "out")
    assert summary.rows == 0
    assert (tmp_path / "out" / "listings.csv").read_text().strip() == ",".join(EXPORT_COLUMNS)
