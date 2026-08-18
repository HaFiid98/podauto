"""CLI stage-handoff tests.

The per-stage commands each filter on `pipeline_status` and advance it. That
chain is the part most likely to break silently: a stage that filters on a
status no upstream stage ever sets processes zero records and still exits 0,
so the pipeline reports success while doing nothing. These tests assert
records actually arrive at the far end.

Paths to config/ are relative, so pytest must run from the repo root -- same
assumption the other test modules already make.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from podauto.cli import main
from podauto.models import ContentPolicyStatus, PipelineStatus, TmStatus
from podauto.store import Store

CSV = """source_url,source_platform,idea_type,niche,raw_title_or_text,bsr_rank,bsr_category,trend_velocity,listings_count
https://amazon.com/dp/B01,amazon,ready_shirt,Pickleball Dad,Pickleball Dad Retro Sunset,45000,Clothing,78,40
https://amazon.com/dp/B03,amazon,ready_shirt,Star Wars Fan,Vintage Star Wars Dad Life,45000,Clothing,78,40
https://reddit.com/r/x/4,reddit,trend_idea,Cannabis Culture,420 Friendly Weed Leaf Design,,,90,30
https://etsy.com/listing/1,etsy,ready_shirt,Broken Row,This row has an invalid platform,,,,
"""

# bsr_captured_at is deliberately omitted. Supplying a fixed date would make
# these tests start failing 30 days after they were written, when the scorer
# begins dropping the reading as stale.


@pytest.fixture
def pipeline(tmp_path):
    """Run every stage once, return the resulting store."""
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(CSV)
    store_path = tmp_path / "records.jsonl"
    rc = main(["--store", str(store_path), "run", str(csv_path)])
    assert rc == 0
    return Store(store_path)


def by_niche(store) -> dict:
    return {r.niche: r for r in store.all()}


# --- the chain actually connects ----------------------------------------

def test_a_clean_record_reaches_the_review_queue(pipeline):
    """The whole point: CSV in, AWAITING_REVIEW out, nothing stuck midway."""
    rec = by_niche(pipeline)["Pickleball Dad"]
    assert rec.pipeline_status is PipelineStatus.AWAITING_REVIEW
    assert rec.listing.title and rec.listing.brand
    assert rec.variations, "prompts stage produced nothing"


def test_no_record_is_stranded_in_an_intermediate_status(pipeline):
    """A stage filtering on a status nobody sets is a silent no-op."""
    stranded = {PipelineStatus.INGESTED, PipelineStatus.DEDUPED,
                PipelineStatus.SCORED, PipelineStatus.TRIAGED,
                PipelineStatus.PROMPTED}
    for rec in pipeline.all():
        assert rec.pipeline_status not in stranded, f"{rec.niche} stuck at {rec.pipeline_status}"


# --- gates stop what they should ----------------------------------------

def test_trademark_block_stops_before_review(pipeline):
    rec = by_niche(pipeline)["Star Wars Fan"]
    assert rec.tm_status is TmStatus.BLOCKED
    assert rec.pipeline_status is PipelineStatus.GATE_REJECTED
    assert not rec.variations, "blocked record must not spend image quota"


def test_policy_failure_stops_before_prompts(pipeline):
    rec = by_niche(pipeline)["Cannabis Culture"]
    assert rec.content_policy_status is ContentPolicyStatus.FAIL
    assert rec.pipeline_status is PipelineStatus.GATE_REJECTED
    assert not rec.variations


def test_invalid_row_is_reported_not_ingested(pipeline):
    """etsy is not a SourcePlatform. It must not silently become a record."""
    assert "Broken Row" not in by_niche(pipeline)


# --- generated text is screened -----------------------------------------

def test_listing_stage_rescreens_generated_text(pipeline):
    """Triage runs before listing exists, so the brand it generates was never
    screened by that pass. cmd_listing re-screens; tm_status must be set on
    records that reached review."""
    rec = by_niche(pipeline)["Pickleball Dad"]
    assert rec.tm_status is not TmStatus.UNCHECKED


# --- re-runnability ------------------------------------------------------

def test_rerunning_is_idempotent(tmp_path):
    """An interrupted run should resume, not duplicate or re-process."""
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(CSV)
    store_path = tmp_path / "records.jsonl"

    assert main(["--store", str(store_path), "run", str(csv_path)]) == 0
    first = {r.id: r.pipeline_status for r in Store(store_path).all()}

    assert main(["--store", str(store_path), "run", str(csv_path)]) == 0
    second = {r.id: r.pipeline_status for r in Store(store_path).all()}

    assert first == second, "second run changed records or added duplicates"


def test_review_command_runs_on_a_populated_store(pipeline, capsys):
    rc = main(["--store", str(pipeline.path), "review"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Pickleball Dad" in out
    # The manual check is the pipeline's safety property; the queue must say so.
    assert "model-rendered" in out


# --- image stages --------------------------------------------------------
#
# No network: httpx.Client is replaced with one wired to a MockTransport. The
# provider config is written per-test so these never touch config/providers.json,
# which is the operator's file and may have any provider enabled.

STUB_PROVIDERS = {
    "rotation": {"order": ["stub"], "cooldown_seconds": 900,
                 "max_attempts_per_image": 6},
    "providers": {"stub": {
        "enabled": True, "commercial_use_confirmed": True,
        "endpoint": "https://stub.test/v1/models/{model}",
        "api_shape": "hf_text_to_image", "model": "stub/model",
        "auth": "bearer", "max_resolution": "1024x1024",
        "keys": ["STUB_KEY"],
    }},
}


def png_response(size=(848, 1024), blank=False) -> httpx.Response:
    """848x1024 is what native_size() asks for at a 1024 cap, so a passing
    fixture has to be that size -- imageqa checks the aspect the request used."""
    img = Image.new("RGB", size, (250, 250, 250))
    if not blank:
        w, h = size
        img.paste((20, 30, 40), (w // 5, h // 5, w * 4 // 5, h * 4 // 5))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return httpx.Response(200, content=buf.getvalue(),
                          headers={"content-type": "image/png"})


@pytest.fixture
def stub_images(monkeypatch, tmp_path):
    """Point `images` at a stub provider and answer every request locally."""
    monkeypatch.setenv("STUB_KEY", "not-a-real-key")
    config = tmp_path / "providers.json"
    config.write_text(json.dumps(STUB_PROVIDERS))
    real_client = httpx.Client
    state = {"response": png_response, "requests": 0}

    def factory(*args, **kwargs):
        def handler(request):
            state["requests"] += 1
            return state["response"]()
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)

    def run(store_path, *extra, blank=False):
        state["response"] = (lambda: png_response(blank=True)) if blank else png_response
        return main(["--store", str(store_path), "images",
                     "--providers", str(config),
                     "--ledger", str(tmp_path / "ledger.json"),
                     "--out-dir", str(tmp_path / "images"), *extra])

    run.state = state
    return run


def test_run_spends_no_quota(pipeline):
    """`run` stops at listing. Image generation is the only irreversible cost in
    the pipeline, so it has to be a command the operator types on purpose."""
    assert all(not v.image_path for r in pipeline.all() for v in r.variations)


def test_images_writes_files_and_leaves_the_record_in_the_review_queue(pipeline,
                                                                      stub_images):
    """Images enrich a record already queued for review. If this advanced the
    status past AWAITING_REVIEW the record would silently leave the queue, and
    that manual check is the pipeline's safety property."""
    assert stub_images(pipeline.path) == 0

    rec = by_niche(Store(pipeline.path))["Pickleball Dad"]
    assert rec.pipeline_status is PipelineStatus.AWAITING_REVIEW
    assert rec.variations[0].image_path
    assert Path(rec.variations[0].image_path).is_file()
    assert rec.variations[0].image_provider == "stub:STUB_KEY"


def test_rerunning_images_does_not_pay_for_the_same_file_twice(pipeline, stub_images):
    assert stub_images(pipeline.path) == 0
    first = stub_images.state["requests"]
    assert first > 0

    assert stub_images(pipeline.path) == 0
    assert stub_images.state["requests"] == first, "re-run re-requested existing files"


def test_review_shows_the_file_to_open(pipeline, stub_images, capsys):
    assert stub_images(pipeline.path) == 0
    capsys.readouterr()

    assert main(["--store", str(pipeline.path), "review"]) == 0
    out = capsys.readouterr().out
    assert ".png" in out


def test_imageqa_passes_a_native_generation(pipeline, stub_images, capsys):
    assert stub_images(pipeline.path) == 0
    capsys.readouterr()

    assert main(["--store", str(pipeline.path), "imageqa"]) == 0
    out = capsys.readouterr().out
    assert "fail=0" in out
    assert all(r.pipeline_status is not PipelineStatus.IMAGES_QA_FAILED
               for r in Store(pipeline.path).all())


def test_imageqa_pulls_a_blank_generation_out_of_the_review_queue(pipeline,
                                                                 stub_images, capsys):
    """A solid frame is a valid PNG, so nothing before QA rejects it. The record
    needs regenerating, not reviewing -- so it must leave the queue."""
    assert stub_images(pipeline.path, blank=True) == 0
    assert main(["--store", str(pipeline.path), "imageqa"]) == 0
    capsys.readouterr()

    failed = [r for r in Store(pipeline.path).all()
              if r.pipeline_status is PipelineStatus.IMAGES_QA_FAILED]
    assert failed, "a blank canvas passed QA"
    assert any("blank or near-uniform" in n for n in failed[0].variations[0].qa_notes)

    assert main(["--store", str(pipeline.path), "review"]) == 0
    assert failed[0].niche not in capsys.readouterr().out


def test_a_fresh_clone_generates_nothing_and_says_why(pipeline, tmp_path, capsys):
    """The shipped template has every provider off. Generating nothing is correct;
    doing it silently is not, so every provider in the rotation has to appear with
    a reason -- otherwise this reads as an empty queue or a network fault."""
    rc = main(["--store", str(pipeline.path), "images",
               "--providers", "config/providers.example.json",
               "--ledger", str(tmp_path / "ledger.json"),
               "--out-dir", str(tmp_path / "images")])
    captured = capsys.readouterr()

    assert rc == 0
    assert "generated=0" in captured.out
    for name in ("hf_nscale", "hf_together", "pollinations", "cloudflare",
                 "gemini", "together", "huggingface"):
        assert f"skip {name}:" in captured.out
    assert not list((tmp_path / "images").iterdir())


# --- print-ready stage ---------------------------------------------------
#
# The print canvas is shrunk for these: `printready` at its real 4500x5400 costs
# a couple of seconds per variation, and nothing here is testing PIL's resizer.

@pytest.fixture
def small_print(monkeypatch):
    """Run `printready` on a 250x300 canvas instead of 4500x5400."""
    import functools

    from podauto import cli
    from podauto.printready import printready_record

    monkeypatch.setattr(cli, "printready_record",
                        functools.partial(printready_record, size=(250, 300)))


def test_printready_produces_a_file_and_keeps_the_record_in_review(pipeline,
                                                                  stub_images,
                                                                  small_print):
    assert stub_images(pipeline.path) == 0
    assert main(["--store", str(pipeline.path), "printready",
                 "--method", "raster"]) == 0

    rec = by_niche(Store(pipeline.path))["Pickleball Dad"]
    assert rec.pipeline_status is PipelineStatus.AWAITING_REVIEW
    var = rec.variations[0]
    assert var.print_path and Path(var.print_path).is_file()
    assert var.image_path and Path(var.image_path).is_file(), "generated file was consumed"


def test_printready_is_idempotent(pipeline, stub_images, small_print, capsys):
    assert stub_images(pipeline.path) == 0
    assert main(["--store", str(pipeline.path), "printready", "--method", "raster"]) == 0
    capsys.readouterr()

    assert main(["--store", str(pipeline.path), "printready", "--method", "raster"]) == 0
    assert "produced=0" in capsys.readouterr().out


def test_printready_before_images_does_nothing_and_says_so(pipeline, capsys):
    assert main(["--store", str(pipeline.path), "printready"]) == 0
    assert "no generated images to convert" in capsys.readouterr().out


def test_print_qa_checks_the_print_file_not_the_generated_one(pipeline, stub_images,
                                                              small_print, capsys):
    """--stage print_ready run against image_path would fail every record on
    dimensions and alpha. Run against a real 250x300 print file it fails on the
    size only, which is the one thing the gate hard-codes."""
    assert stub_images(pipeline.path) == 0
    assert main(["--store", str(pipeline.path), "printready", "--method", "raster"]) == 0
    capsys.readouterr()

    assert main(["--store", str(pipeline.path), "imageqa",
                 "--stage", "print_ready"]) == 0
    out = capsys.readouterr().out
    assert "expected 4500x5400" in out
    assert "no alpha channel" not in out, "the print file has alpha; wrong file checked"


def test_print_qa_before_printready_finds_nothing_rather_than_failing_everything(
        pipeline, stub_images, capsys):
    assert stub_images(pipeline.path) == 0
    capsys.readouterr()

    assert main(["--store", str(pipeline.path), "imageqa",
                 "--stage", "print_ready"]) == 0
    assert "no generated images to check" in capsys.readouterr().out
    assert all(r.pipeline_status is not PipelineStatus.IMAGES_QA_FAILED
               for r in Store(pipeline.path).all())


def test_review_points_at_the_print_file_as_the_one_to_upload(pipeline, stub_images,
                                                             small_print, capsys):
    assert stub_images(pipeline.path) == 0
    assert main(["--store", str(pipeline.path), "printready", "--method", "raster"]) == 0
    capsys.readouterr()

    assert main(["--store", str(pipeline.path), "review"]) == 0
    out = capsys.readouterr().out
    assert "_print.png" in out
    assert "upload this one" in out



# --- review decisions and export -----------------------------------------
#
# The stub art is a near-white ground, never the chroma colour the prompt asked
# for, so every stub variation carries the ignored-ground note -- the same note
# all 3 real generations carry. That makes it the natural fixture for the one
# thing this section has to prove: an approval is not blocked by it.

def print_ready_store(store_path, stub_images, capsys):
    assert stub_images(store_path) == 0
    assert main(["--store", str(store_path), "printready", "--method", "raster"]) == 0
    capsys.readouterr()
    return by_niche(Store(store_path))["Pickleball Dad"]


def test_the_decision_chain_gets_one_listing_out_of_the_store(pipeline, stub_images,
                                                              small_print, tmp_path,
                                                              capsys):
    """CSV in, one uploadable row out. The whole pipeline in one test."""
    rec = print_ready_store(pipeline.path, stub_images, capsys)
    rid = rec.id[:8]

    assert main(["--store", str(pipeline.path), "approve", rid, "--variation", "0"]) == 0
    out = capsys.readouterr().out
    assert "note [0]" in out, "the ignored-ground note was not reported"
    assert "ok   [0]" in out

    # Half reviewed: still in the queue, deliberately.
    assert by_niche(Store(pipeline.path))["Pickleball Dad"].pipeline_status \
        is PipelineStatus.AWAITING_REVIEW

    assert main(["--store", str(pipeline.path), "reject", rid,
                 "--reason", "lettering doubled"]) == 0
    out = capsys.readouterr().out
    assert "already decided" in out, "the approval was overwritten by the sweep"

    rec = by_niche(Store(pipeline.path))["Pickleball Dad"]
    assert rec.pipeline_status is PipelineStatus.APPROVED
    assert rec.variations[0].review_decision == "approved"
    assert rec.variations[1].review_decision == "lettering doubled"

    export_dir = tmp_path / "export"
    assert main(["--store", str(pipeline.path), "export",
                 "--out-dir", str(export_dir), "--copy-files"]) == 0
    assert "1 listings from 1 records" in capsys.readouterr().out

    rows = list(csv.DictReader((export_dir / "listings.csv").open()))
    assert len(rows) == 1
    assert rows[0]["title"] == rec.listing.title
    assert Path(rows[0]["print_file"]).is_file()
    assert len(list((export_dir / "files").iterdir())) == 1

    assert main(["--store", str(pipeline.path), "review"]) == 0
    assert rid not in capsys.readouterr().out, "an approved record stayed in the queue"


def test_an_approval_is_not_blocked_by_the_ground_note_but_is_by_a_collision(
        pipeline, stub_images, small_print, capsys):
    """The gate has to stay meaningful: routine note through, damage note stops."""
    rec = print_ready_store(pipeline.path, stub_images, capsys)
    store = Store(pipeline.path)
    rec.variations[1].qa_notes.append(
        "print ready warning: 4.5% of the art's bounding box is the ground colour "
        "#000000 (limit 1%) -- the design reuses the ground colour, so edges "
        "touching it will have been cut away; check the file before approving")
    store.upsert(rec)

    assert main(["--store", str(pipeline.path), "approve", rec.id[:8],
                 "--variation", "1"]) == 1
    out = capsys.readouterr().out
    assert "blocking QA warning" in out
    assert "reuses the ground colour" in out, "the note that blocked was not shown"
    assert not by_niche(Store(pipeline.path))["Pickleball Dad"].variations[1].review_decision

    assert main(["--store", str(pipeline.path), "approve", rec.id[:8],
                 "--variation", "1", "--force"]) == 0
    assert by_niche(Store(pipeline.path))["Pickleball Dad"] \
        .variations[1].review_decision == "approved"


def test_a_rejection_with_no_reason_changes_nothing(pipeline, stub_images,
                                                    small_print, capsys):
    rec = print_ready_store(pipeline.path, stub_images, capsys)

    assert main(["--store", str(pipeline.path), "reject", rec.id[:8]]) == 1
    assert "--reason" in capsys.readouterr().err
    after = by_niche(Store(pipeline.path))["Pickleball Dad"]
    assert after.pipeline_status is PipelineStatus.AWAITING_REVIEW
    assert all(not v.review_decision for v in after.variations)


def test_an_ambiguous_or_unknown_record_id_exits_nonzero(pipeline, capsys):
    assert main(["--store", str(pipeline.path), "approve", "zzzzzz"]) == 1
    assert "no record id starts with" in capsys.readouterr().err


def test_approving_before_printready_is_refused(pipeline, capsys):
    """Nothing to upload yet. This is the state every record is in until the
    print stage has run."""
    rec = by_niche(pipeline)["Pickleball Dad"]
    assert main(["--store", str(pipeline.path), "approve", rec.id[:8]]) == 1
    out = capsys.readouterr().out
    assert "no print file" in out
    assert "the store is unchanged" in out


def test_export_before_any_approval_says_so_and_writes_no_rows(pipeline, tmp_path,
                                                               capsys):
    export_dir = tmp_path / "export"
    assert main(["--store", str(pipeline.path), "export",
                 "--out-dir", str(export_dir)]) == 0
    assert "nothing approved yet" in capsys.readouterr().out
    assert list(csv.DictReader((export_dir / "listings.csv").open())) == []
