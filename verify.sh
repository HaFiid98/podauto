#!/usr/bin/env bash
# Full verification: config parse -> unit tests -> end-to-end CLI smoke test.
# One script so a single Bash call verifies everything at once.
set -uo pipefail
cd /home/ahssaini/podauto
. .venv/bin/activate

fail=0

echo "=============================================================="
echo "1. CONFIG JSON PARSE"
echo "=============================================================="
python - <<'PY' || fail=1
import json, sys
ok = True
for path, kind in {
    "config/styles.json": "styles",
    "config/denylist.json": "denylist",
    "config/content_policy.json": "policy",
    "config/providers.json": "providers",
    "config/providers.example.json": "providers",
}.items():
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"  FAIL {path}: {e}")
        ok = False
        continue
    print(f"  ok   {path}")
    if kind == "styles":
        styles = d["styles"]
        missing = [s["id"] for s in styles if not s.get("lettering")]
        print(f"       {len(styles)} styles, missing lettering: {missing or 'none'}")
        if len(styles) != 12 or missing:
            ok = False
sys.exit(0 if ok else 1)
PY

echo
echo "=============================================================="
echo "2. UNIT TESTS"
echo "=============================================================="
# -p no:randomly keeps ordering stable if pytest-randomly is installed;
# harmless if it is not.
python -m pytest tests/ -q --tb=short 2>&1 | tail -60 || fail=1

echo
echo "=============================================================="
echo "3. END-TO-END SMOKE TEST"
echo "=============================================================="
rm -f data/smoke.jsonl
python -m podauto.cli --store data/smoke.jsonl run data/sample_input.csv 2>&1 | tail -70 || fail=1

echo
echo "=============================================================="
echo "4. REVIEW QUEUE"
echo "=============================================================="
python -m podauto.cli --store data/smoke.jsonl review 2>&1 | head -100 || fail=1

echo
echo "=============================================================="
echo "5. PROVIDER SAFETY"
echo "=============================================================="
python - <<'PY' || fail=1
import json
from pathlib import Path

from podauto.env import load_env_file
from podauto.providers import Rotator, validate_config

# from_files, not load; candidates() is a generator, so it must be listed.

# (a) The committable template must be inert -- a fresh clone generates nothing.
#     Asserted on the template, not the live copy: once a licence is confirmed
#     the live copy is SUPPOSED to produce candidates, so asserting zero there
#     would turn a correct setup into a test failure.
tpl = Rotator.from_files("config/providers.example.json", "data/smoke_ledger.json")
assert not list(tpl.candidates()), "template must generate nothing"
print("  ok   template is inert -- a fresh clone generates no images")

# (b) The live copy must hold variable NAMES, never credentials.
live = Path("config/providers.json")
if not live.is_file():
    print("  --   no operator-local config/providers.json")
else:
    validate_config(json.loads(live.read_text()))
    print("  ok   live config holds no credentials")

    print(f"  env  loaded from .env: {load_env_file() or 'nothing (already set?)'}")
    rot = Rotator.from_files(live, "data/smoke_ledger.json")
    cands = list(rot.candidates())
    print(f"  live candidates: {len(cands)}"
          + (f" -> {[k.ident for k in cands]}" if cands else ""))
    for name, why in rot.skip_reasons().items():
        print(f"    skip {name}: {why}")
    # Markers that mean "the flag is set but the evidence behind it is not
    # recorded". Both spellings are checked because a warning that cannot fire
    # is the same defect as a config caution no code reads. Restricted to
    # enabled providers: a disabled one is not going to be called, so warning
    # that the rotator will call it would be false.
    UNRECORDED = ("UNVERIFIED", "EVIDENCE NOT YET RECORDED",
                  "SERVICE TERMS UNREAD")
    unconfirmed = [
        n for n, c in json.loads(live.read_text())["providers"].items()
        if c.get("commercial_use_confirmed") and c.get("enabled")
        and any(m in (c.get("licence_notes") or "").upper() for m in UNRECORDED)
    ]
    if unconfirmed:
        print("  WARN commercial_use_confirmed=true but licence_notes records no "
              f"evidence: {unconfirmed}")
        print("       The rotator will call these providers. Nothing here checks "
              "a licence -- only you can.")
PY
rm -f data/smoke_ledger.json

echo
echo "=============================================================="
echo "6. IMAGE PIPELINE READINESS"
echo "=============================================================="
# Information, not assertion: which parts of the image path can actually run
# today. Both imageqa gates now have producers -- `images` for --stage generated
# and `printready` for --stage print_ready.
python - <<'PY' || fail=1
import importlib

from podauto.imagegen import DEFAULT_NATIVE, MAX_NATIVE_H, native_size
from podauto.imageqa import ADVISED_SHORT_EDGE, ASPECT_TOLERANCE, TARGET_H, TARGET_W

# The two modules have to agree, or generation produces only QA failures. This
# was a real defect: the fallback was 832x1024, which is 2.5% off 5:6.
worst = max(abs(native_size({"max_resolution": c})[0]
                / native_size({"max_resolution": c})[1] - 5 / 6)
            for c in ("512x512", "1024x1024", "1216x1216", "2048x2048", "4500x5400"))
assert worst <= ASPECT_TOLERANCE, f"native_size can request {worst:.3f} off 5:6"
print(f"  ok   imagegen sizes are within {ASPECT_TOLERANCE} of 5:6 "
      f"(worst {worst:.4f}); fallback {DEFAULT_NATIVE}, ceiling h<={MAX_NATIVE_H}")

# Optional means optional: `printready --method raster` works without either of
# these, so a MISSING line here is a quality note, not a broken pipeline. rembg
# is deliberately absent -- for flat art on a uniform ground, colour keying gives
# a harder edge than salient-object segmentation and can report why it failed,
# without 34MB of onnxruntime and numpy wheels.
for mod, what in (("vtracer", "trace flat art to SVG (crisp 5.3x upscale)"),
                  ("resvg_py", "rasterize the trace at 4500x5400"),
                  ("PIL", "raster QA, removal, LANCZOS fallback")):
    try:
        importlib.import_module(mod)
        print(f"  ok   {mod:<8} present  ({what})")
    except Exception:                                          # noqa: BLE001
        print(f"  --   {mod:<8} MISSING  ({what})")

print("  note imageqa --stage generated is reachable now.")
print(f"       --stage print_ready asserts {TARGET_W}x{TARGET_H} with alpha; "
      f"`printready` is its producer (stage 7).")
print(f"       generated output is opaque at native size (advised short edge "
      f"{ADVISED_SHORT_EDGE}px).")
PY

echo
echo "=============================================================="
echo "7. PRINT-READY STAGE"
echo "=============================================================="
# Run the real thing at the real deliverable size, on a synthetic generation.
# The smoke store has no images (every provider is disabled by design), so this
# supplies its own input rather than depending on quota being spent. Both cases
# matter: a clean ground must key out to a passing print file, and a ground
# colour reused inside the artwork must survive removal and be reported -- that
# collision was measured in 2 of the first 3 real generations.
python - <<'PY' || fail=1
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from podauto.imageqa import check_print_ready
from podauto.printready import make_print_ready

GROUND = (255, 0, 255)


def art(path, hole=None):
    img = Image.new("RGB", (848, 1024), GROUND)
    d = ImageDraw.Draw(img)
    d.ellipse((170, 200, 680, 620), fill=(20, 40, 90))
    d.rectangle((170, 720, 680, 820), fill=(230, 180, 40))
    if hole:
        d.rectangle(hole, fill=GROUND)
    img.save(path)
    return path


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    clean = make_print_ready(art(tmp / "clean.png"), tmp / "clean_print.png",
                             requested_ground="#FF00FF")
    assert clean.ok, clean.failures
    report = check_print_ready(tmp / "clean_print.png")
    assert report.ok, report.failures
    print(f"  ok   clean ground -> {report.facts['width']}x{report.facts['height']} "
          f"{report.facts['mode']}, dpi {report.facts['dpi']}, "
          f"edge_opaque {report.facts['edge_opaque']}, method {clean.method}")

    dirty = make_print_ready(art(tmp / "dirty.png", hole=(300, 300, 450, 420)),
                             tmp / "dirty_print.png", requested_ground="#FF00FF")
    assert dirty.ok, dirty.failures
    collision = dirty.facts["ground_colour_in_art"]
    assert collision > 0.01, f"palette collision not measured ({collision})"
    assert any("reuses the ground colour" in w for w in dirty.warnings), dirty.warnings
    print(f"  ok   ground reused in the art is kept and reported "
          f"({collision:.1%} of the design)")

    wrong = make_print_ready(art(tmp / "wrong.png"), tmp / "wrong_print.png",
                             requested_ground="#00FF00")
    assert any("ignored the ground instruction" in w for w in wrong.warnings), wrong.warnings
    print("  ok   a model that painted the wrong ground is still salvaged, and said so")
PY

echo
echo "=============================================================="
echo "8. REVIEW DECISIONS AND EXPORT"
echo "=============================================================="
# The one path a human drives, and the only one that gets a listing out of the
# store. Driven through cli.main so the argparse wiring and exit codes are checked
# too, on a temp store so the real data is never touched. The two note strings are
# copied from data/quality2.jsonl: one fires on essentially every generation and
# must not block, one means the removal cut into the design and must.
python - <<'PY' || fail=1
import csv
import tempfile
from pathlib import Path

from PIL import Image

from podauto.cli import main
from podauto.models import (IdeaType, Listing, PipelineStatus, Record,
                            SourcePlatform, StyleVariation)
from podauto.store import Store

IGNORED_GROUND = ("print ready warning: asked for a #FF00FF ground, got #D53536 -- "
                  "the model ignored the ground instruction, so removal is keying "
                  "whatever it painted instead")
COLLISION = ("print ready warning: 4.5% of the art's bounding box is the ground "
             "colour #000000 (limit 1%) -- the design reuses the ground colour, so "
             "edges touching it will have been cut away; check the file before "
             "approving")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    store_path = tmp / "records.jsonl"

    def variation(i, notes):
        print_path = tmp / f"v{i}_print.png"
        Image.new("RGBA", (10, 12)).save(print_path)
        var = StyleVariation(style_id=f"s{i}", style_name=f"Style {i}",
                             graphic_prompt="p", text_spec={"line_1": f"TEXT {i}"},
                             qa_notes=list(notes))
        var.image_path = str(tmp / f"v{i}.png")
        var.print_path = str(print_path)
        return var

    rec = Record(id="stage8-0000-0000", timestamp="2026-08-18T00:00:00",
                 source_url="https://example.com/verify",
                 source_platform=SourcePlatform.AMAZON,
                 idea_type=IdeaType.READY_SHIRT, niche="Verify Niche",
                 raw_title_or_text="Verify Niche")
    rec.pipeline_status = PipelineStatus.AWAITING_REVIEW
    rec.listing = Listing(title="A Verify Title", brand="A Brand", bullet_1="One",
                          bullet_2="Two", description="Desc")
    rec.variations = [variation(0, [IGNORED_GROUND]), variation(1, [COLLISION]),
                      variation(2, [])]
    Store(store_path).upsert(rec)

    def cli(*args):
        return main(["--store", str(store_path), *args])

    def reload():
        return Store(store_path).all()[0]

    assert cli("approve", "stage8", "--variation", "0") == 0
    assert reload().variations[0].review_decision == "approved"
    print("  ok   the ignored-ground note (3 of 3 real generations) did not block")

    assert cli("approve", "stage8", "--variation", "1") == 1, "collision waved through"
    assert not reload().variations[1].review_decision
    assert cli("approve", "stage8", "--variation", "1", "--force") == 0
    print("  ok   a ground-colour collision held the approval until --force")

    assert reload().pipeline_status is PipelineStatus.AWAITING_REVIEW
    print("  ok   a half-decided record is still AWAITING_REVIEW, so it stays in "
          "the queue")

    assert cli("reject", "stage8", "--reason", "misspelled lettering") == 0
    rec = reload()
    assert rec.pipeline_status is PipelineStatus.APPROVED, rec.pipeline_status
    assert rec.variations[2].review_decision == "misspelled lettering"
    assert rec.variations[0].review_decision == "approved", "the sweep overwrote it"
    print("  ok   deciding the last variation rolled the record up to APPROVED")

    assert cli("export", "--out-dir", str(tmp / "export"), "--copy-files") == 0
    rows = list(csv.DictReader((tmp / "export" / "listings.csv").open()))
    assert len(rows) == 2, f"expected one row per approved variation, got {len(rows)}"
    assert {r["intended_text"] for r in rows} == {"TEXT 0", "TEXT 1"}
    assert len(list((tmp / "export" / "files").iterdir())) == 2
    assert reload().pipeline_status is PipelineStatus.APPROVED, "export moved a status"
    print(f"  ok   export wrote {len(rows)} rows (one per approved variation), "
          f"copied 2 files, and changed no status")
PY

echo
echo "=============================================================="
[ "$fail" -eq 0 ] && echo "RESULT: all stages green" || echo "RESULT: failures above"
echo "=============================================================="
exit "$fail"
