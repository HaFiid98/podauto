"""Print-ready stage tests: ground removal, print sizing, and the collision report.

The stage's whole reason to exist is that Merch prints every opaque pixel as ink,
so these tests care about two things above all: that the ground actually becomes
transparent, and that a ground colour reused *inside* the artwork is left alone
and reported rather than silently punched out. That second case is not
hypothetical -- it was measured in 2 of the first 3 real generations.

`size` is passed small on purpose. Rasterizing 24 megapixels per assertion would
make this module slower than the rest of the suite combined, and nothing here is
testing PIL's ability to resize.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from podauto.imageqa import check_print_ready
from podauto.models import PipelineStatus, Record, StyleVariation
from podauto.printready import (
    ALPHA_FLOOR,
    as_rgb,
    border_colour,
    fit_to_canvas,
    make_print_ready,
    print_path_for,
    printready_record,
    printready_variation,
    remove_ground,
)

MAGENTA = (255, 0, 255)
INK = (20, 40, 90)
GOLD = (230, 180, 40)

# A 5:6 canvas, small enough to be fast and big enough that a 1px border vote
# and a bounding box still mean something.
SMALL = (100, 120)
TEST_SIZE = (250, 300)


def art_on_ground(size=SMALL, ground=MAGENTA, hole=None) -> Image.Image:
    """Flat art centred on a uniform ground -- what the generator is asked for.

    `hole` fills a patch inside the art with the ground colour, which is the
    palette-collision case.
    """
    img = Image.new("RGB", size, ground)
    d = ImageDraw.Draw(img)
    w, h = size
    d.ellipse((w // 5, h // 5, w * 4 // 5, h * 3 // 5), fill=INK)
    d.rectangle((w // 5, h * 7 // 10, w * 4 // 5, h * 8 // 10), fill=GOLD)
    if hole:
        d.rectangle(hole, fill=ground)
    return img


@pytest.fixture
def clean_png(tmp_path) -> Path:
    path = tmp_path / "clean.png"
    art_on_ground().save(path)
    return path


def alpha_at(img: Image.Image, xy) -> int:
    return img.convert("RGBA").getchannel("A").getpixel(xy)


# --- colour plumbing -----------------------------------------------------

def test_as_rgb_reads_the_hex_the_config_stores():
    assert as_rgb("#FF00FF") == MAGENTA
    assert as_rgb("00ff00") == (0, 255, 0)
    assert as_rgb((1, 2, 3)) == (1, 2, 3)


def test_as_rgb_rejects_something_that_is_not_a_colour():
    with pytest.raises(ValueError):
        as_rgb("magenta")


def test_border_colour_votes_rather_than_sampling_a_corner():
    """A corner can land on a speck; a modal vote over the whole border cannot."""
    img = art_on_ground()
    img.putpixel((0, 0), (7, 7, 7))
    img.putpixel((99, 119), (9, 9, 9))
    assert border_colour(img) == MAGENTA


# --- removal ------------------------------------------------------------

def test_the_ground_becomes_transparent_and_the_art_stays_opaque():
    removal = remove_ground(art_on_ground())
    out = removal.image
    assert out.mode == "RGBA"
    assert alpha_at(out, (2, 2)) == 0                      # ground
    assert alpha_at(out, (50, 40)) == 255                  # inside the ellipse
    assert removal.facts["ground"] == "#FF00FF"
    assert not removal.warnings


def test_ground_colour_inside_the_art_is_kept_and_reported():
    """The defect this stage exists to survive. A plain colour threshold would
    punch a hole through the design here; only border connectivity keeps it, and
    the leftover is the measurement that tells the reviewer to look."""
    removal = remove_ground(art_on_ground(hole=(40, 30, 60, 50)))

    assert alpha_at(removal.image, (50, 40)) == 255, "ground colour in the art was cut out"
    assert removal.facts["ground_colour_in_art"] > 0.01
    assert any("reuses the ground colour" in w for w in removal.warnings)


def test_a_model_that_ignored_the_requested_ground_is_reported_as_such():
    """Removal still keys what is actually there -- the file is salvageable -- but
    a prompt the model disobeyed is a prompt defect and has to be visible."""
    removal = remove_ground(art_on_ground(ground=(12, 34, 90)), requested="#FF00FF")

    assert alpha_at(removal.image, (2, 2)) == 0
    assert removal.facts["ground"] == "#0C225A"
    assert removal.facts["ground_requested"] == "#FF00FF"
    assert any("ignored the ground instruction" in w for w in removal.warnings)


def test_full_bleed_art_with_no_removable_ground_warns():
    """Merch prints every opaque pixel, so "nothing to remove" is not a pass.
    Deterministic pseudo-noise rather than random: a flaky QA test is worse than
    no QA test."""
    busy = Image.new("RGB", SMALL)
    w, h = SMALL
    seed = 12345
    px = []
    for _ in range(w * h):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        px.append(((seed >> 16) & 255, (seed >> 8) & 255, seed & 255))
    busy.putdata(px)

    removal = remove_ground(busy)
    assert removal.facts["ground_fraction"] < 0.05
    assert any("full-bleed art" in w for w in removal.warnings)


def test_near_ground_pixels_ramp_rather_than_cutting_hard():
    """Ground noise and JPEG ringing sit a few counts off the ground colour. They
    key out, but by a proportional ramp -- a hard 0/255 threshold on a soft edge
    prints as jaggies, and the ramp is what buys the anti-aliasing.

    The probe pixels go in the ground region, not inside the artwork: alpha only
    ramps where the fill actually reached, which is exactly what keeps a
    ground-coloured patch inside the design fully opaque."""
    img = art_on_ground()
    img.putpixel((5, 5), (230, 25, 255))       # 25 counts off -- inside tolerance
    img.putpixel((7, 5), (252, 3, 255))        # 3 counts off -- ringing

    out = remove_ground(img).image
    assert ALPHA_FLOOR <= alpha_at(out, (5, 5)) < 255
    assert alpha_at(out, (7, 5)) == 0, "near-identical ground must go fully clear"


# --- print canvas -------------------------------------------------------

def test_fit_to_canvas_pads_instead_of_stretching():
    """848x1024 is 0.8281 and the canvas is 0.8333. Resizing into it would
    stretch every design 0.6% horizontally; padding costs nothing because
    transparent pixels print as nothing."""
    art = Image.new("RGBA", (4472, 5400), (10, 20, 30, 255))
    out = fit_to_canvas(art, (4500, 5400))

    assert out.size == (4500, 5400)
    assert out.getpixel((0, 2700))[3] == 0, "pad must be transparent"
    assert out.getpixel((2250, 2700)) == (10, 20, 30, 255)


def test_fit_to_canvas_crops_something_oversized_rather_than_failing():
    art = Image.new("RGBA", (300, 400), (10, 20, 30, 255))
    assert fit_to_canvas(art, (100, 120)).size == (100, 120)


# --- the whole stage ----------------------------------------------------

@pytest.mark.parametrize("method", ["raster", "auto"])
def test_the_produced_file_passes_the_print_gate(tmp_path, clean_png, method):
    out = tmp_path / "print.png"
    result = make_print_ready(clean_png, out, requested_ground="#FF00FF",
                              method=method, size=TEST_SIZE)

    assert result.ok, result.failures
    assert result.method in ("raster", "vector")
    report = check_print_ready(out)
    # Size is the one thing the gate hard-codes to 4500x5400, so a small test
    # canvas is expected to trip exactly that and nothing else.
    assert [f for f in report.failures if "expected 4500x5400" not in f] == []
    assert report.facts["edge_opaque"] == 0.0
    assert report.facts["dpi"] == (300, 300)


def test_the_real_print_size_is_produced_at_300_dpi_with_alpha(tmp_path, clean_png):
    """One full-size pass, because 4500x5400 at 300dpi is the actual deliverable
    and a stage that only works at test dimensions is not a stage."""
    out = tmp_path / "full.png"
    result = make_print_ready(clean_png, out, method="raster")

    assert result.ok, result.failures
    assert check_print_ready(out).ok


def test_a_missing_source_fails_cleanly_without_raising(tmp_path):
    result = make_print_ready(tmp_path / "nope.png", tmp_path / "out.png")
    assert not result.ok
    assert "no file at" in result.failures[0]
    assert not (tmp_path / "out.png").exists()


def test_a_source_that_is_not_an_image_fails_cleanly(tmp_path):
    junk = tmp_path / "junk.png"
    junk.write_text("this is not a PNG")
    result = make_print_ready(junk, tmp_path / "out.png")
    assert not result.ok
    assert "could not read" in result.failures[0]


def test_vector_method_reports_failure_instead_of_downgrading_silently(tmp_path,
                                                                      clean_png,
                                                                      monkeypatch):
    """--method vector is an instruction. Quietly returning a soft LANCZOS file
    would look identical until the shirt arrived."""
    import podauto.printready as pr
    monkeypatch.setattr(pr, "vector_upscale",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("no vtracer")))

    result = make_print_ready(clean_png, tmp_path / "out.png", method="vector",
                              size=TEST_SIZE)
    assert not result.ok
    assert "vectorize failed" in result.failures[0]


def test_auto_method_falls_back_and_says_so(tmp_path, clean_png, monkeypatch):
    import podauto.printready as pr
    monkeypatch.setattr(pr, "vector_upscale",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("no vtracer")))

    result = make_print_ready(clean_png, tmp_path / "out.png", method="auto",
                              size=TEST_SIZE)
    assert result.ok
    assert result.method == "raster"
    assert any("fell back to a LANCZOS upscale" in w for w in result.warnings)


# --- record level -------------------------------------------------------

def a_record(tmp_path, n=1, hole=None) -> Record:
    rec = Record(id="rec-1", timestamp="2026-08-17T00:00:00", source_url="u",
                 source_platform="amazon", idea_type="ready_shirt",
                 niche="Pickleball Dad", raw_title_or_text="Pickleball Dad")
    rec.pipeline_status = PipelineStatus.AWAITING_REVIEW
    for i in range(n):
        src = tmp_path / f"gen_{i}.png"
        art_on_ground(hole=hole).save(src)
        rec.variations.append(StyleVariation(
            style_id=f"s{i}", style_name=f"Style {i}", graphic_prompt="p",
            ground_hint="#FF00FF", image_path=str(src)))
    return rec


def test_print_path_sits_beside_the_generated_file_by_default(tmp_path):
    var = StyleVariation("s", "S", "p", image_path=str(tmp_path / "a_0_retro.png"))
    assert print_path_for(var) == tmp_path / "a_0_retro_print.png"
    assert print_path_for(var, tmp_path / "prints") == tmp_path / "prints" / "a_0_retro_print.png"


def test_printready_sets_print_path_without_replacing_the_generated_one(tmp_path):
    """Both files are kept. When print QA rejects a file, the generated original
    is the only evidence for whether the prompt, the provider or the removal is
    at fault."""
    rec = a_record(tmp_path)
    var = rec.variations[0]
    summary = printready_record(rec, method="raster", size=TEST_SIZE)

    assert summary.produced == 1 and summary.failed == 0
    assert var.image_path and Path(var.image_path).is_file()
    assert var.print_path and Path(var.print_path).is_file()
    assert var.print_path != var.image_path


def test_printready_leaves_the_record_in_the_review_queue(tmp_path):
    """Producing a file is not a review. Anything that moved the record out of
    AWAITING_REVIEW would make the manual check skippable by a side effect."""
    rec = a_record(tmp_path)
    printready_record(rec, method="raster", size=TEST_SIZE)
    assert rec.pipeline_status is PipelineStatus.AWAITING_REVIEW


def test_rerunning_skips_a_variation_that_already_has_its_file(tmp_path):
    rec = a_record(tmp_path)
    assert printready_record(rec, method="raster", size=TEST_SIZE).produced == 1

    again = printready_record(rec, method="raster", size=TEST_SIZE)
    assert again.produced == 0 and again.skipped == 1


def test_force_reruns_but_does_not_stack_duplicate_notes(tmp_path):
    """The command is re-runnable, so an accumulating list of identical warnings
    would make the review queue unreadable."""
    rec = a_record(tmp_path, hole=(40, 30, 60, 50))
    for _ in range(3):
        printready_record(rec, method="raster", force=True, size=TEST_SIZE)

    notes = rec.variations[0].qa_notes
    collision = [n for n in notes if "reuses the ground colour" in n]
    assert len(collision) == 1, notes


def test_a_variation_with_no_generated_image_is_skipped_not_failed(tmp_path):
    rec = a_record(tmp_path)
    rec.variations.append(StyleVariation("s9", "S9", "p"))
    summary = printready_record(rec, method="raster", size=TEST_SIZE)
    assert summary.produced == 1 and summary.skipped == 1 and summary.failed == 0


def test_the_collision_warning_reaches_the_reviewer_through_qa_notes(tmp_path):
    rec = a_record(tmp_path, hole=(40, 30, 60, 50))
    printready_record(rec, method="raster", size=TEST_SIZE)
    assert any("reuses the ground colour" in n for n in rec.variations[0].qa_notes)


def test_print_qa_reads_the_print_file_not_the_generated_one(tmp_path):
    """imageqa --stage print_ready pointed at image_path would assert 4500x5400
    with alpha against an opaque 848x1024 native file and fail every record."""
    from podauto.imageqa import qa_variation

    rec = a_record(tmp_path)
    var = rec.variations[0]
    assert qa_variation(var, stage="print_ready") is None, "nothing to check yet"

    printready_record(rec, method="raster")
    report = qa_variation(var, stage="print_ready")
    assert report is not None
    assert report.path == var.print_path
    assert report.ok, report.failures
