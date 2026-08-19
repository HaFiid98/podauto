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

from podauto import lettering
from podauto.imageqa import TARGET_H, TARGET_W, check_print_ready
from podauto.models import PipelineStatus, Record, StyleVariation
from podauto.printready import (
    OVERLAY_LEGIBILITY_PHRASE,
    ALPHA_FLOOR,
    ART_MARGIN,
    as_rgb,
    border_colour,
    fit_to_canvas,
    fit_width,
    inset_box,
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


def full_bleed(size=SMALL, ground=MAGENTA) -> Image.Image:
    """Art that runs off the top and bottom of the frame, leaving ground at the
    sides -- which is what the providers actually return. Measured on 3 of 3 real
    generations: the alpha bbox after removal is (0, 0, 848, 1024), touching every
    edge, because the model paints past the canvas.

    Every other fixture here centres the art with clearance around it, so none of
    them can catch a print file whose art sits on the border. That is exactly how
    2 of 3 real print files came to fail MAX_EDGE_OPAQUE with the suite green.
    """
    img = Image.new("RGB", size, ground)
    d = ImageDraw.Draw(img)
    w, h = size
    d.ellipse((w // 10, -h // 4, w * 9 // 10, h * 5 // 4), fill=INK)
    d.rectangle((w // 5, h * 2 // 5, w * 4 // 5, h * 3 // 5), fill=GOLD)
    return img


def test_the_art_is_inset_so_full_bleed_art_clears_the_border(tmp_path):
    """The regression. Without the inset this same source measures edge_opaque
    0.2836 against a 0.02 limit -- the fixture is built to fail loudly if the
    margin stops being applied, rather than to pass for its own reasons."""
    src = tmp_path / "bleed.png"
    full_bleed().save(src)
    out = tmp_path / "bleed_print.png"

    result = make_print_ready(src, out, requested_ground="#FF00FF", method="raster")

    assert result.ok, result.failures
    assert check_print_ready(out).ok, check_print_ready(out).failures


def test_the_margin_clears_all_four_borders_not_just_the_one_that_failed(tmp_path):
    """Per-border, because the real failure was top-heavy (top 0.1411, left and
    right 0.0000) and a check that only looked at the overall vote -- or at one
    axis -- would have passed a fix that inset the art vertically alone."""
    src = tmp_path / "bleed2.png"
    full_bleed().save(src)
    out = tmp_path / "bleed2_print.png"
    assert make_print_ready(src, out, requested_ground="#FF00FF",
                            method="raster").ok

    with Image.open(out) as img:
        left, top, right, bottom = img.getchannel("A").getbbox()
        clear_x = round(TARGET_W * ART_MARGIN)
        clear_y = round(TARGET_H * ART_MARGIN)

    # Guard the guard: at ART_MARGIN 0 the clearances below are 0 and every
    # assertion here holds vacuously, so the case would survive the margin being
    # switched off -- which is the one change it exists to catch.
    assert clear_x > 0 and clear_y > 0, "no margin is configured to check"
    assert left >= clear_x and TARGET_W - right >= clear_x, (left, right)
    assert top >= clear_y and TARGET_H - bottom >= clear_y, (top, bottom)


def test_art_wider_than_the_canvas_is_clamped_instead_of_overflowing():
    """`vector_upscale` and `raster_upscale` take a height only, so on art wider
    than 5:6 the height fit alone spends the inset sideways. Nothing the providers
    return today is that wide, which is why this is asserted rather than assumed."""
    inner_h, inner_w = inset_box((TARGET_W, TARGET_H))
    wide = Image.new("RGBA", (round(inner_h * 1.4), inner_h), (10, 20, 30, 255))

    fitted = fit_width(wide, inner_w)

    assert fitted.width == inner_w
    assert fitted.height < inner_h, "the aspect was not preserved"
    assert fit_width(Image.new("RGBA", (inner_w - 1, 10)), inner_w).width == inner_w - 1


@pytest.mark.parametrize("lines", [1, 2, 3])
def test_the_inset_is_taken_from_the_art_box_not_the_band(lines):
    """The margin scales the art inside `art_h`, so a band still takes exactly its
    own height off the art and the inset is measured against what is left. If the
    inset were taken from the full canvas instead, the art box and the band would
    overlap by the margin and the byte-exact disjointness case would break --
    stated here directly because that case would report it as a mystery diff."""
    band = round(TEST_SIZE[1] * lettering.BAND_BY_LINES[lines])
    art_h = TEST_SIZE[1] - band

    inner_h, _ = inset_box((TEST_SIZE[0], art_h))

    assert inner_h == round(art_h * (1 - 2 * ART_MARGIN))
    assert inner_h < art_h, "no margin was reserved"
    assert inner_h + band < TEST_SIZE[1], "the art and the band would collide"


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


# --- composited lettering ------------------------------------------------
#
# The band is the whole contract between the typesetter and the artwork: the art
# is upscaled to size[1] - band and seated at the end the text is not on, so the
# two can never overlap. Asserted byte-exactly below, because "they look like they
# do not overlap" is not a property a 250px canvas can show.

COMPOSITED_STYLE = {
    "id": "s0", "name": "Style 0", "graphic": "g", "lettering": "clean sans type",
    "text_layout": {"arrangement": "stacked", "case": "upper",
                    "font_role": "display", "position": "below",
                    "text_strategy": "composited"},
}


def composited_spec(**kw) -> dict:
    return {"line_1": "Pickleball Dad", "arrangement": "stacked", "case": "upper",
            "position": "below", "truncated": False, "strategy": "composited", **kw}


def a_plan(position="below", size=TEST_SIZE):
    layout = dict(COMPOSITED_STYLE["text_layout"], position=position)
    style = dict(COMPOSITED_STYLE, text_layout=layout)
    plan = lettering.plan_from(composited_spec(position=position), style, size)
    assert plan is not None and plan.ok, plan.failures
    return plan


def print_with(tmp_path, clean_png, plan=None, size=TEST_SIZE, name="p.png"):
    out = tmp_path / name
    result = make_print_ready(clean_png, out, requested_ground="#FF00FF",
                              method="raster", size=size, lettering=plan)
    assert result.ok, result.failures
    return Image.open(out), result


def test_passing_no_plan_changes_nothing_about_the_file(tmp_path, clean_png):
    """The default path, and 8 of 12 styles stay on it. The lettering parameter
    defaults to None, so a record synthesized before lettering existed takes the
    same path it always did and no lettering fact appears.

    The bbox moved in from (44, 54, 208, 248) when ART_MARGIN was introduced --
    every print file is inset now, lettered or not. That is the whole point of the
    margin and not a lettering effect, which is what this case pins."""
    plain, result = print_with(tmp_path, clean_png, name="plain.png")
    assert not [k for k in result.facts if "lettering" in k], result.facts
    assert plain.getchannel("A").getbbox() == (47, 58, 205, 244)


def test_a_lettered_file_still_passes_the_print_gate_at_the_real_size(tmp_path,
                                                                      clean_png):
    """One full-size pass with type on it, for the same reason the plain one above
    exists -- and because the gate is what this could break. Type reaching the
    canvas border fails edge_opaque outright, and the margins that keep it clear
    are fractions of 4500x5400, so a 250px canvas cannot exercise them."""
    full = (TARGET_W, TARGET_H)
    _, result = print_with(tmp_path, clean_png, a_plan(size=full),
                           name="real.png", size=full)

    report = check_print_ready(tmp_path / "real.png")
    assert report.ok, report.failures
    assert report.facts["edge_opaque"] == 0.0
    assert result.facts["lettering_band"] > 0


def test_the_band_is_reserved_and_the_artwork_never_enters_it(tmp_path, clean_png):
    """Byte-exact: the region above a `below` band must be the same image as the
    art rendered onto a canvas that short. Any art pixel inside the band, or any
    glyph above it, breaks the comparison."""
    plan = a_plan("below")
    band = plan.band_height(TEST_SIZE)
    lettered, _ = print_with(tmp_path, clean_png, plan, name="below.png")
    short, _ = print_with(tmp_path, clean_png, name="short.png",
                          size=(TEST_SIZE[0], TEST_SIZE[1] - band))

    art_region = lettered.crop((0, 0, TEST_SIZE[0], TEST_SIZE[1] - band))
    assert art_region.tobytes() == short.tobytes(), "art and type are not disjoint"
    band_region = lettered.crop((0, TEST_SIZE[1] - band, *TEST_SIZE))
    assert band_region.getchannel("A").getbbox() is not None, "the band is empty"


def test_position_above_seats_the_art_below_its_band(tmp_path, clean_png):
    """Same contract, mirrored -- and worth its own case because this is where the
    seating offset is applied rather than skipped."""
    plan = a_plan("above")
    band = plan.band_height(TEST_SIZE)
    lettered, _ = print_with(tmp_path, clean_png, plan, name="above.png")
    short, _ = print_with(tmp_path, clean_png, name="short2.png",
                          size=(TEST_SIZE[0], TEST_SIZE[1] - band))

    assert lettered.crop((0, band, *TEST_SIZE)).tobytes() == short.tobytes()
    assert lettered.crop((0, 0, TEST_SIZE[0], band)).getchannel("A").getbbox()


def test_center_overlay_reserves_nothing_and_leaves_the_art_full_height(
        tmp_path, clean_png):
    """These position values already meant this, so the compositor follows them
    instead of inventing a third rule: an overlay lands on top of whatever the
    model painted, at the price of legibility the reviewer has to judge."""
    plain, _ = print_with(tmp_path, clean_png, name="plain2.png")
    lettered, result = print_with(tmp_path, clean_png, a_plan("center_overlay"),
                                 name="overlay.png")

    assert result.facts["lettering_band"] == 0
    top, bottom = plain.getchannel("A").getbbox()[1::2]
    assert lettered.getchannel("A").getbbox()[1::2] == (top, bottom), \
        "the art was scaled down for a band that should not exist"


def test_an_overlaid_design_tells_the_reviewer_to_check_it_reads(tmp_path,
                                                                clean_png):
    """Measured on the first real composited run: Anton over a busy dark
    illustration, spelled perfectly and completely unreadable. Nothing measures
    legibility, and the type and the artwork were produced independently, so the
    only place this can be caught is the review queue -- which means it has to be
    said there."""
    _, result = print_with(tmp_path, clean_png, a_plan("center_overlay"),
                          name="legible.png")
    overlay = [w for w in result.warnings if OVERLAY_LEGIBILITY_PHRASE in w]
    assert len(overlay) == 1, result.warnings
    assert "spelled correctly by construction" in overlay[0]


def test_a_banded_design_says_nothing_about_legibility(tmp_path, clean_png):
    """The band exists precisely so the two do not overlap, so the concern does not
    apply -- and a note that fired on every design would stop being read."""
    _, result = print_with(tmp_path, clean_png, a_plan("below"), name="banded.png")
    assert not [w for w in result.warnings if OVERLAY_LEGIBILITY_PHRASE in w]


def test_the_legibility_note_is_classified_so_it_does_not_gate_every_overlay(
        tmp_path, clean_png):
    """The allowlist blocks anything unclassified, which is the right default -- but
    this note fires on every center_overlay design, and a --force typed every time
    carries no information. The seam is asserted here rather than trusted: the
    warning and the constant review.py matches on live in different statements."""
    from podauto.review import split_notes

    rec = a_record(tmp_path)
    rec.variations[0].text_spec = composited_spec(position="center_overlay")
    layout = dict(COMPOSITED_STYLE["text_layout"], position="center_overlay")
    printready_record(rec, method="raster", size=TEST_SIZE,
                      styles={"s0": dict(COMPOSITED_STYLE, text_layout=layout)})

    blocking, informational = split_notes(rec.variations[0])
    assert any(OVERLAY_LEGIBILITY_PHRASE in n for n in informational)
    assert not [n for n in blocking if OVERLAY_LEGIBILITY_PHRASE in n]


def test_the_same_source_and_plan_produce_the_same_file_twice(tmp_path, clean_png):
    """printready is re-runnable with --force. A file that differed run to run
    would not be the file a reviewer approved."""
    first, _ = print_with(tmp_path, clean_png, a_plan(), name="one.png")
    second, _ = print_with(tmp_path, clean_png, a_plan(), name="two.png")
    assert first.tobytes() == second.tobytes()


def test_the_facts_carry_the_font_and_band_to_the_review_queue(tmp_path, clean_png):
    _, result = print_with(tmp_path, clean_png, a_plan())
    assert "font=Anton" in result.facts["lettering"]
    assert "point_size=" in result.facts["lettering"]
    assert result.facts["lettering_ink_source"] == "sampled from the artwork"
    assert "font_licence" not in result.facts["lettering"], \
        "the licence path belongs in the plan, not in every reviewer's queue line"


def test_a_style_ink_override_beats_sampling_and_is_reported_as_such(tmp_path,
                                                                     clean_png):
    layout = dict(COMPOSITED_STYLE["text_layout"], ink="#FF3366")
    style = dict(COMPOSITED_STYLE, text_layout=layout)
    plan = lettering.plan_from(composited_spec(), style, TEST_SIZE)
    lettered, result = print_with(tmp_path, clean_png, plan, name="ink.png")

    assert result.facts["lettering_ink_source"] == "style override"
    colours = {c for _, c in lettered.getcolors(maxcolors=1 << 20) or []}
    assert (0xFF, 0x33, 0x66, 255) in colours


def test_a_plan_that_cannot_be_drawn_still_writes_a_file_and_warns(tmp_path,
                                                                   clean_png):
    """Throwing away usable artwork over a config mistake would be the wrong
    trade. The file is written, the reason is recorded, and because review.py's
    allowlist blocks anything it has not classified, the note holds the approval
    until a human reads it -- which is the point, since the prompt already told
    the model not to draw text."""
    layout = dict(COMPOSITED_STYLE["text_layout"], arrangement="banner")
    style = dict(COMPOSITED_STYLE, text_layout=layout)
    plan = lettering.plan_from(composited_spec(arrangement="banner"), style, TEST_SIZE)
    assert plan is not None and not plan.ok

    out = tmp_path / "failed.png"
    result = make_print_ready(clean_png, out, requested_ground="#FF00FF",
                              method="raster", size=TEST_SIZE, lettering=plan)
    assert result.ok and out.is_file()
    assert any("banner" in w for w in result.warnings)
    assert "lettering_band" not in result.facts
    assert result.facts["lettering"] == "not drawn"


# --- record level, with the style database -------------------------------

def test_a_composited_style_is_typeset_when_printready_is_given_the_styles(tmp_path):
    """The wiring `cmd_printready` depends on. Without the styles argument there is
    no strategy to read and the design silently comes out blank where the words
    belong -- the prompt has already asked the model not to draw them."""
    rec = a_record(tmp_path)
    rec.variations[0].text_spec = composited_spec()
    printready_record(rec, method="raster", size=TEST_SIZE,
                      styles={"s0": COMPOSITED_STYLE})

    var = rec.variations[0]
    assert var.print_path and Path(var.print_path).is_file()
    band = lettering.band_for("below", 1, TEST_SIZE)
    assert Image.open(var.print_path).crop(
        (0, TEST_SIZE[1] - band, *TEST_SIZE)).getchannel("A").getbbox()


def test_a_record_printed_without_styles_is_never_typeset(tmp_path):
    """printready_record is called from places that have no style database, and
    from records synthesized before text_strategy existed. Both must keep today's
    behaviour rather than half-applying a plan."""
    rec = a_record(tmp_path)
    rec.variations[0].text_spec = composited_spec()
    printready_record(rec, method="raster", size=TEST_SIZE)
    assert not any("lettering" in n for n in rec.variations[0].qa_notes)


def test_a_model_rendered_variation_is_untouched_by_the_style_database(tmp_path):
    rec = a_record(tmp_path)
    rec.variations[0].text_spec = composited_spec(strategy="model_rendered")
    printready_record(rec, method="raster", size=TEST_SIZE,
                      styles={"s0": COMPOSITED_STYLE})
    assert not any("lettering" in n for n in rec.variations[0].qa_notes)


def test_a_lettering_failure_reaches_the_reviewer_through_qa_notes(tmp_path):
    layout = dict(COMPOSITED_STYLE["text_layout"], font="helvetica-neue")
    rec = a_record(tmp_path)
    rec.variations[0].text_spec = composited_spec()
    printready_record(rec, method="raster", size=TEST_SIZE,
                      styles={"s0": dict(COMPOSITED_STYLE, text_layout=layout)})
    assert any("helvetica-neue" in n for n in rec.variations[0].qa_notes)
