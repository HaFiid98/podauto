"""Image QA. No network and no generated images -- every fixture is drawn here.

The gates are tested as a pair rather than individually, because the thing most
likely to go wrong is the split itself: a print-contract check applied to a
freshly generated file fails every file, and a check nobody can pass is a check
nobody reads. `test_the_generated_gate_does_not_apply_the_print_contract` is
the one that pins that down.

Nothing here asserts anything about lettering. Text is model-rendered by
decision and is not verified anywhere in this pipeline.
"""

from pathlib import Path

from PIL import Image

from podauto.imagegen import native_size
from podauto.imageqa import (
    ADVISED_SHORT_EDGE,
    ASPECT_TOLERANCE,
    MAX_EDGE_OPAQUE,
    MIN_INK,
    MIN_SHORT_EDGE,
    TARGET_H,
    TARGET_W,
    check_generated,
    check_print_ready,
    edge_opaque_fraction,
    ink_fraction,
    qa_record,
    qa_variation,
)
from podauto.models import IdeaType, PipelineStatus, Record, SourcePlatform, StyleVariation


def art(path, size=(848, 1024), mode="RGB", ground=(250, 250, 250), **save):
    """A plausible generated image: flat ground with a dark shape on it."""
    img = Image.new(mode, size, ground if mode == "RGB" else ground + (0,))
    w, h = size
    box = (w // 5, h // 5, w * 4 // 5, h * 4 // 5)
    img.paste((20, 30, 40) if mode == "RGB" else (20, 30, 40, 255), box)
    img.save(path, **save)
    return str(path)


def flat(path, size=(848, 1024)):
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return str(path)


# --- reading the file ----------------------------------------------------

def test_a_missing_file_is_a_failure_not_an_exception(tmp_path):
    report = check_generated(tmp_path / "nope.png")
    assert not report.ok
    assert "no file at" in report.failures[0]


def test_a_truncated_download_is_caught(tmp_path):
    """A connection dropped mid-response leaves a PNG that opens fine and only
    fails when something reads its pixels -- which would otherwise be the
    vectorize stage, two steps later."""
    path = tmp_path / "half.png"
    art(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    report = check_generated(path)
    assert not report.ok
    assert "not a readable image" in report.failures[0]


def test_json_saved_under_a_png_name_is_not_an_image(tmp_path):
    """imagegen's magic-byte guard is what stops this, so this is the second
    line of defence rather than the first."""
    path = tmp_path / "err.png"
    path.write_bytes(b'{"error": {"message": "model not found"}}')
    assert not check_generated(path).ok


# --- the generated gate --------------------------------------------------

def test_a_native_generation_passes(tmp_path):
    report = check_generated(art(tmp_path / "a.png"))
    assert report.ok and report.warnings == []
    assert report.facts["width"] == 848 and report.facts["aspect"] == 0.8281


def test_a_blank_canvas_fails_and_says_what_it_measured(tmp_path):
    """Some endpoints answer a filtered prompt with a solid frame, which is a
    perfectly valid PNG. 'QA failed' would send the operator back to the file;
    the measurement does not."""
    report = check_generated(flat(tmp_path / "blank.png"))
    assert not report.ok
    assert report.facts["ink"] == 0.0
    assert "blank or near-uniform" in report.failures[0]
    assert f"{MIN_INK:.0%}" in report.failures[0]


def test_a_square_image_fails_the_aspect_check(tmp_path):
    """5:6 is the Merch aspect; anything else gets cropped, and cropping a design
    with lettering in it usually removes part of the lettering."""
    report = check_generated(art(tmp_path / "sq.png", size=(1024, 1024)))
    assert not report.ok
    assert any("aspect" in f for f in report.failures)


def test_the_8px_rounded_native_size_is_inside_the_tolerance(tmp_path):
    """native_size() floors to a multiple of 8, so a correct generation is never
    exactly 5:6. The tolerance exists for that, not as slack."""
    for size in ((848, 1024), (1016, 1216)):
        assert check_generated(art(tmp_path / f"n{size[0]}.png", size=size)).ok


def test_every_size_imagegen_can_request_passes_this_gate():
    """The two modules have to agree or the pipeline generates only QA failures.
    Asserted across the whole range of caps rather than on the one default,
    because the default was in fact the value that did not agree: 832x1024 is
    2.5% off 5:6 and this gate allows 2%."""
    caps = [None, "UNSET", "1024x1024", "1216x1216", "2048x2048", "512x512",
            "4500x5400", "1024x768", "768x1024"]
    for cap in caps:
        w, h = native_size({} if cap is None else {"max_resolution": cap})
        assert abs((w / h) - (5 / 6)) <= ASPECT_TOLERANCE, (cap, w, h)
        assert min(w, h) >= MIN_SHORT_EDGE, (cap, w, h)


def test_a_placeholder_tile_fails_on_size(tmp_path):
    """A 5:6 tile with art on it passes every other check, so size is the only
    thing that catches an endpoint answering with a thumbnail."""
    report = check_generated(art(tmp_path / "tile.png", size=(120, 144)))
    assert not report.ok
    assert f"minimum {MIN_SHORT_EDGE}px" in " ".join(report.failures)


def test_a_low_resolution_provider_warns_instead_of_failing(tmp_path):
    """A provider capped at 512x512 yields 424x512 -- thin for a 15-inch print,
    but usable art. Failing it would discard work over a preference, so the size
    judgement is a warning and only the error-tile floor is a failure."""
    report = check_generated(art(tmp_path / "small.png", size=(424, 512)))
    assert report.ok
    assert any(f"{ADVISED_SHORT_EDGE}px advised" in w for w in report.warnings)


def test_the_generated_gate_does_not_apply_the_print_contract(tmp_path):
    """The same file passes generation QA and fails print QA. Background removal
    and vectorizing have not run yet, so demanding alpha and 4500x5400 here would
    fail every file that is in fact correct for this stage."""
    path = art(tmp_path / "native.png")
    assert check_generated(path).ok

    later = check_print_ready(path)
    assert not later.ok
    assert any("4500x5400" in f for f in later.failures)
    assert any("no alpha channel" in f for f in later.failures)


# --- the print-ready gate ------------------------------------------------

def printable(path, size=(TARGET_W, TARGET_H), ground=(0, 0, 0, 0),
              dpi=(300, 300), opaque_edge_px=0):
    """What the vectorize stage is supposed to emit: transparent ground, art
    inside, DPI stamped."""
    img = Image.new("RGBA", size, ground)
    w, h = size
    img.paste((20, 30, 40, 255), (w // 8, h // 8, w * 7 // 8, h * 7 // 8))
    for x in range(opaque_edge_px):
        img.putpixel((x, 0), (20, 30, 40, 255))
    img.save(path, **({"dpi": dpi} if dpi else {}))
    return str(path)


def test_a_print_ready_file_passes_clean(tmp_path):
    report = check_print_ready(printable(tmp_path / "p.png"))
    assert report.ok and report.warnings == []
    assert report.facts["dpi"] == (300, 300)
    assert report.facts["edge_opaque"] == 0.0


def test_the_wrong_pixel_count_fails_even_at_the_right_aspect(tmp_path):
    """Aspect and size are separate failures: 2250x2700 is exactly 5:6 and still
    prints at half the resolution Merch asks for."""
    report = check_print_ready(printable(tmp_path / "half.png", size=(2250, 2700)))
    assert not report.ok
    assert "2250x2700, expected 4500x5400" in " ".join(report.failures)
    assert not any("aspect" in f for f in report.failures)


def test_an_opaque_ground_reads_as_background_removal_not_run(tmp_path):
    """This is the failure the whole print gate exists for: an opaque ground gets
    printed as ink, so a 'transparent PNG' that is not transparent is a ruined
    shirt rather than a cosmetic problem."""
    report = check_print_ready(printable(tmp_path / "opaque.png", ground=(255, 255, 255, 255)))
    assert not report.ok
    assert any("border is opaque" in f for f in report.failures)
    assert report.facts["edge_opaque"] == 1.0


def test_art_that_touches_the_edge_is_tolerated(tmp_path):
    """A hard zero would fail files that print correctly: anti-aliased art running
    off the canvas legitimately leaves a few opaque border pixels."""
    allowed = int(2 * (TARGET_W + TARGET_H) * MAX_EDGE_OPAQUE / 2)
    report = check_print_ready(
        printable(tmp_path / "bleed.png", opaque_edge_px=allowed))
    assert report.ok
    assert 0 < report.facts["edge_opaque"] < MAX_EDGE_OPAQUE


def test_a_missing_dpi_header_warns_because_the_pixels_already_decide(tmp_path):
    """4500x5400 IS 15x18 inches at 300dpi. The header only tells software what to
    assume when nothing else does, so it cannot be a failure."""
    report = check_print_ready(printable(tmp_path / "nodpi.png", dpi=None))
    assert report.ok
    assert report.facts["dpi"] is None
    assert any("no DPI header" in w for w in report.warnings)


def test_a_wrong_dpi_header_warns_and_names_the_value(tmp_path):
    report = check_print_ready(printable(tmp_path / "dpi72.png", dpi=(72, 72)))
    assert report.ok
    assert any("says 72" in w for w in report.warnings)


def test_edge_opaque_fraction_is_none_without_alpha(tmp_path):
    """None and 0.0 mean opposite things -- fully transparent border versus no
    alpha channel at all -- so they must not collapse into one value."""
    assert edge_opaque_fraction(Image.new("RGB", (64, 76), (255, 255, 255))) is None
    assert edge_opaque_fraction(Image.new("RGBA", (64, 76), (0, 0, 0, 0))) == 0.0


def test_ink_fraction_is_zero_on_a_flat_canvas():
    assert ink_fraction(Image.new("RGB", (100, 120), (17, 42, 99))) == 0.0
    assert ink_fraction(Image.new("RGB", (100, 120), (255, 255, 255))) == 0.0


# --- record level --------------------------------------------------------

def make_record(*paths) -> Record:
    return Record(
        id="rec1", timestamp="2026-08-16T00:00:00Z",
        source_url="https://example.test/1",
        source_platform=SourcePlatform.AMAZON, idea_type=IdeaType.READY_SHIRT,
        niche="Gardening Grandma", raw_title_or_text="Garden Nan",
        variations=[StyleVariation(style_id=f"style{i}", style_name="S",
                                   graphic_prompt="a cat", image_path=p)
                    for i, p in enumerate(paths)],
    )


def test_failures_reach_qa_notes_with_the_stage_that_found_them(tmp_path):
    """The reviewer needs to know which gate objected: 'no alpha' from the
    generated gate would be a bug in the gate, from the print gate it is a
    missing background-removal step."""
    record = make_record(flat(tmp_path / "blank.png"))
    record.pipeline_status = PipelineStatus.IMAGES_GENERATED

    qa_record(record, stage="generated")
    assert record.variations[0].qa_notes
    assert record.variations[0].qa_notes[0].startswith("image qa (generated):")


def test_a_record_fails_qa_only_when_no_variation_survived(tmp_path):
    """One bad variation out of two is a variation to drop, not a record to
    abandon -- the same reasoning as imagegen. This record passed four gates."""
    record = make_record(art(tmp_path / "good.png"), flat(tmp_path / "bad.png"))
    record.pipeline_status = PipelineStatus.IMAGES_GENERATED

    summary = qa_record(record)
    assert (summary.passed, summary.failed) == (1, 1)
    assert record.pipeline_status == PipelineStatus.IMAGES_GENERATED


def test_every_variation_failing_marks_the_record(tmp_path):
    record = make_record(flat(tmp_path / "b1.png"), flat(tmp_path / "b2.png"))
    record.pipeline_status = PipelineStatus.IMAGES_GENERATED

    summary = qa_record(record)
    assert (summary.passed, summary.failed) == (0, 2)
    assert record.pipeline_status == PipelineStatus.IMAGES_QA_FAILED


def test_a_failing_file_is_left_on_disk(tmp_path):
    """Deleting it would remove the only evidence for deciding whether the prompt,
    the provider or the removal step is at fault."""
    path = flat(tmp_path / "bad.png")
    record = make_record(path)
    qa_record(record)

    assert Path(path).is_file()
    assert record.variations[0].image_path == path


def test_a_variation_with_no_image_is_counted_not_failed(tmp_path):
    """A variation whose generation failed already carries its reason in qa_notes.
    Counting it as a QA failure would report the same problem twice and could flip
    a record to IMAGES_QA_FAILED for something QA never looked at."""
    record = make_record(art(tmp_path / "ok.png"), None)
    summary = qa_record(record)

    assert (summary.passed, summary.failed, summary.missing) == (1, 0, 1)
    assert record.variations[1].qa_notes == []
    assert qa_variation(record.variations[1]) is None


def test_warnings_are_recorded_but_do_not_fail_the_record(tmp_path):
    record = make_record(art(tmp_path / "small.png", size=(424, 512)))
    summary = qa_record(record)

    assert summary.passed == 1 and summary.failed == 0
    assert any("warning" in n for n in record.variations[0].qa_notes)
    assert record.pipeline_status != PipelineStatus.IMAGES_QA_FAILED



