"""Lettering compositor tests.

This module exists because text was the pipeline's measured defect: of the first
three real generations the artwork was usable in all three and the *text* failed
in two -- one doubled it, one misspelled it. Typesetting removes that defect
class, so what these tests care about is the ways typesetting could introduce a
new one:

* the wrong strategy, drawing type onto a design whose prompt also asked the
  model for text -- two sets of lettering on one shirt;
* type that reaches the canvas border, which fails `check_print_ready`'s
  `edge_opaque` gate outright;
* a composited style the compositor silently cannot draw, which is worse than a
  misspelling because the prompt has already told the model not to draw text --
  the shirt comes out blank where the words belong;
* type that is not deterministic, which would make `printready` non-idempotent.

`size` is small in most cases for the same reason as test_printready.py, but the
margin and print-QA assertions use the real 4500x5400 canvas: they are about the
gate a real file has to pass, and a 100px canvas cannot exercise it.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from podauto.imageqa import TARGET_H, TARGET_W
from podauto.lettering import (
    BAND_BY_LINES,
    DEFAULT_INK,
    FONTS,
    MARGIN_X,
    MARGIN_Y,
    MAX_LINES,
    ROLE_FACE,
    band_for,
    choose_lines,
    plan_from,
    render,
    sample_ink,
    text_box,
)

PRINT = (TARGET_W, TARGET_H)
SMALL = (300, 360)
TEXT = "Pickleball Dad Retro Sunset"

INK = (20, 40, 90)
GOLD = (230, 180, 40)


def style(font_role="display", strategy="composited", **layout) -> dict:
    """A style entry shaped like config/styles.json's, where font_role and the
    layout keys live inside text_layout rather than at the top level."""
    return {
        "id": "test_style", "name": "Test Style", "graphic": "g", "lettering": "l",
        "text_layout": {"arrangement": "stacked", "case": "upper",
                        "font_role": font_role, "position": "below",
                        "text_strategy": strategy, **layout},
    }


def spec(strategy="composited", **kw) -> dict:
    """A text_spec as prompts.TextSpec.to_dict() writes it."""
    return {"line_1": TEXT, "arrangement": "stacked", "case": "upper",
            "position": "below", "truncated": False, "strategy": strategy, **kw}


def art(size=(848, 1024)) -> Image.Image:
    """Ground-removed flat art: opaque shapes on transparency, which is what
    remove_ground hands the compositor."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w, h = size
    d.ellipse((w // 5, h // 5, w * 4 // 5, h * 3 // 5), fill=(*INK, 255))
    d.rectangle((w // 5, h * 7 // 10, w * 4 // 5, h * 8 // 10), fill=(*GOLD, 255))
    return img


# --- who gets typeset ----------------------------------------------------

def test_a_model_rendered_variation_gets_no_plan_at_all():
    """The default, and it has to stay silent: 8 of 12 styles are model_rendered
    and nothing about that is worth a note in the review queue."""
    assert plan_from(spec(strategy="model_rendered"), style(), PRINT) is None


def test_a_text_spec_with_no_strategy_key_is_treated_as_model_rendered():
    """Records generated before the strategy field existed still load, and they
    must not suddenly acquire typeset text on a re-run of printready."""
    old = {"line_1": TEXT, "arrangement": "stacked", "case": "upper",
           "position": "below", "truncated": False}
    assert plan_from(old, style(), PRINT) is None


def test_a_missing_text_spec_or_style_is_not_an_error():
    """printready is called for every variation, including ones synthesized before
    styles existed, and it must never raise on the way to writing a file."""
    assert plan_from(None, style(), PRINT) is None
    assert plan_from(spec(), None, PRINT) is None


def test_a_composited_variation_resolves_a_font_and_lines():
    plan = plan_from(spec(), style(font_role="display"), PRINT)
    assert plan is not None and plan.ok
    assert plan.face.key == "anton"
    assert plan.lines and all(line.strip() for line in plan.lines)


# --- font resolution -----------------------------------------------------

def test_font_role_selects_the_face_and_text_layout_font_overrides_it():
    """font_role describes the look asked of the model, which is the right coarse
    key -- but blueprint_technical is `sans` and wants technical monospace, so the
    override has to win."""
    assert plan_from(spec(), style(font_role="sans"), PRINT).face.key == "jost"
    override = style(font_role="sans", font="spacemono")
    assert plan_from(spec(), override, PRINT).face.key == "spacemono"


def test_every_registered_face_is_on_disk_with_its_licence():
    """The licence is the load-bearing fact -- these fonts go on merchandise. It
    ships beside the file for the same reason providers.json records a model
    licence, and a face whose OFL.txt is missing is a face nobody can prove is
    usable."""
    for key, face in FONTS.items():
        assert face.path.is_file(), f"{key}: {face.path}"
        assert "OFL" in face.licence
        assert (face.path.parent / "OFL.txt").is_file(), key


def test_every_font_role_maps_to_a_registered_face():
    """styles.json documents five roles; a role with no face would fall back to
    Anton silently and set a display face where a serif was asked for."""
    for role, key in ROLE_FACE.items():
        assert key in FONTS, role


def test_an_unregistered_face_key_fails_loudly_instead_of_guessing():
    plan = plan_from(spec(), style(font="helvetica-neue"), PRINT)
    assert plan is not None and not plan.ok
    assert "helvetica-neue" in plan.failures[0]
    assert "asked the model not to draw it" in plan.failures[0]


def test_a_font_file_that_is_not_installed_reports_the_path(tmp_path, monkeypatch):
    """The one failure a fresh clone hits. It must name the file, because the fix
    is to fetch it and nothing else in the message would say so."""
    from dataclasses import replace

    import podauto.lettering as lettering
    ghost = replace(FONTS["anton"], path=tmp_path / "nope" / "Anton-Regular.ttf")
    monkeypatch.setitem(lettering.FONTS, "anton", ghost)

    plan = plan_from(spec(), style(font_role="display"), PRINT)
    assert not plan.ok
    assert "font file missing" in plan.failures[0]
    assert "Anton-Regular.ttf" in plan.failures[0]


# --- arrangements this slice does not draw -------------------------------

@pytest.mark.parametrize("arrangement", ["arc_over", "arc_under", "banner", "none"])
def test_an_unsupported_arrangement_fails_rather_than_drawing_nothing(arrangement):
    """The dangerous config edit: marking an arc style composited stops the prompt
    asking the model for text, so a compositor that quietly declines leaves a shirt
    with no lettering anywhere. The failure has to say which key to put back."""
    plan = plan_from(spec(arrangement=arrangement), style(), PRINT)
    assert plan is not None and not plan.ok
    assert arrangement in plan.failures[0]
    assert "model_rendered" in plan.failures[0]


def test_empty_text_is_a_failure_not_a_blank_shirt():
    plan = plan_from(spec(line_1="   "), style(), PRINT)
    assert not plan.ok and "line_1" in plan.failures[0]


def test_truncated_text_is_flagged_because_the_shirt_says_less_than_the_title():
    plan = plan_from(spec(truncated=True), style(), PRINT)
    assert plan.ok, "truncation is a warning, not a refusal"
    assert any("truncated" in w for w in plan.warnings)


# --- layout --------------------------------------------------------------

def test_case_comes_from_the_text_spec():
    """prompts.synthesize already resolved the style's case into text_spec, so this
    reads it rather than re-deriving it -- one place decides."""
    assert plan_from(spec(case="lower"), style(), PRINT).text == TEXT.lower()
    assert plan_from(spec(case="upper"), style(), PRINT).text == TEXT.upper()
    assert plan_from(spec(case="title"), style(), PRINT).text == TEXT.title()


def test_a_long_string_wraps_to_at_most_three_lines_without_splitting_a_word():
    """27 characters set as one 4500px line is a caption, not a shirt."""
    lines = choose_lines(TEXT.upper(), "anton", 0.02, PRINT, "below")
    assert 1 <= len(lines) <= MAX_LINES
    assert " ".join(lines) == TEXT.upper(), "a word was split or a space was lost"


def test_a_single_word_stays_on_one_line():
    assert choose_lines("PICKLEBALL", "anton", 0.02, PRINT, "below") == ["PICKLEBALL"]


def test_above_and_below_reserve_a_band_and_an_overlay_does_not():
    """These position values already meant this, so the compositor follows them
    rather than inventing a third rule: a band is reserved at that end and the art
    scales into the rest, so the two never overlap."""
    assert band_for("below", 2, PRINT) == round(PRINT[1] * BAND_BY_LINES[2])
    assert band_for("above", 2, PRINT) == round(PRINT[1] * BAND_BY_LINES[2])
    assert band_for("center_overlay", 2, PRINT) == 0


def test_more_lines_reserve_a_taller_band():
    assert band_for("below", 1, PRINT) < band_for("below", 2, PRINT) \
        < band_for("below", 3, PRINT)


def test_the_band_sits_at_the_end_the_position_names():
    below = text_box("below", PRINT, band_for("below", 2, PRINT))
    above = text_box("above", PRINT, band_for("above", 2, PRINT))
    assert below[1] > PRINT[1] / 2, "a 'below' band is in the bottom half"
    assert above[3] < PRINT[1] / 2, "an 'above' band is in the top half"


# --- rendering -----------------------------------------------------------

def rendered(position="below", **kw):
    plan = plan_from(spec(position=position, **kw), style(position=position), PRINT)
    return plan, render(plan, PRINT, "#102030")


def test_the_type_is_drawn_at_all():
    plan, layer = rendered()
    assert layer.getchannel("A").getbbox() is not None, "nothing was drawn"
    assert plan.facts["point_size"] > 0


@pytest.mark.parametrize("position", ["below", "above", "center_overlay"])
def test_the_type_never_reaches_the_canvas_border(position):
    """check_print_ready fails above MAX_EDGE_OPAQUE (0.02) of the 1px border, so
    this is a print-QA constraint and not an aesthetic one. Asserted on the real
    canvas because the margins are fractions of it."""
    _, layer = rendered(position)
    left, top, right, bottom = layer.getchannel("A").getbbox()
    assert left >= PRINT[0] * MARGIN_X - 1
    assert right <= PRINT[0] * (1 - MARGIN_X) + 1
    assert top >= PRINT[1] * MARGIN_Y - 1
    assert bottom <= PRINT[1] * (1 - MARGIN_Y) + 1


def test_type_positioned_below_stays_inside_its_own_band():
    """The band is the contract with the artwork: the art is scaled to
    size[1] - band, so type escaping the band would overlap it."""
    plan, layer = rendered("below")
    band = plan.band_height(PRINT)
    top = layer.getchannel("A").getbbox()[1]
    assert top >= PRINT[1] - band, f"type at y={top} left a band of {band}"


def test_type_positioned_above_stays_inside_its_own_band():
    plan, layer = rendered("above")
    bottom = layer.getchannel("A").getbbox()[3]
    assert bottom <= plan.band_height(PRINT)


def test_rendering_the_same_plan_twice_is_byte_identical():
    """printready is re-runnable and idempotent. Type that moved between runs
    would make the print file differ from the one a human approved."""
    first = rendered()[1].tobytes()
    second = rendered()[1].tobytes()
    assert first == second


def test_a_failed_plan_renders_an_empty_layer_rather_than_raising():
    plan = plan_from(spec(arrangement="banner"), style(), PRINT)
    layer = render(plan, SMALL, "#000000")
    assert layer.size == SMALL
    assert layer.getchannel("A").getbbox() is None


def test_the_facts_name_the_font_and_its_licence():
    """These reach the review queue via PrintResult.facts, and the licence is the
    part someone auditing a listing would need."""
    plan, _ = rendered()
    assert plan.facts["font"] == "Anton"
    assert "OFL" in plan.facts["font_licence"]
    assert plan.facts["lines"] == len(plan.lines)


# --- ink -----------------------------------------------------------------

def test_ink_is_sampled_from_a_colour_the_artwork_already_uses():
    """Sampling keeps the limited-palette screen-print look the prompts ask for:
    the type introduces no colour the design does not already contain."""
    ink, how = sample_ink(art())
    assert how == "sampled from the artwork"
    r, g, b = (int(ink[i:i + 2], 16) for i in (1, 3, 5))
    # The navy ellipse is the only dark mass in the fixture; bucketing rounds it.
    assert abs(r - INK[0]) < 32 and abs(g - INK[1]) < 32 and abs(b - INK[2]) < 32


def test_sampling_is_deterministic():
    assert sample_ink(art()) == sample_ink(art())


def test_artwork_with_nothing_dark_enough_falls_back_and_says_so():
    """Nothing in the pipeline records a garment colour, so type has to be dark
    enough to print on a light shirt. When the art offers nothing, the fallback
    must be visible in the facts rather than looking like a sampled result."""
    pale = Image.new("RGBA", (64, 77), (250, 245, 240, 255))
    ink, how = sample_ink(pale)
    assert ink == DEFAULT_INK
    assert "dark enough" in how


def test_fully_transparent_art_falls_back_without_dividing_by_zero():
    ink, how = sample_ink(Image.new("RGBA", (64, 77), (0, 0, 0, 0)))
    assert ink == DEFAULT_INK
    assert "no opaque artwork" in how


def test_the_residual_ground_colour_is_never_chosen_as_ink():
    """Measured on the first real composited generation: the model painted a
    #FB57AE ground, removal kept 3.4% of it inside the art's bounding box, and at
    luma 139 that pink outvoted the actual line work. The type came out in the one
    colour the prompt had told the artwork to exclude -- and sat on the fringe of
    the same colour, so it read as a smudge rather than words."""
    img = Image.new("RGBA", (64, 77), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 63, 50), fill=(251, 87, 174, 255))     # kept ground, the majority
    d.rectangle((0, 55, 63, 76), fill=(*INK, 255))            # the actual line work

    naive, _ = sample_ink(img)
    assert naive.upper() == "#F050B0", "fixture no longer reproduces the defect"

    ink, how = sample_ink(img, avoid="#FB57AE")
    assert ink.upper() != "#F050B0"
    assert "residual ground colour excluded" in how
    r = int(ink[1:3], 16)
    assert abs(r - INK[0]) < 32, f"picked {ink}, not the line work"


def test_art_whose_only_dark_colour_is_the_ground_falls_back_visibly():
    """Excluding the ground can leave nothing to sample. Returning the ground
    anyway would be the defect above; returning a near-white would be unprintable.
    The named fallback is the only honest answer."""
    img = Image.new("RGBA", (64, 77), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((0, 0, 63, 60), fill=(251, 87, 174, 255))

    ink, how = sample_ink(img, avoid="#FB57AE")
    assert ink == DEFAULT_INK
    assert "only dark colour is the residual ground" in how


def test_a_ground_colour_that_the_art_does_not_reuse_changes_nothing():
    """The common case -- printready reports a 0.0% collision on most designs --
    and it must not pick up a note that implies one happened."""
    plain, how_plain = sample_ink(art())
    guarded, how_guarded = sample_ink(art(), avoid="#FF00FF")
    assert (plain, how_plain) == (guarded, how_guarded)
    assert "excluded" not in how_guarded


def test_an_unparseable_ground_hint_is_ignored_rather_than_raising():
    """ground_hint is config text and a variation synthesized before it existed
    has none at all. Neither may cost a print file."""
    assert sample_ink(art(), avoid=None) == sample_ink(art(), avoid="not a colour")
    assert sample_ink(art(), avoid=(20, 40, 90))[0] != "#103050"


def test_a_style_can_override_the_ink():
    plan = plan_from(spec(), style(ink="#FF3366"), PRINT)
    assert plan.ink == "#FF3366"


def test_the_ink_actually_reaches_the_pixels():
    plan = plan_from(spec(), style(), PRINT)
    layer = render(plan, PRINT, "#FF3366")
    colours = {c for _, c in layer.getcolors(maxcolors=1 << 20) or []}
    assert (0xFF, 0x33, 0x66, 255) in colours
