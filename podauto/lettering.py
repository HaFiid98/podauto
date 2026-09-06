"""Typeset lettering, so the shirt says what the record says it says.

The pipeline's measured defect. Of the first three real generations, the artwork
was usable in all three and the *text* failed in two: one doubled it
("PICKLEBALL" twice over a mangled "DAD"), one misspelled it ("RERTICES" where
"RETRO SUNSET" belongs). That is not bad luck -- `FLUX.1-schnell` is a 4-step
distilled model, the weakest of the FLUX family at text, and the strings this
pipeline asks for run 23 to 33 characters. Every such defect costs a generation
and lands in the manual review queue.

So for styles that opt in, the model is asked for artwork only and the lettering
is typeset here with freetype. `text_spec["line_1"]` is rendered exactly, every
time, from the same string `export` writes to `intended_text`.

WHAT THIS DOES NOT DO, stated plainly because "composited" invites the opposite
assumption: it guarantees the *intended* string is spelled correctly. It cannot
guarantee the model did not scribble letterforms into the artwork itself. OCR
verification was declined deliberately, so residual gibberish in the art remains
a review-queue matter exactly as it is today. `build_prompt` asks for "no text,
no letters, no words" for these styles, and like the ground instruction that is a
request the model may ignore.

Opt-in is per style (`text_layout.text_strategy` in config/styles.json) rather
than global, because several styles specify lettering that is a *treatment*
freetype cannot reproduce -- y2k's "chrome-effect with sharp bevels and a glossy
highlight", grunge_punk's "ransom-note cut-paper with mismatched letterforms",
distressed_collegiate's "heavy distress and crackle across the letterforms".
Clean type where those were asked for is a downgrade, so they stay
model_rendered. Absent the key, nothing changes for a style.

Only the `stacked` arrangement renders here. Arcs, banners and split layouts are
the next slice; a style configured for one of those and marked composited is a
loud failure rather than a silently text-free shirt, because the prompt will
already have stopped asking the model for text.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

FONT_ROOT = Path(__file__).resolve().parent.parent / "assets" / "fonts"

# Type is measured at this size and scaled; ratios are size-independent, so the
# line-wrap decision does not depend on the point size it later produces.
REF_SIZE = 100

# Line advance in em multiples. Tighter than a text setting on purpose: stacked
# merch lettering is a lockup, not a paragraph.
LINE_H = 1.12

# Free margin, as a fraction of the canvas. MARGIN_X is also the print-QA
# constraint: `imageqa.check_print_ready` fails above MAX_EDGE_OPAQUE (0.02) of
# the 1px border, so type that bleeds to the edge fails outright. 0.06 of 4500px
# is 270px of clearance, which is far more than that gate needs and also just
# looks right on a garment.
MARGIN_X = 0.06
MARGIN_Y = 0.04

# Height of the band reserved for text, by line count, as a fraction of the
# canvas. The art is scaled into what is left, so this is the one number that
# trades design size against lettering size.
BAND_BY_LINES = {1: 0.15, 2: 0.21, 3: 0.27}

# For center_overlay nothing is reserved -- the type goes over the art at its
# native size -- but the block still needs a height to size against.
CENTER_BAND = 0.28

MAX_LINES = 3
MIN_SIZE, MAX_SIZE = 8, 3000

# Type dark enough to read on a light garment. Nothing in the pipeline records a
# garment colour, so this is the assumption; the ink is still sampled from the
# artwork so it belongs to the design either way.
INK_MAX_LUMA = 150
DEFAULT_INK = "#1A1A1A"

# A sampled colour must cover at least this much of the opaque artwork to count
# as part of the palette rather than an anti-aliasing artefact.
INK_MIN_SHARE = 0.01

# Colour buckets for ink sampling: 8 levels per channel, each reported as its
# midpoint. Wide enough that anti-aliasing collapses into the fill it edges.
_BUCKET = 32
_BUCKET_LUT = [i // _BUCKET * _BUCKET + _BUCKET // 2 for i in range(256)]

# --- overlay legibility ----------------------------------------------------
#
# A band guarantees the type and the art do not overlap. An overlay guarantees
# nothing, and the two halves were produced independently: measured on the first
# real composited run, Anton over a busy dark illustration was spelled perfectly
# and could not be read at all. So the overlay case is measured rather than merely
# announced to the reviewer.

# The garment the type is assumed to sit on. Art alpha is not opaque everywhere --
# ground removal is what makes a print file -- and where it is transparent the
# shirt itself is the background. Nothing in the pipeline records a garment
# colour, so this is the same light-garment assumption INK_MAX_LUMA already
# encodes, and it is the generous reading on purpose: a gate that fired on the
# reviewer's unstated colour choice would be noise, not information.
GARMENT_RGB = (255, 255, 255)

# Contrast ratio, by WCAG's formula, that overlaid type has to reach against the
# artwork under it. 3.0 is that standard's large-text threshold; merch lettering is
# an order of magnitude bigger than the 18pt which qualifies, so this is
# deliberately the loose end of the scale. The question being asked is "does the
# word read on a shirt", not "does this pass an accessibility audit".
MIN_CONTRAST = 3.0

# How much of the type's own area may fall below MIN_CONTRAST before the design is
# called unreadable. Not zero: a descender clipping the corner of one dark shape is
# still a legible word, and every real generation has dark detail somewhere. A
# tenth of the letterforms lost is about where a word stops reading.
MAX_LOW_CONTRAST = 0.10

# Height the measurement works at. A print canvas is 24 megapixels and the answer
# does not change with resolution -- at 5400px every stem is hundreds of pixels
# across, so a reduction to this still leaves each one many pixels wide.
_CONTRAST_H = 360

# Glyph coverage a reduced pixel needs to count as type rather than as the
# anti-aliased edge of it. Area-averaging the mask down makes an edge pixel part
# background, and measuring contrast there would compare the art against itself.
_GLYPH_ALPHA = 250


@dataclass(frozen=True)
class Face:
    """One font file, with the licence that lets it go on merchandise.

    `licence` is recorded here for the same reason `config/providers.json` records
    a model licence: the permission is the load-bearing fact, and it has to live
    next to the thing it permits. All four are OFL 1.1, which explicitly allows
    embedding and sale of the rendered output; the full text ships beside each
    file so a fresh clone carries it.
    """

    key: str
    family: str
    path: Path
    licence: str
    weight: int | None = None      # variable-font wght axis, None if static
    tracking: float = 0.03         # extra letter-spacing, in em


FONTS: dict[str, Face] = {
    "anton": Face(
        "anton", "Anton", FONT_ROOT / "anton" / "Anton-Regular.ttf",
        "OFL 1.1 -- assets/fonts/anton/OFL.txt", tracking=0.02),
    "jost": Face(
        "jost", "Jost", FONT_ROOT / "jost" / "Jost[wght].ttf",
        "OFL 1.1 -- assets/fonts/jost/OFL.txt", weight=600, tracking=0.05),
    "spacemono": Face(
        "spacemono", "Space Mono", FONT_ROOT / "spacemono" / "SpaceMono-Bold.ttf",
        "OFL 1.1 -- assets/fonts/spacemono/OFL.txt", tracking=0.06),
    "baloo2": Face(
        "baloo2", "Baloo 2", FONT_ROOT / "baloo2" / "Baloo2[wght].ttf",
        "OFL 1.1 -- assets/fonts/baloo2/OFL.txt", weight=700, tracking=0.02),
}

# font_role in config/styles.json describes the look asked of the model, so it is
# the right coarse key here too. A style whose role is too coarse overrides it
# with text_layout.font -- blueprint_technical is `sans` but wants technical
# monospace, which is a different thing from a geometric sans.
ROLE_FACE = {
    "display": "anton",
    "sans": "jost",
    "serif": "jost",
    "slab": "anton",
    "script": "baloo2",
}

COMPOSITED = "composited"
MODEL_RENDERED = "model_rendered"
SUPPORTED_ARRANGEMENTS = ("stacked",)


@lru_cache(maxsize=512)
def load_font(key: str, size: int) -> ImageFont.FreeTypeFont:
    """A face at a point size, cached because fitting binary-searches ~12 sizes.

    Variable faces are set to the weight the Face declares. If that fails the
    default instance renders instead -- lighter type, still correct -- rather
    than losing the lettering over a font-format detail.
    """
    face = FONTS[key]
    font = ImageFont.truetype(str(face.path), size)
    if face.weight is not None:
        try:
            font.set_variation_by_axes([face.weight])
        except Exception:                                          # noqa: BLE001
            pass
    return font


@dataclass
class LetteringPlan:
    """Everything needed to draw the type, decided before the art is upscaled.

    Deciding the band up front is what lets `make_print_ready` rasterize the
    artwork straight to its final height instead of scaling it twice -- a second
    resample would undo the point of tracing to vector.
    """

    text: str
    face: Face
    lines: list[str] = field(default_factory=list)
    position: str = "center_overlay"
    ink: str | None = None                  # style override; None means sample
    facts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def band_height(self, size: tuple[int, int]) -> int:
        """Canvas height to reserve for type, 0 when the type overlays the art."""
        if not self.ok:
            return 0
        return band_for(self.position, len(self.lines), size)


def _measure(key: str, size: int, tracking: float,
             lines: list[str]) -> tuple[float, float, float, int]:
    """(widest advance, ink height, ink top, line step) for a block of lines.

    Width comes from summed per-glyph advances plus tracking, because that is how
    `render` draws it. Height comes from the real ink extent rather than the font
    metrics, so a line of all caps centres optically instead of hanging from an
    ascender nothing occupies.
    """
    font = load_font(key, size)
    step = max(1, round(size * LINE_H))
    widest = 0.0
    for line in lines:
        adv = sum(font.getlength(ch) for ch in line)
        adv += tracking * size * max(0, len(line) - 1)
        widest = max(widest, adv)
    tops, bottoms = [], []
    for i, line in enumerate(lines):
        _, top, _, bottom = font.getbbox(line, anchor="ls")
        tops.append(i * step + top)
        bottoms.append(i * step + bottom)
    return widest, max(bottoms) - min(tops), min(tops), step


def _balanced_split(words: list[str], count: int,
                    measure: Callable[[str], float]) -> list[str]:
    """Split into exactly `count` lines, minimising the widest one.

    Brute force over the split points: MAX_TEXT_WORDS caps the input at 5 words
    and MAX_LINES at 3, so this is at most six candidates. The tie-break prefers
    the evenest block, which is what makes a two-line lockup look deliberate.
    """
    if count <= 1 or len(words) <= 1:
        return [" ".join(words)]
    if count >= len(words):
        return list(words)
    best: list[str] = []
    best_cost: tuple[float, float] | None = None
    for cuts in itertools.combinations(range(1, len(words)), count - 1):
        bounds = (0, *cuts, len(words))
        lines = [" ".join(words[bounds[i]:bounds[i + 1]]) for i in range(count)]
        widths = [measure(line) for line in lines]
        cost = (max(widths), -min(widths))
        if best_cost is None or cost < best_cost:
            best, best_cost = lines, cost
    return best


def _fit_size(key: str, tracking: float, lines: list[str],
              box: tuple[int, int]) -> int:
    """Largest integer point size at which the block fits inside `box`."""
    box_w, box_h = box
    lo, hi = MIN_SIZE, MAX_SIZE
    while lo < hi:
        mid = (lo + hi + 1) // 2
        width, height, _, _ = _measure(key, mid, tracking, lines)
        if width <= box_w and height <= box_h:
            lo = mid
        else:
            hi = mid - 1
    return lo


def band_for(position: str, line_count: int, size: tuple[int, int]) -> int:
    """Canvas height reserved for type: a band for above/below, nothing for an
    overlay. One rule, so the box the wrap chooser measures against and the height
    `make_print_ready` takes off the artwork cannot drift apart."""
    if position not in ("above", "below"):
        return 0
    return round(size[1] * BAND_BY_LINES.get(line_count, CENTER_BAND))


def text_box(position: str, size: tuple[int, int], band: int) -> tuple[int, int, int, int]:
    """The rectangle the type is centred in, as (left, top, right, bottom).

    `above` and `below` mean what they say -- the band is reserved at that end
    and the art occupies the rest, so the two never overlap and no third rule is
    needed. `center_overlay` reserves nothing and the type goes over the art.
    """
    canvas_w, canvas_h = size
    mx = round(canvas_w * MARGIN_X)
    my = round(canvas_h * MARGIN_Y)
    if position == "below" and band:
        return mx, canvas_h - band, canvas_w - mx, canvas_h - my
    if position == "above" and band:
        return mx, my, canvas_w - mx, band
    half = round(canvas_h * CENTER_BAND / 2)
    return mx, canvas_h // 2 - half, canvas_w - mx, canvas_h // 2 + half


def choose_lines(text: str, key: str, tracking: float, size: tuple[int, int],
                 position: str) -> list[str]:
    """Wrap into the line count that yields the biggest type.

    text_spec carries one string, but `stacked` means what it says: 27 characters
    set as one 4500px line is a caption, not a shirt. More lines buy a wider band
    and a shorter line, fewer lines buy a taller line -- so the choice is made by
    measuring which one actually renders larger in the box `position` really
    gives it, not by a rule of thumb.
    """
    words = text.split()
    if not words:
        return []
    ref = load_font(key, REF_SIZE)
    measure = ref.getlength
    best_lines, best_size = [], -1
    for count in range(1, min(MAX_LINES, len(words)) + 1):
        lines = _balanced_split(words, count, measure)
        band = band_for(position, len(lines), size)
        left, top, right, bottom = text_box(position, size, band)
        fitted = _fit_size(key, tracking, lines, (right - left, bottom - top))
        if fitted > best_size:
            best_lines, best_size = lines, fitted
    return best_lines


def is_composited(text_spec: dict[str, Any] | None) -> bool:
    """Whether this variation's lettering is typeset rather than drawn.

    One definition, because two readers need it for different reasons and a drift
    between them would be silent: `plan_from` decides whether to typeset, and the
    CLI decides what to tell a reviewer to look at. Those instructions are
    opposites -- on a model_rendered design you check the spelling of the letters
    the model drew; on a composited one the spelling is exact by construction and
    what is worth hunting for is stray letterforms the model drew anyway, in the
    artwork, having been asked not to.
    """
    return bool(text_spec) and text_spec.get("strategy", MODEL_RENDERED) == COMPOSITED


def plan_from(text_spec: dict[str, Any] | None, style: dict[str, Any] | None,
              size: tuple[int, int]) -> LetteringPlan | None:
    """A plan, or None when this variation's text is the model's job.

    None is the common case and is silent -- `model_rendered` is the default and
    nothing about it is worth a note. A composited variation that cannot be
    rendered returns a plan carrying `failures` instead, so the reason reaches
    `qa_notes` and stops an approval: by then the prompt has already told the
    model not to draw text, and a silently text-free shirt is worse than a
    visible failure.
    """
    if not text_spec or not style:
        return None
    if not is_composited(text_spec):
        return None

    layout = style.get("text_layout", {})
    key = layout.get("font") or ROLE_FACE.get(layout.get("font_role", ""), "anton")
    face = FONTS.get(key)
    plan = LetteringPlan(text="", face=face or FONTS["anton"],
                         position=text_spec.get("position", "center_overlay"),
                         ink=layout.get("ink"))
    if face is None:
        plan.failures.append(
            f"no font registered for {key!r} -- lettering was not drawn, and the "
            f"prompt already asked the model not to draw it")
        return plan
    if not face.path.is_file():
        plan.failures.append(
            f"font file missing: {face.path} -- run the asset download; the "
            f"design has no lettering at all until it is there")
        return plan

    arrangement = text_spec.get("arrangement", "stacked")
    if arrangement not in SUPPORTED_ARRANGEMENTS:
        plan.failures.append(
            f"arrangement {arrangement!r} is not composited yet (only "
            f"{', '.join(SUPPORTED_ARRANGEMENTS)}), so this design has no "
            f"lettering -- set text_strategy back to model_rendered for this style")
        return plan

    text = str(text_spec.get("line_1") or "").strip()
    if not text:
        plan.failures.append("text_spec has no line_1 to set")
        return plan

    plan.text = _cased(text, text_spec.get("case", "upper"))
    plan.lines = choose_lines(plan.text, face.key, face.tracking, size,
                              plan.position)
    plan.facts.update(font=face.family, font_licence=face.licence,
                      lines=len(plan.lines), arrangement=arrangement,
                      position=plan.position)
    if text_spec.get("truncated"):
        plan.warnings.append(
            "the lettering was truncated to 5 words when the prompt was built, "
            "so the shirt says less than the source title")
    return plan


def _cased(text: str, case: str) -> str:
    """Case as the style asked for it.

    Deliberately a copy of `prompts._apply_case` rather than an import: printready
    must not depend on the prompt synthesizer to draw a file, and this is four
    lines. The two are tied by a test instead.
    """
    if case == "upper":
        return text.upper()
    if case == "title":
        return text.title()
    if case == "lower":
        return text.lower()
    return text


def sample_ink(art: Image.Image,
               avoid: str | tuple[int, int, int] | None = None) -> tuple[str, str]:
    """Pick a type colour out of the artwork's own palette.

    Sampling rather than choosing keeps the limited-palette screen-print look the
    prompts ask for: the type introduces no colour the design does not already
    contain, and the same art always yields the same answer. Only reasonably dark
    colours qualify, because nothing in the pipeline records a garment colour and
    light type on a white shirt is unprintable.

    `avoid` is the ground colour removal actually keyed, and excluding it is not a
    refinement -- it is the difference between typeset and unreadable. Measured on
    the first composited generation: the model painted a #FB57AE ground, 3.4% of
    the art's bounding box came back that colour (the collision printready already
    warns about), and at luma 139 it passed the darkness test and won the vote. The
    type came out in the one colour the prompt had asked the artwork to exclude,
    matching the fringe it was surrounded by. Excluded within one bucket width,
    which is the quantization error plus the anti-aliased drift around it.

    Returns (hex, how) -- `how` says which rule fired, so a fallback or an
    exclusion is visible in the facts instead of looking like a plain sample.
    """
    small = art.convert("RGBA").resize((64, 77), Image.Resampling.NEAREST)

    # Bucketed through a LUT rather than a per-pixel Python loop: flat art has a
    # handful of colours but anti-aliased edges give every one of them a hundred
    # near-neighbours, and without bucketing the edge shades outvote the fills.
    # Split first -- point() on an RGBA image applies the LUT to all four bands,
    # which would bucket alpha too and blur the opacity test below.
    bands = small.split()
    bucketed = Image.merge("RGBA", tuple(b.point(_BUCKET_LUT) for b in bands[:3])
                           + (bands[3],))
    counts: dict[tuple[int, int, int], int] = {}
    opaque = 0
    for n, (r, g, b, a) in bucketed.getcolors(maxcolors=1 << 20) or []:
        if a < 200:
            continue
        opaque += n
        counts[(r, g, b)] = counts.get((r, g, b), 0) + n
    if not opaque:
        return DEFAULT_INK, "no opaque artwork to sample"

    def luma(c: tuple[int, int, int]) -> float:
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

    # Sorted by share, then darkness, then the colour itself -- the last key
    # matters because two different colours can share a luma, and getcolors makes
    # no promise about its order. printready has to be idempotent.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], luma(kv[0]), kv[0]))
    dark = [c for c, n in ranked
            if luma(c) <= INK_MAX_LUMA and n / opaque >= INK_MIN_SHARE]

    ground = _as_rgb(avoid)
    kept = [c for c in dark
            if ground is None or max(abs(a - b) for a, b in zip(c, ground)) > _BUCKET]
    if kept:
        # Reported only when the exclusion changed the answer. Saying so every
        # time a pink pixel was skipped would make the fact meaningless on the one
        # design where it mattered.
        note = " (residual ground colour excluded)" if kept[0] != dark[0] else ""
        return "#%02X%02X%02X" % kept[0], "sampled from the artwork" + note
    if dark:
        # Every dark colour in the design IS the ground. Setting type in it would
        # be invisible against the fringe, so this is the fallback's case even
        # though the artwork is not pale.
        return DEFAULT_INK, "artwork's only dark colour is the residual ground"

    darkest = min(counts, key=luma)
    if luma(darkest) <= 200:
        return "#%02X%02X%02X" % darkest, "darkest colour in the artwork"
    return DEFAULT_INK, "artwork has no colour dark enough to set type in"


def _as_rgb(colour: str | tuple[int, int, int] | None) -> tuple[int, int, int] | None:
    """`#RRGGBB` or a tuple to a tuple. A deliberate four-line copy of
    printready.as_rgb: the dependency runs the other way, and importing back up
    would be a cycle."""
    if colour is None:
        return None
    if isinstance(colour, str):
        text = colour.strip().lstrip("#")
        if len(text) != 6:
            return None
        try:
            return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))    # type: ignore[return-value]
        except ValueError:
            return None
    return tuple(colour)[:3]                                            # type: ignore[return-value]


@lru_cache(maxsize=1)
def _luminance_luts() -> tuple[list[float], list[float], list[float]]:
    """Per-channel relative-luminance contributions for each 8-bit value.

    WCAG's transfer function -- sRGB decoded to linear light, then weighted --
    folded into three lookups so the per-pixel work below is two adds. Built once
    rather than inlined because the whole point of measuring under the glyphs is
    that there are a lot of pixels.
    """
    linear = [v / 255 / 12.92 if v / 255 <= 0.04045
              else ((v / 255 + 0.055) / 1.055) ** 2.4 for v in range(256)]
    return ([0.2126 * c for c in linear], [0.7152 * c for c in linear],
            [0.0722 * c for c in linear])


def contrast_ratio(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """WCAG contrast ratio between two opaque colours: 1.0 identical, 21.0 for
    black on white. Public because it is the one number the overlay gate is
    expressed in, and a threshold nobody can check by hand is a threshold nobody
    trusts."""
    lr, lg, lb = _luminance_luts()
    la = lr[a[0]] + lg[a[1]] + lb[a[2]]
    lc = lr[b[0]] + lg[b[1]] + lb[b[2]]
    return (max(la, lc) + 0.05) / (min(la, lc) + 0.05)


@dataclass(frozen=True)
class Legibility:
    """What overlaid type measured against the artwork beneath it.

    `low_share` is the gate; the other two are for the reviewer. `worst` is a 5th
    percentile rather than the true minimum because the minimum is ~1.0 on any
    real illustration -- the art's own anti-aliased edges put some shade of the
    ink colour somewhere under the type on essentially every design, and a gate
    keyed on that would fire always and mean nothing.
    """

    low_share: float = 0.0
    median: float = 0.0
    worst: float = 0.0
    sampled: int = 0

    @property
    def ok(self) -> bool:
        return self.low_share <= MAX_LOW_CONTRAST


def measure_contrast(layer: Image.Image, art: Image.Image,
                     ink: str | tuple[int, int, int]) -> Legibility:
    """How well the type on `layer` reads against the artwork on `art`.

    Measured *under the glyphs* rather than across the canvas, which is the whole
    design of this function: type over a busy illustration fails exactly where the
    illustration is dark, and a whole-canvas average of a mostly-pale design
    reports "fine" while the word sits in the one dark corner. So the mask the type
    already produced is the sampling region.

    Transparent art counts as GARMENT_RGB rather than being skipped -- a word half
    over the design and half over bare shirt is legible, and skipping those pixels
    would score it on its worst half alone.

    Never raises: a bad size or an empty layer comes back as `sampled == 0`, which
    `printready` reads as "nothing to say" rather than as a pass or a failure.
    """
    ink_rgb = _as_rgb(ink) or _as_rgb(DEFAULT_INK) or (0, 0, 0)
    scale = min(1.0, _CONTRAST_H / max(1, layer.height))
    work = (max(1, round(layer.width * scale)), max(1, round(layer.height * scale)))

    # BOX is an area average, so a glyph that survives the reduction survives
    # because it is solid, not because a sample point happened to land on it.
    mask = layer.getchannel("A").resize(work, Image.Resampling.BOX)
    under = art.convert("RGBA").resize(work, Image.Resampling.BOX)

    lr, lg, lb = _luminance_luts()
    ink_lum = lr[ink_rgb[0]] + lg[ink_rgb[1]] + lb[ink_rgb[2]]
    seen: dict[tuple[int, int, int, int], float] = {}
    ratios: list[float] = []
    for cover, pixel in zip(mask.getdata(), under.getdata()):
        if cover < _GLYPH_ALPHA:
            continue
        ratio = seen.get(pixel)
        if ratio is None:
            r, g, b, a = pixel
            # Composite the art over the garment first: at a < 255 the shirt is
            # part of what the eye sees behind the letter, so scoring the raw art
            # colour would report a contrast the print does not have.
            f = a / 255
            bg = (round(r * f + GARMENT_RGB[0] * (1 - f)),
                  round(g * f + GARMENT_RGB[1] * (1 - f)),
                  round(b * f + GARMENT_RGB[2] * (1 - f)))
            lum = lr[bg[0]] + lg[bg[1]] + lb[bg[2]]
            ratio = ((max(ink_lum, lum) + 0.05) / (min(ink_lum, lum) + 0.05))
            seen[pixel] = ratio
        ratios.append(ratio)

    if not ratios:
        return Legibility()
    ratios.sort()
    low = sum(1 for r in ratios if r < MIN_CONTRAST) / len(ratios)
    return Legibility(low_share=low,
                      median=ratios[len(ratios) // 2],
                      worst=ratios[int(len(ratios) * 0.05)],
                      sampled=len(ratios))


def render(plan: LetteringPlan, size: tuple[int, int], ink: str,
           band: int | None = None) -> Image.Image:
    """A transparent `size` layer with the type drawn on it.

    Glyph by glyph rather than one `ImageDraw.text` call per line. That buys
    letter-spacing, which condensed display faces need at merch scale, and it is
    the same primitive curved baselines need next -- per-glyph placement, plus a
    rotation. The cost is kerning pairs, which added tracking hides.
    """
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if not plan.ok or not plan.lines:
        return layer

    band = plan.band_height(size) if band is None else band
    left, top, right, bottom = text_box(plan.position, size, band)
    key, tracking = plan.face.key, plan.face.tracking
    point = _fit_size(key, tracking, plan.lines, (right - left, bottom - top))
    _, ink_h, ink_top, step = _measure(key, point, tracking, plan.lines)

    font = load_font(key, point)
    draw = ImageDraw.Draw(layer)
    # Centre the ink, not the metrics: `first_baseline` is placed so the block's
    # real top and bottom sit equidistant inside the box.
    first_baseline = top + (bottom - top - ink_h) / 2 - ink_top
    for i, line in enumerate(plan.lines):
        advance = sum(font.getlength(ch) for ch in line)
        advance += tracking * point * max(0, len(line) - 1)
        x = left + (right - left - advance) / 2
        y = first_baseline + i * step
        for ch in line:
            if not ch.isspace():
                draw.text((x, y), ch, font=font, fill=ink, anchor="ls")
            x += font.getlength(ch) + tracking * point

    plan.facts.update(point_size=point, ink=ink,
                      text_box=f"{right - left}x{bottom - top}+{left}+{top}")
    return layer
