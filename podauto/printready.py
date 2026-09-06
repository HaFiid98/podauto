"""Generated art -> the file Merch actually accepts: 4500x5400, transparent.

Three operations, in this order, because each depends on the previous one being
exact:

1. **Strip the ground.** The generation is a full-bleed rectangle -- the model
   paints a background whether asked to or not -- and Merch prints every opaque
   pixel as ink, so a ground left in place prints as a coloured slab around the
   design. Removal here is a border-seeded colour key, not salient-object
   segmentation. That is a deliberate choice, not a shortcut: this pipeline
   generates *flat vector art on a uniform ground*, which is the one case where
   colour keying beats a segmentation model -- it gives a hard, exact edge with
   no soft matte, needs no 300MB of onnxruntime, and, unlike a model, it can
   report *why* it went wrong (see the collision measurement below).

2. **Scale to print.** By tracing to vector and rasterizing at the target size
   (`vtracer` -> `resvg`), not by interpolating pixels. The source is 848x1024
   and the target is 5.3x that; LANCZOS on flat art turns crisp screen-print
   edges into soft ramps, while re-rasterizing a traced curve is sharp at any
   size. If either library is missing this falls back to LANCZOS and says so in
   `method`, because a soft print is worth more than no print.

3. **Typeset the lettering**, for the styles that opt in. Optional and off by
   default: pass no plan and this module behaves exactly as it did before. See
   `lettering.py` for why -- text is the pipeline's measured defect, and drawing
   it here instead of asking the model for it removes the defect class. The band
   is taken off the canvas *before* step 2 so the artwork rasterizes straight to
   its final height; scaling it again afterwards would undo the point of tracing.

The collision measurement is the reason this module reports facts rather than
just writing a file. Removal can only fail one way that matters: when the ground
colour *also appears in the artwork*. Then the fill either eats into the design
or leaves the same colour stranded inside it. Counting ground-coloured pixels
that are NOT reachable from the border measures exactly that -- observed in 2 of
the first 3 real generations, where the model reused a palette colour as the
ground. `prompts.py` now asks for a chroma ground that the palette excludes; this
is the check that the model obeyed.

Nothing here advances `pipeline_status`. A print-ready file enriches a record
that is already queued for human review, and that review is the pipeline's
load-bearing safety property -- it must not be skippable as a side effect of
producing a file.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from . import lettering as _lettering
from .imageqa import TARGET_H, TARGET_W
from .models import Record, StyleVariation

PRINT_DPI = 300

# Per-channel distance from the ground colour still counted as ground. A
# diffusion model's "flat" ground carries a few counts of noise and JPEG routes
# add ringing, so zero tolerance keys out nothing; but a wide tolerance eats
# anti-aliased edges and any artwork sitting near the ground colour. 32 clears
# JPEG ringing on a flat field while staying far from a distinct fill colour.
GROUND_TOLERANCE = 32

# Alpha below this snaps to fully transparent. The ramp between ground and art
# is where chroma fringe lives: a pixel at alpha 20 still carries the ground's
# RGB and prints as a faint halo on a light shirt. Snapping the bottom of the
# ramp is cheaper and more predictable than un-premultiplying colour, and on
# flat art the ramp is only a pixel or two wide.
ALPHA_FLOOR = 40

# Ground-coloured pixels stranded inside the art's bounding box, as a fraction
# of that box, above which the ground colour is judged to be reused by the
# artwork. Not zero: a few stray pixels are noise, not a palette collision.
COLLISION_WARN = 0.01

# The fill must reach at least this fraction of the canvas for a removable
# ground to exist at all. Below it, the generation is full-bleed art rather than
# a subject on a ground, and there is nothing to key out.
MIN_GROUND = 0.05

# Above this, almost nothing survived removal -- the fill escaped through the
# artwork and ate it, which is the failure mode a palette collision produces.
MAX_GROUND = 0.98

# Free margin the artwork is scaled to fit inside, as a fraction of the canvas.
# Not cosmetic: the generation is full-bleed -- alpha bbox (0,0,848,1024) on 3 of 3
# real sources, touching every edge -- so scaling the art to fill the canvas puts
# whatever the model ran off the frame onto the 1px border `imageqa` votes on.
# Measured through the vector path on those 3 files: edge_opaque 0.0328 / 0.0253 /
# 0.0027 became 0.0000 / 0.0000 / 0.0000 at this value, where 2 of the 3 had been
# failing MAX_EDGE_OPAQUE outright and so could not be approved at all. The cost is
# 4% of design height, which is why it is this small rather than lettering's 0.04/0.06.
ART_MARGIN = 0.02

# The phrase that identifies the "model painted a different ground" warning in a
# variation's qa_notes. Named because review.py classifies that warning as
# informational rather than blocking, and matching on the whole sentence would put
# the wording in two files. Measured 2026-08-17: 3 of 3 real generations ignored
# the requested ground, so the note fires on essentially every design while the
# files themselves pass print QA. See review.INFORMATIONAL_NOTES.
IGNORED_GROUND_PHRASE = "ignored the ground instruction"

# Same contract as above: review.py matches on this substring, so rewording the
# warning without rewording the constant silently reclassifies it as blocking.
# This is the *measured pass* -- an overlay whose type was checked and reads. It
# stays informational because it fires on every overlay that is fine, and a note on
# every design is a note nobody reads.
OVERLAY_LEGIBILITY_PHRASE = "position center_overlay"

# The measured *failure*, and deliberately NOT on review.INFORMATIONAL_NOTES: an
# overlay the contrast measurement says is unreadable is a defect in the file, so it
# blocks the approval the way any unclassified note does. The trap this constant
# exists to avoid is the two messages sharing wording -- the allowlist matches on
# substrings, so a failure that happened to contain OVERLAY_LEGIBILITY_PHRASE would
# be silently waved through as the pass it is not. Asserted in the tests.
OVERLAY_CONTRAST_PHRASE = "overlaid type is low-contrast"

PRINT_SUFFIX = "_print.png"


@dataclass
class Removal:
    """What removal produced, and what it measured while producing it."""

    image: Image.Image
    facts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PrintResult:
    path: str
    method: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return f"pass ({self.method})"
        parts = [f"FAIL: {f}" for f in self.failures]
        parts += [f"warn: {w}" for w in self.warnings]
        return "; ".join(parts)


def as_rgb(colour: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """Accept `#RRGGBB` as well as a tuple, since the ground hint is config text."""
    if isinstance(colour, str):
        text = colour.strip().lstrip("#")
        if len(text) != 6:
            raise ValueError(f"not a #RRGGBB colour: {colour!r}")
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))   # type: ignore[return-value]
    return tuple(colour)                                          # type: ignore[return-value]


def border_colour(img: Image.Image) -> tuple[int, int, int]:
    """The most common colour on the 1px border.

    The border rather than a single corner pixel: a corner can land on a stray
    speck or a vignette, and a modal vote over ~3700 pixels cannot.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    strips = [rgb.crop((0, 0, w, 1)), rgb.crop((0, h - 1, w, h)),
              rgb.crop((0, 0, 1, h)), rgb.crop((w - 1, 0, w, h))]
    counts: dict[tuple[int, int, int], int] = {}
    for strip in strips:
        for n, colour in strip.getcolors(maxcolors=w * h * 2) or []:
            counts[colour] = counts.get(colour, 0) + n
    if not counts:
        return (255, 255, 255)
    return max(counts.items(), key=lambda kv: kv[1])[0]


def ground_distance(img: Image.Image, ground: tuple[int, int, int]) -> Image.Image:
    """Per-pixel max-channel distance from `ground`, as an 8-bit L image.

    Max-of-channels rather than a Euclidean norm, and built entirely from PIL
    band operations so it runs at C speed over ~870k pixels. Channel maximum is
    the right metric for keying a flat colour: it treats a 40-count shift in one
    channel as a real difference, where a norm would average it away.
    """
    diff = ImageChops.difference(img.convert("RGB"),
                                 Image.new("RGB", img.size, ground))
    r, g, b = diff.split()
    return ImageChops.lighter(ImageChops.lighter(r, g), b)


def flood_from_border(inside: bytearray, w: int, h: int) -> bytearray:
    """Which ground-coloured pixels are reachable from the canvas border.

    Connectivity is the whole point. A plain "every pixel within tolerance of
    the ground is transparent" threshold punches holes through the middle of the
    artwork wherever the design legitimately uses that colour -- a navy sky and a
    navy ground are the same number to a threshold and completely different
    things to a print. Seeding only from the border and spreading through
    contiguous ground keeps interior matches opaque, and the ones left behind are
    the collision measurement.

    Scanline span fill, so each pixel is visited O(1) times and the stack holds
    spans rather than pixels; a per-pixel recursive fill overflows on a 870k-pixel
    canvas that is mostly ground.
    """
    reached = bytearray(w * h)
    stack: list[tuple[int, int]] = []

    def seed(x: int, y: int) -> None:
        i = y * w + x
        if inside[i] and not reached[i]:
            stack.append((x, y))

    for x in range(w):
        seed(x, 0)
        seed(x, h - 1)
    for y in range(h):
        seed(0, y)
        seed(w - 1, y)

    while stack:
        x, y = stack.pop()
        row = y * w
        if reached[row + x] or not inside[row + x]:
            continue

        x0 = x
        while x0 > 0 and inside[row + x0 - 1] and not reached[row + x0 - 1]:
            x0 -= 1
        x1 = x
        while x1 < w - 1 and inside[row + x1 + 1] and not reached[row + x1 + 1]:
            x1 += 1
        reached[row + x0:row + x1 + 1] = b"\x01" * (x1 - x0 + 1)

        for ny in (y - 1, y + 1):
            if not 0 <= ny < h:
                continue
            nrow = ny * w
            xi = x0
            while xi <= x1:
                # One push per contiguous run, not per pixel: without this the
                # stack grows with area instead of with the number of spans.
                if inside[nrow + xi] and not reached[nrow + xi]:
                    stack.append((xi, ny))
                    while xi <= x1 and inside[nrow + xi] and not reached[nrow + xi]:
                        xi += 1
                else:
                    xi += 1
    return reached


def remove_ground(img: Image.Image, requested: str | tuple[int, int, int] | None = None,
                  tolerance: int = GROUND_TOLERANCE) -> Removal:
    """Key out the border-connected ground and return RGBA plus measurements.

    `requested` is the ground colour the prompt asked for. It is NOT used as the
    key -- the key is always the colour actually on the border, because a model
    that ignored the instruction still produced a file worth salvaging. What
    `requested` buys is the ability to say the model ignored it, which is a
    prompt problem rather than a removal problem and needs to be reported as one.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    detected = border_colour(rgb)
    dist = ground_distance(rgb, detected)

    inside = bytearray(dist.point(lambda v: 1 if v <= tolerance else 0).tobytes())
    reached = flood_from_border(inside, w, h)

    # Alpha is the distance ramp inside the keyed region and fully opaque
    # outside it, which anti-aliases the cut for free: a pixel halfway between
    # ground and ink gets halfway alpha.
    ramp = dist.point(lambda v: 255 if v >= tolerance else round(v * 255 / tolerance))
    ramp = ramp.point(lambda v: 0 if v < ALPHA_FLOOR else v)
    keyed = Image.frombytes("L", (w, h), bytes(reached)).point(lambda v: 255 if v else 0)
    alpha = Image.composite(ramp, Image.new("L", (w, h), 255), keyed)

    out = rgb.convert("RGBA")
    out.putalpha(alpha)

    total = w * h
    n_reached = sum(reached)
    stranded = sum(inside) - n_reached
    ground_fraction = n_reached / total

    art = alpha.point(lambda v: 255 if v > 8 else 0)
    bbox = art.getbbox()
    box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox else 0
    collision = stranded / box_area if box_area else 0.0
    hist = alpha.histogram()

    facts: dict[str, Any] = {
        "ground": "#%02X%02X%02X" % detected,
        "ground_fraction": round(ground_fraction, 4),
        "art_bbox": bbox,
        "ground_colour_in_art": round(collision, 4),
        "fringe": round(sum(hist[1:255]) / total, 5),
        "tolerance": tolerance,
    }

    warnings: list[str] = []
    if requested is not None:
        want = as_rgb(requested)
        facts["ground_requested"] = "#%02X%02X%02X" % want
        if max(abs(a - b) for a, b in zip(want, detected)) > tolerance:
            warnings.append(
                f"asked for a #{'%02X%02X%02X' % want} ground, got "
                f"#{'%02X%02X%02X' % detected} -- the model "
                f"{IGNORED_GROUND_PHRASE}, so removal is keying whatever it "
                f"painted instead")

    if ground_fraction < MIN_GROUND:
        warnings.append(
            f"only {ground_fraction:.1%} of the canvas keyed out as ground "
            f"(minimum {MIN_GROUND:.0%}) -- this looks like full-bleed art with "
            f"no removable ground, and Merch will print it as a slab")
    elif ground_fraction > MAX_GROUND:
        warnings.append(
            f"{ground_fraction:.1%} of the canvas keyed out (maximum "
            f"{MAX_GROUND:.0%}) -- the fill escaped through the artwork and "
            f"removed it, not just the ground")

    if collision > COLLISION_WARN:
        warnings.append(
            f"{collision:.1%} of the art's bounding box is the ground colour "
            f"#{'%02X%02X%02X' % detected} (limit {COLLISION_WARN:.0%}) -- the "
            f"design reuses the ground colour, so edges touching it will have "
            f"been cut away; check the file before approving")

    return Removal(image=out, facts=facts, warnings=warnings)


# --- scaling to the print canvas -----------------------------------------

def vector_upscale(rgba: Image.Image, height: int = TARGET_H) -> Image.Image:
    """Trace to vector, rasterize at `height`. Raises if the tools are absent.

    Only `height` is passed to the rasterizer: resvg derives the other dimension
    from the SVG's own aspect, so asking for both would either stretch the art or
    be silently ignored (measured: a 4500x5400 request on an 848x1024 trace
    returns 4472x5400). The 28px shortfall is squared up by `fit_to_canvas`,
    where it becomes transparent padding rather than a distortion.
    """
    import resvg_py                     # local: optional dependency
    import vtracer

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.png"
        svg = Path(tmp) / "out.svg"
        rgba.save(src)
        # Tuned for flat art: `spline` for smooth curves over stair-stepped
        # polygons, filter_speckle to drop single-pixel noise that would
        # otherwise become its own path, color_precision high enough to keep a
        # limited palette exact. Transparent regions are not traced at all, so
        # the alpha from removal survives into the SVG as absence of geometry.
        vtracer.convert_image_to_svg_py(
            str(src), str(svg), colormode="color", hierarchical="stacked",
            mode="spline", filter_speckle=4, color_precision=6, path_precision=8)
        png = resvg_py.svg_to_bytes(svg_path=str(svg), height=height)

    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


def raster_upscale(rgba: Image.Image, height: int = TARGET_H) -> Image.Image:
    """LANCZOS fallback, aspect preserved. Softer edges than tracing, but a
    soft print beats no print when vtracer/resvg are not installed."""
    width = round(rgba.width * height / rgba.height)
    return rgba.resize((width, height), Image.Resampling.LANCZOS)


def inset_box(size: tuple[int, int] = (TARGET_W, TARGET_H)) -> tuple[int, int]:
    """The (height, width) the artwork is scaled to fit inside `size`.

    Returned height-first because that is the order the upscalers take: both
    `vector_upscale` and `raster_upscale` are height-driven, and the width is a
    clamp applied afterwards by `fit_width`.
    """
    return (round(size[1] * (1 - 2 * ART_MARGIN)),
            round(size[0] * (1 - 2 * ART_MARGIN)))


def fit_width(img: Image.Image, limit: int) -> Image.Image:
    """Scale `img` down, aspect preserved, if it is wider than `limit`.

    The upscalers only take a height, so on art wider than the canvas aspect the
    height fit alone overflows sideways and spends the inset on the left and right
    borders -- the two edges the current generations happen not to touch. Inert at
    the 848x1024 (0.828) the providers return, load-bearing for anything wider
    than 5:6, and this is the whole reason the inset is a box and not one number.
    """
    if img.width <= limit:
        return img
    return img.resize((limit, max(1, round(img.height * limit / img.width))),
                      Image.Resampling.LANCZOS)


def fit_to_canvas(img: Image.Image, size: tuple[int, int] = (TARGET_W, TARGET_H)
                  ) -> Image.Image:
    """Centre `img` on an exactly `size` transparent canvas, padding or cropping.

    Padding rather than resizing to the exact target is what keeps the artwork
    undistorted. The generation is 848x1024 (0.8281) and the print canvas is 5:6
    (0.8333) -- resizing into it would stretch every design by 0.6% horizontally
    for no reason, where padding costs only transparent pixels that print as
    nothing at all. `make_print_ready` scales the art into `inset_box` first, so
    in practice the pad it adds is that margin plus the aspect difference.
    """
    tw, th = size
    if img.width > tw or img.height > th:
        left = max(0, (img.width - tw) // 2)
        top = max(0, (img.height - th) // 2)
        img = img.crop((left, top, left + min(tw, img.width),
                        top + min(th, img.height)))
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.alpha_composite(img.convert("RGBA"),
                           ((tw - img.width) // 2, (th - img.height) // 2))
    return canvas


def _note_overlay(result: "PrintResult", layer: Image.Image,
                  art: Image.Image, ink: str) -> None:
    """Measure overlaid type against the art under it and say so, either way.

    Both outcomes are reported. A silent pass would leave the reviewer unable to
    tell a design that was checked from one produced before anything checked it,
    and the pass note is also where the numbers live for the one judgement the
    measurement cannot make -- whether the lockup looks right.

    The failure is an ordinary qa_note, so it blocks the approval; the pass is on
    review.INFORMATIONAL_NOTES and does not. That asymmetry is the entire gate.
    """
    read = _lettering.measure_contrast(layer, art, ink)
    if not read.sampled:
        # No glyph survived the reduction, so there is nothing to measure and
        # nothing to claim. A plan that drew no type has already warned about it.
        return

    result.facts["overlay_contrast"] = (
        f"median {read.median:.1f}:1, 5th pct {read.worst:.1f}:1, "
        f"{read.low_share:.0%} below {_lettering.MIN_CONTRAST:.0f}:1")
    if read.ok:
        result.warnings.append(
            f"typeset text is overlaid on the artwork rather than banded "
            f"({OVERLAY_LEGIBILITY_PHRASE}) and measures {read.median:.1f}:1 "
            f"against it, {read.low_share:.0%} of the type below "
            f"{_lettering.MIN_CONTRAST:.0f}:1 -- it is spelled correctly by "
            f"construction and it reads, so what is left to judge by eye is "
            f"whether the lockup sits well on what the model painted")
        return
    result.warnings.append(
        f"{OVERLAY_CONTRAST_PHRASE}: {read.low_share:.0%} of the typeset text "
        f"is under {_lettering.MIN_CONTRAST:.0f}:1 against the artwork behind "
        f"it (median {read.median:.1f}:1, 5th pct {read.worst:.1f}:1, limit "
        f"{_lettering.MAX_LOW_CONTRAST:.0%}) -- the words will not read on the "
        f"shirt. Set the style's ink explicitly, or band the text instead of "
        f"overlaying it")


def make_print_ready(src: str | Path, out: str | Path,
                     requested_ground: str | None = None,
                     method: str = "auto",
                     tolerance: int = GROUND_TOLERANCE,
                     size: tuple[int, int] = (TARGET_W, TARGET_H),
                     lettering: _lettering.LetteringPlan | None = None,
                     ) -> PrintResult:
    """Ground-removed, 4500x5400, 300dpi PNG at `out`. Never raises on bad input.

    `method` is "auto" (trace if the tools import, else LANCZOS), "vector", or
    "raster". "auto" degrades rather than failing, and records which path it took
    in `PrintResult.method` -- a silent downgrade to a soft upscale would be
    indistinguishable from a crisp one until the shirt arrives.

    `size` defaults to Merch's standard placement and exists so a second
    placement size does not need a second code path; tests use a small one to
    avoid rasterizing 24 megapixels per assertion.

    `lettering` defaults to None, which is the model_rendered path and byte-for-byte
    what this function did before it existed. Given a plan, the type is typeset
    over the finished canvas and the artwork is scaled into whatever the text band
    leaves. A plan that cannot be rendered is a *warning*, not a failure: the file
    is still worth writing, and an unclassified qa_note already stops the approval
    (see review.INFORMATIONAL_NOTES), so a human looks at a design that lost its
    lettering instead of it exporting silently.
    """
    result = PrintResult(path=str(out))
    src = Path(src)
    if not src.is_file():
        result.failures.append(f"no file at {src}")
        return result

    try:
        with Image.open(src) as handle:
            handle.load()
            removal = remove_ground(handle, requested_ground, tolerance)
    except Exception as exc:                                       # noqa: BLE001
        result.failures.append(f"could not read {src.name}: {type(exc).__name__}: {exc}")
        return result

    result.facts.update(removal.facts)
    result.warnings.extend(removal.warnings)
    result.facts["source"] = f"{src.name} {removal.image.width}x{removal.image.height}"

    layer, band = None, 0
    if lettering is not None:
        result.warnings.extend(lettering.failures)
        result.warnings.extend(lettering.warnings)
        if lettering.ok:
            ink = lettering.ink
            if ink:
                result.facts["lettering_ink_source"] = "style override"
            else:
                # The keyed ground, not the requested one: what contaminates the
                # palette is the colour the model actually painted.
                ink, how = _lettering.sample_ink(removal.image,
                                                 avoid=removal.facts.get("ground"))
                result.facts["lettering_ink_source"] = how
            band = lettering.band_height(size)
            layer = _lettering.render(lettering, size, ink, band)
            # The overlay legibility note is NOT decided here. It depends on what
            # the artwork looks like under the glyphs, and the artwork does not
            # exist yet at this point -- see `_note_overlay` below.
            result.facts["lettering_band"] = band
        result.facts["lettering"] = (
            " ".join(f"{k}={v}" for k, v in lettering.facts.items()
                     if k != "font_licence") or "not drawn")

    art_h = size[1] - band
    # The art is scaled into an inset box rather than the full one, and then padded
    # back out by `fit_to_canvas`. `art_h` still sets the outer box, so a band keeps
    # taking exactly its height off the art and the two stay disjoint.
    inner_h, inner_w = inset_box((size[0], art_h))

    scaled = None
    if method in ("auto", "vector"):
        try:
            scaled = vector_upscale(removal.image, inner_h)
            result.method = "vector"
        except Exception as exc:                                   # noqa: BLE001
            if method == "vector":
                result.failures.append(
                    f"vectorize failed: {type(exc).__name__}: {exc}")
                return result
            result.warnings.append(
                f"vectorize unavailable ({type(exc).__name__}), fell back to a "
                f"LANCZOS upscale -- edges will be softer than a traced print")
    if scaled is None:
        scaled = raster_upscale(removal.image, inner_h)
        result.method = result.method or "raster"

    scaled = fit_width(scaled, inner_w)
    final = fit_to_canvas(scaled, (size[0], art_h))
    if band:
        # The art canvas is short by `band`; seat it at the end the text is not on.
        seated = Image.new("RGBA", size, (0, 0, 0, 0))
        seated.alpha_composite(final, (0, band if lettering.position == "above" else 0))
        final = seated
    if layer is not None:
        if not band:
            # The last moment both halves exist separately, which is the only
            # moment the measurement is possible: one more line and the type is
            # part of the artwork it was supposed to be measured against.
            _note_overlay(result, layer, final, ink)
        final.alpha_composite(layer)
    result.facts["scaled"] = f"{scaled.width}x{scaled.height}"
    result.facts["canvas"] = f"{final.width}x{final.height}"
    # Recorded so a reviewer reading the facts can tell the art was inset rather
    # than scaled small by a failed trace.
    result.facts["art_margin"] = ART_MARGIN

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 4500x5400 already IS 15"x18" at 300dpi; the header only tells other
        # software to assume that instead of guessing 72.
        final.save(out_path, "PNG", dpi=(PRINT_DPI, PRINT_DPI))
    except Exception as exc:                                       # noqa: BLE001
        result.failures.append(f"could not write {out_path}: {exc}")
        return result

    result.facts["bytes"] = out_path.stat().st_size
    return result


# --- record level --------------------------------------------------------

@dataclass
class PrintSummary:
    produced: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[PrintResult] = field(default_factory=list)


def print_path_for(variation: StyleVariation, out_dir: str | Path | None = None) -> Path:
    """`<generated stem>_print.png`, beside the source unless redirected.

    Derived from the generated file's own name rather than from the record id, so
    the pair is obvious in a directory listing and a re-run overwrites instead of
    accumulating -- the same reasoning as imagegen.image_path_for.
    """
    src = Path(variation.image_path or "")
    name = src.stem + PRINT_SUFFIX
    return (Path(out_dir) if out_dir else src.parent) / name


def printready_variation(variation: StyleVariation, out_dir: str | Path | None = None,
                         method: str = "auto", force: bool = False,
                         tolerance: int = GROUND_TOLERANCE,
                         size: tuple[int, int] = (TARGET_W, TARGET_H),
                         style: dict[str, Any] | None = None,
                         ) -> PrintResult | None:
    """Produce one print-ready file, or None if there is nothing to work from.

    Warnings land in `qa_notes` so the review queue shows them: a palette
    collision is precisely the thing a human should look at the file for. Notes
    are de-duplicated because this command is re-runnable and an accumulating
    list of identical warnings makes the queue unreadable.

    `style` is the style's config entry, and it is what decides whether the
    lettering is typeset here or was the model's job -- `lettering.plan_from`
    returns None for every style that has not opted in, so omitting it keeps the
    old behaviour rather than changing it.
    """
    if not variation.image_path:
        return None

    target = print_path_for(variation, out_dir)
    if not force and variation.print_path and Path(variation.print_path).is_file():
        return None

    result = make_print_ready(
        variation.image_path, target, requested_ground=variation.ground_hint,
        method=method, tolerance=tolerance, size=size,
        lettering=_lettering.plan_from(variation.text_spec, style, size))
    if result.ok:
        variation.print_path = str(target)
    for warning in result.warnings:
        note = f"print ready warning: {warning}"
        if note not in variation.qa_notes:
            variation.qa_notes.append(note)
    for failure in result.failures:
        note = f"print ready: {failure}"
        if note not in variation.qa_notes:
            variation.qa_notes.append(note)
    return result


def printready_record(record: Record, out_dir: str | Path | None = None,
                      method: str = "auto", force: bool = False,
                      tolerance: int = GROUND_TOLERANCE,
                      size: tuple[int, int] = (TARGET_W, TARGET_H),
                      styles: dict[str, dict[str, Any]] | None = None
                      ) -> PrintSummary:
    """Every generated variation of one record. Does not touch pipeline_status:
    the record stays in the review queue it is already in.

    `styles` is StyleDatabase.styles -- already a style_id -> config dict, so no
    lookup type is introduced for this.
    """
    summary = PrintSummary()
    styles = styles or {}
    for variation in record.variations:
        result = printready_variation(variation, out_dir, method, force,
                                      tolerance, size,
                                      styles.get(variation.style_id))
        if result is None:
            summary.skipped += 1
            continue
        summary.results.append(result)
        if result.ok:
            summary.produced += 1
        else:
            summary.failed += 1
    return summary






