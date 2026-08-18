"""Image QA: reject an unusable file before a human ever opens it.

Two gates, not one, because what is true of a file depends on where it is in the
pipeline. Straight out of `imagegen` an image is provider-native (e.g. 848x1024)
and fully opaque -- background removal has not run yet. Asserting the print
contract there would fail every file, so `check_generated()` checks only what is
already supposed to hold, and `check_print_ready()` checks the print contract
after the removal and vectorize steps.

What this module deliberately does not check: **lettering**. Text is
model-rendered by decision, so spelling is not verified anywhere in this
pipeline. Misspelled type reaches the review queue by design; the `review`
command prints the intended string next to each variation so the check is a
comparison rather than a guess.

The checks that exist are the ones a human should not have to make: an endpoint
that answered with a 64x64 error tile, a blank canvas, a file that decodes as a
truncated PNG, art whose aspect ratio will be cropped by Merch. Each failure
names the measurement that produced it, because "QA failed" sends the operator
back to the file while "aspect 1.000, expected 0.833 +/-0.02" does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .models import PipelineStatus, Record, StyleVariation

# Merch's print target, reached by the vectorize stage. 4500x5400 already IS
# 15"x18" at 300dpi -- the DPI header is metadata that changes no pixels, which
# is why a wrong one is a warning here and a wrong pixel count is a failure.
TARGET_W, TARGET_H = 4500, 5400
EXPECTED_DPI = 300

ASPECT = 5 / 6
ASPECT_TOLERANCE = 0.02

# Two different questions, so two different numbers. MIN_SHORT_EDGE asks "is
# this a generation at all, or an error tile?" -- a provider that answers a
# failure with a 64x64 placeholder still returns a valid PNG. ADVISED_SHORT_EDGE
# asks "is there enough source detail for a 15-inch print?", which is a judgement
# and therefore a warning: a provider capped at 512x512 yields 424x512, below
# what is comfortable but well above nothing, and failing it would discard usable
# art over a preference. Conflating the two is what makes a gate unpassable.
MIN_SHORT_EDGE = 256
ADVISED_SHORT_EDGE = 768

# Fraction of the canvas that must differ from the single most common colour.
# A flat ground with no art on it is the failure mode this catches: some
# endpoints return a solid frame when the prompt trips a filter, and a solid
# frame is a perfectly valid PNG.
MIN_INK = 0.02

# Border pixels allowed to be opaque once alpha exists. Not zero: anti-aliased
# art that reaches the edge legitimately leaves a few opaque pixels, and a
# hard zero would fail files that print correctly.
MAX_EDGE_OPAQUE = 0.02
ALPHA_OPAQUE = 8          # alpha above this counts as "not transparent"


@dataclass
class QaReport:
    """Findings for one file. `facts` is what was measured, so a report is
    readable without re-opening the image."""

    path: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "pass"
        parts = [f"FAIL: {f}" for f in self.failures]
        parts += [f"warn: {w}" for w in self.warnings]
        return "; ".join(parts)


def _load(path: str | Path, report: QaReport) -> Image.Image | None:
    """Open `path`, or record why it could not be used as an image.

    Opened twice on purpose: `verify()` is the only way PIL detects a truncated
    file, and it leaves the instance unusable afterwards. A truncated PNG is the
    realistic outcome of a connection dropped mid-download, and it opens fine
    until something reads its pixels.
    """
    p = Path(path)
    if not p.is_file():
        report.failures.append(f"no file at {p}")
        return None
    try:
        with Image.open(p) as probe:
            probe.verify()
        img = Image.open(p)
        img.load()
    except Exception as exc:                                   # noqa: BLE001
        report.failures.append(f"not a readable image: {type(exc).__name__}: {exc}")
        return None
    return img


def ink_fraction(img: Image.Image) -> float:
    """Fraction of the canvas that is not the single most common colour.

    Measured on a 64x64 nearest-neighbour reduction quantized to 16 colours.
    Nearest-neighbour because interpolated downscaling invents intermediate
    colours and would make a flat canvas look busy; the quantization is what
    keeps JPEG ringing and dither on a flat ground from reading as art.

    Known limitation: this detects a *flat* empty canvas, not an empty one with a
    gradient sky -- a smooth gradient measures ~90% ink. That case is left to the
    human reviewer, because distinguishing "gradient with no subject" from
    "gradient that is the design" is not a pixel-counting problem.
    """
    small = img.convert("RGB").resize((64, 64), Image.Resampling.NEAREST)
    hist = small.quantize(colors=16).histogram()
    total = 64 * 64
    return 1.0 - (max(hist) / total)


def edge_opaque_fraction(img: Image.Image) -> float | None:
    """Fraction of the 1px border that is not transparent, or None without alpha."""
    if img.mode not in ("RGBA", "LA") and "transparency" not in img.info:
        return None
    alpha = img.convert("RGBA").getchannel("A")
    w, h = alpha.size
    strips = [
        alpha.crop((0, 0, w, 1)), alpha.crop((0, h - 1, w, h)),
        alpha.crop((0, 0, 1, h)), alpha.crop((w - 1, 0, w, h)),
    ]
    opaque = total = 0
    for strip in strips:
        counts = strip.histogram()
        opaque += sum(counts[ALPHA_OPAQUE + 1:])
        total += sum(counts)
    return opaque / total if total else 0.0


def _check_shape(img: Image.Image, report: QaReport) -> None:
    """Aspect and ink checks, shared by both gates."""
    w, h = img.size
    aspect = w / h
    report.facts.update(width=w, height=h, aspect=round(aspect, 4), mode=img.mode)

    if abs(aspect - ASPECT) > ASPECT_TOLERANCE:
        report.failures.append(
            f"aspect {aspect:.3f}, expected {ASPECT:.3f} +/-{ASPECT_TOLERANCE} "
            f"(5:6) -- Merch crops anything else")

    ink = ink_fraction(img)
    report.facts["ink"] = round(ink, 4)
    if ink < MIN_INK:
        report.failures.append(
            f"blank or near-uniform canvas: {ink:.1%} of pixels differ from the "
            f"dominant colour, minimum {MIN_INK:.0%}")


def check_generated(path: str | Path) -> QaReport:
    """Gate straight after generation, before removal or vectorizing.

    Checks aspect, a floor on size, and that the canvas is not blank. It does
    NOT require alpha or the print size: neither exists yet at this point, and a
    check that always fails is a check nobody reads.
    """
    report = QaReport(path=str(path))
    img = _load(path, report)
    if img is None:
        return report
    with img:
        _check_shape(img, report)
        short = min(img.size)
        if short < MIN_SHORT_EDGE:
            report.failures.append(
                f"short edge {short}px, minimum {MIN_SHORT_EDGE}px -- too small "
                f"to be a generation rather than an error tile")
        elif short < ADVISED_SHORT_EDGE:
            report.warnings.append(
                f"short edge {short}px, under the {ADVISED_SHORT_EDGE}px advised "
                f"for a 15-inch print; this provider's max_resolution is low")
    return report


def check_print_ready(path: str | Path) -> QaReport:
    """Gate after removal and vectorizing: the print contract, asserted exactly.

    Pixel dimensions are a failure and the DPI header is a warning, because the
    file is already 15"x18" at 300dpi by virtue of being 4500x5400 -- the header
    only tells software what to assume when it is not told.
    """
    report = QaReport(path=str(path))
    img = _load(path, report)
    if img is None:
        return report

    with img:
        _check_shape(img, report)
        if img.size != (TARGET_W, TARGET_H):
            report.failures.append(
                f"{img.width}x{img.height}, expected {TARGET_W}x{TARGET_H}")

        edge = edge_opaque_fraction(img)
        report.facts["edge_opaque"] = None if edge is None else round(edge, 4)
        if edge is None:
            report.failures.append(
                f"no alpha channel (mode {img.mode}) -- Merch prints the "
                f"background as ink")
        elif edge > MAX_EDGE_OPAQUE:
            report.failures.append(
                f"{edge:.1%} of the border is opaque, maximum "
                f"{MAX_EDGE_OPAQUE:.0%} -- background removal did not run or "
                f"left a frame")

        dpi = img.info.get("dpi")
        report.facts["dpi"] = tuple(round(d) for d in dpi) if dpi else None
        if not dpi:
            report.warnings.append(
                f"no DPI header; the pixels are already {TARGET_W}x{TARGET_H} "
                f"so this is cosmetic, but set it to {EXPECTED_DPI}")
        elif round(dpi[0]) != EXPECTED_DPI:
            report.warnings.append(
                f"DPI header says {round(dpi[0])}, expected {EXPECTED_DPI}")
    return report


# --- record level --------------------------------------------------------

@dataclass
class QaSummary:
    passed: int = 0
    failed: int = 0
    missing: int = 0
    reports: list[QaReport] = field(default_factory=list)


def qa_variation(variation: StyleVariation, stage: str = "generated") -> QaReport | None:
    """Run one gate against one variation, recording failures in qa_notes.

    Each stage checks its own file. The print gate reads `print_path`, not
    `image_path`: pointing it at the generated file would assert 4500x5400 with
    alpha against an opaque 848x1024 native output and fail every record, which
    is the "check that always fails is a check nobody reads" failure this module
    was split in two to avoid.

    The file is left on disk when it fails. Deleting it would remove the only
    evidence the operator has for deciding whether the prompt, the provider or
    the removal step is at fault.
    """
    print_ready = stage == "print_ready"
    path = variation.print_path if print_ready else variation.image_path
    if not path:
        return None
    report = (check_print_ready if print_ready else check_generated)(path)
    for failure in report.failures:
        variation.qa_notes.append(f"image qa ({stage}): {failure}")
    for warning in report.warnings:
        variation.qa_notes.append(f"image qa ({stage}) warning: {warning}")
    return report


def qa_record(record: Record, stage: str = "generated") -> QaSummary:
    """QA every generated variation of one record.

    A record is only marked IMAGES_QA_FAILED when *no* variation passed. One bad
    variation out of three is a variation to drop, not a record to abandon --
    the same reasoning as in imagegen: this record passed four gates to get here.
    """
    summary = QaSummary()
    for variation in record.variations:
        report = qa_variation(variation, stage)
        if report is None:
            summary.missing += 1
            continue
        summary.reports.append(report)
        if report.ok:
            summary.passed += 1
        else:
            summary.failed += 1

    if summary.failed and not summary.passed:
        record.pipeline_status = PipelineStatus.IMAGES_QA_FAILED
    return summary
