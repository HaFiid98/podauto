"""Design prompt synthesis.

Text strategy is MODEL_RENDERED: each variation emits one prompt carrying both
artwork and lettering, and the model draws the letters itself. Image models draw
letterforms rather than typeset them, so spelling is not guaranteed and no
verification stage is configured -- misspelled or malformed lettering reaches
the manual review queue. That tradeoff was chosen deliberately; the design here
keeps it reversible.

Reversibility hinges on one thing: StyleVariation.text_spec is populated with
the intended string even in model_rendered mode, where nothing reads it. Turning
on OCR verification or composed text later reads from that field instead of
requiring a schema change and a re-run of everything already generated.

Two corrections to the original spec, both load-bearing:

* No Midjourney "--no ..." suffix. Flux/SDXL endpoints have no negative-prompt
  parameter in their prompt string; best case those tokens are ignored, plausible
  case they are read as positive prompt text -- so "--no mockup, shadows" asks
  for a mockup with shadows.
* No "solid black background". It contradicted the transparent-PNG requirement.
  The model paints a ground whether asked to or not, so the prompt names a chroma
  ground the style's own palette excludes and the print stage keys it out. See
  `ground_for_style` for why naming it matters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dedupe import concept_tokens
from .models import IdeaType, PipelineStatus, Record, StyleVariation, TmStatus
from .scoring import variations_allowed

# The ground the model is asked to paint behind the art, so the print stage can
# key it out. Chroma colours, in preference order, each paired with the colour
# words that would make it collide with a style's own palette.
#
# This exists because "isolated on a plain uniform background" -- the previous
# wording -- does not forbid reusing a palette colour as the ground, and in 2 of
# the first 3 real generations the model did exactly that (a burnt-orange ground
# under near-ground-tone silhouettes; a navy ground under a navy land mass).
# Removing such a ground cuts into the design, because at that point the ground
# and the artwork are the same number. Naming a colour the palette excludes is
# the only prompt-level lever available at all.
#
# MEASURED 2026-08-17, and it does not work: 3 of 3 real generations on the HF
# routes painted their own ground instead of the requested one --
#   #FF00FF -> #D53536,  #00FF00 -> #000000,  #FF00FF -> #F40034
# -- and the #000000 case then collided with its own black outlines at 4.5% of the
# art's bounding box, which is exactly the failure this wording was added to
# prevent. What saved all three files was printready keying the border colour it
# actually finds, not this instruction. So:
#
#   * ground_hint is a DIAGNOSTIC, not a contract. printready compares it to the
#     detected border colour and warns when they differ; nothing depends on the
#     model having complied.
#   * review.py classifies that warning as informational for the same reason --
#     it fires on essentially every design, so gating approvals on it would make
#     --force routine.
#   * do not read ground_for_style as steering the model. It steers the warning.
#
# The list is kept because it costs nothing, occasionally lands, and the hint is
# what makes the ignored-ground defect visible instead of silent.
GROUND_CANDIDATES: list[tuple[str, tuple[str, ...]]] = [
    ("#FF00FF", ("magenta", "pink", "purple", "violet", "lilac", "fuchsia",
                 "rose", "plum", "mauve", "orchid")),
    ("#00FF00", ("green", "sage", "mint", "olive", "avocado", "lime",
                 "emerald", "teal", "forest")),
    ("#00FFFF", ("cyan", "teal", "turquoise", "aqua", "sky", "blue")),
    ("#FFFF00", ("yellow", "gold", "mustard", "cream", "wheat", "amber",
                 "tan", "beige", "sand", "ochre")),
]

GROUND_NAMES = {"#FF00FF": "magenta", "#00FF00": "green",
                "#00FFFF": "cyan", "#FFFF00": "yellow"}


def ground_for_style(style: dict[str, Any]) -> str:
    """Pick a chroma ground whose colour family the style's own palette avoids.

    Checked against the style's graphic and lettering text as well as its
    palette_hint, because the colour a model actually reaches for is named in the
    prose ("electric blue and magenta on black") at least as often as in the hint.

    If every candidate collides the first is returned anyway rather than raising.
    A style that wants all four chroma families is a style whose ground will
    collide no matter what is asked for, and printready measures that collision
    directly -- failing prompt synthesis over it would block a design that a
    human can still judge.
    """
    text = " ".join(str(style.get(k, "")) for k in
                    ("palette_hint", "graphic", "lettering")).lower()
    for colour, words in GROUND_CANDIDATES:
        if not any(word in text for word in words):
            return colour
    return GROUND_CANDIDATES[0][0]


def output_suffix(ground: str) -> str:
    """The tail of every prompt: medium, then the ground contract.

    Plain descriptive language only -- no CLI-style flags. The ground is stated
    twice on purpose, once as what to paint and once as what the artwork must not
    contain, because the second half is the part that makes it removable.
    """
    name = GROUND_NAMES.get(ground, "chroma")
    return (
        "flat vector t-shirt graphic, bold clean shapes, screen print aesthetic, "
        "limited colour palette, crisp edges, centred composition, "
        f"the subject sits on a completely flat solid {name} background "
        f"({ground}) that fills every corner, "
        f"no {name} anywhere in the artwork itself, "
        "no gradient or texture in the background, "
        "no garment, no fabric, no mockup"
    )


# Model text reliability drops sharply with length, so the synthesizer refuses
# to ask for long strings rather than generating art destined for the reject pile.
MAX_TEXT_WORDS = 5


class StyleDatabase:
    def __init__(self, data: dict[str, Any]):
        self.meta = data.get("_meta", {})
        self.styles = {s["id"]: s for s in data.get("styles", [])}

    @classmethod
    def load(cls, path: str | Path) -> StyleDatabase:
        return cls(json.loads(Path(path).read_text()))

    def get(self, style_id: str) -> dict[str, Any]:
        if style_id not in self.styles:
            raise KeyError(f"unknown style: {style_id!r}")
        return self.styles[style_id]

    def match_for_niche(self, niche: str, exclude: set[str] | None = None) -> str:
        """Pick the style whose good_for tags best overlap the niche.

        Used for the ready_shirt case, where variation 1 should echo the
        original design's aesthetic rather than impose an unrelated style.
        """
        exclude = exclude or set()
        tokens = set(concept_tokens(niche))
        best, best_score = None, -1
        for sid, style in self.styles.items():
            if sid in exclude:
                continue
            score = affinity(tokens, style)
            if score > best_score:
                best, best_score = sid, score
        if best is None:
            raise ValueError("style database is empty")
        return best


def affinity(tokens: set[str], style: dict[str, Any]) -> int:
    """How well a style's good_for tags fit a set of concept tokens.

    Prefix-tolerant in both directions, because exact equality missed the
    obvious cases: "gardening" never matched the "garden" tag, so Gardening
    Grandma drew a distressed collegiate athletic look while
    cottagecore_botanical -- the style written for precisely that niche --
    scored zero.

    The 4-character floor keeps this conservative. Without it, three-letter
    tokens prefix-match far too much ("cat" into "category"); with it, only
    substantial stems count, which catches gardening/garden and grandmas/grandma
    without taking on a stemmer dependency.
    """
    score = 0
    for raw_tag in style.get("good_for", []):
        tag = raw_tag.lower()
        for token in tokens:
            if (token == tag
                    or (len(tag) >= 4 and token.startswith(tag))
                    or (len(token) >= 4 and tag.startswith(token))):
                score += 1
                break
    return score


@dataclass
class TextSpec:
    """The intended lettering.

    Populated even in model_rendered mode, where no code reads `line_1` back.
    That is deliberate: it is the hook that makes switching strategies later a
    config change rather than a migration.
    """

    line_1: str
    arrangement: str = "stacked"
    case: str = "upper"
    position: str = "center_overlay"
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_1": self.line_1,
            "arrangement": self.arrangement,
            "case": self.case,
            "position": self.position,
            "truncated": self.truncated,
            "strategy": "model_rendered",
        }


def derive_text(record: Record) -> TextSpec:
    """Pull the shirt's lettering from the concept, capped at MAX_TEXT_WORDS."""
    words = record.raw_title_or_text.split()
    truncated = len(words) > MAX_TEXT_WORDS
    return TextSpec(line_1=" ".join(words[:MAX_TEXT_WORDS]), truncated=truncated)


def _apply_case(text: str, case: str) -> str:
    if case == "upper":
        return text.upper()
    if case == "title":
        return text.title()
    if case == "lower":
        return text.lower()
    return text


def build_prompt(style: dict[str, Any], record: Record, text: TextSpec,
                 ground: str | None = None) -> str:
    """One prompt carrying artwork, lettering and the removable ground together."""
    layout = style.get("text_layout", {})
    rendered = _apply_case(text.line_1, layout.get("case", text.case))
    ground = ground or ground_for_style(style)

    parts = [
        f"{record.niche} t-shirt design",
        style["graphic"],
    ]
    if style.get("lettering"):
        parts.append(style["lettering"])
    # Quoting the string and stating it plainly is the one prompt-level lever
    # that reliably helps text accuracy.
    parts.append(f'the design includes the text "{rendered}", spelled exactly as written')
    if style.get("palette_hint"):
        parts.append(f"palette: {style['palette_hint']}")
    parts.append(output_suffix(ground))

    return ", ".join(parts)


def select_styles(record: Record, db: StyleDatabase, count: int,
                  selected: list[str] | None = None) -> list[str]:
    """ready_shirt: aesthetic match first, then fill. trend_idea: fill only.

    Two behaviours beyond the original spec, both learned from the first
    end-to-end run rather than reasoned about in advance:

    * Styles carrying a `caution` are dropped for records triage has already
      flagged. distressed_collegiate's own caution says never to pair it with a
      "Property of" construction -- and the first smoke run handed it exactly
      that record, as the lead style. A warning in config that no code reads is
      a comment, not a control.

    * The fill rotates per record instead of always starting at the top of the
      pool. Without it every record drew the same opening styles and a 12-style
      library collapsed to 3 -- visibly repetitive across a storefront, and it
      wastes the scaffold.

    An explicit `selected` list is an instruction from the operator, so it is
    honoured verbatim: no rotation, no caution filtering.
    """
    explicit = selected is not None
    pool = list(selected) if explicit else list(db.styles.keys())

    banned: set[str] = set()
    if not explicit and record.tm_status is TmStatus.NEEDS_REVIEW:
        banned = {sid for sid, s in db.styles.items() if s.get("caution")}
        pool = [sid for sid in pool if sid not in banned]

    chosen: list[str] = []
    if record.idea_type is IdeaType.READY_SHIRT and count > 0:
        chosen.append(db.match_for_niche(record.niche, exclude=banned))

    # Keyed on dedupe_key so the same record always draws the same styles --
    # the pipeline has to stay idempotent across re-runs. Deliberately not
    # hash(): that is salted per process, so it would reshuffle every run.
    if not explicit and pool:
        seed = record.dedupe_key or record.niche or record.raw_title_or_text
        offset = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % len(pool)
        pool = pool[offset:] + pool[:offset]

    for sid in pool:
        if len(chosen) >= count:
            break
        if sid not in chosen:
            chosen.append(sid)

    return chosen[:count]


def synthesize(record: Record, db: StyleDatabase, configured_variations: int,
               selected_styles: list[str] | None = None) -> Record:
    """Populate record.variations. Count comes from the scorer, not a fixed X.

    An under-evidenced pass gets one variation instead of the configured X, so
    image quota follows the evidence.
    """
    count = variations_allowed(record, configured_variations)
    if count == 0:
        record.variations = []
        return record

    text = derive_text(record)
    record.variations = []

    for style_id in select_styles(record, db, count, selected_styles):
        style = db.get(style_id)
        layout = dict(style.get("text_layout", {}))
        spec = TextSpec(
            line_1=text.line_1,
            arrangement=layout.get("arrangement", text.arrangement),
            case=layout.get("case", text.case),
            position=layout.get("position", text.position),
            truncated=text.truncated,
        )
        ground = ground_for_style(style)
        record.variations.append(StyleVariation(
            style_id=style_id,
            style_name=style["name"],
            graphic_prompt=build_prompt(style, record, spec, ground),
            text_spec=spec.to_dict(),
            ground_hint=ground,
        ))

    record.pipeline_status = PipelineStatus.PROMPTED
    return record
