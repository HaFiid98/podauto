"""Core record schema for the POD pipeline.

The Record is the spine: every stage reads it, mutates its own fields, and
advances `pipeline_status`. Nothing else holds state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class SourcePlatform(str, Enum):
    AMAZON = "amazon"
    REDBUBBLE = "redbubble"
    REDDIT = "reddit"
    TIKTOK = "tiktok"
    GOOGLE_TRENDS = "google_trends"


class IdeaType(str, Enum):
    READY_SHIRT = "ready_shirt"
    TREND_IDEA = "trend_idea"


class TmStatus(str, Enum):
    """Trademark triage outcome.

    There is deliberately no CLEARED value. A keyword screen cannot clear a
    mark -- it can only fail to find one. NO_FLAGS_FOUND says exactly that.
    """

    UNCHECKED = "unchecked"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"
    NO_FLAGS_FOUND = "no_flags_found"


class ContentPolicyStatus(str, Enum):
    """Amazon content-policy gate.

    Unlike trademark, this checks our own generated text against published
    rules, so PASS is a claim we can actually stand behind.
    """

    UNCHECKED = "unchecked"
    PASS = "pass"
    FAIL = "fail"


class PipelineStatus(str, Enum):
    """Where a record is in the pipeline.

    There is deliberately no PRINT_READY value. Producing the 4500x5400 file
    enriches a record that is already AWAITING_REVIEW, exactly as image
    generation does, and a status that advanced past AWAITING_REVIEW would drop
    the record out of the human review queue as a side effect of writing a file.
    That review is the pipeline's load-bearing safety property -- lettering is
    model-rendered and verified nowhere else. Presence of
    StyleVariation.print_path is the marker for "print file exists"; a print file
    that fails QA reuses IMAGES_QA_FAILED, which already means "regenerate, do
    not review".
    """

    INGESTED = "ingested"
    DEDUPED = "deduped"
    SCORED = "scored"
    GATE_HELD = "gate_held"
    GATE_REJECTED = "gate_rejected"
    TRIAGED = "triaged"
    PROMPTED = "prompted"
    IMAGES_GENERATED = "images_generated"
    IMAGES_QA_FAILED = "images_qa_failed"
    LISTED = "listed"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    REJECTED_BY_HUMAN = "rejected_by_human"
    UPLOADED = "uploaded"
    ERROR = "error"


class ScoreConfidence(str, Enum):
    HIGH = "high"
    PARTIAL = "partial"
    LOW = "low"


class Gate(str, Enum):
    PASS = "pass"
    HOLD = "hold"
    REJECT = "reject"


@dataclass
class StyleVariation:
    style_id: str
    style_name: str
    graphic_prompt: str
    text_spec: dict[str, Any] = field(default_factory=dict)
    # The ground colour the prompt asked the model to paint behind the art, as
    # #RRGGBB. Removal keys on what the border actually is, not on this -- the
    # hint exists so the print stage can report that the model ignored it, which
    # is a prompt defect and needs to be visible as one.
    ground_hint: str | None = None
    image_path: str | None = None
    image_provider: str | None = None
    # The 4500x5400 transparent PNG. A separate field, not an overwrite of
    # image_path: when print QA rejects a file the generated original is the only
    # evidence for whether the prompt, the provider or the removal is at fault.
    print_path: str | None = None
    qa_notes: list[str] = field(default_factory=list)
    # "" until a human decides, then APPROVED_MARK or the rejection reason. Free
    # text rather than an enum because for a rejection the reason IS the value --
    # it is the only feedback that ever makes it back to the styles and prompts,
    # and an enum would discard it. Decisions are per variation, not per record:
    # one record holds several designs sharing a title, and the measured rate is
    # a third to two thirds usable, so a record-level verdict would ship the
    # misspelled ones alongside the good one.
    review_decision: str = ""


@dataclass
class Listing:
    title: str = ""
    brand: str = ""
    bullet_1: str = ""
    bullet_2: str = ""
    description: str = ""


@dataclass
class Record:
    id: str
    timestamp: str
    source_url: str
    source_platform: SourcePlatform
    idea_type: IdeaType
    niche: str
    raw_title_or_text: str

    dedupe_key: str = ""

    # Demand signals. bsr_category and bsr_captured_at are required for the
    # BSR bands to mean anything -- rank is category-relative and decays.
    bsr_rank: int | None = None
    bsr_category: str | None = None
    bsr_captured_at: str | None = None

    # External signals, absent unless a collector supplies them.
    trend_velocity: float | None = None
    listings_count: int | None = None

    # Scoring
    hitter_score: float | None = None
    score_confidence: ScoreConfidence | None = None
    score_components: dict[str, Any] = field(default_factory=dict)
    gate: Gate | None = None

    # Compliance
    tm_status: TmStatus = TmStatus.UNCHECKED
    tm_flag_reason: str = ""
    matched_terms: list[str] = field(default_factory=list)
    content_policy_status: ContentPolicyStatus = ContentPolicyStatus.UNCHECKED
    content_policy_reason: str = ""

    variations: list[StyleVariation] = field(default_factory=list)
    listing: Listing = field(default_factory=Listing)

    pipeline_status: PipelineStatus = PipelineStatus.INGESTED
    error_log: str = ""

    review_decision: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
