"""Per-stage CLI.

n8n calls these commands; it does not hold business logic. Debugging logic
inside n8n nodes is miserable and untestable, so the orchestrator stays a
scheduler and HTTP glue while the decisions live in tested Python.

Stages run cost-ascending, so every filter runs before anything pricier:

    ingest -> score -> triage -> policy -> prompts -> listing -> images
           -> imageqa -> printready -> imageqa --stage print_ready -> review
           -> approve/reject -> export

`run` stops at listing. Everything up to there is free arithmetic and text, so
it is safe to re-run; `images` spends provider quota and is invoked deliberately.
`printready` is local and free again, so it sits outside `run` only because it
has nothing to work on until `images` has run.

`approve`/`reject` are the only stages a human drives, and `export` is the only
one that reads the store without writing it.

Each command is independently re-runnable and reads/writes the same store, so
an interrupted run resumes rather than restarts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .content_policy import ContentPolicy, check_record
from .env import load_env_file
from .imagegen import generate_for_record
from .imageqa import qa_record
from .ingest import ingest, load_rows
from .listing import generate as generate_listing
from .models import ContentPolicyStatus, Gate, PipelineStatus, TmStatus
from .printready import GROUND_TOLERANCE, printready_record
from .prompts import StyleDatabase, synthesize
from .providers import Rotator
from .review import ReviewError, decide, export, resolve
from .scoring import score
from .store import Store
from .triage import Denylist, triage_record

DEFAULT_STORE = "data/records.jsonl"
DEFAULT_STYLES = "config/styles.json"
DEFAULT_DENYLIST = "config/denylist.json"
DEFAULT_POLICY = "config/content_policy.json"
DEFAULT_PROVIDERS = "config/providers.json"
DEFAULT_LEDGER = "data/usage_ledger.json"
DEFAULT_IMAGE_DIR = "data/images"
DEFAULT_EXPORT_DIR = "data/export"

# Generation is slow -- a diffusion endpoint under load takes tens of seconds --
# and httpx defaults to 5s, which would turn every busy provider into a timeout
# the rotator reads as a dead key.
IMAGE_TIMEOUT_S = 180.0


def cmd_ingest(args) -> int:
    store = Store(args.store)
    rows = load_rows(args.input)
    result = ingest(rows, known_keys=store.existing_keys())
    store.upsert_many(result.accepted)

    print(f"ingest: {result.summary()}")
    for _, text in result.duplicates:
        print(f"  duplicate: {text[:60]!r}")
    for row_num, why in result.invalid:
        print(f"  invalid row {row_num}: {why}", file=sys.stderr)
    return 0


def cmd_score(args) -> int:
    store = Store(args.store)
    records = store.all()
    touched = []

    for rec in records:
        if rec.pipeline_status is not PipelineStatus.DEDUPED and not args.force:
            continue
        score(rec)
        rec.pipeline_status = {
            Gate.PASS: PipelineStatus.SCORED,
            Gate.HOLD: PipelineStatus.GATE_HELD,
            Gate.REJECT: PipelineStatus.GATE_REJECTED,
        }[rec.gate]
        touched.append(rec)

    store.upsert_many(touched)

    passed = sum(1 for r in touched if r.gate is Gate.PASS)
    held = sum(1 for r in touched if r.gate is Gate.HOLD)
    rejected = sum(1 for r in touched if r.gate is Gate.REJECT)
    print(f"score: scored={len(touched)} pass={passed} hold={held} reject={rejected}")

    for rec in touched:
        missing = rec.score_components.get("missing", [])
        flag = f"  [{rec.score_confidence.value}]" if missing else ""
        print(f"  {rec.hitter_score:>5} {rec.gate.value:<7} {rec.niche[:32]:<32}{flag}")
        for note in missing:
            print(f"        missing: {note}")
    return 0


def cmd_triage(args) -> int:
    store = Store(args.store)
    denylist = Denylist.load(args.denylist)

    if denylist.is_stale():
        days = denylist.staleness_days()
        print(f"WARNING: denylist last reviewed {days}d ago, past its refresh "
              f"cadence. New IP will be missed.", file=sys.stderr)

    touched = []
    for rec in store.all():
        if rec.pipeline_status is not PipelineStatus.SCORED and not args.force:
            continue
        triage_record(rec, denylist)
        if rec.tm_status is TmStatus.BLOCKED:
            rec.pipeline_status = PipelineStatus.GATE_REJECTED
        else:
            rec.pipeline_status = PipelineStatus.TRIAGED
        touched.append(rec)

    store.upsert_many(touched)

    blocked = sum(1 for r in touched if r.tm_status is TmStatus.BLOCKED)
    review = sum(1 for r in touched if r.tm_status is TmStatus.NEEDS_REVIEW)
    clean = sum(1 for r in touched if r.tm_status is TmStatus.NO_FLAGS_FOUND)
    print(f"triage: checked={len(touched)} blocked={blocked} "
          f"needs_review={review} no_flags_found={clean}")
    print("  (no_flags_found is not clearance -- a keyword screen cannot clear a mark)")

    for rec in touched:
        if rec.tm_status is not TmStatus.NO_FLAGS_FOUND:
            print(f"  {rec.tm_status.value:<14} {rec.niche[:28]:<28} {rec.tm_flag_reason}")
    return 0


def cmd_policy(args) -> int:
    store = Store(args.store)
    policy = ContentPolicy.load(args.policy)

    touched = []
    for rec in store.all():
        if rec.pipeline_status is not PipelineStatus.TRIAGED and not args.force:
            continue
        check_record(rec, policy)
        if rec.content_policy_status is ContentPolicyStatus.FAIL:
            rec.pipeline_status = PipelineStatus.GATE_REJECTED
        touched.append(rec)

    store.upsert_many(touched)

    failed = sum(1 for r in touched if r.content_policy_status is ContentPolicyStatus.FAIL)
    print(f"policy: checked={len(touched)} pass={len(touched) - failed} fail={failed}")
    for rec in touched:
        if rec.content_policy_status is ContentPolicyStatus.FAIL:
            print(f"  FAIL {rec.niche[:28]:<28} {rec.content_policy_reason}")
    return 0


def cmd_prompts(args) -> int:
    store = Store(args.store)
    db = StyleDatabase.load(args.styles)

    selected = args.styles_list.split(",") if args.styles_list else None
    touched = []
    for rec in store.all():
        if rec.pipeline_status is not PipelineStatus.TRIAGED and not args.force:
            continue
        if rec.content_policy_status is ContentPolicyStatus.FAIL:
            continue
        synthesize(rec, db, args.variations, selected)
        touched.append(rec)

    store.upsert_many(touched)

    total = sum(len(r.variations) for r in touched)
    print(f"prompts: records={len(touched)} variations={total}")
    print("  note: text is model-rendered and unverified -- spelling errors "
          "will reach the review queue")
    for rec in touched:
        for var in rec.variations:
            print(f"  {rec.niche[:24]:<24} {var.style_name}")
    return 0


def cmd_listing(args) -> int:
    """Generate listing fields, then re-screen the text that was just created.

    Triage ran on concept text only, because at that point this text did not
    exist yet -- so the generated brand and title had never been screened. That
    is what the plan means by "brand, run through triage before use".

    Overwriting tm_status here cannot downgrade an earlier finding: triage_record
    always includes the concept text in what it checks, and every tier is
    monotonic in its input (more text yields the same flags or more), so a
    re-screen on a superset can only hold or escalate.
    """
    store = Store(args.store)
    denylist = Denylist.load(args.denylist)
    touched, all_issues, blocked = [], 0, 0

    for rec in store.all():
        if rec.pipeline_status is not PipelineStatus.PROMPTED and not args.force:
            continue
        _, issues = generate_listing(rec)
        if issues:
            all_issues += len(issues)
            for issue in issues:
                print(f"  issue {rec.niche[:24]:<24} {issue}", file=sys.stderr)

        triage_record(rec, denylist)
        if rec.tm_status is TmStatus.BLOCKED:
            blocked += 1
            rec.pipeline_status = PipelineStatus.GATE_REJECTED
            print(f"  BLOCKED on re-screen {rec.niche[:24]:<24} "
                  f"{rec.tm_flag_reason}", file=sys.stderr)
        else:
            rec.pipeline_status = PipelineStatus.AWAITING_REVIEW
        touched.append(rec)

    store.upsert_many(touched)
    print(f"listing: generated={len(touched)} issues={all_issues} "
          f"blocked_on_rescreen={blocked}")
    return 0


def cmd_images(args) -> int:
    """Generate artwork for records that already have listings.

    Runs after `listing`, not before it, because listing is free text and every
    image costs quota -- the cost-ascending rule the rest of the pipeline follows.
    Kept out of `run` for the same reason: `run` should be safe to re-run, and
    this command spends money.
    """
    import httpx                       # local: only this command needs the client

    store = Store(args.store)
    rotator = Rotator.from_files(args.providers, args.ledger)

    queue = [r for r in store.all()
             if r.pipeline_status is PipelineStatus.AWAITING_REVIEW
             or (args.force and r.variations)]
    if args.limit:
        queue = queue[: args.limit]

    if not queue:
        print("images: nothing to generate")
        for name, why in rotator.skip_reasons().items():
            print(f"  skip {name}: {why}")
        return 0

    print(f"images: {len(queue)} records, "
          f"{sum(len(r.variations) for r in queue)} variations")
    for name, why in rotator.skip_reasons().items():
        print(f"  skip {name}: {why}")

    touched, totals = [], {"generated": 0, "skipped": 0, "failed": 0}
    with httpx.Client(timeout=IMAGE_TIMEOUT_S, follow_redirects=True) as client:
        for rec in queue:
            before = rec.pipeline_status
            summary = generate_for_record(rec, rotator, client,
                                          out_dir=args.out_dir, force=args.force)
            # Images enrich a record that is already queued for review; they do
            # not move it. Leaving it at IMAGES_GENERATED would drop it out of
            # the review queue, and that manual check is the pipeline's safety
            # property -- it must not be skippable by a side effect.
            if before is PipelineStatus.AWAITING_REVIEW:
                rec.pipeline_status = before

            for field_name in totals:
                totals[field_name] += getattr(summary, field_name)
            for err in summary.errors:
                print(f"  fail {err}", file=sys.stderr)
            print(f"  {rec.niche[:28]:<28} generated={summary.generated} "
                  f"skipped={summary.skipped} failed={summary.failed}")
            touched.append(rec)

    store.upsert_many(touched)
    print(f"images: generated={totals['generated']} skipped={totals['skipped']} "
          f"failed={totals['failed']}")
    return 0


def cmd_printready(args) -> int:
    """Strip the generated ground and write the 4500x5400 transparent PNG.

    Free and local -- no provider quota -- so unlike `images` this is safe to
    re-run, and it is idempotent: a variation whose print file already exists is
    skipped unless --force.

    It does not advance `pipeline_status`. The record stays in the review queue,
    for the same reason `images` leaves it there: producing a file is not a
    review, and the manual check is the pipeline's safety property.
    """
    store = Store(args.store)
    touched, totals = [], {"produced": 0, "skipped": 0, "failed": 0}

    for rec in store.all():
        if not any(v.image_path for v in rec.variations):
            continue
        before = rec.pipeline_status
        summary = printready_record(rec, out_dir=args.out_dir or None,
                                    method=args.method, force=args.force,
                                    tolerance=args.tolerance)
        rec.pipeline_status = before
        for field_name in totals:
            totals[field_name] += getattr(summary, field_name)
        for result in summary.results:
            print(f"  {Path(result.path).name}: {result.summary()}")
            print(f"      {result.facts}")
        touched.append(rec)

    store.upsert_many(touched)
    if not touched:
        print("printready: no generated images to convert")
        return 0

    print(f"printready: produced={totals['produced']} skipped={totals['skipped']} "
          f"failed={totals['failed']}")
    print("  next: podauto imageqa --stage print_ready")
    return 0


def cmd_imageqa(args) -> int:
    """Check generated files. No text verification -- lettering is model-rendered
    by decision, so the review queue prints the intended string instead."""
    store = Store(args.store)
    touched, totals = [], {"passed": 0, "failed": 0, "missing": 0}
    # Each stage looks at its own file, so the "is there anything to check here"
    # filter has to look at the same one qa_variation will read. Filtering on
    # image_path for the print gate would pull in records that have no print
    # file yet and report them all as failures.
    has_file = ((lambda v: v.print_path) if args.stage == "print_ready"
                else (lambda v: v.image_path))

    for rec in store.all():
        if not any(has_file(v) for v in rec.variations):
            continue
        summary = qa_record(rec, stage=args.stage)
        for field_name in totals:
            totals[field_name] += getattr(summary, field_name)
        for report in summary.reports:
            if not report.ok or report.warnings:
                print(f"  {Path(report.path).name}: {report.summary()}")
        touched.append(rec)

    store.upsert_many(touched)
    if not touched:
        print("imageqa: no generated images to check")
        return 0

    failed_records = [r for r in touched
                      if r.pipeline_status is PipelineStatus.IMAGES_QA_FAILED]
    print(f"imageqa ({args.stage}): pass={totals['passed']} "
          f"fail={totals['failed']} no_image={totals['missing']}")
    for rec in failed_records:
        print(f"  every variation failed: {rec.niche} -- regenerate, do not review")
    return 0


def cmd_review(args) -> int:
    """Human review queue. The manual check is the pipeline's safety property,
    so this prints everything a ten-second decision needs and nothing else."""
    store = Store(args.store)
    queue = [r for r in store.all()
             if r.pipeline_status is PipelineStatus.AWAITING_REVIEW]

    if not queue:
        print("review queue: empty")
        return 0

    print(f"review queue: {len(queue)} records\n")
    for rec in queue:
        print(f"--- {rec.id[:8]}  {rec.niche}")
        print(f"    score      {rec.hitter_score} ({rec.score_confidence.value})")
        print(f"    tm         {rec.tm_status.value}"
              + (f"  <- {rec.tm_flag_reason}" if rec.tm_flag_reason else ""))
        print(f"    policy     {rec.content_policy_status.value}"
              + (f"  <- {rec.content_policy_reason}" if rec.content_policy_reason else ""))
        print(f"    title      {rec.listing.title} ({len(rec.listing.title)} chars)")
        print(f"    brand      {rec.listing.brand}")
        for i, var in enumerate(rec.variations):
            text = var.text_spec.get("line_1", "")
            decided = f"  [{var.review_decision}]" if var.review_decision else ""
            print(f"    [{i}] {var.style_name} -- text: {text!r}{decided}")
            if var.image_path:
                print(f"        {var.image_path}")
            if var.print_path:
                print(f"        {var.print_path}  <- upload this one")
            for note in var.qa_notes:
                print(f"        ! {note}")
        print(f"    source     {rec.source_url}")
        print()

    print("Check generated lettering against the text shown above -- it is "
          "model-rendered and not verified.")
    print(f"Then: podauto approve {queue[0].id[:8]} --variation 0"
          f"   /   podauto reject {queue[0].id[:8]} --variation 1 --reason ...")
    return 0


def cmd_approve(args) -> int:
    """Record a human approval on chosen variations.

    Per variation, not per record: a record holds several designs sharing one
    title, and the lettering is model-rendered with no verification stage, so
    approving a whole record would ship the misspelled designs with the good one.
    """
    return _decide(args, approve=True)


def cmd_reject(args) -> int:
    """Record a human rejection, with the reason that makes it useful.

    --reason is required. It is the only feedback that ever reaches the styles
    and prompts, and it is unrecoverable once the reviewer has moved on.
    """
    return _decide(args, approve=False)


def _decide(args, approve: bool) -> int:
    store = Store(args.store)
    records = store.all()
    try:
        rec = resolve(records, args.record)
        summary = decide(rec, approve=approve, indices=args.variation or None,
                         reason=getattr(args, "reason", "") or getattr(args, "note", ""),
                         force=args.force)
    except ReviewError as exc:
        print(f"{'approve' if approve else 'reject'}: {exc}", file=sys.stderr)
        return 1

    for d in summary.decisions:
        # Informational notes print above the verdict, blocking ones below it --
        # a forced approval has to show what it overrode, not hide it.
        for note in d.informational:
            print(f"  note [{d.index}] {note}")
        mark = "ok  " if d.applied else "skip"
        print(f"  {mark} [{d.index}] {d.style_name}: {d.detail}")
        for note in d.blocking:
            print(f"       ! {note}")

    if not any(d.applied for d in summary.decisions):
        print("nothing recorded -- the store is unchanged")
        return 1

    store.upsert(rec)
    print(f"{rec.id[:8]} {rec.niche} -> {summary.status.value} "
          f"({summary.summary()})")
    if summary.undecided:
        # Deliberate: a half-reviewed record stays in the queue. Leaving it
        # silently would make the manual check skippable by getting bored.
        print("  still in the review queue until every variation is decided")
    elif summary.status is PipelineStatus.APPROVED:
        print("  next: podauto export")
    return 0


def cmd_export(args) -> int:
    """Write listings.csv for every approved variation.

    Reads the store and never writes it. Writing a CSV is not an upload to
    Amazon, so this does not set UPLOADED -- the same rule `images` and
    `printready` follow, where producing a file is not a stage transition.
    """
    store = Store(args.store)
    records = store.all()
    approved = [r for r in records if r.pipeline_status is PipelineStatus.APPROVED]

    summary = export(records, out_dir=args.out_dir, copy_files=args.copy_files)

    for reason in summary.skipped:
        print(f"  skip {reason}", file=sys.stderr)

    if not summary.rows:
        pending = sum(1 for r in records
                      if r.pipeline_status is PipelineStatus.AWAITING_REVIEW)
        print(f"export: nothing approved yet ({len(approved)} approved records, "
              f"{pending} awaiting review)")
        print("  approve something first: podauto review")
        return 0

    print(f"export: {summary.rows} listings from {summary.records} records "
          f"-> {summary.path}")
    if args.copy_files:
        print(f"  {summary.copied} print file(s) copied to "
              f"{Path(args.out_dir) / 'files'}")
    print("  each row is one Merch product; check the lettering in the print "
          "file against intended_text before uploading")
    return 0


def cmd_status(args) -> int:
    store = Store(args.store)
    counts: dict[str, int] = {}
    for rec in store.all():
        counts[rec.pipeline_status.value] = counts.get(rec.pipeline_status.value, 0) + 1
    if not counts:
        print("store is empty")
        return 0
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {status}")
    return 0


def cmd_run(args) -> int:
    """The free stages, in cost-ascending order.

    Stops before `images` on purpose: everything here is arithmetic and text, so
    re-running it costs nothing, while image generation spends quota that does not
    come back. Run `podauto images` when you mean to spend it.
    """
    for fn in (cmd_ingest, cmd_score, cmd_triage, cmd_policy, cmd_prompts, cmd_listing):
        rc = fn(args)
        if rc != 0:
            return rc
        print()
    print("next: podauto images   (spends provider quota)\n")
    return cmd_status(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="podauto", description=__doc__)
    p.add_argument("--store", default=DEFAULT_STORE)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text):
        sp = sub.add_parser(name, help=help_text)
        sp.set_defaults(func=fn)
        sp.add_argument("--force", action="store_true",
                        help="re-run on records already past this stage")
        return sp

    sp = add("ingest", cmd_ingest, "parse rows into the store, applying dedupe")
    sp.add_argument("input", help="CSV or JSON file")

    add("score", cmd_score, "score and gate")

    sp = add("triage", cmd_triage, "trademark keyword screen (never clears)")
    sp.add_argument("--denylist", default=DEFAULT_DENYLIST)

    sp = add("policy", cmd_policy, "Amazon content policy gate")
    sp.add_argument("--policy", default=DEFAULT_POLICY)

    sp = add("prompts", cmd_prompts, "synthesize design prompts")
    sp.add_argument("--styles", default=DEFAULT_STYLES)
    sp.add_argument("--variations", type=int, default=3)
    sp.add_argument("--styles-list", default="", help="comma-separated style ids")

    sp = add("listing", cmd_listing, "generate Amazon listing fields")
    sp.add_argument("--denylist", default=DEFAULT_DENYLIST)

    sp = add("images", cmd_images, "generate artwork (spends provider quota)")
    sp.add_argument("--providers", default=DEFAULT_PROVIDERS)
    sp.add_argument("--ledger", default=DEFAULT_LEDGER)
    sp.add_argument("--out-dir", default=DEFAULT_IMAGE_DIR)
    sp.add_argument("--limit", type=int, default=0,
                    help="stop after N records; 0 means all")

    sp = add("imageqa", cmd_imageqa, "check generated files (no text check)")
    sp.add_argument("--stage", choices=("generated", "print_ready"),
                    default="generated",
                    help="'generated' checks native output; 'print_ready' "
                         "checks the 4500x5400 transparent PNG")

    sp = add("printready", cmd_printready,
             "remove the ground and write the 4500x5400 transparent PNG")
    sp.add_argument("--out-dir", default="",
                    help="where print files go; default is beside the generated image")
    sp.add_argument("--method", choices=("auto", "vector", "raster"), default="auto",
                    help="'auto' traces to vector if vtracer/resvg are installed "
                         "and falls back to LANCZOS; 'raster' never traces")
    sp.add_argument("--tolerance", type=int, default=GROUND_TOLERANCE,
                    help="per-channel distance from the ground colour still "
                         "counted as ground")

    add("review", cmd_review, "print the human review queue")

    # Decisions are per variation. --variation is repeatable; omitting it decides
    # every design on the record. The id is a prefix because `review` prints
    # rec.id[:8], so that is what the reviewer has in front of them.
    def add_decision(name, fn, help_text):
        sp = add(name, fn, help_text)
        sp.add_argument("record", help="record id or unambiguous prefix")
        sp.add_argument("--variation", type=int, action="append", default=[],
                        metavar="N",
                        help="variation index, repeatable; default is all")
        return sp

    sp = add_decision("approve", cmd_approve, "approve variations for upload")
    sp.add_argument("--note", default="", help="optional reviewer note")

    sp = add_decision("reject", cmd_reject, "reject variations, with a reason")
    sp.add_argument("--reason", default="",
                    help="required -- the only feedback that reaches the prompts")

    sp = add("export", cmd_export, "write listings.csv for approved variations")
    sp.add_argument("--out-dir", default=DEFAULT_EXPORT_DIR)
    sp.add_argument("--copy-files", action="store_true",
                    help="gather the print PNGs into <out-dir>/files/")

    add("status", cmd_status, "record counts by pipeline stage")

    sp = add("run", cmd_run, "all stages in order")
    sp.add_argument("input", help="CSV or JSON file")
    sp.add_argument("--denylist", default=DEFAULT_DENYLIST)
    sp.add_argument("--policy", default=DEFAULT_POLICY)
    sp.add_argument("--styles", default=DEFAULT_STYLES)
    sp.add_argument("--variations", type=int, default=3)
    sp.add_argument("--styles-list", default="")

    return p


def main(argv: list[str] | None = None) -> int:
    # Before anything else: providers.json names the variables holding the image
    # keys but cannot hold the values, so .env has to reach the environment or
    # every provider skips with "no key value in the environment".
    load_env_file()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
