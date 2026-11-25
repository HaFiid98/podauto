Project analysis, 2026-11-22

The pipeline works through the covered scenarios, but approval, artifact changes,
and reruns have gaps that can allow unusable or blocked records to reach export.
The next implementation pass should address those gaps before adding automation.

This pass reviewed the CLI stage transitions, record store, scoring, policy and
trademark checks, prompt synthesis, provider rotation, image generation, image QA,
print conversion, and review/export. Application code was not changed. Existing
workspace changes were preserved.

The project is a Python 3.10+ CLI for Amazon Merch print-on-demand listings.
Its required runtime dependencies are HTTPX and Pillow. Its free `run` command
performs ingestion through listing generation. Image generation is a separate
command that consumes provider quota. Print conversion and QA run locally;
approval is per variation; export produces a CSV and optionally copies files.

| Area | Current implementation |
| --- | --- |
| Input | CSV or JSON imports; no collectors committed in this checkout |
| Filtering | Stable dedupe keys, confidence-aware scores, trademark keyword triage, configured content-policy rules |
| Design | Deterministic style selection, model-rendered lettering, retained intended text |
| Images | Provider/key fallback, quota ledger, native image downloads |
| Print | Background removal, raster upscale, optional vector conversion, 4500×5400 transparent PNG |
| Review | Per-variation human decisions, record status derived from decisions |
| Output | Listings CSV and optional file copies; no Amazon upload implementation |
| Persistence | JSONL records and JSON quota ledger; no transactional database |

Validation: `/tmp/podauto-analysis-venv/bin/python -m pytest -q -p no:cacheprovider`
returned **306 passed, 1 skipped** in 9.17 seconds. The skipped test requires the
operator-local `config/providers.json`, which is absent. Dependencies were
installed into a temporary virtual environment. Six additional local scenarios
reproduced the findings below using temporary files. No live image-generation
requests or uploads were made. Passing tests cover existing behavior and do not
resolve the uncovered scenarios.

1. **High priority: a credential is hardcoded in launcher scripts.**
   `setup.sh:4` and `verf.sh:4` contain the same API key literal. Both files were
   already untracked when this analysis started. The credential's validity was
   not tested. Remove the literal, read the key from the environment, and rotate
   the credential with its provider. Do not include its value in reports or
   commits. These scripts configure development tools, rather than the POD
   pipeline itself.

2. **High priority: approval bypasses record-level gates.**
   `podauto/review.py:194` checks variation decisions, the presence of a print
   path, and QA notes, but does not check pipeline status, trademark status,
   content-policy status, or score gate. `roll_up` can then replace a rejection
   status with `approved`; export checks that status and the variation marker.
   Reproduction: a record with a valid print file, `gate_rejected`, policy `fail`,
   and trademark `blocked` was approved without `--force` and yielded one export
   row. Such records can retain variations after a forced policy/triage rerun.
   Define approval eligibility explicitly and enforce it again at export.

3. **High priority: generated listings miss the content-policy check.**
   `podauto/cli.py:192` generates listing fields and repeats trademark triage,
   but does not repeat `check_record` from the content-policy module. The policy
   stage runs before those listing fields exist. Reproduction: importing the
   concept `Buy now cat` through `run` returned success and reached
   `awaiting_review` with stored policy `pass`. Checking the completed record
   returned `fail` because its title and first bullet contained `buy now`, which
   the configured listing rules prohibit. Check all generated fields before
   queueing images or review, using the policy configuration selected by the
   operator.

4. **High priority: changed print files retain earlier human approval.**
   `podauto/printready.py:457` replaces print files without invalidating decisions;
   `podauto/cli.py:289` restores the record's previous status after conversion.
   Reproduction: approve a design, change its source image, run
   `printready --force --method raster`; the print file hash changed while its
   decision and record status stayed `approved`, and export still emitted it.
   Image regeneration also retains existing print paths and decisions in
   `podauto/imagegen.py:419`. Bind QA and approval to an artifact version or hash.
   Invalidate downstream print files when source images change and require a
   fresh review after the printable artifact changes.

5. **High priority: approval does not require successful print QA.**
   `podauto/review.py:226` accepts a nonempty print path with no blocking notes;
   `podauto/review.py:279` only verifies that the file exists at export.
   Reproduction: an opaque, blank 64×64 PNG was approved and emitted as an export
   row without running QA. There is no recorded proof that the print gate ran
   against the current file. Require current print QA before approval, either
   through a recorded result bound to the file hash or by running the local
   checks during approval. Keep explicit warning overrides separate from missing
   or failed required checks.

6. **Medium priority: corrected QA failures do not recover.**
   `podauto/imageqa.py:261` appends findings without replacing old results;
   `podauto/imageqa.py:287` sets a failure status but never restores an eligible
   record after success. Reproduction: QA rejected a blank image, then passed
   valid artwork written to the same path; the record stayed `images_qa_failed`
   and retained the obsolete blocking note. `podauto/cli.py:246` excludes that
   status from normal image generation. Replace current findings by stage and
   artifact version, preserve history separately, and define recovery transitions
   that return eligible records to review without reviving other gate failures.

7. **Medium priority: concurrent store writes lose updates.**
   `podauto/store.py:83` and `:93` read the full store, modify their snapshots,
   and atomically replace the file. Replacement prevents a partial file but
   does not serialize the read/modify/write operation. Reproduction: two writers
   inserting different records after reading the same snapshot completed, but
   only one record remained. The quota ledger likewise uses an ordinary JSON
   rewrite in `podauto/providers.py:180`. Serialize pipeline jobs as an interim
   measure. Use transactions and version checks for overlapping runs; a lock
   around file replacement alone does not protect stale stage snapshots.

Other observations for a later pass:

- `verify.sh:5` hardcodes `/home/ahssaini/podauto`, while this checkout is under
  `/home/ahssaini/Desktop/podauto`. It also assumes an existing `.venv`; it is not
  portable verification for this checkout.
- `pyproject.toml` has no console-script entry for `podauto`. The supported
  invocation in this checkout is `python -m podauto.cli`, despite help text using
  `podauto ...`.
- Listing validation issues are printed but not persisted with the record, so
  the review queue lacks those findings. Export shares one title across a
  record's variations, despite similarity checks constructing style-specific
  candidate titles.
- Real provider readiness, current provider terms, optional vector conversion,
  market-signal quality, and lettering accuracy were not validated by this pass.

Recommended implementation order: remove the credential literals and rotate the
key; enforce approval/export eligibility and generated-listing policy checks;
tie file changes to QA/review invalidation; repair QA recovery; make persistence
safe for overlapping jobs; then address packaging and verification portability.

Provider follow-up, 2026-11-24: corrected the Cloudflare, Pollinations, Fal, Gemini, and SiliconFlow adapters/configuration. Cloudflare and Pollinations generated valid images and print files; 324 tests passed. Nscale's retired image route is disabled. SiliconFlow needs a valid account key, Fal needs credits, and Gemini needs image-generation quota. See PROVIDERS.md and data/images/provider_checks/2026-11-24-fixed/report.json for the current details. The other review, policy, and persistence findings above remain unresolved.
