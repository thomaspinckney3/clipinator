# Source-centric project layout + migration

Status: **DRAFT — decisions marked (tentative) await Tom's confirmation** (asked 2026-07-04; proceeding on recommended defaults, all reversible before the destructive apply).

## Goal (Tom, 2026-07-04)
1. New projects are written in a source-centric directory layout.
2. Re-organize all existing runs into that layout, **fully on the T9 volume** (pull off-volume masters on).
3. Human-understandable names for **sources** and **clips/extracts/ads**.

## Target layout — one project per source video
```
/Volumes/T9/extract/<source-slug>/
  project.json            # manifest: source meta, provenance (orig run dir + timestamp), pointers, schema_version
  media/source.mp4        # the master, ON T9 (copied in if it was in ~/Downloads etc.)
  transcript/             # transcript.json, transcript.srt, raw/
  outline/                # outline.json, outline.md
  evidence/               # was video-evidence/<stem>/ : manifest.json, contact-sheet.jpg
  visual-index/  subject-index/
  plan/                   # strategy.json moments.json plan.json critique.json critique-package.json revised-plan.json
  extracts/               # one folder per clip (see decision Q4)
    NN-title-slug/        # e.g. 01-everyone-knows-the-moon-quote/
      draft.mp4  final.mp4  captions.srt  clip.otio  metadata.json
  export/                 # one-way editor handoff bundle (edit-list.otio incl. storyboard markers)
  .cache/                 # _response-cache, _critique-assets, _stage4-review-assets (hidden, per-project)
```

## Decisions (recommended defaults in effect)
- **Q1 Clip names (tentative): `NN-title-slug`** — ordered prefix (stable sort even if title edited) + slug of `concept_title` (lowercase, punctuation stripped, ~6 words / 50 chars, de-duped).
- **Q2 Multi-run (tentative): latest promoted, older archived** — one clean project per source from its newest complete run; older/dup runs → `<project>/_archive/<timestamp>/`.
- **Q3 Orphans/junk (tentative): quarantine to `/Volumes/T9/extract/_legacy/`** — nothing deleted; loose top-level artifacts, smoke/cache-test dirs, `latest/` moved there untouched.
- **Q4 Extract layout (tentative): per-extract folders** — `extracts/NN-slug/` each self-contained.

## Naming rules
- **source-slug**: `safe_stem` of the source title; strip cloudflare hash suffixes (`-<32hex>.mp4`) and `_master`; lowercase; e.g. `ergo-braver-1.4-nietzsche`, `nasa-explorers-artemis-generation`.
- **clip-slug**: from `concept_title`; fallback to `opening_hook.text`; fallback `clip_ad_NN`.
- Collisions resolved with a numeric suffix.

## TRUE SCOPE (deep scan, 2026-07-04) — bigger than it first looked
- **50 pipeline runs total**, 35 GB — NOT 4. Most are nested inside two collections:
  - `lecture-tests/` (**30 GB**, ~30+ runs): Ergo-Braver lecture series (base + `-excerpt-insight-<ts>`, `.invalid-truncated-<ts>`, `.partial-whisper-<ts>` variants), cs50x / mit-6006 / stanford-cs229 / yale runs, loose source .mp4s, `transcripts/`, smoke dirs, and `.migration-backup/` dirs from the *prior* bucket-1 migration.
  - `cloudflare-captures/` (**5.9 GB**): Ergo masters w/ cloudflare-hash names + `.json` sidecars + excerpt/rerun runs.
- **Off-volume masters** are referenced by NESTED Ergo runs (`/Users/tp/Downloads/Ergo-Braver-1.1B…`, `Ergo-Hamkins…`), not top-level ones.
- **3 clean top-level projects** map trivially: `nasa-explorers-artemis-generation` (1 run, 3 clips), `nasa-explorers-moon-rocks` (2 runs, 3 clips), `nasa-explorers-space-school` (run dir `full-pipeline-20260430T134606Z`).
- **Semantic ambiguities only Tom can resolve:** (a) is "re-org the current ones" meant to include the 30 GB `lecture-tests` test corpus (much looks like throwaway smoke/variant/invalid data), or the "real" projects only? (b) what counts as one *source* — each Ergo lecture separately, or grouped? (c) keep the `-excerpt`/`-rerun`/`-insight` variant runs, or promote one and archive the rest? (d) prior `.migration-backup/` dirs — drop them?

## Proposed phasing given the true scope
- **Pilot on the 3 clean NASA projects first** (low risk, ~730 MB): prove the migrator + new layout end to end, get Tom's eyes on a real result, THEN decide how far to take `lecture-tests`/`cloudflare-captures`.
- Do NOT bulk-migrate 30 GB of heterogeneous nested runs on best-judgment.

## Findings that shaped this (from the live T9 scan, 2026-07-04)
- Some masters are OFF-volume (`/Users/tp/Downloads/Ergo-Braver-…`, `Ergo-Hamkins-…`) → copy onto T9 into the project's `media/`.
- `/Volumes/T9/extract` is itself used as a run root (loose top-level `renders/` 305M, `transcripts/`, `openai-runs/` 27M, `video-evidence/`).
- Path case inconsistency `/Volumes/t9` vs `/Volumes/T9` → normalize.
- 35 GB total → migration moves **in place** (rename), never copy, except pulling off-volume masters on.
- Test/cache junk: `*-self-contained-smoke`, `*-cache-hit-test`, `*-title-prompt-smoke-*`, `_response-cache`, `_critique-assets`, `_stage4-review-assets`, `latest/`.

## Execution plan (phased, non-destructive until apply)
1. **Inventory (read-only)** — scan library, classify entries, group runs by source, enumerate clips + proposed slugs, flag off-volume masters + orphans/junk. Emit a report + machine plan. *(built first; run before anything moves)*
2. **Migrator `apply`** — execute the plan: in-place moves, copy-in off-volume masters, rewrite manifests/paths (relative where possible), `.migration-backup/` + undo map, idempotent, `--dry-run` default. Show full dry-run before real apply.
3. **Pipeline writes new layout for new projects** — refactor `PipelinePaths` to the source-centric schema; update the renderer output paths (per-extract), review-server discovery, and cache locations.
4. Update `review_server` discovery + `otio_edl`/`edit_list` run-root detection for the new layout.

## Risks / must-verify
- Cache-key invalidation if any path feeds a fingerprint (the rename bucket-1 work already moved cache literals — check none of these new moves change a `prompt_cache_key` or fingerprint `paths`).
- review-server must keep discovering both old (during transition) and new layouts.
- Clip-slug stability: title edits change the slug → keep `clip_id` inside `metadata.json` as the durable key; slug is cosmetic.
