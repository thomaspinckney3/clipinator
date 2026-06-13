# Session changes audit — stage rename, timebase, edit list, streaming, Py3.13, output-label

## 2026-06-12

### Verdict

Audited this session's uncommitted changes: the stage rename (bucket 1 +
cache keys + `migrate_run_layout.py`), `timebase.py` frame-grid quantization,
`edit_list.py` (TOML edit list + static HTML + import), `openai_responses.py`
streaming helper wired into all five OpenAI stages, the Python 3.13 switch, and
the `render_plan.py` `--output-label` change. The rename, timebase, 3.13 switch,
and `--output-label` change are clean and verified by live runs (full Gemes
pipeline end-to-end, migration across 23 runs on T9, re-rendered finals). The
streaming fix is functionally correct and proven (it turned a 3×-failing stage
into a clean success) but carries a cost-exposure caveat. **The edit-list
feature has one high-severity functional gap: human caption-text edits are a
no-op at render time under the default pipeline config.** No change writes a
half-applied/corrupt artifact (imports and migration are all-or-nothing with
loud failure). Net: passes structurally; one high-severity functional finding
(F1) should be fixed before the OTIO work builds caption support on top of it.

### Findings

1. **(HIGH, functional) Human caption-text edits never reach rendered output.**
   - evidence:
     - `src/extractor/edit_list.py:671-717` (rebuild writes `caption_plan` only)
     - `src/extractor/render_plan.py:1217-1221` (`caption_plan_entries` returns `[]` unless `caption_plan_source` ∈ {absolute_render_time, render_time, srt} or `use_caption_plan_for_srt`)
     - `src/extractor/render_plan.py:3642-3644` (default `auto` mode prefers `transcript_caption_entries`)
     - default `--caption-source` is `auto` (verified: `build_parser().parse_args(...).caption_source == 'auto'`); pipeline does not override it
   - `rebuild_clip` writes the edited captions into `clip["caption_plan"]` but never sets `caption_plan_source`/`use_caption_plan_for_srt`, and even if it did, the renderer's default `auto` mode re-derives caption text from the transcript by timestamp. So edited caption text is silently dropped at render. The export still emits `[[clips.captions]]` blocks inviting edits that cannot take effect. Title edits and segment trims work (they drive the timeline/source ranges directly); captions do not.
   - commands used:
     - `grep -n "caption_plan_source" src/extractor/render_plan.py`
     - `.venv/bin/python -c "from extractor.render_plan import build_parser; ..."`

2. **(MEDIUM, functional) Unedited export→import round-trip can fail on short segments.**
   - evidence:
     - `src/extractor/edit_list.py:88` (export keeps segments with `end - start > 0.05`)
     - `src/extractor/edit_list.py:557` (`validate_segments` rejects `< MIN_SEGMENT_SECONDS` = 0.25)
   - A plan containing a legitimate 0.05–0.25s segment exports fine, then `import` of the unedited file fails. Fails loudly (good) but breaks the identity round-trip the user did not cause. Pick one threshold.

3. **(MEDIUM, integrity) `openai_responses` retry re-bills output on a late-stream disconnect.**
   - evidence:
     - `src/extractor/openai_responses.py:63-92` (retry wraps the whole `stream(...)` + `get_final_response()`)
   - A connection drop after substantial generation retries the entire request; output tokens already produced on the failed attempt are billed again. Correctness is fine (stateless, `store=False`), but the docstring's "retrying is safe" conflates correctness with cost. Real but bounded cost exposure.

4. **(MEDIUM, functional) Migration undo map cannot cleanly reverse files that were both content-edited and renamed.**
   - evidence:
     - `src/extractor/migrate_run_layout.py:246-264` (content edits + backups keyed by OLD relative path, recorded before renames run)
   - Files like `review-package.json`→`critique-package.json` are backed up at their old path, then moved; a reverse tool using `content_backups` (old paths) would restore to a stale location and miss the renamed-and-edited file at its new path. The undo records renames and backups independently without the composition. No live impact unless an undo is attempted.

5. **(MEDIUM, functional) Out-of-range / empty-segment caption mapping degrades silently.**
   - evidence:
     - `src/extractor/edit_list.py:112-120` (`render_offset_to_source` clamps any past-end offset into the last segment)
     - `src/extractor/edit_list.py:197-207` (empty `segments` → all plan captions map to `None`, then `chunk_captions([])` → `[]`; no warning)
   - A title-card-only clip, or one whose segments were all trimmed sub-threshold, drops captions end-to-end with no diagnostic. Compounds F1.

6. **(LOW, functional) `resolve_recorded_path` relink-by-name can silently resolve a same-named file from another run.**
   - evidence:
     - `src/extractor/edit_list.py:257-263` (matches on `Path(raw).name` only, no hash/size check)
     - mitigated for the plan (scoped under `openai-runs/<dir>`, line ~752) but not for `source_video` (line ~773, bare name vs `[run_root.parent, run_root]`)
   - Reused source-video filenames across runs could mis-resolve. Prints a `note:` but does not verify identity.

7. **(LOW, functional) TOML written to disk before the parse-guard runs.**
   - evidence:
     - `src/extractor/edit_list.py:527-528` (`toml_path.write_text(...)` then `tomllib.loads(...)`)
   - A malformed emission leaves a corrupt `edit-list.toml` on disk (and no HTML) before raising a raw decode error. Guard should run on the in-memory string before writing.

8. **(LOW, functional) Import bounds-check silently disabled when source video unresolved.**
   - evidence:
     - `src/extractor/edit_list.py:775-802` (ffprobe failure / missing video → `duration=None`)
     - `src/extractor/edit_list.py:564` (upper-bound check skipped when `duration is None`)
   - Warns, but still produces a plan whose segments may exceed the real video length. All-or-nothing write preserved; validation weakened.

9. **(LOW, integrity) Streaming "heartbeat" cannot detect an idle hang.**
   - evidence:
     - `src/extractor/openai_responses.py:65-79` (heartbeat check is inside `for event in stream`)
   - The heartbeat only prints while events flow; during a true idle stall no events arrive so it never fires — contradicting the "keeps connection active" framing. The real keep-alive benefit (active socket read) still holds; only the hang-detection/progress framing is wrong. Also `httpx.TransportError` (line 24-30) is broader than needed (would retry client-side `LocalProtocolError`/`UnsupportedProtocol` pointlessly).

10. **(LOW, latent) `composed_path_pairs` is incorrect for ≥2 levels of nested renames.**
    - evidence:
      - `src/extractor/migrate_run_layout.py:143-158` (second ancestor slice uses `len(dir_old)` against an already-rewritten `final_new`)
    - Currently masked: the layout nests files only one level inside a renamed run dir (`openai-runs/` itself is not renamed), so it never triggers. A latent trap if the layout deepens.

11. **(INVESTIGATED — NOT A BUG) Migration `stage_outputs` key mapping.**
    - evidence:
      - migrated: `['strategy','moments','plan','critique','revised_plan']`
      - backup original: `['stage1','stage2','stage3','stage4','stage5_stage3']`
    - The migration agent flagged a missing `stage5` mapping. Verified false against real T9 data: the old key was `stage5_stage3` (not `stage5`), and `STAGE_OUTPUT_KEYS` maps it correctly. No action.
    - commands used:
      - `python -c "import json; print(list(json.load(open('.../pipeline-manifest.json'))['stage_outputs'].keys()))"` on the migrated file and its `.migration-backup/` original

### Completeness

No BRIEF.md exists for this session. Against the user's stated requests:

- Stage rename (bucket 1 + cache keys + migration): done; verified by the live Gemes run (cache hits, all stages) and migration of 23 T9 runs. Model-visible legacy keys intentionally retained (documented). Clean.
- Timebase quantization: done; applied at render + subject-index ingest (transient, not persisted — matches design). Clean.
- Edit list (TOML + HTML + import): delivered, but the caption sub-feature is functionally a no-op at render (F1). Title/segment/drop/rework editing works.
- Streaming helper across 5 stages: done and proven; cost caveat (F3) and heartbeat caveat (F9).
- Python 3.13 switch: done; OTIO works in-process, all adapters present, imports/tests pass.
- `--output-label`: done; finals render without `.draft`, verified on the real Gemes finals.
- Deferred: OTIO export/import (task #5) — explicitly the next step, not descoped from this batch.

### Structural Assessment

- Architecture: clean. `openai_responses` removes 5× duplicated `responses.create` call sites behind one helper. `edit_list` reuses `render_plan.transcript_words_from_data` rather than re-deriving. `migrate_run_layout` is appropriately standalone. Renderer still consumes plan-format JSON unchanged (edit-list import and OTIO both target that boundary).
- Abstraction: clean. No notable duplication introduced. F1 is a missing-integration bug, not an abstraction defect.
- Semantic families: captions are the one split — the edit-list path writes `caption_plan` while the renderer's default reads transcript words. These should converge (F1).
- Patterns: consistent (argparse, env helpers, read_json/write_json, loud failure on error).
- Integrity: F3 (re-bill on retry) and F8 (bounds disabled on missing video) weaken stated guarantees without failing loudly; F7 (write-before-guard) leaves a corrupt artifact. None are bug cover-ups.
- Scope: clean. `.env.example` rename and README updates are in-scope for the rename/feature work; no unrelated drive-by changes.

### Forward Impact

- F1 is the key forward-impact risk: task #5 (OTIO) plans to carry captions in OTIO metadata. Building caption interchange on top of a render path that ignores edited captions would propagate the gap. Fix caption rendering (route edited `caption_plan` to the renderer, or write an SRT the renderer consumes) before/with the OTIO caption mapping.
- `timebase.OUTPUT_FRAME_RATE = 30` is coupled by comment to the renderer's hardcoded `fps=30` (render_plan.py:2747/3198/3280). If output fps ever becomes configurable, both must change together. Low risk, worth a shared constant later.
- Migration is not crash-safe/restartable (partial apply → preflight refuses recovery); fine for a one-shot tool run under supervision, but note it if it's ever automated.

### Remediation Mapping

- Before OTIO (task #5):
  - F1: make `rebuild_clip` captions actually render — set `use_caption_plan_for_srt`/`caption_plan_source` on the rebuilt clip and/or have the edit-list path emit an SRT the renderer burns; reconcile with the default `auto` transcript-words behavior. Add F5 (empty-segment / out-of-range) handling in the same change.
- Edit-list hardening (low, batch when convenient):
  - F2 unify segment-length thresholds; F6 add identity check (or scope) to relink; F7 move parse-guard before write; F8 fail or warn-hard when bounds can't be checked.
- Streaming (low):
  - F3 reword docstring re cost; consider capping retries tighter for late-stream failures. F9 narrow the exception tuple and/or move heartbeat to a time-based watchdog (or rename it to "progress").
- Migration (low, only if undo/automation needed):
  - F4 record composed (post-rename) paths in the undo map; F10 fix `composed_path_pairs` for multi-level nesting if the layout ever deepens.

## 2026-06-12 — remediation

All findings from the audit above were fixed and verified the same day.

- **F1 (caption no-op) — fixed.** Root cause was deeper than reported: the
  default `auto` caption mode re-derives verbatim captions from the transcript,
  and the revise LLM also sets `caption_plan_source` with *paraphrased*
  captions, so naively honoring that flag would have regressed every render
  from verbatim to paraphrase. Fix distinguishes human-authoritative captions:
  `edit_list.rebuild_clip` sets a new `human_edited_captions` flag only when the
  edited text actually differs from a verbatim transcript re-seed (and clears
  `caption_plan_source` otherwise); `render_plan` auto mode honors plan captions
  over transcript only when that flag is set. Verified: an edited caption now
  appears in the rendered SRT; an unedited clip keeps verbatim transcript; Gemes
  finals are unaffected (revise sets `caption_plan_source` but not the new flag).
  `export_captions` now seeds from the transcript (mirroring what renders) and
  preserves prior human edits. `evidence: render_plan.py:3639-3654`,
  `edit_list.py:rebuild_clip` + `build_caption_plan`.
- **F2 — fixed.** Single `MIN_SEGMENT_SECONDS = 0.05` used by both export
  extraction and import validation; every exported segment now re-imports.
- **F5 — fixed.** `render_offset_to_source` returns None for out-of-range
  offsets instead of clamping into the last segment; import warns when no
  captions can be placed.
- **F6 — fixed.** Relink refuses to guess: exactly one name match relinks,
  multiple raise an ambiguous-relink error.
- **F7 — fixed.** TOML parse-guard runs before the file is written.
- **F8 — fixed.** Loud WARNING when source-video bounds cannot be validated.
- **F3 — fixed.** Docstring corrected re re-billing; retry budget called out.
- **F9 — fixed.** Connection-error set narrowed to genuinely transient httpx
  errors (excludes LocalProtocolError/UnsupportedProtocol/ProxyError); the
  "heartbeat" renamed to progress logging with a note that it cannot detect an
  idle stall (that is httpx's read timeout).
- **F4 — fixed.** Undo map records each content backup's composed post-rename
  `final_path` plus an `undo_order` note, so a reversal is unambiguous.
- **F10 — fixed.** `final_location` / `_apply_dir_renames` compose directory
  renames transitively; verified on the nested `_stage4-review-assets` case and
  non-renamed manifests moved by a parent rename.
- **F11 — confirmed not a bug** (migration `stage_outputs` key), verified
  against a real T9 backup.

Verification: all modules compile and import on 3.13; edit-list export→edit→
import→render cycle confirmed to carry an edited caption into the SRT;
validation failures still write nothing; migration fixture exercised for nested
renames, undo `final_path` existence, and idempotence.
