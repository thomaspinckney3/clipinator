# Clipinator review editor — v1 spec

A local web app for a **video editor (non-programmer)** to review, examine
against the real video, and update the AI-proposed edit lists, then re-render.

## Audience & launch constraint

- The operator is an editor, not a programmer: no terminal commands, no venv,
  no file paths typed by hand.
- **Launch: double-click `.command`** in the repo. First run sets up the venv +
  checks ffmpeg, then starts the local server and opens the browser; later runs
  just start it. (Graduate to a menu-bar app later if handed to others.)
- All pipeline vocabulary (plans, OTIO, stages, run dirs) stays hidden. The
  editor sees **Projects → Clips**.
- No API key required for the core loop (editing + local render are API-free).
  LLM "rework" is the only key-dependent feature; off by default.

## Architecture

- **Python local server** (stdlib `http.server` or FastAPI) that:
  - auto-discovers runs in a configured **library folder** (anything with a
    `plan-output.json` + render manifest); editor never types a path.
  - serves the source video with **HTTP range requests** (required for
    scrubbing a large MP4).
  - exposes endpoints that wrap the existing `otio-edl`/`edit-list` import and
    `render-plan` — the app never reimplements merge/validate/render.
  - catches all errors and returns plain-language messages (never tracebacks).
- **Single-page browser front-end**, transcript-first (below).
- Edits are written back as the existing edit-list overlay (TOML/OTIO) and
  funnel through `edit_list.apply_edits` → `plan-output.human.json`.

## Primary editing surface: transcript-first

- The edit surface is the **word-level transcript**, not a filmstrip. Trim by
  selecting/striking words; in/out snap to word boundaries; **captions follow
  the transcript automatically** (cut and captions are one object).
- Per clip: keep / drop / rework, title (+ title card), notes.
- The AI's rationale and the critique findings are shown inline as
  accept/dismiss suggestions (edit *with* the machine's intent visible).
- `rework` = type a natural-language note → (later) LLM proposes a revised cut.

## Render model: hybrid

- **Instant in-browser preview** for the editing loop: play the source segments
  back-to-back, with a 9:16 CSS crop and a caption overlay (same font / safe
  zone) to approximate framing. Accurate for content/pacing decisions.
- **"Render this clip" button** runs the real `render-plan` for one clip on
  demand (~1 min) for true final look.
- Preview is faithful to *content* decisions (which words, order, pacing) but
  NOT to final look — 9:16 subject-aware punch-in, color grade, transitions,
  burned-caption styling, title-card render, and **B-roll/asset cutaways** are
  renderer policy. Clips with augmentation assets / media_request_ids show a
  prominent "preview approximate — render to confirm" cue.

## Dependencies (confirmed)

- **The v1 editor app needs no third-party pip packages.** Importing the
  editor-path modules (`edit_list`, `render_plan`, `timebase`) pulls in zero
  heavy deps — `opentimelineio`/`openai`/`httpx` live only in the LLM stages,
  `otio_edl`, and transcription, none of which the edit/preview/save/render loop
  uses. So the `.command` just needs a Python 3.11+ and `PYTHONPATH=src`; no
  venv, pip, or Homebrew for the core loop. (The one thing a `.command` can't
  bootstrap is the base Python itself — bundle a `.app` later for fresh Macs.)
- **ffmpeg is optional, render-only.** Browse/edit/preview (browser's own
  decoder)/save all work without it; only "Render this clip" and the optional
  end-of-video bounds check use ffmpeg/ffprobe. The render button is gated with
  a friendly install prompt when ffmpeg is absent.
- **v1 editor uses the TOML/JSON overlay path** (`edit_list.apply_edits`) to
  stay dependency-free; OTIO remains a separate CLI handoff for NLE users.

## Pointing at projects (settled)

Standard run-root layout (`pipeline-manifest.json` +
`renders/<mode>/<variant>/render-manifest.json`) lets the app be **pointed at a
project directly** at startup instead of assuming a canonical library — the same
scan handles both: point at one run root → its variants; point at a folder →
many. The `.command` forwards a dropped/passed folder; the last folder is
remembered in `~/.clipinator-review/config.json`; fallback default
`~/Documents/clipinator`. No symlinks/copying — projects live anywhere (e.g.
external drive). Labels derive from run-root folder name + variant.

- B-roll frequency unknown → default to showing the "render to confirm" badge,
  emphasized when a clip references assets.

## Status (2026-06-13)

Built & tested: `review_server.py` (stdlib: discovery, range video serving,
JSON API), the `.command` launcher, `tests/test_review_server.py`.
**Transcript-first editing UI built**: per clip the transcript shows every word
in the clip span flagged in/out; clicking a word toggles it; on save contiguous
in-runs become the segments (in/out snap to word boundaries). Keep/drop/rework,
editable title, notes, and an instant 9:16 cut preview (source segments played
back-to-back with a caption overlay). `POST /api/save` builds the edit overlay,
runs `edit_list.apply_edits`, and writes `<run>/review/<variant>/
plan-output.human.json`. Verified end-to-end on the Gemes run (word flags, save
with a drop → valid human plan; word-boundary reconstruction faithful, merges
sub-second gaps as expected). Inline JS syntax-checked.

**"Render this clip" built**: `POST /api/render` saves the current edits then
runs single-clip `render-plan` in a background thread; the front-end polls
`/api/render-status` and swaps in the new render. ffmpeg-gated (button disabled
with a friendly note when absent); previews land in `<run>/review/<variant>/
preview-renders/` (discovery skips `review/`) and are served via an allowlist.
Video streaming tolerates client disconnects (normal while scrubbing) without
tracebacks. Verified live on the Gemes run (job → done → served 200).

NOT yet built: inline AI critique suggestions; conversational rework via LLM
(both add the only API-key dependency).

## Why this shape (innovation thesis)

This is not NLE editing and shouldn't imitate one. The pipeline uniquely has
word transcripts + an LLM + structured editorial rationale, so editing moves up
a level: the human edits *content and intent* (which sentences, what to tighten),
while captions/title cards/transitions/crops are derived and regenerate
consistently. Differentiated core = transcript-first editing + visible AI
reasoning + conversational rework.

## v1 scope (build order)

1. `.command` launcher + Python server skeleton + project discovery.
2. Range-serving the source video; serve the front-end.
3. Transcript-first UI: clip list, source player, clickable/strike transcript,
   keep/drop, title, notes; instant preview (9:16 crop + caption overlay).
4. Save → write edit list → `apply_edits` → `plan-output.human.json`.
5. "Render this clip" endpoint → `render-plan` single clip + progress.
6. (later) inline AI critique suggestions; conversational rework.

## AI integration (built — 2026-06-13)

- **A — critique surfaced inline (no key):** each clip card shows the run's
  existing critique (verdict, score, one-sentence assessment, issues with
  severity + suggested fix). Matched by clip_id, prefix-fallback for final
  clips' `_R2` suffix. Read-only.
- **B — conversational rework (needs OPENAI_API_KEY):** per-clip instruction box
  → focused Responses call (numbered sentences of the clip neighborhood + current
  in/out) returns sentences to keep → applied to the word in-set; cut + captions
  update; human keeps editing. Gated by an `ai` flag; openai imported lazily;
  key loaded from .env by a minimal stdlib loader so the core stays dep-free.
  ~6s round-trip at low reasoning. Verified in-browser (clip 74.8s→47.6s).
- NOT built: on-demand re-critique of the *edited* cut (was the "full layer"
  option, deferred).

## Framing override (built — 2026-06-14)

Rationale: word/sentence selection is trivial in the transcript UI, so it's a
weak use of the AI. The high-value editorial decisions are the ones the text UI
*can't* express — first among them, **horizontal framing** of the 9:16 punch-in
when the subject-crop detector mis-centers the speaker.

- **Manual, no key.** A per-clip horizontal **framing nudge** (slider + ◀/▶,
  range ±0.4) biases the renderer's subject-crop center. Stored as
  `clip["framing_offset"]` only when non-zero (untouched clips keep automatic
  framing). Carried browser → `clipPayload` → `/api/save` →
  `edit_list.rebuild_clip` → human plan; honored in `render_plan` by
  `apply_framing_offset` after the subject/motion guards.
- **Preview honesty.** The in-browser cut preview shifts `object-position-x` to
  approximate the reframe direction, with a "preview approximate — render to
  confirm framing" cue; the real crop is verified via "render this clip".
- NOT built yet: **AI auto-center** (a vision pass that sets the offset for
  off-center subjects) and the **AI cost guardrail** (confirm-before-call +
  actual-cost readout) that will wrap every billable editor call.
