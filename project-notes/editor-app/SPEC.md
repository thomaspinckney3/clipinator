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

## Status (foundation built — 2026-06-13)

Built & tested: `review_server.py` (stdlib: discovery, range video serving,
JSON API), the `Clipinator Review Editor.command` launcher, a minimal
front-end, `tests/test_review_server.py`. Verified on the real Gemes run (range
serving on the 672 MB source; path traversal blocked; pointing directly at a
project). NOT yet built: transcript-first editing UI, save → `apply_edits`,
"render this clip".

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
