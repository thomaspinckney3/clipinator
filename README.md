# Video Clip Extractor

Initial pipeline experiments for extracting interesting clips from general-audience videos.

## Setup

Install system tools with Homebrew:

```bash
brew bundle
```

Required:

```text
ffmpeg
ffprobe
```

Optional tools recorded in `Brewfile` for upcoming stages:

```text
yt-dlp
media-info
exiftool
imagemagick
tesseract
sox
```

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python package dependencies are recorded in both `pyproject.toml` and
`requirements.txt`. For a non-editable dependency install, use:

```bash
pip install -r requirements.txt
```

Put local config and API keys in `.env`:

```bash
OPENAI_API_KEY=sk-...
TRANSCRIBE_MODEL=whisper-1
TRANSCRIBE_LANGUAGE=en
TRANSCRIBE_OUTPUT_DIR=/Volumes/t9/extract/transcripts
TRANSCRIBE_PROMPT=NASA general-audience video about Artemis, astronauts, moon rocks, space exploration, scientific instruments, and training.
TRANSCRIBE_AUDIO_BITRATE_KBPS=64
TRANSCRIBE_MAX_UPLOAD_MB=24
TRANSCRIBE_WHISPER_1_USD_PER_MINUTE=0.006
TRANSCRIBE_CHUNK_SECONDS=
```

`.env` is ignored by git. `.env.example` is the shareable template.

## Transcribe With OpenAI Whisper

The first transcription backend uses OpenAI `whisper-1` because it supports `verbose_json` with word and segment timestamps. The CLI extracts compressed mono MP3 chunks from each video, keeps chunks below the upload limit, transcribes them, offsets timestamps back to the full video timeline, and writes normalized JSON plus SRT.

Dry-run the downloaded NASA videos:

```bash
PYTHONPATH=src python -m extractor.transcribe /Volumes/t9/extract/*.mp4 \
  --dry-run
```

Estimate API cost without making API calls:

```bash
PYTHONPATH=src python -m extractor.transcribe /Volumes/t9/extract/*.mp4 \
  --estimate-cost
```

Run transcription:

```bash
PYTHONPATH=src python -m extractor.transcribe /Volumes/t9/extract/*.mp4 \
  --approve-api-cost 0.20
```

The CLI refuses non-dry-run API calls unless `--approve-api-cost` is provided.
Use the estimate first, then pass a cap at or above that estimate.

Expected outputs per input video:

```text
<video-stem>.whisper-1.json
<video-stem>.whisper-1.srt
raw/<video-stem>.chunk-000.whisper-1.json
```

The normalized JSON is the stable downstream contract for clip selection.

## Local Video Preprocessing

Create a local video evidence bundle for later LLM clip-ad planning. This uses
only `ffmpeg` and `ffprobe`; it makes no API calls.

```bash
preprocess-video /Volumes/t9/extract/*.mp4 \
  --transcript-dir /Volumes/t9/extract/transcripts \
  --output-dir /Volumes/t9/extract/video-evidence \
  --write-markdown
```

Each video gets:

```text
manifest.json
manifest.md
contact-sheet.jpg
frames/frame-0000-00000.000s.jpg
```

The manifest includes media metadata, sampled frame references, scene-change
timestamps, silence regions, transcript QA summary, and guidance for building a
stable prompt-cache prefix.

## Self-Contained Pipeline

For system tests, use the single-entry pipeline rather than manually assembling
intermediate artifacts. The only required input is the video file:

```bash
extract-clips /path/to/video.mp4 --approve-api-cost 25.00
```

The wrapper creates a run directory next to the input video and runs:

```text
video -> whisper-1 transcript -> local evidence -> stage 1 -> stage 2
-> dense visual index from stage 2 candidates -> stage 3 -> render
-> stage 4 critique -> stage 5 re-edit -> final render
```

It performs a dry-run cost estimate before each API stage and spends from the
single `--approve-api-cost` cap. If the cap is omitted, the pipeline stops before
the first uncached API call and prints the required estimate. Local response
caches are still honored by the individual OpenAI stage commands.

## Dense Visual Indexing

For object-level clip planning, build a local 1 fps visual index. This is useful
when a clip hook depends on a concrete visible object or action, such as a hammer,
spacesuit glove, tool strike, or underwater suit movement. The index stays local
and makes no API calls.

```bash
PYTHONPATH=src python -m extractor.visual_index \
  /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --stage2-output /Volumes/t9/extract/openai-runs/space-school-stage2-gpt-5.5-high/stage2-output.json \
  --output-dir /Volumes/t9/extract/visual-index \
  --frame-interval-seconds 1 \
  --write-markdown
```

Outputs include:

```text
manifest.json
manifest.md
frames/frame-000000.jpg
sheets/window-*.jpg
sheets/candidate-*.jpg
```

Candidate sheets are built from Stage 2 windows with padding. They are intended
as visual-proof packages: Stage 3 should cite exact sheet timestamps for named
objects/actions before rendering a clip around them.

## Transcript QA

Run local transcript quality checks without making API calls:

```bash
qa-transcript /Volumes/t9/extract/transcripts/*.whisper-1.json
```

The QA command reports a score, timing and density metrics, Whisper confidence
signals, and suspicious ranges worth manual review. JSON output is available for
automation:

```bash
qa-transcript /Volumes/t9/extract/transcripts/*.whisper-1.json --json
```

## Micro-Trailer Ad Planning

Create local test plans for clips that advertise the full video. This stage
summarizes the full-video promise, generates deterministic ad-angle hypotheses,
grounds those angles in transcript windows, applies transcript QA penalties, and
outputs edit plans using standard clip tactics. It does not make API calls.

```bash
plan-clip-ads /Volumes/t9/extract/transcripts/*.whisper-1.json \
  --output-dir /Volumes/t9/extract/ad-plans \
  --write-markdown
```

Each plan includes:

```text
full_video_ad_brief
clip_ad_plans[]
opening hook
timestamp range
edit methods
audio/video sync policy
visual requirements to verify
QA notes
```

Current editorial default: when a candidate has a clean payoff or surprising
line, lead with it as the first spoken beat. Use a title/tagline card first if
needed, then jump into the payoff, then backfill context.

If there is no clear payoff, treat the candidate as weak or reject it. The
pipeline should not try to fix a payoff-free clip with faster cuts or captions.

## OpenAI Stage 1: Video Brief And Ad Angles

Run the first LLM stage against the real OpenAI API. This consolidates the full
video overview, why-to-watch summary, visual inventory, and proposed clip-ad
angles into one structured output. It sends a compact transcript timeline, the
local preprocessing manifest, and optionally the contact sheet image.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_stage1 \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --dry-run
```

Run with an explicit approved cap:

```bash
PYTHONPATH=src python -m extractor.openai_stage1 \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --approve-api-cost 5.00
```

The command refuses non-dry-run API calls unless `--approve-api-cost` is
provided. The command also computes a deterministic local response-cache key
from the model, prompt, schema, transcript, manifest, contact sheet bytes, image
detail, reasoning effort, and output cap. When the same request is already in
`STAGE1_CACHE_DIR`, it reuses the cached response locally and makes no API call,
so no approval flag is needed.

To bypass a cached response intentionally, pass `--refresh-cache`; that path is
treated as a fresh API call and still requires `--approve-api-cost`.

Seed the cache from an existing run without calling the API:

```bash
PYTHONPATH=src python -m extractor.openai_stage1 \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --seed-cache-from-run /Volumes/t9/extract/openai-runs/space-school-stage1-gpt-5.5-high
```

Outputs are written under `STAGE1_OUTPUT_DIR` and include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
raw-response.json
stage1-output.json
usage-and-cost.json
summary.md
```

## OpenAI Stage 2: Transcript Moment Analysis

Run the second LLM stage after stage 1 has produced a video brief and ad-angle
hypotheses. This stage reuses the same stable video evidence prefix and adds the
stage-1 output, then asks for concrete timestamped transcript windows for each
ad angle. It still uses the local response cache and the explicit API approval
gate.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_stage2 \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --stage1-output /Volumes/t9/extract/openai-runs/space-school-stage1-gpt-5.5-high/stage1-output.json \
  --dry-run
```

Run with an explicit approved cap:

```bash
PYTHONPATH=src python -m extractor.openai_stage2 \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --stage1-output /Volumes/t9/extract/openai-runs/space-school-stage1-gpt-5.5-high/stage1-output.json \
  --approve-api-cost 1.00
```

Outputs include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
raw-response.json
stage2-output.json
usage-and-cost.json
summary.md
```

## OpenAI Stage 3: Edit Plans

Run the third LLM stage after stage 2 has produced ranked candidate moments.
This stage creates concrete editor-facing cut plans for the strongest short-form
clip ads: source ranges, opening hook, audio/video strategy, B-roll coverage,
caption/overlay notes, verification steps, and rendering assumptions. It does
not render video.

Stage 3 treats the payoff as the hook by default: title/tagline card first,
strongest payoff line as the first spoken beat, then context. It should avoid
silent source-video lead-ins unless the visual action itself is the hook.
Candidates without a clear payoff should be deferred instead of turned into edit
plans.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_stage3 \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --visual-index-manifest /Volumes/t9/extract/visual-index/nasa-explorers-space-school/manifest.json \
  --stage1-output /Volumes/t9/extract/openai-runs/space-school-stage1-gpt-5.5-high/stage1-output.json \
  --stage2-output /Volumes/t9/extract/openai-runs/space-school-stage2-gpt-5.5-high/stage2-output.json \
  --dry-run
```

Run with an explicit approved cap:

```bash
PYTHONPATH=src python -m extractor.openai_stage3 \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --visual-index-manifest /Volumes/t9/extract/visual-index/nasa-explorers-space-school/manifest.json \
  --stage1-output /Volumes/t9/extract/openai-runs/space-school-stage1-gpt-5.5-high/stage1-output.json \
  --stage2-output /Volumes/t9/extract/openai-runs/space-school-stage2-gpt-5.5-high/stage2-output.json \
  --approve-api-cost 8.00
```

Outputs include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
raw-response.json
stage3-output.json
usage-and-cost.json
summary.md
```

## Local Draft Rendering

Render local draft cuts from a stage-3 edit plan. This uses only `ffmpeg`; it
does not call any API. The renderer uses the stage-3 timeline source ranges,
separates dialogue audio from visual B-roll when the edit plan provides usable
visual time cues, formats drafts as vertical `9:16`, writes sidecar SRT
captions, and muxes a subtitle-track version when possible. If the installed
`ffmpeg` lacks `drawtext`, captions are not burned into the video.

```bash
PYTHONPATH=src python -m extractor.render_stage3 \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --stage3-output /Volumes/t9/extract/openai-runs/space-school-stage3-gpt-5.5-high/stage3-output.json \
  --output-dir /Volumes/t9/extract/renders/space-school-stage3-drafts
```

If a planned silent cold open creates dead air, repair that clip by borrowing
the beginning of the following narration beat under the opening visual:

```bash
PYTHONPATH=src python -m extractor.render_stage3 \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --stage3-output /Volumes/t9/extract/openai-runs/space-school-stage3-gpt-5.5-high/stage3-output.json \
  --output-dir /Volumes/t9/extract/renders/space-school-stage3-final \
  --repair-silent-lead-clip clip_ad_02_moon_hammer
```

For a cleaner silent opening, replace the first silent beat with a baked-in
black title card. On macOS the renderer uses `tools/render_title_card.swift`
for crisp system text, with a dependency-free bitmap fallback. The renderer does
this automatically when the first silent timeline beat is described as a title
card; `--black-title-card-clip` forces it for older plans:

```bash
PYTHONPATH=src python -m extractor.render_stage3 \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --stage3-output /Volumes/t9/extract/renders/moon-hammer-alternatives/moon-hammer-alternatives-stage3.json \
  --output-dir /Volumes/t9/extract/renders/moon-hammer-alternatives/renders-black-title \
  --black-title-card-clip clip_ad_02a_moon_hammer_geology_setup \
  --black-title-card-clip clip_ad_02b_moon_hammer_payoff_first
```

Use `--transition-duration 0.18` to render short audio/video crossfades
between planned segments when hard cuts feel too abrupt.

Outputs include one `.draft.mp4`, one `.draft.srt`, and one
`.draft.subtitled.mp4` per clip, plus `render-manifest.json`.

Stage 5 can put reusable rendering defaults in the plan itself. To use those
settings while still allowing CLI overrides, pass:

```bash
PYTHONPATH=src python -m extractor.render_stage3 \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --stage3-output /Volumes/t9/extract/openai-runs/<stage5-run>/stage3-output.json \
  --output-dir /Volumes/t9/extract/renders/<stage5-render> \
  --use-plan-render-settings
```

## OpenAI Stage 4: Post-Render Review

Run the fourth LLM stage after local rendering. This stage critiques rendered
draft clips using the human-review rules from this project: payoff first, no
confusing mid-sentence opening, no mismatched talking head under narration, no
repeated B-roll padding, smooth cuts, and no ending that drifts into a different
story.

The rendered MP4 files stay local. The stage sends local evidence images:
storyboard contact sheets sampled from the rendered clips, transition strips
around segment boundaries and internal multi-cut B-roll edits, and audio waveform timelines. The prompt also includes the
render manifest, SRT captions, local silence/caption timing diagnostics,
source-range overlap warnings, and the optional stage-3 edit plan.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_stage4_review \
  --render-manifest /Volumes/t9/extract/renders/space-school-underwater-fixed/renders-tight/render-manifest.json \
  --stage3-output /Volumes/t9/extract/renders/space-school-underwater-fixed/space-school-underwater-tight-stage3.json \
  --dry-run
```

Run with an explicit approved cap:

```bash
PYTHONPATH=src python -m extractor.openai_stage4_review \
  --render-manifest /Volumes/t9/extract/renders/space-school-underwater-fixed/renders-tight/render-manifest.json \
  --stage3-output /Volumes/t9/extract/renders/space-school-underwater-fixed/space-school-underwater-tight-stage3.json \
  --approve-api-cost 1.00
```

Review only one clip from a multi-clip render:

```bash
PYTHONPATH=src python -m extractor.openai_stage4_review \
  --render-manifest /Volumes/t9/extract/renders/space-school-payoff-first-other-clips/renders/render-manifest.json \
  --stage3-output /Volumes/t9/extract/renders/space-school-payoff-first-other-clips/space-school-other-clips-payoff-first-stage3.json \
  --clip-id clip_ad_03_personal_spacecraft_payoff_first \
  --dry-run
```

Outputs include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
review-package.json
raw-response.json
stage4-output.json
usage-and-cost.json
summary.md
```

Local review assets are written under `_stage4-review-assets/` and include one
`review-contact-sheet.jpg`, `transition-contact-sheet.jpg`, and
`audio-waveform.jpg` per reviewed clip when the default evidence layers are
enabled. Use `--no-transition-strips` or `--no-waveform` for a cheaper
storyboard-only review.

## OpenAI Stage 5: Critique-Driven Re-Edit

Run the fifth LLM stage after Stage 4 has critiqued a render. This stage turns
the critique back into a full revised Stage-3-compatible edit plan. It is not a
manual clip patch: the prompt asks the model to apply general rules such as
payoff-first structure, no dead source video before audio, no mid-word endings,
no mismatched talking head under narration, no repeated B-roll padding, and
topic-consistent endings.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_stage5_reedit \
  --stage3-output /Volumes/t9/extract/renders/moon-hammer-polish/moon-hammer-polish-stage3-v2.json \
  --render-manifest /Volumes/t9/extract/renders/moon-hammer-polish/renders-v2/render-manifest.json \
  --stage4-output /Volumes/t9/extract/openai-runs/renders-v2.stage4.gpt-5.5.high.20260430T131557Z/stage4-output.json \
  --review-package /Volumes/t9/extract/openai-runs/renders-v2.stage4.gpt-5.5.high.20260430T131557Z/review-package.json \
  --dry-run
```

Include the full transcript/video evidence prefix when the re-edit may need new
sentence boundaries or alternate B-roll ranges:

```bash
PYTHONPATH=src python -m extractor.openai_stage5_reedit \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --stage3-output /Volumes/t9/extract/renders/moon-hammer-polish/moon-hammer-polish-stage3-v2.json \
  --render-manifest /Volumes/t9/extract/renders/moon-hammer-polish/renders-v2/render-manifest.json \
  --stage4-output /Volumes/t9/extract/openai-runs/renders-v2.stage4.gpt-5.5.high.20260430T131557Z/stage4-output.json \
  --review-package /Volumes/t9/extract/openai-runs/renders-v2.stage4.gpt-5.5.high.20260430T131557Z/review-package.json \
  --approve-api-cost 1.50
```

Outputs include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
raw-response.json
stage5-output.json
stage3-output.json
usage-and-cost.json
summary.md
```

Render the revised output with:

```bash
PYTHONPATH=src python -m extractor.render_stage3 \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --stage3-output /Volumes/t9/extract/openai-runs/<stage5-run>/stage3-output.json \
  --output-dir /Volumes/t9/extract/renders/<stage5-render> \
  --use-plan-render-settings
```

## Experimental Video API Probe

Probe whether provider APIs accept MP4 files as model inputs. This is separate
from Stage 4 so failed provider experiments do not disturb the working
storyboard review path. The probe creates a small local MP4 sample by default,
then estimates a conservative upper-bound cost before any API call.

OpenAI dry run:

```bash
PYTHONPATH=src python -m extractor.video_api_probe \
  --provider openai \
  --video /Volumes/t9/extract/renders/space-school-underwater-fixed/renders-tight/clip_ad_01_underwater_spacewalk_payoff_first_tight.draft.mp4 \
  --dry-run
```

OpenAI real probe after reviewing the estimate:

```bash
PYTHONPATH=src python -m extractor.video_api_probe \
  --provider openai \
  --video /Volumes/t9/extract/renders/space-school-underwater-fixed/renders-tight/clip_ad_01_underwater_spacewalk_payoff_first_tight.draft.mp4 \
  --approve-api-cost 0.50
```

Anthropic dry run:

```bash
PYTHONPATH=src python -m extractor.video_api_probe \
  --provider anthropic \
  --video /Volumes/t9/extract/renders/space-school-underwater-fixed/renders-tight/clip_ad_01_underwater_spacewalk_payoff_first_tight.draft.mp4 \
  --dry-run
```

Anthropic requires `ANTHROPIC_API_KEY` in `.env` before a real probe can run.
The default Anthropic mode uploads the MP4 through the beta Files API and then
tries to reference it as a document block. Use `--anthropic-mode
base64-document` to test direct base64 message input instead.

Outputs include:

```text
request-manifest.json
provider-result.json
usage-and-cost.json
summary.md
```
