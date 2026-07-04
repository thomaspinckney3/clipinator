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

Install Python dependencies. The project targets Python 3.13 (the
`opentimelineio` interchange library has no working Python 3.14 build yet):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python package dependencies are recorded in both `pyproject.toml` and
`requirements.txt`. For a non-editable dependency install, use:

```bash
pip install -r requirements.txt
```

## Tests

The test suite is `pytest`-based and lives in `tests/`. Install the dev extra
and run it:

```bash
pip install -e ".[dev]"
pytest
```

Tests are pure and fast except a few that build a tiny video with `ffmpeg`;
those skip automatically when `ffmpeg`/`ffprobe` are absent. Coverage focuses on
the pure-logic core — frame-grid quantization (`test_timebase`), the edit-list
export/import and caption mapping (`test_edit_list`), the run-layout migration
(`test_migrate_run_layout`), and the streaming retry helper
(`test_openai_responses`) — and includes a regression test for each fix from the
code audit.

Put local config and API keys in `.env`:

```bash
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
OPENAI_RESPONSE_CACHE_MODE=strict
TRANSCRIBE_PROVIDER=elevenlabs
TRANSCRIBE_MODEL=
TRANSCRIBE_OPENAI_MODEL=whisper-1
TRANSCRIBE_ELEVENLABS_MODEL=scribe_v2
TRANSCRIBE_LANGUAGE=en
TRANSCRIBE_OUTPUT_DIR=/Volumes/t9/extract/transcripts
TRANSCRIBE_PROMPT=NASA general-audience video about Artemis, astronauts, moon rocks, space exploration, scientific instruments, and training.
TRANSCRIBE_AUDIO_BITRATE_KBPS=64
TRANSCRIBE_MAX_UPLOAD_MB=24
TRANSCRIBE_WHISPER_1_USD_PER_MINUTE=0.006
TRANSCRIBE_ELEVENLABS_USD_PER_HOUR=0.40
TRANSCRIBE_CHUNK_SECONDS=
```

`.env` is ignored by git. `.env.example` is the complete, commented reference
for every setting: per-stage model/pricing/limit families (`STRATEGY_*`,
`MOMENTS_*`, `PLAN_*`, `CRITIQUE_*`, `REVISE_*`), transcription
(`TRANSCRIBE_*`), Cloudflare capture (`CLOUDFLARE_*`), the video API probe
(`VIDEO_PROBE_*`), and the local response cache mode
(`OPENAI_RESPONSE_CACHE_MODE`). CLI flags always override `.env` values.

## Capture From Cloudflare Stream

Download owned Cloudflare Stream videos or ready live-input recordings into the
local pipeline workspace. The utility uses the supported Stream downloads API:
it can create the downloadable MP4 or M4A asset, poll until Cloudflare marks it
ready, and then save the returned download URL locally.

Add Cloudflare credentials to `.env`:

```bash
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_CAPTURE_OUTPUT_DIR=/Volumes/t9/extract/cloudflare-captures
CLOUDFLARE_CAPTURE_INDEX=/Volumes/t9/extract/cloudflare-captures/cloudflare-stream-index.json
```

The token needs `Stream Read` permission to inspect videos and downloads. It
also needs `Stream Write` when `cloudflare-stream-capture` has to generate a new
downloadable MP4 or M4A asset.

If `CLOUDFLARE_ACCOUNT_ID` is omitted, the utility tries to discover it from the
API token. This only works when the token can list exactly one account. To inspect
visible accounts:

```bash
cloudflare-stream-capture --list-accounts
```

Capture a known Stream video UID, also commonly called the Stream ID:

```bash
cloudflare-stream-capture \
  --stream-id <VIDEO_UID> \
  --output-dir /Volumes/t9/extract/cloudflare-captures
```

If the API token only has `Stream Read`, capture from the playback manifest
instead of generating a downloadable MP4 asset:

```bash
cloudflare-stream-capture \
  --stream-id <VIDEO_UID> \
  --capture-method playback \
  --output-dir /Volumes/t9/extract/cloudflare-captures
```

Playback capture uses the video's HLS manifest and local `ffmpeg` to remux the
stream to MP4. For private videos that require signed playback URLs, add
`--generate-playback-token`; the token must have permission to create Stream
playback tokens.

Successful captures update a local JSON index mapping Stream IDs to downloaded
files. By default it is written as `cloudflare-stream-index.json` in the capture
output directory. To backfill the index from an existing sidecar metadata file:

```bash
cloudflare-stream-capture \
  --record-metadata /Volumes/t9/extract/cloudflare-captures/example.mp4.json
```

List recordings created by a live input:

```bash
cloudflare-stream-capture \
  --live-input-uid <LIVE_INPUT_UID> \
  --list-recordings
```

Capture the newest ready recording for a live input:

```bash
cloudflare-stream-capture \
  --live-input-uid <LIVE_INPUT_UID> \
  --latest-ready \
  --output-dir /Volumes/t9/extract/cloudflare-captures
```

Use `--type audio` to generate and download the audio-only M4A instead of the
default MP4. Cloudflare only exposes downloadable MP4s after a video is ready;
for live recordings, Cloudflare documents MP4 downloads only for recordings under
four hours.

## Transcribe With OpenAI Whisper Or ElevenLabs Scribe

The default transcription backend uses ElevenLabs Scribe (`scribe_v2`). The CLI
extracts compressed mono MP3 chunks from each video, keeps chunks below the
upload limit, transcribes them, offsets timestamps back to the full video
timeline, and writes normalized JSON plus SRT. The source video is never
uploaded directly, only the extracted MP3 chunks.

OpenAI Whisper remains available with `--provider openai --model whisper-1`.

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

Explicitly select ElevenLabs Scribe, which is also the default:

```bash
PYTHONPATH=src python -m extractor.transcribe /Volumes/t9/extract/*.mp4 \
  --provider elevenlabs \
  --approve-api-cost 0.20
```

The CLI refuses non-dry-run API calls unless `--approve-api-cost` is provided.
Use the estimate first, then pass a cap at or above that estimate.

Expected ElevenLabs outputs per input video:

```text
<video-stem>.elevenlabs-scribe_v2.json
<video-stem>.elevenlabs-scribe_v2.srt
raw/<video-stem>.elevenlabs-scribe_v2.chunk-000.json
```

OpenAI Whisper outputs omit the provider name:

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

Preprocessing samples evidence frames every `1s` by default. Frames are
downscaled to `480px` wide, and the overview contact sheet uses smaller `180px`
cells with 12 columns, so 4K source videos do not produce 4K prompt images. The
preprocessor also writes labeled 60-second window sheets in `sheets/window-*.jpg`;
each cell is stamped with `frame_index timecode`, and the manifest maps every
cell back to the source timestamp.

`--ffmpeg-bin auto` prefers `/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg` when it
exists so the sheet generator can use `drawtext`. On macOS,
`--ffmpeg-hwaccel auto` uses VideoToolbox hardware decode when the selected
ffmpeg binary supports it.

## Self-Contained Pipeline

For system tests, use the single-entry pipeline rather than manually assembling
intermediate artifacts. The only required input is the video file:

```bash
extract-clips /path/to/video.mp4 --approve-api-cost 25.00
```

Run self-contained excerpt mode instead of clip-ad mode:

```bash
extract-clips /path/to/video.mp4 \
  --content-mode excerpt \
  --approve-api-cost 25.00
```

Excerpt mode creates self-contained extracts rather than ads for the full video.
It can select up to one extract per started 10 minutes of source video, but that
is an upper bound rather than a quota. The pipeline should first choose the
smallest coherent semantic unit: sometimes a single atomic moment is enough, and
sometimes the right output is a longer chapter extract whose setup, complication,
reveal, and explanation only make sense together. Each extract renders with a
neutral opening title card plus a final fade-out.

Use OpenAI Whisper for the transcription stage instead:

```bash
extract-clips /path/to/video.mp4 \
  --transcribe-provider openai \
  --transcribe-model whisper-1 \
  --approve-api-cost 25.00
```

The wrapper creates one artifact directory per input file, next to the source
video and named after the input file stem. That directory holds transcripts,
local evidence, visual indexes, renders, OpenAI run outputs, and the local
response cache at `openai-runs/_response-cache`. Re-running the same input
resumes from that directory unless `--force` or a distinct `--run-id` is used.

It runs:

```text
video -> provider transcript -> local evidence -> outline -> strategy -> moments
-> dense visual index from moments candidates -> plan
-> local subject index -> draft render -> critique -> revise
-> local subject index -> final render
```

It performs a dry-run cost estimate before each API stage and spends from the
single `--approve-api-cost` cap. If the cap is omitted, the pipeline stops before
the first uncached API call and prints the required estimate. Local response
caches are still honored by the individual OpenAI stage commands.

`--frame-interval-seconds` controls both preprocessing evidence frames and the
dense visual index. By default the pipeline samples local frames every `1s`,
stores downscaled frame JPEGs at `480px` wide, and builds the preprocessing
overview contact sheet with `180px` cells. The pipeline also forwards
`--ffmpeg-bin auto` and `--ffmpeg-hwaccel auto` into local frame extraction,
visual indexing, and rendering, which resolves to ffmpeg-full plus VideoToolbox
on supported Macs.

If an input MP4 is truncated, the pipeline keeps the decodable prefix instead of
failing or planning against misleading container metadata. Audio extraction
records the actual extracted duration, preprocessing records both container and
effective decoded duration, and downstream OpenAI stages receive a transcript
clipped to the usable video duration.

### Run Directory Layout

`extract-clips` creates one run root beside the input video (or at
`--output-root`), named after the input file stem:

```text
<video-stem>/
  pipeline-manifest.json          # stage outputs, costs, cache settings for this run
  transcripts/
    <stem>.<provider>-<model>.json / .srt
    <stem>.<provider>-<model>.available.json   # present only for truncated inputs
  video-evidence/<stem>/
    manifest.json  manifest.md  contact-sheet.jpg  frames/  sheets/
  visual-index/<stem>/            # 1 fps dense index from moments candidates
    manifest.json  frames/  sheets/
  subject-index/
    <base>-plan/manifest.json     # subject/crop analysis for the draft render
    <base>-revise/manifest.json   # subject/crop analysis for the final render
  openai-runs/
    _response-cache/              # local response cache shared by all stages
    _critique-assets/<base>-critique/   # contact sheets, transition strips, waveforms
    <base>-outline/   outline-output.json  outline.md   # source content map for editors
    <base>-strategy/  strategy-output.json
    <base>-moments/   moments-output.json
    <base>-plan/      plan-output.json
    <base>-critique/  critique-output.json  critique-package.json
    <base>-revise/    revise-output.json  plan-output.json   # revised plan
  review/
    draft/                        # human edit list for the draft plan
      edit-list.toml  edit-list.html  plan-output.human.json
    final/                        # human edit list for the revised plan
  renders/
    [excerpt/]draft/              # ad mode omits the content-mode directory
      <clip-id>.draft.mp4 / .draft.srt / .draft.subtitled.mp4
      render-manifest.json
    [excerpt/]final/              # final deliverables, no stage word in the name
      <clip-id>.mp4 / .srt / .subtitled.mp4
      render-manifest.json
    [excerpt/]human/              # renders of a human-edited plan
```

`<base>` is the input stem, plus the content mode when it is not `ad`, plus the
`--run-id` suffix when one is given (for example `my-video-excerpt-strategy`).
With `--run-id`, the render directories also gain the variant prefix
(`renders/[excerpt/]<run-id>-draft`), so variants never overwrite each other.
Each `openai-runs/<base>-<stage>/` directory also records the exact request:
`request-manifest.json`, `source-materials.txt`, `task-prompt.txt`,
`raw-response.json`, `usage-and-cost.json`, and a human-readable `summary.md`.

### Stages At A Glance

| Stage | Command | Consumes | Produces |
|---|---|---|---|
| transcribe | `transcribe-video` | video | normalized transcript JSON + SRT |
| preprocess | `preprocess-video` | video + transcript | evidence manifest, contact sheet, frames |
| outline | `openai-outline` | transcript | `outline-output.json` + `outline.md` (source content map) |
| strategy | `openai-strategy` | transcript, evidence manifest, contact sheet | `strategy-output.json` |
| moments | `openai-moments` | the above + strategy output | `moments-output.json` |
| visual-index | `visual-index` (local) | video + moments output | dense frame index + candidate sheets |
| plan | `openai-plan` | the above + moments output, visual index | `plan-output.json` (master edit plan) |
| subject-index | `subject-index` (local) | video + plan | subject/crop analysis manifest |
| draft-render | `render-plan` (local) | video + plan + subject index | draft MP4s, SRT, `render-manifest.json` |
| critique | `openai-critique` | render manifest + plan | `critique-output.json`, `critique-package.json` |
| revise | `openai-revise` | plan + critique + package + render manifest | revised `plan-output.json` |
| final-render | `render-plan` (local) | video + revised plan + subject index | final MP4s + manifest |

The plan and revise stages emit the same plan format; the renderer does not
care which one produced its input. Only the `openai-*` stages call an API;
everything else is local ffmpeg/ffprobe work.

## Caching, Cost Approval, And Resume

Three separate mechanisms reduce repeat work; they are easy to conflate.

**Local response cache.** Every OpenAI stage hashes its full request into a
cache key and stores the raw response under the cache directory
(`openai-runs/_response-cache` for pipeline runs, `<STAGE>_CACHE_DIR` for
standalone commands). A repeat request replays the cached response with no API
call and no approval flag. `OPENAI_RESPONSE_CACHE_MODE` (or `--cache-mode`)
selects the identity mode: `strict` keys on the LLM query hash plus hashes of
the stage source files and prompt templates that produced it, so any code or
prompt change is a cache miss; `query-only` keys on the query alone, which
keeps cache hits across code-only changes; `off` disables the cache.
`--refresh-cache` forces a fresh API call (and therefore requires
`--approve-api-cost`); `--seed-cache-from-run` imports an existing run
directory into the cache without an API call.

**OpenAI server-side prompt caching.** The strategy, moments, and plan stages
share one stable prompt prefix (transcript + video evidence) and send a shared
`prompt_cache_key` (prefix `strategy-`), so the provider can reuse the cached
prefix across consecutive stages. Critique and revise use their own keys
(`critique-`, `revise-`). This affects provider-side token pricing only, not
local behavior.

**Pipeline resume.** `extract-clips` skips any stage whose expected output file
already exists, so re-running the same input resumes where it stopped. Use
`--force` to regenerate local artifacts, or a distinct `--run-id` to build a
parallel variant without disturbing earlier outputs. `pipeline-manifest.json`
records each stage's estimated and actual cost for the run.

**Cost approval.** No command makes an API call without an explicit cap. Every
API stage supports `--dry-run` (transcription uses `--estimate-cost`) to print
an upper-bound estimate; pass `--approve-api-cost <USD>` at or above it.
`extract-clips` takes one total cap and spends from it across all stages,
estimating before each call; with no cap it stops at the first uncached API
call and prints the required estimate.

## Dense Visual Indexing

For object-level clip planning, build a local 1 fps visual index. This is useful
when a clip hook depends on a concrete visible object or action, such as a hammer,
spacesuit glove, tool strike, or underwater suit movement. The index stays local
and makes no API calls.

```bash
PYTHONPATH=src python -m extractor.visual_index \
  /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --moments-output /Volumes/t9/extract/openai-runs/space-school-moments-gpt-5.5-high/moments-output.json \
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

Candidate sheets are built from moments-stage windows with padding. They are intended
as visual-proof packages: the plan stage should cite exact sheet timestamps for named
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

## Prompt Templates

OpenAI stage prompts are stored as editable Markdown templates outside the Python
modules:

```text
prompts/extractor/
```

The stage code loads those files and injects runtime JSON such as stage outputs,
render manifests, and review packages. Prompt-template changes are included in
the local response-cache source hash.

Set `CLIPINATOR_PROMPT_DIR` to point at an alternate prompt directory with the
same filenames.

## OpenAI Outline Stage: Source Content Map

The outline stage labels every logical content unit of the source (the same
transcript segments the storyboard review UX uses) with a short title and a
one-sentence gist, so a human editor can see what building blocks the source
offers without reading the transcript or scrubbing the video. It runs early in
the pipeline, right after transcription.

```bash
PYTHONPATH=src python -m extractor.openai_outline \
  --transcript /Volumes/t9/extract/transcripts/example.elevenlabs-scribe_v2.json \
  --approve-api-cost 5.00
```

It labels units in batched calls and caches the result to disk keyed by the
transcript, so a repeat run is free. Outputs under `OUTLINE_OUTPUT_DIR`:

```text
outline-output.json   # units: [{index, start, end, title, summary}]
outline.md            # a scannable outline an editor can read top-to-bottom
request-manifest.json  raw-response.json  usage-and-cost.json
```

The review editor's Storyboard tab reads `outline-output.json` and shows each
unit's title on the timeline, with the one-line gist on hover. Per-stage env
config lives under the `OUTLINE_*` family (model, effort, pricing), falling back
to `STRATEGY_*`.

## OpenAI Strategy Stage: Video Brief And Ad Angles

Run the first LLM stage (`openai-strategy`) against the real OpenAI API. This consolidates the full
video overview, why-to-watch summary, visual inventory, and proposed clip-ad
angles into one structured output. It sends a compact transcript timeline, the
local preprocessing manifest, and optionally the contact sheet image.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_strategy \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --dry-run
```

Run with an explicit approved cap:

```bash
PYTHONPATH=src python -m extractor.openai_strategy \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --approve-api-cost 5.00
```

The command refuses non-dry-run API calls unless `--approve-api-cost` is
provided. The command also computes a deterministic local response-cache key
from the model, prompt, schema, transcript, manifest, contact sheet bytes, image
detail, reasoning effort, and output cap. When the same request is already in
`STRATEGY_CACHE_DIR`, it reuses the cached response locally and makes no API call,
so no approval flag is needed.

To bypass a cached response intentionally, pass `--refresh-cache`; that path is
treated as a fresh API call and still requires `--approve-api-cost`.

Seed the cache from an existing run without calling the API:

```bash
PYTHONPATH=src python -m extractor.openai_strategy \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --seed-cache-from-run /Volumes/t9/extract/openai-runs/space-school-strategy-gpt-5.5-high
```

Outputs are written under `STRATEGY_OUTPUT_DIR` and include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
raw-response.json
strategy-output.json
usage-and-cost.json
summary.md
```

## OpenAI Moments Stage: Transcript Moment Analysis

Run the second LLM stage (`openai-moments`) after the strategy stage has produced
a video brief and ad-angle hypotheses. This stage reuses the same stable video
evidence prefix and adds the strategy output, then asks for concrete timestamped transcript windows for each
ad angle. It still uses the local response cache and the explicit API approval
gate.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_moments \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --strategy-output /Volumes/t9/extract/openai-runs/space-school-strategy-gpt-5.5-high/strategy-output.json \
  --dry-run
```

Run with an explicit approved cap:

```bash
PYTHONPATH=src python -m extractor.openai_moments \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --strategy-output /Volumes/t9/extract/openai-runs/space-school-strategy-gpt-5.5-high/strategy-output.json \
  --approve-api-cost 1.00
```

Outputs include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
raw-response.json
moments-output.json
usage-and-cost.json
summary.md
```

## OpenAI Plan Stage: Edit Plans

Run the third LLM stage (`openai-plan`) after the moments stage has produced
ranked candidate moments.
This stage creates concrete editor-facing cut plans for the strongest short-form
clip ads: source ranges, opening hook, audio/video strategy, B-roll coverage,
caption/overlay notes, verification steps, and rendering assumptions. It does
not render video.

The plan stage treats the payoff as the hook by default: title/tagline card first,
strongest payoff line as the first spoken beat, then context. It should avoid
silent source-video lead-ins unless the visual action itself is the hook.
Candidates without a clear payoff should be deferred instead of turned into edit
plans.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_plan \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --visual-index-manifest /Volumes/t9/extract/visual-index/nasa-explorers-space-school/manifest.json \
  --strategy-output /Volumes/t9/extract/openai-runs/space-school-strategy-gpt-5.5-high/strategy-output.json \
  --moments-output /Volumes/t9/extract/openai-runs/space-school-moments-gpt-5.5-high/moments-output.json \
  --dry-run
```

Run with an explicit approved cap:

```bash
PYTHONPATH=src python -m extractor.openai_plan \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --visual-index-manifest /Volumes/t9/extract/visual-index/nasa-explorers-space-school/manifest.json \
  --strategy-output /Volumes/t9/extract/openai-runs/space-school-strategy-gpt-5.5-high/strategy-output.json \
  --moments-output /Volumes/t9/extract/openai-runs/space-school-moments-gpt-5.5-high/moments-output.json \
  --approve-api-cost 8.00
```

Outputs include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
raw-response.json
plan-output.json
usage-and-cost.json
summary.md
```

## Local Draft Rendering

Render local draft cuts from an edit plan. This uses only `ffmpeg`; it
does not call any API. The renderer uses the plan's timeline source ranges,
separates dialogue audio from visual B-roll when the edit plan provides usable
visual time cues, formats drafts as vertical `9:16`, writes sidecar SRT
captions, and muxes a subtitle-track version when possible. If the installed
`ffmpeg` lacks `drawtext`, captions are not burned into the video.

```bash
PYTHONPATH=src python -m extractor.render_plan \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --plan-output /Volumes/t9/extract/openai-runs/space-school-plan-gpt-5.5-high/plan-output.json \
  --output-dir /Volumes/t9/extract/renders/space-school-plan-drafts
```

If a planned silent cold open creates dead air, repair that clip by borrowing
the beginning of the following narration beat under the opening visual:

```bash
PYTHONPATH=src python -m extractor.render_plan \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --plan-output /Volumes/t9/extract/openai-runs/space-school-plan-gpt-5.5-high/plan-output.json \
  --output-dir /Volumes/t9/extract/renders/space-school-plan-final \
  --repair-silent-lead-clip clip_ad_02_moon_hammer
```

For a cleaner silent opening, replace the first silent beat with a baked-in
black title card. On macOS the renderer uses `tools/render_title_card.swift`
for crisp system text, with a dependency-free bitmap fallback. The renderer does
this automatically when the first silent timeline beat is described as a title
card; `--black-title-card-clip` forces it for plans whose first silent beat is
not described as a title card:

```bash
PYTHONPATH=src python -m extractor.render_plan \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --plan-output /Volumes/t9/extract/renders/moon-hammer-alternatives/moon-hammer-alternatives-plan.json \
  --output-dir /Volumes/t9/extract/renders/moon-hammer-alternatives/renders-black-title \
  --black-title-card-clip clip_ad_02a_moon_hammer_geology_setup \
  --black-title-card-clip clip_ad_02b_moon_hammer_payoff_first
```

Use `--transition-duration 0.18` to render short audio/video crossfades
between planned segments when hard cuts feel too abrupt.

For subject-aware vertical crops, build a local subject index first and pass it
to the renderer. The index samples the planned source spans, identifies the
primary speaker/subject and optional secondary content, then caps and centers
foreground punch-ins so the speaker is not pushed half off-screen:

```bash
PYTHONPATH=src python -m extractor.subject_index /Volumes/t9/extract/source.mp4 \
  --plan-output /Volumes/t9/extract/openai-runs/example-plan/plan-output.json \
  --output-dir /Volumes/t9/extract/subject-index \
  --run-id example-plan

PYTHONPATH=src python -m extractor.render_plan \
  --source-video /Volumes/t9/extract/source.mp4 \
  --plan-output /Volumes/t9/extract/openai-runs/example-plan/plan-output.json \
  --subject-index-manifest /Volumes/t9/extract/subject-index/example-plan/manifest.json \
  --output-dir /Volumes/t9/extract/renders/example-plan \
  --use-plan-render-settings \
  --auto-foreground-scale
```

Outputs include one `.draft.mp4`, one `.draft.srt`, and one
`.draft.subtitled.mp4` per clip, plus `render-manifest.json`. The `.draft`
infix is controlled by `--output-label` (default `draft`); the pipeline passes
an empty label for the final render, so final deliverables are named
`<clip-id>.mp4` / `.srt` / `.subtitled.mp4`.

### Frame-Accurate Timing

Plans carry times as float seconds, but cuts land on discrete frames. At
ingest, the renderer and the subject indexer snap every plan time onto an
exact frame grid (`extractor/timebase.py`): source-addressing times (timeline
`source_start`/`source_end`, `source_ranges`, manual-review ranges) snap to
the source video's native frame rate probed from the file (for example
`24000/1001`), while render-clock times (screen times, caption offsets,
durations) snap to the renderer's 30 fps output grid. Snapping uses exact
rational arithmetic, so repeated runs are stable, and `render-manifest.json`
records the applied rates and how many values moved under `timebase`.

### Augmentation assets

The plan stage and the renderer can experiment with related local assets for excerpt
cutaways, graphics, and quiet audio beds. The assets must already be local and
rights-reviewed; the renderer does not download web media or verify licenses.

Create a JSON catalog:

```json
{
  "assets": [
    {
      "id": "infinity-stars",
      "type": "video",
      "path": "assets/infinity-stars.mp4",
      "license": "CC0",
      "source_url": "https://example.org/source",
      "description": "Slow abstract star field for moments about mathematical infinity.",
      "tags": ["infinity", "cosmos", "abstract"],
      "usage_notes": "Use only as metaphorical cutaway; do not imply the lecture is about astronomy."
    },
    {
      "id": "soft-pad",
      "type": "audio",
      "path": "assets/soft-pad.m4a",
      "license": "locally generated",
      "volume": 0.12,
      "description": "Subtle neutral audio bed for title cards or quiet reflection."
    }
  ]
}
```

Pass it to the plan stage so the model can plan with the catalog:

```bash
PYTHONPATH=src python -m extractor.openai_plan \
  --transcript /Volumes/t9/extract/transcripts/example.elevenlabs-scribe_v2.json \
  --manifest /Volumes/t9/extract/video-evidence/example/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/example/contact-sheet.jpg \
  --strategy-output /Volumes/t9/extract/openai-runs/example-strategy/strategy-output.json \
  --moments-output /Volumes/t9/extract/openai-runs/example-moments/moments-output.json \
  --content-mode excerpt \
  --augmentation-assets /Volumes/t9/extract/assets/augmentation-assets.json \
  --approve-api-cost 4.00
```

A timeline beat can reference a visual asset by putting `asset://id` or
`asset://id#t=12.5` in `visual_source`. Source dialogue still comes from
`source_start`/`source_end`, so the asset becomes a visual cutaway over the
lecture audio. A beat can also mention `asset://id` in `audio_source` as a
quiet bed; source dialogue remains primary unless `audio_source` starts with
the asset URI, in which case the asset is used as that segment's audio.

The plan stage can also emit planned media requests when the useful cutaway, audio, or
graphic is not already local. Those appear in top-level `media_requests`, and
clips list the relevant `media_request_ids`. Third-party requests include a
search query, suggested source types, license requirements, review notes, and a
rights-risk rating. Generated requests include a source-faithful prompt,
negative prompt, duration/aspect-ratio notes, style notes, and placement
instructions. `media_request://id` is only a planning placeholder; final renders
need that media reviewed or generated, saved locally, and added to the catalog
as a real `asset://id`. The renderer records unresolved request IDs in
`render-manifest.json` but falls back to source visuals/audio for the draft.

Render with the same catalog:

```bash
PYTHONPATH=src python -m extractor.render_plan \
  --source-video /Volumes/t9/extract/example.mp4 \
  --plan-output /Volumes/t9/extract/openai-runs/example-plan/plan-output.json \
  --transcript /Volumes/t9/extract/transcripts/example.elevenlabs-scribe_v2.json \
  --output-dir /Volumes/t9/extract/renders/example-augmented \
  --use-plan-render-settings \
  --auto-foreground-scale \
  --burn-captions \
  --augmentation-assets /Volumes/t9/extract/assets/augmentation-assets.json
```

`run_pipeline` also accepts `--augmentation-assets` and forwards it to the plan
stage and both render passes.

The revise stage can put reusable rendering defaults in the plan itself. To use those
settings while still allowing CLI overrides, pass:

```bash
PYTHONPATH=src python -m extractor.render_plan \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --plan-output /Volumes/t9/extract/openai-runs/<revise-run>/plan-output.json \
  --output-dir /Volumes/t9/extract/renders/<revise-render> \
  --use-plan-render-settings
```

## Human Review: Edit List Export And Import

Between the draft render and the final render, a human can adjust the proposed
clips without touching the plan JSON. `edit-list export` writes two files into
`review/draft/` or `review/final/` in the run root, depending on which plan is
being reviewed:

```bash
edit-list export \
  --plan-output <run>/openai-runs/<base>-plan/plan-output.json \
  --render-manifest <run>/renders/excerpt/draft/render-manifest.json
```

- `edit-list.toml` - the editable artifact. One block per clip with
  `decision` (`keep` / `drop` / `rework`), `title`, `title_card_seconds`,
  `notes`, the source `segments` (in/out points in source-video seconds,
  annotated with the transcript words at each boundary), and the `captions`.
- `edit-list.html` - a read-only, self-contained review page: the source
  video and each draft preview play inline, and every transcript word is
  clickable to seek the source player, with a millisecond clock under each
  player for copying timestamps into the TOML.

Edit the TOML in any text editor, then apply it:

```bash
edit-list import <run>/review/draft/edit-list.toml
```

This validates the edits (ordered, non-overlapping segments inside the source
duration; known clip ids; non-empty titles) and writes
`plan-output.human.json` - a normal plan-format file the renderer consumes
directly:

```bash
render-plan --source-video <video> \
  --plan-output <run>/review/draft/plan-output.human.json \
  --transcript <run>/transcripts/<stem>.<provider>-<model>.json \
  --output-dir <run>/renders/excerpt/human
```

Mechanics worth knowing:

- All paths embedded in the TOML and the HTML are relative to those files, so
  a run directory is relocatable as a unit: it can move to another disk or
  remount under a different volume prefix and the edit list keeps working. If
  a recorded path is stale anyway (for example an absolute path from an old
  export), `import` relinks it by file name inside the run root, the same way
  an NLE relinks media, and says so.
- Caption entries are anchored to source time, not render time, so caption
  text edits survive trims; on import they are clamped onto the kept segments
  and re-timed, and captions whose span was trimmed away go dormant with a
  warning instead of failing. Delete a clip's `[[clips.captions]]` entries to
  regenerate its captions from the word-level transcript.
- When a clip's segments are unchanged (within 75 ms), its original timeline
  entries are preserved, including B-roll and asset cutaway instructions;
  larger changes rebuild plain source-playthrough entries.
- `decision = "rework"` keeps the clip and records `notes` in the output plan
  under `human_review_notes` for a later revision pass.
- The output plan records provenance under `human_edit` (edit-list path,
  content hash, applied time, dropped/rework clip ids).
- Clips missing from the TOML are kept unchanged with a warning; dropping a
  clip is always an explicit `decision = "drop"`.

### OpenTimelineIO / NLE interchange

For editors who work in a non-linear editor, the same review gate is also
available as OpenTimelineIO. `otio-edl export` writes the run's `review/`
directory:

```bash
otio-edl export \
  --plan-output <run>/openai-runs/<base>-revise/plan-output.json \
  --render-manifest <run>/renders/excerpt/final/render-manifest.json
```

- `edit-list.otio` — the canonical interchange. Each excerpt is a video track
  of source-range clips plus a generated title card, with the clipinator
  decision/title/captions/notes payload in per-track metadata. Premiere Pro and
  DaVinci Resolve open `.otio` natively.
- `edit-list.edl` (CMX 3600) and `edit-list.fcpxml` — one-way cuts views via
  OTIO's adapters: a single flattened track of source segments, for editors on
  tools that prefer those formats. They carry cut geometry only — title cards,
  captions, and metadata are dropped by construction, so they are disposable
  views, not the edit list.

Import an edited `.otio` back into a plan-format file:

```bash
otio-edl import <run>/review/final/edit-list.otio
```

This reads the same edit-list overlay the TOML path uses (`clip_id` as the
merge key, decision/title/segments/captions) and funnels through the shared
apply step, so it validates identically and writes `plan-output.human.json`.
Times are frame-accurate (`RationalTime` at the source rate), so a plan →
`.otio` → plan round-trip preserves every segment to within one frame. The
`.otio` is the source of truth; an `.otio` re-exported from an NLE is read
best-effort (cut geometry by track/clip name; metadata an NLE dropped is
regenerated from the transcript).

## OpenAI Critique Stage: Post-Render Review

Run the fourth LLM stage (`openai-critique`) after local rendering. This stage critiques rendered
draft clips using the human-review rules from this project: payoff first, no
confusing mid-sentence opening, no mismatched talking head under narration, no
repeated B-roll padding, smooth cuts, and no ending that drifts into a different
story.

The rendered MP4 files stay local. The stage sends local evidence images:
storyboard contact sheets sampled from the rendered clips, transition strips
around segment boundaries and internal multi-cut B-roll edits, and audio waveform timelines. The prompt also includes the
render manifest, SRT captions, local silence/caption timing diagnostics,
source-range overlap warnings, and the optional edit plan.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_critique \
  --render-manifest /Volumes/t9/extract/renders/space-school-underwater-fixed/renders-tight/render-manifest.json \
  --plan-output /Volumes/t9/extract/renders/space-school-underwater-fixed/space-school-underwater-tight-plan.json \
  --dry-run
```

Run with an explicit approved cap:

```bash
PYTHONPATH=src python -m extractor.openai_critique \
  --render-manifest /Volumes/t9/extract/renders/space-school-underwater-fixed/renders-tight/render-manifest.json \
  --plan-output /Volumes/t9/extract/renders/space-school-underwater-fixed/space-school-underwater-tight-plan.json \
  --approve-api-cost 1.00
```

Review only one clip from a multi-clip render:

```bash
PYTHONPATH=src python -m extractor.openai_critique \
  --render-manifest /Volumes/t9/extract/renders/space-school-payoff-first-other-clips/renders/render-manifest.json \
  --plan-output /Volumes/t9/extract/renders/space-school-payoff-first-other-clips/space-school-other-clips-payoff-first-plan.json \
  --clip-id clip_ad_03_personal_spacecraft_payoff_first \
  --dry-run
```

Outputs include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
critique-package.json
raw-response.json
critique-output.json
usage-and-cost.json
summary.md
```

Local review assets are written under `_critique-assets/` and include one
`review-contact-sheet.jpg`, `transition-contact-sheet.jpg`, and
`audio-waveform.jpg` per reviewed clip when the default evidence layers are
enabled. Use `--no-transition-strips` or `--no-waveform` for a cheaper
storyboard-only review.

## OpenAI Revise Stage: Critique-Driven Re-Edit

Run the fifth LLM stage (`openai-revise`) after the critique stage has reviewed
a render. This stage turns the critique back into a full revised plan-compatible
edit plan. It is not a
manual clip patch: the prompt asks the model to apply general rules such as
payoff-first structure, no dead source video before audio, no mid-word endings,
no mismatched talking head under narration, no repeated B-roll padding, and
topic-consistent endings.

Dry-run the estimated API cost without making a call:

```bash
PYTHONPATH=src python -m extractor.openai_revise \
  --plan-output /Volumes/t9/extract/renders/moon-hammer-polish/moon-hammer-polish-plan-v2.json \
  --render-manifest /Volumes/t9/extract/renders/moon-hammer-polish/renders-v2/render-manifest.json \
  --critique-output /Volumes/t9/extract/openai-runs/renders-v2.critique.gpt-5.5.high.20260430T131557Z/critique-output.json \
  --critique-package /Volumes/t9/extract/openai-runs/renders-v2.critique.gpt-5.5.high.20260430T131557Z/critique-package.json \
  --dry-run
```

Include the full transcript/video evidence prefix when the re-edit may need new
sentence boundaries or alternate B-roll ranges:

```bash
PYTHONPATH=src python -m extractor.openai_revise \
  --transcript /Volumes/t9/extract/transcripts/nasa-explorers-space-school.whisper-1.json \
  --manifest /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/manifest.json \
  --contact-sheet /Volumes/t9/extract/video-evidence/nasa-explorers-space-school/contact-sheet.jpg \
  --plan-output /Volumes/t9/extract/renders/moon-hammer-polish/moon-hammer-polish-plan-v2.json \
  --render-manifest /Volumes/t9/extract/renders/moon-hammer-polish/renders-v2/render-manifest.json \
  --critique-output /Volumes/t9/extract/openai-runs/renders-v2.critique.gpt-5.5.high.20260430T131557Z/critique-output.json \
  --critique-package /Volumes/t9/extract/openai-runs/renders-v2.critique.gpt-5.5.high.20260430T131557Z/critique-package.json \
  --approve-api-cost 1.50
```

Outputs include:

```text
request-manifest.json
source-materials.txt
task-prompt.txt
raw-response.json
revise-output.json
plan-output.json
usage-and-cost.json
summary.md
```

Render the revised output with:

```bash
PYTHONPATH=src python -m extractor.render_plan \
  --source-video /Volumes/t9/extract/nasa-explorers-space-school.mp4 \
  --plan-output /Volumes/t9/extract/openai-runs/<revise-run>/plan-output.json \
  --output-dir /Volumes/t9/extract/renders/<revise-render> \
  --use-plan-render-settings
```

## Experimental Video API Probe

Probe whether provider APIs accept MP4 files as model inputs. This is separate
from the critique stage so failed provider experiments do not disturb the working
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
