from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
import tomllib
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.render_plan import transcript_words_from_data

# Human review gate: `export` turns a plan (plus transcript and optional draft
# render manifest) into an edit-list.toml a person can change in any text
# editor, with a read-only edit-list.html companion for watching previews and
# finding timestamps. `import` merges the edited TOML back over the plan and
# writes a plan-format JSON the existing renderer consumes unchanged.
#
# The TOML is an overlay, not a copy of the plan: it carries only the
# decisions a human makes (keep/drop/rework, title, segment in/out points,
# caption text, notes). Caption entries are anchored to SOURCE time so text
# edits survive later trims; `import` recomputes render-time offsets.

SCHEMA_VERSION = 1
DECISIONS = ("keep", "drop", "rework")
SEGMENT_MATCH_TOLERANCE = 0.075
# Minimum renderable segment length. Used both when extracting segments for
# export and when validating edited segments on import, so every segment the
# export emits passes import unchanged (no round-trip failure on short cuts).
MIN_SEGMENT_SECONDS = 0.05
CAPTION_OVERLAP_SECONDS = 0.2
CAPTION_MAX_CHARS = 42
CAPTION_MAX_WORDS = 7
CAPTION_GAP_SECONDS = 0.9
CAPTION_TAIL_SECONDS = 0.15
CONTEXT_WORDS = 7
TITLE_CARD_AUDIO = "No source dialogue; silent generated title card."
TITLE_CARD_VISUAL = "Generated black title card."


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def toml_string(value: str) -> str:
    # JSON string escaping is valid TOML basic-string escaping.
    return json.dumps(str(value), ensure_ascii=False)


def fmt_seconds(value: float) -> str:
    return f"{float(value):.3f}"


def is_title_card_item(item: dict[str, Any]) -> bool:
    text = f"{item.get('visual_source', '')} {item.get('audio_source', '')}".lower()
    try:
        span = float(item.get("source_end") or 0) - float(item.get("source_start") or 0)
    except (TypeError, ValueError):
        span = 0.0
    return "title card" in text and span <= 0.05


def sorted_timeline(clip: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(clip.get("timeline", []) or [], key=lambda item: int(item.get("order", 0) or 0))


def clip_segments(clip: dict[str, Any]) -> list[dict[str, Any]]:
    """Timeline entries that consume source time, in order."""
    segments: list[dict[str, Any]] = []
    for item in sorted_timeline(clip):
        if is_title_card_item(item):
            continue
        try:
            start = float(item["source_start"])
            end = float(item["source_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end - start <= MIN_SEGMENT_SECONDS:
            continue
        segments.append({"source_in": start, "source_out": end, "timeline_item": item})
    return segments


def title_card_seconds(clip: dict[str, Any]) -> float:
    for item in sorted_timeline(clip):
        if is_title_card_item(item):
            try:
                return max(
                    0.0,
                    float(item.get("screen_time_end") or 0)
                    - float(item.get("screen_time_start") or 0),
                )
            except (TypeError, ValueError):
                break
    hook = clip.get("opening_hook") or {}
    try:
        return max(0.0, float(hook.get("duration_seconds") or 0))
    except (TypeError, ValueError):
        return 0.0


def render_offset_to_source(offset: float, card: float, segments: list[dict[str, Any]]) -> float | None:
    cursor = card
    for segment in segments:
        duration = segment["source_out"] - segment["source_in"]
        if offset <= cursor + duration + 1e-6:
            position = min(max(offset - cursor, 0.0), duration)
            return segment["source_in"] + position
        cursor += duration
    # Offset is past the end of the assembled timeline; do not clamp it into the
    # last segment (that would silently mis-anchor an out-of-range caption).
    return None


def source_to_render_offset(t: float, card: float, segments: list[dict[str, Any]]) -> float | None:
    cursor = card
    for segment in segments:
        if segment["source_in"] - 1e-6 <= t <= segment["source_out"] + 1e-6:
            return cursor + max(0.0, t - segment["source_in"])
        cursor += segment["source_out"] - segment["source_in"]
    return None


def words_in_range(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [w for w in words if float(w["start"]) >= start - 1e-6 and float(w["end"]) <= end + 0.05]


def context_text(words: list[dict[str, Any]], t: float, *, before: bool) -> str:
    if before:
        chosen = [w for w in words if float(w["end"]) <= t + 0.05][-CONTEXT_WORDS:]
    else:
        chosen = [w for w in words if float(w["start"]) >= t - 0.05][:CONTEXT_WORDS]
    text = " ".join(str(w.get("text", "")).strip() for w in chosen).strip()
    if not text:
        return "(no transcript here)"
    return f"...{text}" if before else f"{text}..."


def chunk_captions(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group transcript words into caption lines anchored to source time."""
    captions: list[dict[str, Any]] = []
    line: list[dict[str, Any]] = []

    def flush() -> None:
        if not line:
            return
        captions.append(
            {
                "source_start": float(line[0]["start"]),
                "source_end": float(line[-1]["end"]) + CAPTION_TAIL_SECONDS,
                "text": " ".join(str(w.get("text", "")).strip() for w in line).strip(),
            }
        )
        line.clear()

    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        if line:
            chars = len(" ".join(str(w.get("text", "")).strip() for w in line)) + 1 + len(text)
            gap = float(word["start"]) - float(line[-1]["end"])
            if chars > CAPTION_MAX_CHARS or len(line) >= CAPTION_MAX_WORDS or gap > CAPTION_GAP_SECONDS:
                flush()
        line.append(word)
    flush()
    return captions


def export_captions(
    clip: dict[str, Any],
    card: float,
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Source-anchored caption lines shown in the edit list.

    Mirrors what the renderer will actually use: verbatim transcript words by
    default, except for a clip already marked human_edited_captions, whose
    existing caption_plan is preserved so re-export keeps prior edits.
    """
    if clip.get("human_edited_captions"):
        captions: list[dict[str, Any]] = []
        for item in clip.get("caption_plan", []) or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            try:
                start = float(item.get("start_offset", item.get("start")))
                end = float(item.get("end_offset", item.get("end")))
            except (TypeError, ValueError):
                continue
            if not text or end <= start or end <= card:
                continue
            source_start = render_offset_to_source(max(start, card), card, segments)
            source_end = render_offset_to_source(end, card, segments)
            if source_start is None or source_end is None or source_end <= source_start:
                continue
            captions.append({"source_start": source_start, "source_end": source_end, "text": text})
        if captions:
            return captions
    segment_words: list[dict[str, Any]] = []
    for segment in segments:
        segment_words.extend(words_in_range(words, segment["source_in"], segment["source_out"]))
    return chunk_captions(segment_words)


def relative_path(target: Path | None, base: Path) -> str | None:
    if target is None:
        return None
    try:
        return os.path.relpath(target, base)
    except ValueError:
        return str(target)


def find_run_root(start: Path) -> Path | None:
    """Nearest ancestor that looks like a pipeline run root."""
    for candidate in [start, *start.parents]:
        if (candidate / "openai-runs").is_dir() or (candidate / "pipeline-manifest.json").exists():
            return candidate
    return None


def review_kind(plan_path: Path, manifest_dir: Path | None) -> str:
    """Which plan is being reviewed: the draft plan or the revised (final) one."""
    names = [plan_path.parent.name.lower()]
    if manifest_dir is not None:
        names.append(manifest_dir.name.lower())
    for name in names:
        if "revise" in name or "stage5" in name or "final" in name:
            return "final"
    return "draft"


def resolve_recorded_path(
    raw: str,
    anchor_dir: Path,
    relink_candidates: list[Path],
) -> Path | None:
    """Resolve a path recorded in an edit list or render manifest.

    Relative paths resolve against the file that recorded them, making a run
    directory relocatable as a unit. Stale absolute prefixes (a remounted
    drive, a moved run) fall back to relinking by file name inside the run
    root, mirroring how NLEs relink media.
    """
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (anchor_dir / path).resolve()
    if path.exists():
        return path
    # Relink a stale absolute path by file name within the run root. Only relink
    # when exactly one candidate matches; refuse to guess between same-named
    # files in different locations.
    name = Path(raw).name
    matches = []
    seen: set[Path] = set()
    for base in relink_candidates:
        candidate = (base / name).resolve()
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            matches.append(candidate)
    if len(matches) == 1:
        print(f"note: relinked {name} -> {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous relink for {name!r}: {[str(m) for m in matches]}. "
            "Pass an explicit path to disambiguate."
        )
    return None


def render_manifest_clip(manifest: dict[str, Any] | None, clip_id: str) -> dict[str, Any] | None:
    for entry in (manifest or {}).get("clips", []) or []:
        if str(entry.get("clip_id")) == clip_id:
            return entry
    return None


# ---------------------------------------------------------------------------
# Export


def build_edit_list_toml(
    plan: dict[str, Any],
    plan_path: Path,
    transcript_path: Path,
    source_video: Path,
    words: list[dict[str, Any]],
    render_manifest: dict[str, Any] | None,
    output_dir: Path,
) -> str:
    lines: list[str] = [
        "# clipinator edit list",
        f"# Generated {datetime.now(UTC).isoformat(timespec='seconds')} from {plan_path}",
        "#",
        "# Edit this file, then apply it with:",
        "#   edit-list import <this file>",
        "#",
        '# decision: "keep" renders the clip as specified below, "drop" removes it,',
        '#           "rework" keeps it and records notes for a later revision pass.',
        "# Times are seconds in the SOURCE video; they snap to frame boundaries at",
        "# render time. Captions are anchored to source time, so caption text",
        "# edits survive trims. Delete the [[clips.captions]] entries of a clip to",
        "# regenerate its captions from the transcript on import.",
        "#",
        "# Paths below are relative to this file, so the whole run directory can",
        "# move (or remount under a different prefix) without breaking them.",
        "",
        f"schema_version = {SCHEMA_VERSION}",
        f"content_mode = {toml_string(str(plan.get('content_mode', 'ad')))}",
        f"source_video = {toml_string(relative_path(source_video, output_dir) or str(source_video))}",
        f"plan = {toml_string(relative_path(plan_path, output_dir) or str(plan_path))}",
        f"transcript = {toml_string(relative_path(transcript_path, output_dir) or str(transcript_path))}",
        "",
    ]
    for clip in plan.get("clip_ads", []) or []:
        clip_id = str(clip.get("clip_id", ""))
        segments = clip_segments(clip)
        card = title_card_seconds(clip)
        manifest_clip = render_manifest_clip(render_manifest, clip_id)
        lines.append("[[clips]]")
        lines.append(f"clip_id = {toml_string(clip_id)}")
        lines.append('decision = "keep"')
        lines.append(f"title = {toml_string(str(clip.get('concept_title', '')))}")
        lines.append(f"title_card_seconds = {fmt_seconds(card)}")
        lines.append('notes = ""')
        if manifest_clip and manifest_clip.get("output_path"):
            preview = relative_path(Path(str(manifest_clip["output_path"])), output_dir)
            duration = ((manifest_clip.get("probe") or {}).get("format") or {}).get("duration")
            suffix = f" ({float(duration):.2f}s)" if duration else ""
            lines.append(f"# preview: {preview}{suffix}")
        score = clip.get("self_contained_score")
        if score is not None:
            lines.append(f"# self_contained_score: {score}")
        lines.append("")
        for segment in segments:
            lines.append("  [[clips.segments]]")
            lines.append(
                f"  # in  > {context_text(words, segment['source_in'], before=False)}"
            )
            lines.append(
                f"  # out < {context_text(words, segment['source_out'], before=True)}"
            )
            lines.append(f"  source_in = {fmt_seconds(segment['source_in'])}")
            lines.append(f"  source_out = {fmt_seconds(segment['source_out'])}")
            lines.append("")
        for caption in export_captions(clip, card, segments, words):
            lines.append("  [[clips.captions]]")
            lines.append(f"  source_start = {fmt_seconds(caption['source_start'])}")
            lines.append(f"  source_end = {fmt_seconds(caption['source_end'])}")
            lines.append(f"  text = {toml_string(caption['text'])}")
            lines.append("")
    for deferred in plan.get("deferred_candidates", []) or []:
        lines.append(
            f"# deferred candidate {deferred.get('candidate_id', '?')}: {deferred.get('reason', '')}"
        )
    return "\n".join(lines).rstrip() + "\n"


HTML_STYLE = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem auto; max-width: 70rem;
       color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; margin-top: 2.5rem; }
.meta { color: #666; font-size: 0.85rem; }
.clip { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem;
        margin: 1.5rem 0; }
.players { display: flex; gap: 1.5rem; flex-wrap: wrap; }
video { max-width: 320px; max-height: 420px; background: #000; border-radius: 4px; }
.clock { font-variant-numeric: tabular-nums; font-size: 0.9rem; color: #444; margin-top: 0.25rem; }
table { border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }
td, th { border: 1px solid #ddd; padding: 0.3rem 0.6rem; text-align: left; }
.transcript { line-height: 1.9; margin-top: 0.75rem; font-size: 0.95rem; }
.w { cursor: pointer; padding: 0 1px; border-radius: 3px; }
.w:hover { background: #cde3ff; }
.w.inseg { background: #e4f3e2; }
.w.inseg:hover { background: #bfe6ba; }
.hint { color: #666; font-size: 0.85rem; }
code { background: #f0f0f0; padding: 0 4px; border-radius: 3px; }
"""

HTML_SCRIPT = """
function attachClock(video) {
  if (!video) return;
  var clock = document.getElementById(video.id + '-clock');
  if (!clock) return;
  video.addEventListener('timeupdate', function () {
    clock.textContent = video.currentTime.toFixed(3) + ' s';
  });
}
document.querySelectorAll('video').forEach(attachClock);
var source = document.getElementById('source-video');
document.querySelectorAll('.w').forEach(function (w) {
  w.addEventListener('click', function () {
    if (!source) return;
    source.currentTime = parseFloat(w.dataset.t);
    source.pause();
    source.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  });
});
"""


def build_review_html(
    plan: dict[str, Any],
    source_video: Path,
    words: list[dict[str, Any]],
    render_manifest: dict[str, Any] | None,
    output_dir: Path,
    edit_list_name: str,
) -> str:
    esc = html.escape
    source_rel = relative_path(source_video, output_dir)
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>Edit list review: {esc(str(plan.get('video_title', source_video.name)))}</title>",
        f"<style>{HTML_STYLE}</style></head><body>",
        f"<h1>Edit list review: {esc(str(plan.get('video_title', source_video.name)))}</h1>",
        f'<p class="meta">Mode: {esc(str(plan.get("content_mode", "ad")))} &middot; '
        f"Source: <code>{esc(str(source_video))}</code></p>",
        f'<p class="hint">This page is read-only. Make changes in <code>{esc(edit_list_name)}</code> '
        "and apply them with <code>edit-list import</code>. Click any transcript word to seek "
        "the source player to that moment; the clock under each player shows the timestamp "
        "to copy into the edit list.</p>",
        '<h2>Source video</h2><div class="players"><div>'
        f'<video id="source-video" src="{esc(source_rel or "")}" controls preload="metadata"></video>'
        '<div class="clock" id="source-video-clock">0.000 s</div></div></div>',
    ]
    for index, clip in enumerate(plan.get("clip_ads", []) or []):
        clip_id = str(clip.get("clip_id", ""))
        segments = clip_segments(clip)
        card = title_card_seconds(clip)
        manifest_clip = render_manifest_clip(render_manifest, clip_id)
        parts.append('<div class="clip">')
        parts.append(f"<h2>{esc(str(clip.get('concept_title', clip_id)))}</h2>")
        parts.append(
            f'<p class="meta"><code>{esc(clip_id)}</code> &middot; '
            f"title card {card:.1f}s &middot; "
            f"target {float(clip.get('target_duration_seconds') or 0):.1f}s</p>"
        )
        if manifest_clip and manifest_clip.get("output_path"):
            preview_rel = relative_path(Path(str(manifest_clip["output_path"])), output_dir)
            parts.append(
                f'<div class="players"><div><video id="clip-{index}" src="{esc(preview_rel or "")}" '
                'controls preload="metadata"></video>'
                f'<div class="clock" id="clip-{index}-clock">0.000 s</div></div></div>'
            )
        parts.append("<table><tr><th>#</th><th>source in</th><th>source out</th><th>duration</th></tr>")
        for seg_index, segment in enumerate(segments, start=1):
            parts.append(
                f"<tr><td>{seg_index}</td><td>{segment['source_in']:.3f}</td>"
                f"<td>{segment['source_out']:.3f}</td>"
                f"<td>{segment['source_out'] - segment['source_in']:.3f}</td></tr>"
            )
        parts.append("</table>")
        if segments:
            window_start = max(0.0, min(s["source_in"] for s in segments) - 10.0)
            window_end = max(s["source_out"] for s in segments) + 10.0
            spans: list[str] = []
            for word in words:
                start = float(word["start"])
                if start < window_start or start > window_end:
                    continue
                inside = any(
                    s["source_in"] - 1e-6 <= start <= s["source_out"] + 1e-6 for s in segments
                )
                css = "w inseg" if inside else "w"
                spans.append(
                    f'<span class="{css}" data-t="{start:.3f}" '
                    f'title="{start:.3f}s">{esc(str(word.get("text", "")))}</span>'
                )
            parts.append(
                '<div class="transcript">' + " ".join(spans) + "</div>"
                if spans
                else '<p class="hint">No transcript words near this clip.</p>'
            )
        parts.append("</div>")
    parts.append(f"<script>{HTML_SCRIPT}</script></body></html>")
    return "\n".join(parts) + "\n"


def run_export(args: argparse.Namespace) -> int:
    plan_path = args.plan_output.expanduser().resolve()
    plan = read_json(plan_path)
    render_manifest = None
    manifest_dir: Path | None = None
    if args.render_manifest:
        manifest_path = args.render_manifest.expanduser().resolve()
        render_manifest = read_json(manifest_path)
        manifest_dir = manifest_path.parent

    run_root = find_run_root(manifest_dir or plan_path.parent)

    source_video = args.source_video
    if source_video is None and render_manifest and render_manifest.get("source_video"):
        relink = [run_root.parent, run_root] if run_root else []
        source_video = resolve_recorded_path(
            str(render_manifest["source_video"]), manifest_dir or plan_path.parent, relink
        )
    if source_video is None:
        raise RuntimeError(
            "--source-video is required when the render manifest does not provide a "
            "resolvable path."
        )
    source_video = source_video.expanduser().resolve()

    transcript_path = args.transcript
    if transcript_path is None and render_manifest and render_manifest.get("transcript"):
        relink = [run_root / "transcripts"] if run_root else []
        transcript_path = resolve_recorded_path(
            str(render_manifest["transcript"]), manifest_dir or plan_path.parent, relink
        )
    if transcript_path is None:
        raise RuntimeError(
            "--transcript is required when the render manifest does not provide a "
            "resolvable path."
        )
    transcript_path = transcript_path.expanduser().resolve()
    words = transcript_words_from_data(read_json(transcript_path))

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    elif run_root is not None:
        output_dir = run_root / "review" / review_kind(plan_path, manifest_dir)
    else:
        output_dir = manifest_dir if manifest_dir else plan_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    toml_text = build_edit_list_toml(
        plan, plan_path, transcript_path, source_video, words, render_manifest, output_dir
    )
    tomllib.loads(toml_text)  # round-trip guard: never write unparseable TOML
    toml_path = output_dir / "edit-list.toml"
    toml_path.write_text(toml_text, encoding="utf-8")

    html_path = output_dir / "edit-list.html"
    html_path.write_text(
        build_review_html(plan, source_video, words, render_manifest, output_dir, toml_path.name),
        encoding="utf-8",
    )
    clip_count = len(plan.get("clip_ads", []) or [])
    print(f"Exported {clip_count} clip(s) for review:")
    print(f"  edit list: {toml_path}")
    print(f"  review page: {html_path}")
    print(f"Apply edits with: edit-list import {toml_path}")
    return 0


# ---------------------------------------------------------------------------
# Import


def validate_segments(clip_id: str, segments: list[dict[str, Any]], duration: float | None) -> None:
    if not segments:
        raise RuntimeError(f"{clip_id}: a kept clip needs at least one segment.")
    previous_out = None
    for index, segment in enumerate(segments, start=1):
        try:
            source_in = float(segment["source_in"])
            source_out = float(segment["source_out"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{clip_id}: segment {index} needs numeric source_in/source_out.") from exc
        if source_out - source_in < MIN_SEGMENT_SECONDS:
            raise RuntimeError(
                f"{clip_id}: segment {index} is {source_out - source_in:.3f}s; "
                f"minimum is {MIN_SEGMENT_SECONDS}s."
            )
        if source_in < 0:
            raise RuntimeError(f"{clip_id}: segment {index} starts before 0.")
        if duration is not None and source_out > duration + 0.05:
            raise RuntimeError(
                f"{clip_id}: segment {index} ends at {source_out:.3f}s but the source "
                f"video is {duration:.3f}s."
            )
        if previous_out is not None and source_in < previous_out - 1e-6:
            raise RuntimeError(f"{clip_id}: segments must be in order and non-overlapping.")
        previous_out = source_out


def build_caption_plan(
    anchored: list[dict[str, Any]],
    card: float,
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Map source-anchored captions onto kept segments as render-time offsets.

    Each caption is clamped onto the first kept segment it meaningfully overlaps;
    a caption whose span was trimmed away is reported as dormant rather than
    silently dropped or mis-anchored. Returns (caption_plan, dormant_count).
    """
    caption_plan: list[dict[str, Any]] = []
    dormant = 0
    for caption in anchored:
        overlap = next(
            (
                s
                for s in segments
                if min(caption["source_end"], s["source_out"])
                - max(caption["source_start"], s["source_in"])
                >= CAPTION_OVERLAP_SECONDS
            ),
            None,
        )
        if overlap is None:
            dormant += 1
            continue
        clamped_start = max(caption["source_start"], overlap["source_in"])
        clamped_end = min(caption["source_end"], overlap["source_out"])
        start_offset = source_to_render_offset(clamped_start, card, segments)
        end_offset = source_to_render_offset(clamped_end, card, segments)
        if start_offset is None or end_offset is None or end_offset <= start_offset:
            dormant += 1
            continue
        caption_plan.append(
            {
                "start_offset": round(start_offset, 3),
                "end_offset": round(float(end_offset), 3),
                "text": caption["text"],
                "style_note": "human edit list",
            }
        )
    return caption_plan, dormant


def rebuild_clip(
    plan_clip: dict[str, Any],
    edit: dict[str, Any],
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    clip = deepcopy(plan_clip)
    clip_id = str(clip.get("clip_id", ""))
    title = str(edit.get("title", clip.get("concept_title", ""))).strip()
    if not title:
        raise RuntimeError(f"{clip_id}: title must not be empty.")
    card = float(edit.get("title_card_seconds", title_card_seconds(clip)))
    if card < 0:
        raise RuntimeError(f"{clip_id}: title_card_seconds must be >= 0.")

    new_segments = [
        {"source_in": float(s["source_in"]), "source_out": float(s["source_out"])}
        for s in edit.get("segments", []) or []
    ]
    original_segments = clip_segments(clip)
    preserve = len(new_segments) == len(original_segments) and all(
        abs(new["source_in"] - old["source_in"]) <= SEGMENT_MATCH_TOLERANCE
        and abs(new["source_out"] - old["source_out"]) <= SEGMENT_MATCH_TOLERANCE
        for new, old in zip(new_segments, original_segments)
    )

    timeline: list[dict[str, Any]] = []
    order = 1
    cursor = 0.0
    original_card_item = next(
        (item for item in sorted_timeline(clip) if is_title_card_item(item)), None
    )
    if card > 0:
        card_item = deepcopy(original_card_item) if original_card_item else {
            "audio_source": TITLE_CARD_AUDIO,
            "visual_source": TITLE_CARD_VISUAL,
            "edit_action": "Show opening generated black title card.",
            "purpose": "Opening title card.",
        }
        card_item.update(
            {
                "order": order,
                "screen_time_start": 0.0,
                "screen_time_end": round(card, 3),
                "source_start": 0,
                "source_end": 0,
                "caption_or_overlay": title,
            }
        )
        timeline.append(card_item)
        order += 1
        cursor = card

    for index, segment in enumerate(new_segments):
        duration = segment["source_out"] - segment["source_in"]
        if preserve:
            item = deepcopy(original_segments[index]["timeline_item"])
        else:
            item = {
                "audio_source": (
                    f"Source dialogue {segment['source_in']:.3f}-{segment['source_out']:.3f}."
                ),
                "visual_source": (
                    f"Source video {segment['source_in']:.3f}-{segment['source_out']:.3f}."
                ),
                "edit_action": "Play source segment chosen in the human edit list.",
                "caption_or_overlay": "",
                "purpose": "Human-selected source segment.",
            }
        item.update(
            {
                "order": order,
                "screen_time_start": round(cursor, 3),
                "screen_time_end": round(cursor + duration, 3),
                "source_start": round(segment["source_in"], 3),
                "source_end": round(segment["source_out"], 3),
            }
        )
        timeline.append(item)
        order += 1
        cursor += duration

    # Verbatim transcript words are the renderer's default caption source. Seed
    # from the transcript so the edit list mirrors what renders, and only mark a
    # clip human_edited_captions when the text actually differs — untouched clips
    # keep the renderer's native verbatim path rather than freezing a re-chunk.
    norm_segments = [
        {"source_in": s["source_in"], "source_out": s["source_out"]} for s in new_segments
    ]
    segment_words: list[dict[str, Any]] = []
    for segment in new_segments:
        segment_words.extend(words_in_range(words, segment["source_in"], segment["source_out"]))
    transcript_seed = chunk_captions(segment_words)
    captions_edit = edit.get("captions")
    if captions_edit is None:
        anchored = transcript_seed
        human_edited = False
    else:
        anchored = [
            {
                "source_start": float(c["source_start"]),
                "source_end": float(c["source_end"]),
                "text": str(c.get("text", "")).strip(),
            }
            for c in captions_edit
            if str(c.get("text", "")).strip()
        ]
        human_edited = [c["text"] for c in anchored] != [c["text"] for c in transcript_seed]
    caption_plan, dormant = build_caption_plan(anchored, card, norm_segments)
    if not caption_plan and segment_words:
        print(f"  {clip_id}: no captions could be placed on the kept segments.")

    total = round(cursor, 3)
    clip["concept_title"] = title
    clip["target_duration_seconds"] = total
    hook = dict(clip.get("opening_hook") or {})
    hook["text"] = title
    hook["duration_seconds"] = round(card, 3)
    clip["opening_hook"] = hook
    clip["timeline"] = timeline
    clip["caption_plan"] = caption_plan
    # Only human-edited captions override the renderer's verbatim transcript
    # default (auto mode); otherwise clear the plan-caption flags so the LLM's
    # advisory caption_plan_source does not force paraphrased captions.
    if human_edited:
        clip["caption_plan_source"] = "render_time"
        clip["human_edited_captions"] = True
    else:
        clip["caption_plan_source"] = None
        clip["human_edited_captions"] = False
    clip["source_ranges"] = [
        {
            "start": round(s["source_in"], 3),
            "end": round(s["source_out"], 3),
            "role": "core",
            "reason": "human edit list",
        }
        for s in new_segments
    ]
    notes = str(edit.get("notes", "")).strip()
    if notes:
        clip["human_review_notes"] = notes
    if dormant:
        print(f"  {clip_id}: {dormant} caption(s) fell outside the kept segments and were dropped.")
    return clip


def apply_edits(
    plan: dict[str, Any],
    edits: dict[str, Any],
    words: list[dict[str, Any]],
    duration: float | None,
    *,
    source: str,
) -> dict[str, Any]:
    """Merge an edit-list overlay (TOML- or OTIO-derived) onto a plan.

    Shared by every edit-list serialization: validates decisions and segments,
    rebuilds kept clips, drops/reworks per decision, keeps clips absent from the
    overlay unchanged, and records provenance under ``human_edit``. Raises on
    any invalid edit before returning, so a caller never writes a partial plan.
    """
    plan_clips = {str(clip.get("clip_id")): clip for clip in plan.get("clip_ads", []) or []}
    edited_ids: set[str] = set()
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    reworked: list[str] = []
    for edit in edits.get("clips", []) or []:
        clip_id = str(edit.get("clip_id", ""))
        if clip_id not in plan_clips:
            raise RuntimeError(f"Edit list references unknown clip_id {clip_id!r}.")
        if clip_id in edited_ids:
            raise RuntimeError(f"Edit list contains clip_id {clip_id!r} twice.")
        edited_ids.add(clip_id)
        decision = str(edit.get("decision", "keep")).strip().lower()
        if decision not in DECISIONS:
            raise RuntimeError(
                f"{clip_id}: decision must be one of {', '.join(DECISIONS)}; got {decision!r}."
            )
        if decision == "drop":
            dropped.append(clip_id)
            continue
        validate_segments(clip_id, edit.get("segments", []) or [], duration)
        kept.append(rebuild_clip(plan_clips[clip_id], edit, words))
        if decision == "rework":
            reworked.append(clip_id)

    for clip_id, clip in plan_clips.items():
        if clip_id not in edited_ids:
            print(f"warning: {clip_id} is not in the edit list; keeping it unchanged.")
            kept.append(deepcopy(clip))

    revised = deepcopy(plan)
    revised["clip_ads"] = kept
    revised["human_edit"] = {
        "edit_list": source,
        "applied_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dropped_clip_ids": dropped,
        "rework_clip_ids": reworked,
    }
    return revised


def run_import(args: argparse.Namespace) -> int:
    edit_list_path = args.edit_list.expanduser().resolve()
    toml_text = edit_list_path.read_text(encoding="utf-8")
    edits = tomllib.loads(toml_text)
    if int(edits.get("schema_version", 0)) != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported edit list schema_version {edits.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}."
        )

    toml_dir = edit_list_path.parent
    run_root = find_run_root(toml_dir)

    if args.plan_output:
        plan_path: Path | None = args.plan_output.expanduser().resolve()
    else:
        raw_plan = str(edits.get("plan", ""))
        relink = (
            [run_root / "openai-runs" / Path(raw_plan).parent.name] if run_root else []
        )
        plan_path = resolve_recorded_path(raw_plan, toml_dir, relink)
    if plan_path is None or not plan_path.exists():
        raise RuntimeError(
            f"Plan not found: {edits.get('plan')!r}. Pass --plan-output to override."
        )
    plan = read_json(plan_path)

    if args.transcript:
        transcript_path: Path | None = args.transcript.expanduser().resolve()
    else:
        relink = [run_root / "transcripts"] if run_root else []
        transcript_path = resolve_recorded_path(str(edits.get("transcript", "")), toml_dir, relink)
    words: list[dict[str, Any]] = []
    if transcript_path and transcript_path.exists():
        words = transcript_words_from_data(read_json(transcript_path))
    else:
        print("warning: transcript not found; caption regeneration is unavailable.")

    relink = [run_root.parent, run_root] if run_root else []
    source_video = resolve_recorded_path(str(edits.get("source_video", "")), toml_dir, relink)
    duration: float | None = None
    if source_video is not None and source_video.exists():
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(source_video),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            try:
                duration = float(json.loads(result.stdout)["format"]["duration"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                duration = None
        if duration is None:
            print(
                f"WARNING: could not read duration from {source_video}; segment "
                "end-of-video bounds are NOT validated for this import."
            )
    else:
        print(
            f"WARNING: source video not found at {edits.get('source_video')!r}; "
            "segment end-of-video bounds are NOT validated for this import."
        )

    revised = apply_edits(plan, edits, words, duration, source=str(edit_list_path))
    revised["human_edit"]["edit_list_sha256"] = sha256_text(toml_text)
    kept = revised["clip_ads"]
    dropped = revised["human_edit"]["dropped_clip_ids"]
    reworked = revised["human_edit"]["rework_clip_ids"]

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else edit_list_path.parent / "plan-output.human.json"
    )
    write_json(output_path, revised)
    print(
        f"Applied edit list: {len(kept)} kept ({len(reworked)} marked rework), "
        f"{len(dropped)} dropped."
    )
    print(f"Wrote {output_path}")
    print("Render it with: render-plan --plan-output " + str(output_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a plan as a human-editable TOML edit list plus a static HTML "
            "review page, and import the edited list back into a plan-format JSON."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Write edit-list.toml and edit-list.html from a plan.")
    export.add_argument("--plan-output", type=Path, required=True, help="Plan JSON to review.")
    export.add_argument(
        "--render-manifest",
        type=Path,
        default=None,
        help="Draft render manifest; supplies preview paths, source video, and transcript.",
    )
    export.add_argument("--transcript", type=Path, default=None)
    export.add_argument("--source-video", type=Path, default=None)
    export.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the render-manifest directory, else the plan directory.",
    )
    export.set_defaults(func=run_export)

    apply = sub.add_parser("import", help="Apply an edited edit-list.toml to its plan.")
    apply.add_argument("edit_list", type=Path, help="Path to the edited edit-list.toml.")
    apply.add_argument("--plan-output", type=Path, default=None, help="Override the plan path recorded in the edit list.")
    apply.add_argument("--transcript", type=Path, default=None, help="Override the transcript path recorded in the edit list.")
    apply.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plan path. Defaults to plan-output.human.json beside the edit list.",
    )
    apply.set_defaults(func=run_import)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # surface a clean message; stack traces help nobody here
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
