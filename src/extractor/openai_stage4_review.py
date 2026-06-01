from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor import stage_cache
from extractor.openai_stage1 import (
    CONTENT_MODE_CHOICES,
    DEFAULT_CACHE_SUBDIR,
    DEFAULT_ENV_FILE,
    DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION,
    DEFAULT_GPT55_INPUT_USD_PER_MILLION,
    DEFAULT_GPT55_OUTPUT_USD_PER_MILLION,
    DEFAULT_IMAGE_DETAIL,
    DEFAULT_IMAGE_TOKEN_ALLOWANCE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REASONING_EFFORT,
    Pricing,
    actual_cost,
    cache_entry_dir,
    cache_entry_exists,
    cost_from_counts,
    estimate_text_tokens,
    image_data_url,
    read_json,
    response_output_text,
    sha256_file,
    sha256_text,
    stable_json,
    usage_dict,
    write_json,
)
from extractor.prompt_templates import (
    load_prompt_lines,
    load_prompt_template,
    prompt_template_path,
    render_prompt_template,
)
from extractor.render_stage3 import ffprobe_summary, require_command, run_command, safe_name
from extractor.transcribe import env_float, env_int, env_value, load_env_file


DEFAULT_MAX_OUTPUT_TOKENS = 12_000
DEFAULT_FRAMES_PER_CLIP = 14
DEFAULT_CONTACT_SHEET_COLUMNS = 3
DEFAULT_TRANSITION_PROBE_SECONDS = 0.18
DEFAULT_TRANSITION_STRIP_FRAMES = 7
DEFAULT_TRANSITION_WINDOW_SECONDS = 0.48
DEFAULT_WAVEFORM_WIDTH = 1200
DEFAULT_WAVEFORM_HEIGHT = 220
DEFAULT_SILENCE_NOISE_DB = -35.0
DEFAULT_SILENCE_MIN_DURATION = 0.15


def cache_source_files() -> list[Path]:
    root = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        root / "prompt_templates.py",
        prompt_template_path(DEVELOPER_PROMPT_TEMPLATE),
        prompt_template_path("stage4_ad.md"),
        prompt_template_path("stage4_excerpt.md"),
        prompt_template_path("stage4_rubric_ad.md"),
        prompt_template_path("stage4_rubric_excerpt.md"),
        prompt_template_path("stage4_rubric_common.md"),
        root / "stage_cache.py",
        root / "openai_stage1.py",
        root / "render_stage3.py",
        root / "transcribe.py",
    ]


DEVELOPER_PROMPT_TEMPLATE = "stage4_developer.md"


def developer_instructions() -> str:
    return load_prompt_template(DEVELOPER_PROMPT_TEMPLATE)


STAGE4_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "content_mode",
        "review_summary",
        "global_findings",
        "clip_reviews",
        "recommended_rerender_order",
        "notes_for_pipeline",
    ],
    "properties": {
        "content_mode": {"type": "string", "enum": ["ad", "excerpt"]},
        "review_summary": {"type": "string"},
        "global_findings": {"type": "array", "items": {"type": "string"}},
        "clip_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "clip_id",
                    "verdict",
                    "score_0_to_100",
                    "one_sentence_assessment",
                    "rubric",
                    "issues",
                    "strengths",
                    "recommended_actions",
                    "should_rerender",
                    "should_reject_clip",
                ],
                "properties": {
                    "clip_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["keep", "minor_revision", "major_revision", "reject"],
                    },
                    "score_0_to_100": {"type": "number"},
                    "one_sentence_assessment": {"type": "string"},
                    "rubric": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "clear_payoff",
                            "payoff_first",
                            "title_or_opening",
                            "no_dead_air_before_audio",
                            "no_mid_sentence_start",
                            "audio_visual_alignment",
                            "no_mismatched_talking_head",
                            "no_repeated_broll",
                            "cut_smoothness",
                            "topic_consistent_ending",
                            "visual_story_legible",
                            "captions_support_story",
                        ],
                        "properties": {
                            key: {
                                "type": "string",
                                "enum": ["pass", "warn", "fail", "unknown"],
                            }
                            for key in [
                                "clear_payoff",
                                "payoff_first",
                                "title_or_opening",
                                "no_dead_air_before_audio",
                                "no_mid_sentence_start",
                                "audio_visual_alignment",
                                "no_mismatched_talking_head",
                                "no_repeated_broll",
                                "cut_smoothness",
                                "topic_consistent_ending",
                                "visual_story_legible",
                                "captions_support_story",
                            ]
                        },
                    },
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "category",
                                "severity",
                                "evidence",
                                "why_it_matters",
                                "suggested_fix",
                            ],
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": [
                                        "payoff",
                                        "opening",
                                        "audio_video_sync",
                                        "mid_sentence",
                                        "mismatched_talking_head",
                                        "repeated_broll",
                                        "abrupt_cut",
                                        "topic_drift",
                                        "semantic_unit",
                                        "accessibility",
                                        "portfolio_balance",
                                        "visual_relevance",
                                        "captioning",
                                        "technical",
                                        "needs_playback_review",
                                    ],
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high", "blocking"],
                                },
                                "evidence": {"type": "string"},
                                "why_it_matters": {"type": "string"},
                                "suggested_fix": {"type": "string"},
                            },
                        },
                    },
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "recommended_actions": {"type": "array", "items": {"type": "string"}},
                    "should_rerender": {"type": "boolean"},
                    "should_reject_clip": {"type": "boolean"},
                },
            },
        },
        "recommended_rerender_order": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["clip_id", "priority", "reason"],
                "properties": {
                    "clip_id": {"type": "string"},
                    "priority": {"type": "integer"},
                    "reason": {"type": "string"},
                },
            },
        },
        "notes_for_pipeline": {"type": "array", "items": {"type": "string"}},
    },
}


def response_text_config() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "stage4_render_review",
            "schema": STAGE4_SCHEMA,
            "strict": False,
        },
        "verbosity": "medium",
    }


def srt_seconds(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value.strip())
    if not match:
        return 0.0
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def read_srt_entries(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
        entries.append(
            {
                "start": round(srt_seconds(start_text), 3),
                "end": round(srt_seconds(end_text), 3),
                "text": " ".join(lines[2:]).strip(),
            }
        )
    return entries


def unique_times(times: list[float], duration: float) -> list[float]:
    clipped = sorted(max(0.05, min(duration - 0.08, time)) for time in times if duration > 0.2)
    result: list[float] = []
    for time in clipped:
        if not result or abs(time - result[-1]) >= 0.35:
            result.append(round(time, 3))
    return result


def sample_times_for_clip(
    duration: float,
    srt_entries: list[dict[str, Any]],
    max_frames: int,
) -> list[float]:
    times: list[float] = [0.75, 1.7, 2.4]
    for entry in srt_entries:
        start = float(entry.get("start") or 0)
        end = float(entry.get("end") or start)
        if end <= start:
            continue
        times.extend([start + 0.35, (start + end) / 2, end - 0.25])
    if duration > 0 and max_frames > 0:
        for index in range(max_frames):
            times.append((index + 0.5) * duration / max_frames)

    unique = unique_times(times, duration)
    if len(unique) <= max_frames:
        return unique

    # Preserve the opening/title and ending while spreading the rest through time.
    keep = [unique[0]]
    middle = unique[1:-1]
    slots = max(0, max_frames - 2)
    if slots:
        for slot in range(slots):
            keep.append(middle[math.floor(slot * len(middle) / slots)])
    keep.append(unique[-1])
    return unique_times(keep, duration)[:max_frames]


def extract_frame(video_path: Path, timestamp: float, frame_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuvj420p",
            "-strict",
            "unofficial",
            str(frame_path),
        ]
    )


def make_contact_sheet(frames_dir: Path, frame_count: int, columns: int, output_path: Path) -> None:
    rows = max(1, math.ceil(frame_count / columns))
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "1",
            "-i",
            str(frames_dir / "frame-%03d.jpg"),
            "-vf",
            f"scale=300:-2,tile={columns}x{rows}:padding=8:margin=8,format=yuvj420p",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuvj420p",
            "-strict",
            "unofficial",
            str(output_path),
        ]
    )


def probe_duration_seconds(path: Path) -> float:
    probe = ffprobe_summary(path)
    return float((probe.get("format") or {}).get("duration") or 0)


def segment_plan_duration(plan: dict[str, Any]) -> float:
    path_text = plan.get("path")
    if path_text:
        path = Path(str(path_text)).expanduser()
        if path.exists():
            try:
                return probe_duration_seconds(path)
            except RuntimeError:
                pass
    audio_start = plan.get("audio_source_start")
    audio_end = plan.get("audio_source_end")
    if isinstance(audio_start, (int, float)) and isinstance(audio_end, (int, float)) and audio_end > audio_start:
        return float(audio_end - audio_start)
    visual_durations = [
        float(span.get("duration") or 0)
        for span in plan.get("visual_spans", []) or []
        if isinstance(span, dict)
    ]
    return sum(duration for duration in visual_durations if duration > 0)


def rendered_segment_timeline(
    clip: dict[str, Any],
    rendered_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plans = clip.get("segment_plans", []) or []
    transition_duration = float(clip.get("transition_duration") or 0)
    if transition_duration <= 0:
        transition_duration = 0.0

    segments: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    cursor = 0.0
    for index, plan in enumerate(plans):
        duration = max(0.0, segment_plan_duration(plan))
        start = cursor
        end = min(rendered_duration, start + duration) if rendered_duration > 0 else start + duration
        visual_spans = plan.get("visual_spans", []) or []
        internal_visual_transition_duration = max(
            0.0,
            float(plan.get("internal_visual_transition_duration") or 0),
        )
        segments.append(
            {
                "index": index,
                "order": index + 1,
                "render_start": round(start, 3),
                "render_end": round(end, 3),
                "duration": round(max(0.0, end - start), 3),
                "source_segment_path": plan.get("path"),
                "audio_source_start": plan.get("audio_source_start"),
                "audio_source_end": plan.get("audio_source_end"),
                "visual_spans": visual_spans,
                "separated_audio_video": bool(plan.get("separated_audio_video")),
                "silent_audio": bool(plan.get("silent_audio")),
                "multiple_visual_cuts": bool(plan.get("multiple_visual_cuts")),
                "black_title_card": bool(plan.get("black_title_card")),
                "foreground_scale": plan.get("foreground_scale"),
                "internal_visual_transition_duration": internal_visual_transition_duration,
            }
        )
        visual_cursor = start
        for span_index, (left_span, right_span) in enumerate(zip(visual_spans, visual_spans[1:])):
            left_duration = float(left_span.get("duration") or 0) if isinstance(left_span, dict) else 0.0
            if left_duration <= 0:
                continue
            visual_cursor += left_duration
            transition_start = min(max(start, visual_cursor), end)
            transition_end = min(end, transition_start + internal_visual_transition_duration)
            transition_mid = (
                (transition_start + transition_end) / 2
                if transition_end > transition_start
                else transition_start
            )
            transitions.append(
                {
                    "transition_type": "internal_visual_cut",
                    "segment_order": index + 1,
                    "from_segment_order": index + 1,
                    "to_segment_order": index + 1,
                    "from_visual_span_index": span_index,
                    "to_visual_span_index": span_index + 1,
                    "transition_start": round(transition_start, 3),
                    "transition_mid": round(transition_mid, 3),
                    "transition_end": round(transition_end, 3),
                    "transition_duration": round(max(0.0, transition_end - transition_start), 3),
                    "expected_softening": (
                        "internal_visual_crossfade"
                        if transition_end > transition_start
                        else "hard_cut_inside_segment"
                    ),
                    "left_visual_span": left_span,
                    "right_visual_span": right_span,
                    "why_included": (
                        "Renderer joins multiple visual spans inside a segment; these "
                        "internal B-roll edits need review for smoothness."
                    ),
                }
            )
        if index < len(plans) - 1 and duration > 0:
            overlap = min(max(transition_duration, 0.0), duration / 2)
            if overlap <= 0:
                transition_start = end
                transition_end = end
            else:
                transition_start = max(start, end - overlap)
                transition_end = end
            transitions.append(
                {
                    "transition_type": "segment_boundary",
                    "from_segment_order": index + 1,
                    "to_segment_order": index + 2,
                    "from_visual_span_index": None,
                    "to_visual_span_index": None,
                    "transition_start": round(transition_start, 3),
                    "transition_mid": round((transition_start + transition_end) / 2, 3),
                    "transition_end": round(transition_end, 3),
                    "transition_duration": round(max(0.0, transition_end - transition_start), 3),
                    "expected_softening": "crossfade" if overlap > 0 else "hard_cut_or_unknown",
                    "why_included": "Segment boundary between timeline beats.",
                }
            )
        cursor = end - min(max(transition_duration, 0.0), duration / 2) if index < len(plans) - 1 else end
    transitions = sorted(transitions, key=lambda item: float(item.get("transition_mid") or 0))
    for index, transition in enumerate(transitions):
        transition["index"] = index
    return segments, transitions


def transition_offsets(window_seconds: float, frames_per_transition: int) -> list[float]:
    if frames_per_transition <= 1:
        return [0.0]
    window = max(0.0, window_seconds)
    return [
        round(-window + (2 * window * index / (frames_per_transition - 1)), 3)
        for index in range(frames_per_transition)
    ]


def make_transition_sheet(
    video_path: Path,
    clip_assets_dir: Path,
    transitions: list[dict[str, Any]],
    duration: float,
    frames_per_transition: int,
    window_seconds: float,
) -> dict[str, Any] | None:
    if not transitions or frames_per_transition <= 0:
        return None

    transition_dir = clip_assets_dir / "transition_frames"
    transition_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in transition_dir.glob("frame-*.jpg"):
        old_frame.unlink()

    sampled_frames: list[dict[str, Any]] = []
    offsets = transition_offsets(window_seconds, frames_per_transition)
    frame_index = 0
    for transition in transitions:
        center = float(transition.get("transition_mid") or transition.get("transition_end") or 0)
        for offset in offsets:
            timestamp = max(0.05, min(max(0.08, duration - 0.08), center + offset))
            frame_path = transition_dir / f"frame-{frame_index:03d}.jpg"
            extract_frame(video_path, timestamp, frame_path)
            sampled_frames.append(
                {
                    "index": frame_index,
                    "transition_index": transition.get("index"),
                    "transition_type": transition.get("transition_type"),
                    "segment_order": transition.get("segment_order"),
                    "from_segment_order": transition.get("from_segment_order"),
                    "to_segment_order": transition.get("to_segment_order"),
                    "from_visual_span_index": transition.get("from_visual_span_index"),
                    "to_visual_span_index": transition.get("to_visual_span_index"),
                    "render_time_seconds": round(timestamp, 3),
                    "offset_from_transition_mid_seconds": round(offset, 3),
                    "frame_path": str(frame_path),
                }
            )
            frame_index += 1

    sheet_path = clip_assets_dir / "transition-contact-sheet.jpg"
    make_contact_sheet(
        frames_dir=transition_dir,
        frame_count=len(sampled_frames),
        columns=frames_per_transition,
        output_path=sheet_path,
    )
    return {
        "transition_sheet_path": str(sheet_path),
        "transition_sheet_sha256": sha256_file(sheet_path),
        "columns": frames_per_transition,
        "read_order": (
            "Each row is one edit transition. Rows include both segment-boundary edits "
            "and internal visual-span cuts inside multi-cut B-roll segments. Frames read "
            "left-to-right from before the transition midpoint to after it."
        ),
        "sampled_frames": sampled_frames,
    }


def make_waveform_image(
    video_path: Path,
    output_path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    try:
        run_command(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-filter_complex",
                f"showwavespic=s={width}x{height}:colors=0x2f80ed",
                "-frames:v",
                "1",
                "-pix_fmt",
                "yuvj420p",
                "-strict",
                "unofficial",
                str(output_path),
            ]
        )
    except RuntimeError as exc:
        return {"waveform_path": None, "error": str(exc)}
    return {
        "waveform_path": str(output_path),
        "waveform_sha256": sha256_file(output_path),
        "width": width,
        "height": height,
        "read_order": "The waveform spans the full rendered clip from left to right.",
    }


def detect_silence(video_path: Path, noise_db: float, min_duration: float) -> dict[str, Any]:
    try:
        result = run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-v",
                "info",
                "-i",
                str(video_path),
                "-af",
                f"silencedetect=noise={noise_db}dB:d={min_duration}",
                "-f",
                "null",
                "-",
            ]
        )
    except RuntimeError as exc:
        return {"error": str(exc), "silence_ranges": [], "first_audible_time": None}

    silence_ranges: list[dict[str, Any]] = []
    open_start: float | None = None
    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start: ([0-9.]+)", line)
        if start_match:
            open_start = float(start_match.group(1))
            continue
        end_match = re.search(r"silence_end: ([0-9.]+) \| silence_duration: ([0-9.]+)", line)
        if end_match:
            end = float(end_match.group(1))
            duration = float(end_match.group(2))
            start = open_start if open_start is not None else max(0.0, end - duration)
            silence_ranges.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(duration, 3),
                }
            )
            open_start = None

    first_audible_time = 0.0
    if silence_ranges and silence_ranges[0]["start"] <= 0.05:
        first_audible_time = silence_ranges[0]["end"]
    return {
        "noise_db": noise_db,
        "min_duration": min_duration,
        "silence_ranges": silence_ranges,
        "first_audible_time": round(first_audible_time, 3),
    }


def caption_timing_evidence(
    srt_entries: list[dict[str, Any]],
    rendered_duration: float,
    first_audible_time: float | None = None,
) -> dict[str, Any]:
    if not srt_entries:
        return {
            "first_caption_start": None,
            "first_caption_text": None,
            "first_spoken_caption_start": None,
            "first_spoken_caption_text": None,
            "caption_gaps": [],
            "last_caption_end": None,
            "caption_tail_after_last_caption": None,
        }
    gaps: list[dict[str, Any]] = []
    previous_end = 0.0
    for entry in srt_entries:
        start = float(entry.get("start") or 0)
        if start - previous_end >= 0.35:
            gaps.append(
                {
                    "start": round(previous_end, 3),
                    "end": round(start, 3),
                    "duration": round(start - previous_end, 3),
                    "before_caption": entry.get("text"),
                }
            )
        previous_end = max(previous_end, float(entry.get("end") or previous_end))
    last_end = float(srt_entries[-1].get("end") or 0)
    first_spoken_entry = None
    if first_audible_time is not None:
        for entry in srt_entries:
            start = float(entry.get("start") or 0)
            if start >= first_audible_time - 0.25:
                first_spoken_entry = entry
                break
    return {
        "first_caption_start": round(float(srt_entries[0].get("start") or 0), 3),
        "first_caption_text": srt_entries[0].get("text"),
        "first_spoken_caption_start": (
            round(float(first_spoken_entry.get("start") or 0), 3)
            if first_spoken_entry
            else None
        ),
        "first_spoken_caption_text": first_spoken_entry.get("text") if first_spoken_entry else None,
        "caption_gaps": gaps,
        "last_caption_end": round(last_end, 3),
        "caption_tail_after_last_caption": round(max(0.0, rendered_duration - last_end), 3),
    }


def visual_source_overlap_warnings(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for segment in segments:
        for span_index, span in enumerate(segment.get("visual_spans", []) or []):
            if not isinstance(span, dict) or span.get("source_start") is None:
                continue
            start = float(span.get("source_start") or 0)
            duration = float(span.get("duration") or 0)
            if duration <= 0:
                continue
            spans.append(
                {
                    "segment_order": segment.get("order"),
                    "span_index": span_index,
                    "source_start": start,
                    "source_end": start + duration,
                    "reason": span.get("reason"),
                }
            )

    warnings: list[dict[str, Any]] = []
    for left_index, left in enumerate(spans):
        for right in spans[left_index + 1 :]:
            overlap_start = max(left["source_start"], right["source_start"])
            overlap_end = min(left["source_end"], right["source_end"])
            if overlap_end - overlap_start >= 0.1:
                warnings.append(
                    {
                        "left_segment_order": left["segment_order"],
                        "right_segment_order": right["segment_order"],
                        "overlap_start": round(overlap_start, 3),
                        "overlap_end": round(overlap_end, 3),
                        "overlap_duration": round(overlap_end - overlap_start, 3),
                        "why_flagged": "Overlapping visual source ranges can create repeated B-roll in a short clip.",
                    }
                )
    return warnings


def compact_clip_for_prompt(clip: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": clip.get("clip_id"),
        "concept_title": clip.get("concept_title"),
        "output_path": clip.get("output_path"),
        "subtitled_output_path": clip.get("subtitled_output_path"),
        "srt_path": clip.get("srt_path"),
        "implemented_edit_features": clip.get("implemented_edit_features"),
        "transition_duration": clip.get("transition_duration"),
        "internal_visual_transition_duration": clip.get("internal_visual_transition_duration"),
        "foreground_scale": clip.get("foreground_scale"),
        "foreground_brightness": clip.get("foreground_brightness"),
        "foreground_contrast": clip.get("foreground_contrast"),
        "foreground_saturation": clip.get("foreground_saturation"),
        "caption_plan_source": clip.get("caption_plan_source"),
        "caption_entries": clip.get("caption_entries"),
        "probe": clip.get("probe"),
        "segment_plans": clip.get("segment_plans"),
    }


def find_stage3_clip(stage3_output: dict[str, Any] | None, clip_id: str) -> dict[str, Any] | None:
    if not stage3_output:
        return None
    for clip in stage3_output.get("clip_ads", []) or []:
        if clip.get("clip_id") == clip_id:
            return clip
    return None


def build_review_package(
    render_manifest_path: Path,
    stage3_output_path: Path | None,
    clip_ids: set[str],
    assets_dir: Path,
    max_frames_per_clip: int,
    contact_sheet_columns: int,
    transition_strip_frames: int,
    transition_window_seconds: float,
    waveform_width: int,
    waveform_height: int,
    silence_noise_db: float,
    silence_min_duration: float,
    include_transition_strips: bool,
    include_waveform: bool,
) -> tuple[dict[str, Any], list[Path]]:
    render_manifest = read_json(render_manifest_path)
    stage3_output = read_json(stage3_output_path) if stage3_output_path else None
    content_mode = str(
        (stage3_output or {}).get("content_mode")
        or render_manifest.get("content_mode")
        or "ad"
    )
    assets_dir.mkdir(parents=True, exist_ok=True)

    clips = render_manifest.get("clips", []) or []
    if clip_ids:
        clips = [clip for clip in clips if str(clip.get("clip_id")) in clip_ids]
    if not clips:
        raise RuntimeError("No clips selected for review.")

    package_clips: list[dict[str, Any]] = []
    evidence_images: list[Path] = []
    for clip in clips:
        clip_id = str(clip.get("clip_id") or "clip")
        video_path = Path(str(clip.get("output_path") or clip.get("subtitled_output_path"))).expanduser()
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        srt_path = Path(str(clip.get("srt_path"))).expanduser() if clip.get("srt_path") else None
        srt_entries = read_srt_entries(srt_path)
        probe = ffprobe_summary(video_path)
        duration = float((probe.get("format") or {}).get("duration") or 0)
        segments, transitions = rendered_segment_timeline(clip, duration)
        sample_times = sample_times_for_clip(duration, srt_entries, max_frames_per_clip)

        clip_assets_dir = assets_dir / safe_name(clip_id)
        clip_assets_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in clip_assets_dir.glob("frame-*.jpg"):
            old_frame.unlink()
        for index, timestamp in enumerate(sample_times):
            extract_frame(video_path, timestamp, clip_assets_dir / f"frame-{index:03d}.jpg")

        contact_sheet_path = clip_assets_dir / "review-contact-sheet.jpg"
        make_contact_sheet(
            frames_dir=clip_assets_dir,
            frame_count=len(sample_times),
            columns=contact_sheet_columns,
            output_path=contact_sheet_path,
        )
        evidence_images.append(contact_sheet_path)

        transition_sheet = None
        if include_transition_strips:
            transition_sheet = make_transition_sheet(
                video_path=video_path,
                clip_assets_dir=clip_assets_dir,
                transitions=transitions,
                duration=duration,
                frames_per_transition=transition_strip_frames,
                window_seconds=transition_window_seconds,
            )
            if transition_sheet and transition_sheet.get("transition_sheet_path"):
                evidence_images.append(Path(str(transition_sheet["transition_sheet_path"])))

        waveform = None
        if include_waveform:
            waveform_path = clip_assets_dir / "audio-waveform.jpg"
            waveform = make_waveform_image(
                video_path=video_path,
                output_path=waveform_path,
                width=waveform_width,
                height=waveform_height,
            )
            if waveform.get("waveform_path"):
                evidence_images.append(Path(str(waveform["waveform_path"])))

        audio_timing = detect_silence(
            video_path=video_path,
            noise_db=silence_noise_db,
            min_duration=silence_min_duration,
        )
        caption_timing = caption_timing_evidence(
            srt_entries,
            duration,
            first_audible_time=audio_timing.get("first_audible_time"),
        )
        visual_overlap_warnings = visual_source_overlap_warnings(segments)

        stage3_clip = find_stage3_clip(stage3_output, clip_id)
        package_clips.append(
            {
                "clip_id": clip_id,
                "rendered_video_path": str(video_path),
                "subtitled_video_path": clip.get("subtitled_output_path"),
                "contact_sheet_path": str(contact_sheet_path),
                "contact_sheet_sha256": sha256_file(contact_sheet_path),
                "transition_sheet": transition_sheet,
                "audio_waveform": waveform,
                "sampled_frames": [
                    {
                        "index": index,
                        "render_time_seconds": timestamp,
                        "frame_path": str(clip_assets_dir / f"frame-{index:03d}.jpg"),
                    }
                    for index, timestamp in enumerate(sample_times)
                ],
                "srt_entries": srt_entries,
                "local_timing_evidence": {
                    "rendered_duration": round(duration, 3),
                    "rendered_segment_timeline": segments,
                    "edit_transitions": transitions,
                    "audio_silence": audio_timing,
                    "caption_timing": caption_timing,
                    "visual_source_overlap_warnings": visual_overlap_warnings,
                },
                "render_manifest_clip": compact_clip_for_prompt(clip),
                "stage3_plan_clip": stage3_clip,
            }
        )

    package = {
        "schema_version": 3,
        "content_mode": content_mode,
        "created_at": datetime.now(UTC).isoformat(),
        "render_manifest_path": str(render_manifest_path),
        "stage3_output_path": str(stage3_output_path) if stage3_output_path else None,
        "review_assets_dir": str(assets_dir),
        "review_evidence_note": (
            "Attached images include storyboard sheets, edit-transition strips, and audio waveform timelines. "
            "Read storyboard frames left-to-right, top-to-bottom and map them to sampled_frames by index. "
            "Read transition sheets one row per edit, left-to-right from before the transition midpoint to after it; "
            "rows include segment boundaries and internal visual-span cuts created by multi-cut B-roll. "
            "Waveform images span the full rendered clip left-to-right; use local_timing_evidence for captions, "
            "silence, segment boundaries, source-range overlap warnings, and internal cut metadata. "
            "Cuts that happen inside the original source footage can only be caught when sampled frames reveal them; "
            "if a motion-only issue still cannot be verified from local evidence, mark it as needs_playback_review."
        ),
        "evidence_images": [str(path) for path in evidence_images],
        "human_feedback_rubric": load_prompt_lines(f"stage4_rubric_{content_mode}.md")
        + load_prompt_lines("stage4_rubric_common.md"),
        "clips": package_clips,
    }
    write_json(assets_dir / "review-package.json", package)
    return package, evidence_images


def build_source_materials(package: dict[str, Any]) -> str:
    prompt_package = dict(package)
    prompt_package.pop("created_at", None)
    content_mode = str(prompt_package.get("content_mode") or "ad")
    label = "rendered draft excerpts" if content_mode == "excerpt" else "rendered draft clip ads"
    return "\n\n".join(
        [
            "STAGE 4 POST-RENDER REVIEW MATERIALS",
            f"This is a critique of already-{label}, not a request to make edit plans from scratch.",
            "The rendered MP4 files remain local. The LLM receives local evidence images: storyboard contact sheets, transition strips around segment boundaries and internal multi-cut B-roll edits, and audio waveform timelines, plus SRT captions, render metadata, and the edit plan when available.",
            "REVIEW PACKAGE JSON:",
            json.dumps(prompt_package, ensure_ascii=False, indent=2),
        ]
    )


def build_task_prompt(content_mode: str) -> str:
    return render_prompt_template(f"stage4_{content_mode}.md")


def estimated_stage4_cost(
    source_materials: str,
    task_prompt: str,
    image_count: int,
    max_output_tokens: int,
    image_token_allowance: int,
    pricing: Pricing,
) -> dict[str, Any]:
    text_tokens = estimate_text_tokens(source_materials) + estimate_text_tokens(task_prompt)
    image_tokens = image_count * image_token_allowance
    input_tokens = text_tokens + image_tokens
    return {
        "method": "rough upper-bound estimate using chars/4 for text, configured per-image allowance, and max_output_tokens",
        "text_tokens_estimate": text_tokens,
        "image_count": image_count,
        "image_tokens_allowance": image_tokens,
        "input_tokens_estimate": input_tokens,
        "max_output_tokens": max_output_tokens,
        "estimated_usd": round(
            cost_from_counts(
                input_tokens=input_tokens,
                cached_input_tokens=0,
                output_tokens=max_output_tokens,
                pricing=pricing,
            ),
            6,
        ),
    }


def request_fingerprint_payload(
    args: argparse.Namespace,
    source_materials: str,
    task_prompt: str,
    evidence_images: list[Path],
) -> dict[str, Any]:
    return {
        "cache_schema_version": 3,
        "stage": "stage4_render_review",
        "api": "openai.responses.create",
        "model": args.model,
        "input": [
            {"role": "developer", "content": developer_instructions()},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "sha256": sha256_text(source_materials)},
                    *[
                        {
                            "type": "input_image",
                            "sha256": sha256_file(path),
                            "detail": args.image_detail,
                        }
                        for path in evidence_images
                    ],
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "sha256": sha256_text(task_prompt)}],
            },
        ],
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": args.max_output_tokens,
        "store": False,
        "text": response_text_config(),
    }


def write_cache_entry(
    entry_dir: Path,
    cache_key: str,
    fingerprint_payload: dict[str, Any],
    response_dict: dict[str, Any],
    output_text: str,
    parsed_output: dict[str, Any],
    usage_and_cost: dict[str, Any],
    source: str,
    cache_identity: dict[str, Any] | None = None,
) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        entry_dir / "cache-manifest.json",
        stage_cache.cache_manifest_payload(
            cache_key=cache_key,
            query_fingerprint_payload=fingerprint_payload,
            source=source,
            cache_identity=cache_identity,
            created_at=datetime.now(UTC).isoformat(),
        ),
    )
    write_json(entry_dir / "raw-response.json", response_dict)
    (entry_dir / "output-text.json").write_text(output_text.rstrip() + "\n", encoding="utf-8")
    write_json(entry_dir / "stage4-output.json", parsed_output)
    write_json(entry_dir / "usage-and-cost.json", usage_and_cost)


def load_cached_response(entry_dir: Path) -> dict[str, Any]:
    response_dict = read_json(entry_dir / "raw-response.json")
    output_text_path = entry_dir / "output-text.json"
    output_text = (
        output_text_path.read_text(encoding="utf-8")
        if output_text_path.exists()
        else response_output_text(None, response_dict)
    )
    parsed_path = entry_dir / "stage4-output.json"
    parsed_output = read_json(parsed_path) if parsed_path.exists() else json.loads(output_text)
    return {
        "response_dict": response_dict,
        "output_text": output_text,
        "parsed_output": parsed_output,
    }


def write_markdown(
    path: Path,
    output: dict[str, Any],
    cost: dict[str, Any],
    cache_info: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Stage 4 Render Review",
        "",
        "## Summary",
        "",
        str(output.get("review_summary", "")).strip(),
        "",
        "## Clip Reviews",
        "",
    ]
    for review in output.get("clip_reviews", []) or []:
        lines.append(f"### {review.get('clip_id', '')}")
        lines.append(f"- Verdict: `{review.get('verdict', '')}`")
        lines.append(f"- Score: `{review.get('score_0_to_100', '')}`")
        lines.append(f"- Assessment: {review.get('one_sentence_assessment', '')}")
        if review.get("issues"):
            lines.append("- Issues:")
            for issue in review.get("issues", []) or []:
                lines.append(
                    f"  - `{issue.get('severity', '')}` `{issue.get('category', '')}`: "
                    f"{issue.get('evidence', '')} Fix: {issue.get('suggested_fix', '')}"
                )
        if review.get("recommended_actions"):
            lines.append("- Recommended actions:")
            for action in review.get("recommended_actions", []) or []:
                lines.append(f"  - {action}")
        lines.append("")
    lines.extend(["## Cost", ""])
    if cache_info and cache_info.get("hit"):
        original_cost = cache_info.get("original_cost") or {}
        lines.extend(
            [
                "- Local response cache: `hit`",
                "- API call made for this run: `false`",
                f"- Original response estimated USD: `${original_cost.get('estimated_usd', 0):.6f}`",
            ]
        )
    elif cache_info:
        lines.extend(
            [
                "- Local response cache: `miss`",
                f"- Stored cache key: `{cache_info.get('key', '')}`",
            ]
        )
    lines.extend(
        [
            f"- Input tokens: `{cost.get('input_tokens', 0)}`",
            f"- Cached input tokens: `{cost.get('cached_input_tokens', 0)}`",
            f"- Output tokens: `{cost.get('output_tokens', 0)}`",
            f"- Estimated USD: `${cost.get('estimated_usd', 0):.6f}`",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_response_artifacts(
    run_dir: Path,
    response_dict: dict[str, Any],
    output_text: str,
    parsed_output: dict[str, Any],
    usage_and_cost: dict[str, Any],
    summary_cost: dict[str, Any],
    cache_info: dict[str, Any] | None,
) -> None:
    write_json(run_dir / "raw-response.json", response_dict)
    (run_dir / "output-text.json").write_text(output_text.rstrip() + "\n", encoding="utf-8")
    write_json(run_dir / "stage4-output.json", parsed_output)
    write_json(run_dir / "usage-and-cost.json", usage_and_cost)
    write_markdown(run_dir / "summary.md", parsed_output, summary_cost, cache_info=cache_info)


def run_stage4(args: argparse.Namespace) -> dict[str, Any]:
    render_manifest_path = args.render_manifest.expanduser().resolve()
    stage3_output_path = args.stage3_output.expanduser().resolve() if args.stage3_output else None
    if not render_manifest_path.exists():
        raise FileNotFoundError(render_manifest_path)
    if stage3_output_path and not stage3_output_path.exists():
        raise FileNotFoundError(stage3_output_path)

    require_command("ffmpeg")
    require_command("ffprobe")

    pricing = Pricing(
        input_usd_per_million=args.input_usd_per_million,
        cached_input_usd_per_million=args.cached_input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,
    )
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    assets_dir = (
        args.assets_dir.expanduser().resolve()
        if args.assets_dir
        else output_root / "_stage4-review-assets" / safe_name(render_manifest_path.parent.name)
    )
    package, evidence_images = build_review_package(
        render_manifest_path=render_manifest_path,
        stage3_output_path=stage3_output_path,
        clip_ids=set(args.clip_id),
        assets_dir=assets_dir,
        max_frames_per_clip=args.frames_per_clip,
        contact_sheet_columns=args.contact_sheet_columns,
        transition_strip_frames=args.transition_strip_frames,
        transition_window_seconds=args.transition_window_seconds,
        waveform_width=args.waveform_width,
        waveform_height=args.waveform_height,
        silence_noise_db=args.silence_noise_db,
        silence_min_duration=args.silence_min_duration,
        include_transition_strips=not args.no_transition_strips,
        include_waveform=not args.no_waveform,
    )
    source_materials = build_source_materials(package)
    content_mode = args.content_mode or str(package.get("content_mode") or "ad")
    task_prompt = build_task_prompt(content_mode)
    preflight_cost = estimated_stage4_cost(
        source_materials=source_materials,
        task_prompt=task_prompt,
        image_count=len(evidence_images),
        max_output_tokens=args.max_output_tokens,
        image_token_allowance=args.image_token_allowance,
        pricing=pricing,
    )

    print("API cost estimate:")
    print(
        f"  model: {args.model}; reasoning: {args.reasoning_effort}; "
        f"max output tokens: {args.max_output_tokens}"
    )
    print(f"  review package: {assets_dir / 'review-package.json'}")
    print(
        f"  estimated input tokens: {preflight_cost['input_tokens_estimate']} "
        f"({preflight_cost['text_tokens_estimate']} text + "
        f"{preflight_cost['image_tokens_allowance']} image allowance across "
        f"{preflight_cost['image_count']} images)"
    )
    print(f"  estimated upper-bound cost: ${preflight_cost['estimated_usd']:.6f}")

    stable_prefix_cache_material = "\n".join(
        [
            source_materials,
            args.model,
            args.reasoning_effort,
            "\n".join(sha256_file(path) for path in evidence_images),
        ]
    )
    prompt_cache_key = args.prompt_cache_key or f"stage4-{sha256_text(stable_prefix_cache_material)[:48]}"
    fingerprint_payload = request_fingerprint_payload(
        args=args,
        source_materials=source_materials,
        task_prompt=task_prompt,
        evidence_images=evidence_images,
    )
    cache_mode = "off" if args.no_local_cache else args.cache_mode
    cache_identity = stage_cache.build_cache_identity(
        query_fingerprint_payload=fingerprint_payload,
        source_files=cache_source_files(),
        cache_mode=cache_mode,
    )
    local_cache_key = cache_identity["cache_key"]
    cache_dir = None if cache_identity["mode"] == "off" else args.cache_dir.expanduser().resolve()
    entry_dir = cache_entry_dir(cache_dir, local_cache_key) if cache_dir else None
    cache_hit = bool(entry_dir and cache_entry_exists(entry_dir) and not args.refresh_cache)
    if cache_dir:
        print(f"  local response cache mode: {cache_identity['mode']}")
        print(f"  local response cache key: {local_cache_key}")
        print(f"  query hash: {cache_identity['query_hash']}")
        print(f"  implementation hash: {cache_identity['implementation_hash']}")
        print(f"  local response cache: {'hit' if cache_hit else 'miss'}")
    else:
        print(f"  local response cache: disabled ({cache_identity['mode']})")

    if args.dry_run:
        return {
            "preflight_cost": preflight_cost,
            "review_package": str(assets_dir / "review-package.json"),
            "evidence_images": [str(path) for path in evidence_images],
            "source_materials_sha256": sha256_text(source_materials),
            "task_prompt_sha256": sha256_text(task_prompt),
            "prompt_cache_key": prompt_cache_key,
            "local_response_cache": stage_cache.response_cache_status_payload(
                enabled=cache_dir is not None,
                hit=cache_hit,
                cache_identity=cache_identity,
                entry_dir=entry_dir,
            ),
        }

    if not cache_hit:
        if args.approve_api_cost is None:
            raise RuntimeError(
                "API calls require explicit approval. Re-run with "
                f"--approve-api-cost {preflight_cost['estimated_usd']:.4f} or a higher USD cap."
            )
        if preflight_cost["estimated_usd"] > args.approve_api_cost:
            raise RuntimeError(
                f"Estimated API cost ${preflight_cost['estimated_usd']:.6f} exceeds approved cap "
                f"${args.approve_api_cost:.6f}."
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")

    run_id = args.run_id or (
        f"{render_manifest_path.parent.name}.stage4.{args.model}.{args.reasoning_effort}."
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(
        run_dir / "request-manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "stage": "stage4_render_review",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "image_detail": args.image_detail,
            "frames_per_clip": args.frames_per_clip,
            "contact_sheet_columns": args.contact_sheet_columns,
            "transition_strip_frames": args.transition_strip_frames,
            "transition_window_seconds": args.transition_window_seconds,
            "waveform_width": args.waveform_width,
            "waveform_height": args.waveform_height,
            "silence_noise_db": args.silence_noise_db,
            "silence_min_duration": args.silence_min_duration,
            "include_transition_strips": not args.no_transition_strips,
            "include_waveform": not args.no_waveform,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": args.prompt_cache_retention,
            "paths": {
                "render_manifest": str(render_manifest_path),
                "stage3_output": str(stage3_output_path) if stage3_output_path else None,
                "review_assets_dir": str(assets_dir),
                "evidence_images": [str(path) for path in evidence_images],
            },
            "hashes": {
                "source_materials_sha256": sha256_text(source_materials),
                "task_prompt_sha256": sha256_text(task_prompt),
                "evidence_image_sha256": [sha256_file(path) for path in evidence_images],
            },
            "preflight_cost": preflight_cost,
            "pricing": {
                "input_usd_per_million": pricing.input_usd_per_million,
                "cached_input_usd_per_million": pricing.cached_input_usd_per_million,
                "output_usd_per_million": pricing.output_usd_per_million,
            },
            "local_response_cache": {
                "enabled": cache_dir is not None,
                "hit": cache_hit,
                "refresh_requested": args.refresh_cache,
                "mode": cache_identity["mode"],
                "cache_key": local_cache_key if cache_dir else None,
                "entry_dir": str(entry_dir) if entry_dir else None,
                "query_hash": cache_identity["query_hash"],
                "implementation_hash": cache_identity["implementation_hash"],
                "source_files": cache_identity["source_files"],
                "query_fingerprint_payload": fingerprint_payload if cache_dir else None,
                "cache_identity": cache_identity if cache_dir else None,
            },
        },
    )
    (run_dir / "source-materials.txt").write_text(source_materials, encoding="utf-8")
    (run_dir / "task-prompt.txt").write_text(task_prompt, encoding="utf-8")
    write_json(run_dir / "review-package.json", package)

    if cache_hit:
        assert entry_dir is not None
        cached = load_cached_response(entry_dir)
        usage = usage_dict(cached["response_dict"])
        original_cost = actual_cost(usage, pricing)
        zero_cost = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "estimated_usd": 0.0,
        }
        usage_and_cost = {
            "api_call_made": False,
            "cache": {
                "hit": True,
                "key": local_cache_key,
                "entry_dir": str(entry_dir),
                "mode": cache_identity["mode"],
                "query_hash": cache_identity["query_hash"],
                "implementation_hash": cache_identity["implementation_hash"],
            },
            "usage": {},
            "cost": zero_cost,
            "original_api_usage": usage,
            "original_api_cost": original_cost,
            "pricing": {
                "input_usd_per_million": pricing.input_usd_per_million,
                "cached_input_usd_per_million": pricing.cached_input_usd_per_million,
                "output_usd_per_million": pricing.output_usd_per_million,
            },
        }
        cache_info = {
            "hit": True,
            "key": local_cache_key,
            "entry_dir": str(entry_dir),
            "mode": cache_identity["mode"],
            "query_hash": cache_identity["query_hash"],
            "implementation_hash": cache_identity["implementation_hash"],
            "original_cost": original_cost,
        }
        write_response_artifacts(
            run_dir=run_dir,
            response_dict=cached["response_dict"],
            output_text=cached["output_text"],
            parsed_output=cached["parsed_output"],
            usage_and_cost=usage_and_cost,
            summary_cost=zero_cost,
            cache_info=cache_info,
        )
        print(f"Local response cache hit: {entry_dir}")
        print("No OpenAI API call made.")
        print(f"Wrote {run_dir / 'stage4-output.json'}")
        print(f"Wrote {run_dir / 'summary.md'}")
        return {
            "run_dir": str(run_dir),
            "output": cached["parsed_output"],
            "cost": zero_cost,
            "usage": {},
            "local_response_cache": cache_info,
        }

    from openai import OpenAI

    client = OpenAI(timeout=env_float("OPENAI_CLIENT_TIMEOUT_SECONDS", 1800.0))
    content: list[dict[str, Any]] = [{"type": "input_text", "text": source_materials}]
    for evidence_image in evidence_images:
        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url(evidence_image),
                "detail": args.image_detail,
            }
        )
    input_messages: list[dict[str, Any]] = [
        {"role": "developer", "content": developer_instructions()},
        {"role": "user", "content": content},
        {"role": "user", "content": [{"type": "input_text", "text": task_prompt}]},
    ]

    print(
        f"Approved API cost cap: ${args.approve_api_cost:.6f}; "
        f"estimated cost: ${preflight_cost['estimated_usd']:.6f}"
    )
    print(f"Writing run artifacts to {run_dir}")
    response = client.responses.create(
        model=args.model,
        input=input_messages,
        reasoning={"effort": args.reasoning_effort},
        max_output_tokens=args.max_output_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=args.prompt_cache_retention,
        store=False,
        text=response_text_config(),
    )

    response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
    output_text = response_output_text(response, response_dict)
    try:
        parsed_output = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model output was not valid JSON. Raw response saved to {run_dir}") from exc

    usage = usage_dict(response_dict)
    cost = actual_cost(usage, pricing)
    usage_and_cost = {
        "api_call_made": True,
        "cache": {
            "hit": False,
            "stored": entry_dir is not None,
            "key": local_cache_key if entry_dir else None,
            "entry_dir": str(entry_dir) if entry_dir else None,
            "mode": cache_identity["mode"],
            "query_hash": cache_identity["query_hash"],
            "implementation_hash": cache_identity["implementation_hash"],
        },
        "usage": usage,
        "cost": cost,
        "pricing": {
            "input_usd_per_million": pricing.input_usd_per_million,
            "cached_input_usd_per_million": pricing.cached_input_usd_per_million,
            "output_usd_per_million": pricing.output_usd_per_million,
        },
    }
    cache_info = {
        "hit": False,
        "key": local_cache_key if entry_dir else None,
        "entry_dir": str(entry_dir) if entry_dir else None,
        "mode": cache_identity["mode"],
        "query_hash": cache_identity["query_hash"],
        "implementation_hash": cache_identity["implementation_hash"],
    }
    write_response_artifacts(
        run_dir=run_dir,
        response_dict=response_dict,
        output_text=output_text,
        parsed_output=parsed_output,
        usage_and_cost=usage_and_cost,
        summary_cost=cost,
        cache_info=cache_info if entry_dir else None,
    )
    if entry_dir is not None:
        write_cache_entry(
            entry_dir=entry_dir,
            cache_key=local_cache_key,
            fingerprint_payload=fingerprint_payload,
            response_dict=response_dict,
            output_text=output_text,
            parsed_output=parsed_output,
            usage_and_cost=usage_and_cost,
            source=f"api_run:{run_dir}",
            cache_identity=cache_identity,
        )
        print(f"Stored local response cache entry: {entry_dir}")
    print(
        f"Actual token usage: input={cost['input_tokens']}, "
        f"cached_input={cost['cached_input_tokens']}, output={cost['output_tokens']}"
    )
    print(f"Estimated billed cost from usage: ${cost['estimated_usd']:.6f}")
    print(f"Wrote {run_dir / 'stage4-output.json'}")
    print(f"Wrote {run_dir / 'summary.md'}")
    return {
        "run_dir": str(run_dir),
        "output": parsed_output,
        "cost": cost,
        "usage": usage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenAI stage 4: post-render critique of draft clips."
    )
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--stage3-output", type=Path, default=None)
    parser.add_argument("--content-mode", choices=CONTENT_MODE_CHOICES, default=None)
    parser.add_argument("--clip-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--frames-per-clip", type=int, default=None)
    parser.add_argument("--contact-sheet-columns", type=int, default=None)
    parser.add_argument("--transition-strip-frames", type=int, default=None)
    parser.add_argument("--transition-window-seconds", type=float, default=None)
    parser.add_argument("--waveform-width", type=int, default=None)
    parser.add_argument("--waveform-height", type=int, default=None)
    parser.add_argument("--silence-noise-db", type=float, default=None)
    parser.add_argument("--silence-min-duration", type=float, default=None)
    parser.add_argument("--no-transition-strips", action="store_true")
    parser.add_argument("--no-waveform", action="store_true")
    parser.add_argument("--image-detail", choices=["low", "high", "auto", "original"], default=None)
    parser.add_argument("--image-token-allowance", type=int, default=None)
    parser.add_argument("--prompt-cache-key", default=None)
    parser.add_argument("--prompt-cache-retention", choices=["in_memory", "24h"], default="24h")
    parser.add_argument(
        "--cache-mode",
        choices=stage_cache.CACHE_MODE_CHOICES,
        default=None,
        help=(
            "Local response cache identity mode. strict includes LLM query and "
            "stage source-file hashes; query-only ignores source-file hash drift; "
            "off disables the local response cache."
        ),
    )
    parser.add_argument("--no-local-cache", action="store_true")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass any local cached response and make a fresh approved API call.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--approve-api-cost",
        type=float,
        default=None,
        metavar="USD",
        help="Explicitly approve this OpenAI API call up to the preflight estimate.",
    )
    parser.add_argument("--input-usd-per-million", type=float, default=None)
    parser.add_argument("--cached-input-usd-per-million", type=float, default=None)
    parser.add_argument("--output-usd-per-million", type=float, default=None)
    return parser


def apply_env_defaults(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = Path(env_value("STAGE4_OUTPUT_DIR") or env_value("STAGE1_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    if args.cache_dir is None:
        args.cache_dir = Path(
            env_value("STAGE4_CACHE_DIR")
            or env_value("STAGE3_CACHE_DIR")
            or env_value("OPENAI_RESPONSE_CACHE_DIR")
            or (args.output_dir / DEFAULT_CACHE_SUBDIR)
        )
    if args.model is None:
        args.model = env_value("STAGE4_MODEL") or env_value("STAGE1_MODEL") or DEFAULT_MODEL
    if args.reasoning_effort is None:
        args.reasoning_effort = (
            env_value("STAGE4_REASONING_EFFORT")
            or env_value("STAGE1_REASONING_EFFORT")
            or DEFAULT_REASONING_EFFORT
        )
    if args.max_output_tokens is None:
        args.max_output_tokens = env_int("STAGE4_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
    if args.frames_per_clip is None:
        args.frames_per_clip = env_int("STAGE4_FRAMES_PER_CLIP", DEFAULT_FRAMES_PER_CLIP)
    if args.contact_sheet_columns is None:
        args.contact_sheet_columns = env_int("STAGE4_CONTACT_SHEET_COLUMNS", DEFAULT_CONTACT_SHEET_COLUMNS)
    if args.transition_strip_frames is None:
        args.transition_strip_frames = env_int("STAGE4_TRANSITION_STRIP_FRAMES", DEFAULT_TRANSITION_STRIP_FRAMES)
    if args.transition_window_seconds is None:
        args.transition_window_seconds = env_float(
            "STAGE4_TRANSITION_WINDOW_SECONDS",
            DEFAULT_TRANSITION_WINDOW_SECONDS,
        )
    if args.waveform_width is None:
        args.waveform_width = env_int("STAGE4_WAVEFORM_WIDTH", DEFAULT_WAVEFORM_WIDTH)
    if args.waveform_height is None:
        args.waveform_height = env_int("STAGE4_WAVEFORM_HEIGHT", DEFAULT_WAVEFORM_HEIGHT)
    if args.silence_noise_db is None:
        args.silence_noise_db = env_float("STAGE4_SILENCE_NOISE_DB", DEFAULT_SILENCE_NOISE_DB)
    if args.silence_min_duration is None:
        args.silence_min_duration = env_float(
            "STAGE4_SILENCE_MIN_DURATION",
            DEFAULT_SILENCE_MIN_DURATION,
        )
    if args.image_detail is None:
        args.image_detail = env_value("STAGE4_IMAGE_DETAIL") or env_value("STAGE1_IMAGE_DETAIL") or DEFAULT_IMAGE_DETAIL
    if args.image_token_allowance is None:
        args.image_token_allowance = env_int(
            "STAGE4_IMAGE_TOKEN_ALLOWANCE",
            env_int("STAGE1_IMAGE_TOKEN_ALLOWANCE", DEFAULT_IMAGE_TOKEN_ALLOWANCE),
        )
    if args.input_usd_per_million is None:
        args.input_usd_per_million = env_float(
            "STAGE4_INPUT_USD_PER_MILLION",
            env_float("STAGE1_INPUT_USD_PER_MILLION", DEFAULT_GPT55_INPUT_USD_PER_MILLION),
        )
    if args.cached_input_usd_per_million is None:
        args.cached_input_usd_per_million = env_float(
            "STAGE4_CACHED_INPUT_USD_PER_MILLION",
            env_float("STAGE1_CACHED_INPUT_USD_PER_MILLION", DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION),
        )
    if args.output_usd_per_million is None:
        args.output_usd_per_million = env_float(
            "STAGE4_OUTPUT_USD_PER_MILLION",
            env_float("STAGE1_OUTPUT_USD_PER_MILLION", DEFAULT_GPT55_OUTPUT_USD_PER_MILLION),
        )
    if args.cache_mode is None:
        args.cache_mode = (
            env_value("STAGE4_CACHE_MODE")
            or env_value("OPENAI_RESPONSE_CACHE_MODE")
            or stage_cache.DEFAULT_CACHE_MODE
        )
    args.cache_mode = stage_cache.normalize_cache_mode(args.cache_mode)
    if args.frames_per_clip <= 0:
        raise ValueError("STAGE4_FRAMES_PER_CLIP / --frames-per-clip must be positive.")
    if args.contact_sheet_columns <= 0:
        raise ValueError("STAGE4_CONTACT_SHEET_COLUMNS / --contact-sheet-columns must be positive.")
    if args.transition_strip_frames <= 0:
        raise ValueError("STAGE4_TRANSITION_STRIP_FRAMES / --transition-strip-frames must be positive.")
    if args.transition_window_seconds < 0:
        raise ValueError("STAGE4_TRANSITION_WINDOW_SECONDS / --transition-window-seconds must be non-negative.")
    if args.waveform_width <= 0:
        raise ValueError("STAGE4_WAVEFORM_WIDTH / --waveform-width must be positive.")
    if args.waveform_height <= 0:
        raise ValueError("STAGE4_WAVEFORM_HEIGHT / --waveform-height must be positive.")
    if args.silence_min_duration <= 0:
        raise ValueError("STAGE4_SILENCE_MIN_DURATION / --silence-min-duration must be positive.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        load_env_file(args.env_file)
        apply_env_defaults(args)
        run_stage4(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
