from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.timebase import probe_frame_rate, quantize_plan_times
from extractor.preprocess_video import (
    DEFAULT_FFMPEG_BIN,
    DEFAULT_FFMPEG_HWACCEL,
    FFMPEG_HWACCEL_CHOICES,
    ffmpeg_hwaccel_input_args,
    resolve_ffmpeg_bin,
    resolve_ffmpeg_hwaccel,
)


DEFAULT_OUTPUT_DIR = Path("subject-index")
DEFAULT_SAMPLE_WIDTH = 192
DEFAULT_SAMPLE_HEIGHT = 108
DEFAULT_FPS = 1.0
DEFAULT_MAX_FRAMES = 8
DEFAULT_WINDOW_SECONDS = 12.0
DEFAULT_WINDOW_STRIDE_SECONDS = 10.0
MIN_SUBJECT_SCORE = 900.0
FFMPEG_BIN = "ffmpeg"
FFMPEG_HWACCEL: str | None = None


@dataclass(frozen=True)
class SourceWindow:
    clip_id: str
    timeline_order: int
    source_start: float
    source_end: float
    reason: str

    @property
    def duration(self) -> float:
        return max(0.0, self.source_end - self.source_start)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "subject-index"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found on PATH: {name}")


def run_command(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def should_skip_timeline_item(item: dict[str, Any]) -> bool:
    audio_source = str(item.get("audio_source", "")).lower()
    visual_source = str(item.get("visual_source", "")).lower()
    text = f"{audio_source} {visual_source} {item.get('edit_action', '')}".lower()
    if "asset://" in visual_source:
        return True
    if "generated black title card" in text or "title card" in text:
        try:
            return float(item.get("source_end", 0)) - float(item.get("source_start", 0)) <= 0.05
        except (TypeError, ValueError):
            return True
    return False


def source_windows_from_plan(plan_output: dict[str, Any]) -> list[SourceWindow]:
    windows: list[SourceWindow] = []
    for clip in plan_output.get("clip_ads", []) or []:
        clip_id = str(clip.get("clip_id") or "clip")
        timeline = sorted(
            clip.get("timeline", []) or [],
            key=lambda item: int(item.get("order", 0) or 0),
        )
        for item in timeline:
            if should_skip_timeline_item(item):
                continue
            try:
                source_start = float(item["source_start"])
                source_end = float(item["source_end"])
            except (KeyError, TypeError, ValueError):
                continue
            if source_end - source_start <= 0.25:
                continue
            windows.append(
                SourceWindow(
                    clip_id=clip_id,
                    timeline_order=int(item.get("order", 0) or 0),
                    source_start=source_start,
                    source_end=source_end,
                    reason="plan_timeline",
                )
            )
    return merge_duplicate_windows(windows)


def merge_duplicate_windows(windows: list[SourceWindow]) -> list[SourceWindow]:
    merged: list[SourceWindow] = []
    seen: set[tuple[str, int, float, float]] = set()
    for window in windows:
        key = (
            window.clip_id,
            window.timeline_order,
            round(window.source_start, 2),
            round(window.source_end, 2),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(window)
    return merged


def analysis_slices(window: SourceWindow, window_seconds: float, stride_seconds: float) -> list[tuple[float, float]]:
    duration = window.duration
    if duration <= window_seconds + 0.25:
        return [(window.source_start, duration)]
    starts: list[float] = []
    cursor = window.source_start
    last_start = max(window.source_start, window.source_end - window_seconds)
    while cursor < last_start:
        starts.append(cursor)
        cursor += max(1.0, stride_seconds)
    starts.append(last_start)
    return [(start, min(window_seconds, window.source_end - start)) for start in starts]


def extract_rgb_frames(
    video: Path,
    source_start: float,
    duration: float,
    width: int,
    height: int,
    fps: float,
    max_frames: int,
) -> list[bytes]:
    frame_size = width * height * 3
    frame_count = max(1, min(max_frames, int(math.ceil(duration * fps)) + 1))
    args = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{source_start:.3f}",
        "-t",
        f"{duration:.3f}",
        *ffmpeg_hwaccel_input_args(FFMPEG_HWACCEL),
        "-i",
        str(video),
        "-vf",
        f"fps={fps:.3f},scale={width}:{height},format=rgb24",
        "-frames:v",
        str(frame_count),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        result = run_command(args)
    except subprocess.CalledProcessError:
        return []
    payload = result.stdout
    return [
        payload[index : index + frame_size]
        for index in range(0, len(payload) - frame_size + 1, frame_size)
    ]


def luma(red: int, green: int, blue: int) -> int:
    return int(0.2126 * red + 0.7152 * green + 0.0722 * blue)


def skin_score(red: int, green: int, blue: int) -> float:
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    if red <= 45 or green <= 30 or blue <= 20:
        return 0.0
    if maximum - minimum <= 15 or abs(red - green) <= 10:
        return 0.0
    if red <= green or red <= blue:
        return 0.0
    saturation_hint = min(1.0, (maximum - minimum) / 80.0)
    brightness_hint = 1.0 - min(1.0, abs(luma(red, green, blue) - 145) / 145.0)
    return 0.55 + 0.45 * saturation_hint * brightness_hint


def percentile_bounds(scores: list[float], lower_fraction: float = 0.05, upper_fraction: float = 0.95) -> tuple[float, float, float]:
    total = sum(scores)
    if total <= 0 or len(scores) <= 1:
        return 0.0, 1.0, 1.0
    lower_target = total * lower_fraction
    upper_target = total * upper_fraction
    lower_index = 0
    upper_index = len(scores) - 1
    cumulative = 0.0
    for index, value in enumerate(scores):
        cumulative += value
        if cumulative >= lower_target:
            lower_index = index
            break
    cumulative = 0.0
    for index, value in enumerate(scores):
        cumulative += value
        if cumulative >= upper_target:
            upper_index = index
            break
    divisor = max(1, len(scores) - 1)
    left = lower_index / divisor
    right = upper_index / divisor
    return left, right, max(0.0, right - left)


def weighted_center(scores: list[float]) -> float:
    total = sum(scores)
    if total <= 0 or len(scores) <= 1:
        return 0.5
    divisor = len(scores) - 1
    return max(0.0, min(1.0, sum((i / divisor) * score for i, score in enumerate(scores)) / total))


def find_secondary_content(
    edge_columns: list[float],
    skin_columns: list[float],
    subject_left: float,
    subject_right: float,
) -> dict[str, Any] | None:
    total_edges = sum(edge_columns)
    if total_edges <= 0:
        return None
    width = len(edge_columns)
    candidates: list[tuple[float, int, int]] = []
    min_width = max(8, int(width * 0.18))
    for side_left, side_right in ((0, int(width * max(0.0, subject_left - 0.04))), (int(width * min(1.0, subject_right + 0.04)), width)):
        if side_right - side_left < min_width:
            continue
        edge_score = sum(edge_columns[side_left:side_right])
        skin_score_total = sum(skin_columns[side_left:side_right])
        score = edge_score * max(0.0, 1.0 - min(0.8, skin_score_total / max(1.0, edge_score)))
        candidates.append((score, side_left, side_right))
    if not candidates:
        return None
    score, left_index, right_index = max(candidates, key=lambda item: item[0])
    score_ratio = score / total_edges
    if score_ratio < 0.18:
        return None
    divisor = max(1, width - 1)
    return {
        "type": "secondary_content",
        "bbox": [
            round(left_index / divisor, 3),
            0.08,
            round((right_index - 1) / divisor, 3),
            0.92,
        ],
        "confidence": round(min(0.85, 0.25 + score_ratio), 3),
        "method": "non_skin_edge_region",
    }


def connected_skin_components(
    mask: list[bool],
    width: int,
    height: int,
    preferred_center_x: float,
) -> list[dict[str, Any]]:
    visited = [False] * (width * height)
    components: list[dict[str, Any]] = []
    for start_index, is_skin in enumerate(mask):
        if not is_skin or visited[start_index]:
            continue
        stack = [start_index]
        visited[start_index] = True
        count = 0
        sum_x = 0.0
        sum_y = 0.0
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0
        while stack:
            index = stack.pop()
            x = index % width
            y = index // width
            count += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                neighbor = ny * width + nx
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if count < 6:
            continue
        bbox_width = (max_x - min_x + 1) / width
        bbox_height = (max_y - min_y + 1) / height
        if bbox_width <= 0 or bbox_height <= 0:
            continue
        aspect = bbox_width / bbox_height
        compactness = min(1.0, max(0.0, 1.6 - abs(math.log(max(0.15, min(6.0, aspect))))))
        center_y = (sum_y / count) / max(1, height - 1)
        face_band_weight = 1.0 - min(1.0, abs(center_y - 0.36) / 0.36)
        center_x = (sum_x / count) / max(1, width - 1)
        center_weight = 1.0 - min(1.0, abs(center_x - preferred_center_x) / 0.45)
        score = (
            count
            * (0.35 + 0.65 * compactness)
            * (0.45 + 0.55 * face_band_weight)
            * (0.35 + 0.65 * center_weight)
        )
        components.append(
            {
                "count": count,
                "score": score,
                "center_x": center_x,
                "center_y": center_y,
                "bbox": [
                    min_x / max(1, width - 1),
                    min_y / max(1, height - 1),
                    max_x / max(1, width - 1),
                    max_y / max(1, height - 1),
                ],
                "aspect": aspect,
            }
        )
    return sorted(components, key=lambda item: float(item["score"]), reverse=True)


def analyze_frames(frames: list[bytes], width: int, height: int) -> dict[str, Any]:
    if not frames:
        return {
            "available": False,
            "reason": "no_frames",
            "primary_subject": None,
            "secondary_content": None,
            "recommended_crop": {"center_x": 0.5, "max_scale": 1.35, "confidence": 0.0},
        }

    subject_columns = [0.0] * width
    subject_rows = [0.0] * height
    skin_columns = [0.0] * width
    skin_rows = [0.0] * height
    edge_columns = [0.0] * width
    previous_luma: list[int] | None = None
    motion_columns = [0.0] * width
    skin_hits = [0] * (width * height)

    for frame in frames:
        luma_values = [0] * (width * height)
        skin_values = [0.0] * (width * height)
        for y in range(height):
            y_norm = y / max(1, height - 1)
            vertical_subject_weight = 0.35 + 0.65 * (1.0 - min(1.0, abs(y_norm - 0.48) / 0.5))
            row_offset = y * width * 3
            for x in range(width):
                offset = row_offset + x * 3
                red = frame[offset]
                green = frame[offset + 1]
                blue = frame[offset + 2]
                index = y * width + x
                lum = luma(red, green, blue)
                luma_values[index] = lum
                skin = skin_score(red, green, blue)
                if skin:
                    upper_body_weight = 1.0 if 0.10 <= y_norm <= 0.70 else 0.45
                    skin *= upper_body_weight
                    skin_values[index] = skin
                    skin_hits[index] += 1
                    skin_columns[x] += skin
                    skin_rows[y] += skin
                    subject_columns[x] += skin * 5.0 * vertical_subject_weight
                    subject_rows[y] += skin * 5.0

        for y in range(1, height - 1):
            y_norm = y / max(1, height - 1)
            vertical_subject_weight = 0.30 + 0.70 * (1.0 - min(1.0, abs(y_norm - 0.52) / 0.5))
            for x in range(1, width - 1):
                index = y * width + x
                dx = abs(luma_values[index + 1] - luma_values[index - 1])
                dy = abs(luma_values[index + width] - luma_values[index - width])
                edge = max(0.0, float(dx + dy - 28))
                if edge <= 0:
                    continue
                edge_columns[x] += edge
                subject_columns[x] += edge * 0.18 * vertical_subject_weight
                subject_rows[y] += edge * 0.18

        if previous_luma is not None:
            for index, (current, previous) in enumerate(zip(luma_values, previous_luma)):
                delta = abs(current - previous)
                if delta < 12:
                    continue
                x = index % width
                y = index // width
                y_norm = y / max(1, height - 1)
                vertical_subject_weight = 0.40 + 0.60 * (1.0 - min(1.0, abs(y_norm - 0.55) / 0.5))
                motion = float(delta)
                motion_columns[x] += motion
                subject_columns[x] += motion * 0.55 * vertical_subject_weight
                subject_rows[y] += motion * 0.55
        previous_luma = luma_values

    total_subject = sum(subject_columns)
    total_skin = sum(skin_columns)
    total_motion = sum(motion_columns)
    if total_subject < MIN_SUBJECT_SCORE:
        return {
            "available": False,
            "reason": "low_subject_score",
            "frame_count": len(frames),
            "primary_subject": None,
            "secondary_content": None,
            "recommended_crop": {
                "center_x": 0.5,
                "max_scale": 1.35,
                "confidence": 0.0,
                "reason": "no confident subject; use wide centered crop",
            },
            "scores": {
                "subject": round(total_subject, 3),
                "skin": round(total_skin, 3),
                "motion": round(total_motion, 3),
            },
        }

    scene_center = weighted_center(subject_columns)
    skin_threshold = max(1, int(math.ceil(len(frames) * 0.20)))
    skin_mask = [hits >= skin_threshold for hits in skin_hits]
    skin_components = connected_skin_components(
        skin_mask,
        width,
        height,
        preferred_center_x=scene_center,
    )
    face_component = skin_components[0] if skin_components else None
    if total_skin >= 800.0 and face_component is not None:
        skin_center = float(face_component["center_x"])
        face_bbox = face_component["bbox"]
        # A compact skin component is treated as a face/hand anchor. Expand to
        # a conservative upper-body region so the crop can punch in without
        # losing the speaker.
        left = max(0.0, skin_center - 0.24)
        right = min(1.0, skin_center + 0.24)
        top = max(0.0, float(face_bbox[1]) - 0.10)
        bottom = min(1.0, max(0.88, float(face_bbox[3]) + 0.42))
        bbox_width = max(0.0, right - left)
        bbox_height = max(0.0, bottom - top)
        center_x = skin_center
        subject_method = "skin_component_anchored_speaker"
    else:
        left, right, bbox_width = percentile_bounds(subject_columns, 0.04, 0.96)
        top, bottom, bbox_height = percentile_bounds(subject_rows, 0.04, 0.96)
        center_x = weighted_center(subject_columns)
        subject_method = "motion_edge_saliency"
    subject_confidence = min(
        0.95,
        0.30
        + min(0.35, total_skin / max(1.0, total_subject))
        + min(0.20, total_motion / max(1.0, total_subject * 1.8))
        + min(0.10, max(0.0, 0.75 - bbox_width)),
    )
    subject = {
        "type": "speaker",
        "bbox": [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)],
        "center_x": round(center_x, 3),
        "confidence": round(subject_confidence, 3),
        "method": subject_method,
    }
    secondary = find_secondary_content(edge_columns, skin_columns, left, right)

    protected_left = left
    protected_right = right
    protect_regions = ["primary_subject"]
    protected_width = max(0.18, protected_right - protected_left)
    safe_margin = 0.11 if subject_confidence >= 0.55 else 0.16
    visible_width = max(0.44, min(1.0, protected_width + safe_margin * 2))
    max_scale = max(1.0, min(3.0, 1.0 / visible_width))
    if subject_confidence < 0.45:
        max_scale = min(max_scale, 1.45)

    crop_center = max(
        visible_width / 2,
        min(1.0 - visible_width / 2, (protected_left + protected_right) / 2),
    )
    return {
        "available": True,
        "frame_count": len(frames),
        "primary_subject": subject,
        "secondary_content": secondary,
        "recommended_crop": {
            "center_x": round(crop_center, 3),
            "max_scale": round(max_scale, 3),
            "visible_width": round(visible_width, 3),
            "protect_regions": protect_regions,
            "confidence": round(subject_confidence, 3),
            "reason": "protect detected speaker and optional adjacent content",
        },
        "scores": {
            "subject": round(total_subject, 3),
            "skin": round(total_skin, 3),
            "motion": round(total_motion, 3),
            "bbox_width": round(bbox_width, 3),
            "bbox_height": round(bbox_height, 3),
        },
    }


def build_subject_index(args: argparse.Namespace) -> dict[str, Any]:
    video = args.video.expanduser().resolve()
    plan_output = args.plan_output.expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    if not plan_output.exists():
        raise FileNotFoundError(plan_output)

    plan = read_json(plan_output)
    source_frame_rate = probe_frame_rate(video)
    if source_frame_rate is not None:
        quantize_plan_times(plan, source_frame_rate)
    windows = source_windows_from_plan(plan)
    run_id = safe_name(args.run_id or plan_output.parent.name or video.stem)
    output_root = args.output_dir.expanduser().resolve()
    run_dir = output_root / run_id
    entries: list[dict[str, Any]] = []
    for window in windows:
        for slice_start, slice_duration in analysis_slices(
            window,
            args.window_seconds,
            args.window_stride_seconds,
        ):
            frames = extract_rgb_frames(
                video=video,
                source_start=slice_start,
                duration=slice_duration,
                width=args.sample_width,
                height=args.sample_height,
                fps=args.fps,
                max_frames=args.max_frames,
            )
            analysis = analyze_frames(frames, args.sample_width, args.sample_height)
            entries.append(
                {
                    "clip_id": window.clip_id,
                    "timeline_order": window.timeline_order,
                    "source_start": round(window.source_start, 3),
                    "source_end": round(window.source_end, 3),
                    "analysis_start": round(slice_start, 3),
                    "analysis_end": round(slice_start + slice_duration, 3),
                    "reason": window.reason,
                    **analysis,
                }
            )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "api_calls": False,
        "video": str(video),
        "plan_output": str(plan_output),
        "run_id": run_id,
        "sample": {
            "width": args.sample_width,
            "height": args.sample_height,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "window_seconds": args.window_seconds,
            "window_stride_seconds": args.window_stride_seconds,
            "ffmpeg_bin": FFMPEG_BIN,
            "ffmpeg_hwaccel": FFMPEG_HWACCEL or "none",
        },
        "window_count": len(windows),
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local subject/crop index for Stage 3 render plans."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sample-width", type=int, default=DEFAULT_SAMPLE_WIDTH)
    parser.add_argument("--sample-height", type=int, default=DEFAULT_SAMPLE_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--window-stride-seconds", type=float, default=DEFAULT_WINDOW_STRIDE_SECONDS)
    parser.add_argument(
        "--ffmpeg-bin",
        default=DEFAULT_FFMPEG_BIN,
        help="ffmpeg executable. Use auto to prefer the full Homebrew install when present.",
    )
    parser.add_argument(
        "--ffmpeg-hwaccel",
        choices=FFMPEG_HWACCEL_CHOICES,
        default=DEFAULT_FFMPEG_HWACCEL,
        help="Hardware decode mode for local frame analysis.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    global FFMPEG_BIN, FFMPEG_HWACCEL
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        FFMPEG_BIN = resolve_ffmpeg_bin(args.ffmpeg_bin)
        FFMPEG_HWACCEL = resolve_ffmpeg_hwaccel(args.ffmpeg_hwaccel, FFMPEG_BIN) or "none"
        require_command("ffprobe")
        manifest = build_subject_index(args)
        print(f"Subject index entries: {manifest['entry_count']}")
        print(f"Wrote {Path(args.output_dir).expanduser().resolve() / manifest['run_id'] / 'manifest.json'}")
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
