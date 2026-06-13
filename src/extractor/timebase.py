from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

# Plan JSON carries times as float seconds. Float-second arithmetic drifts off
# the frame grid, which is where off-by-a-frame cut and caption bugs breed.
# This module snaps plan times onto exact rational frame grids at ingest:
# source_* fields onto the source video's frame grid, and screen-time/caption
# fields onto the renderer's output grid. All snapping math is exact Fraction
# arithmetic; floats only re-enter when writing values back into the plan.

OUTPUT_FRAME_RATE = Fraction(30)  # matches the fps=30 filter in render_plan

SOURCE_TIME_KEYS = ("source_start", "source_end", "start", "end")
SCREEN_TIME_KEYS = (
    "screen_time_start",
    "screen_time_end",
    "start_offset",
    "end_offset",
    "duration_seconds",
    "target_duration_seconds",
)


def parse_frame_rate(text: Any) -> Fraction | None:
    """Parse an ffprobe rate string such as '30000/1001' or '25'."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw or raw in {"0", "0/0", "N/A"}:
        return None
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            rate = Fraction(int(numerator), int(denominator))
        else:
            rate = Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def probe_frame_rate(video: Path, ffprobe_bin: str = "ffprobe") -> Fraction | None:
    """Return the source video frame rate as an exact rational, or None."""
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None
    if not streams:
        return None
    stream = streams[0]
    return parse_frame_rate(stream.get("avg_frame_rate")) or parse_frame_rate(
        stream.get("r_frame_rate")
    )


def quantize_seconds(value: Any, rate: Fraction) -> float | None:
    """Snap a float-seconds value to the nearest frame boundary on `rate`."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    frames = round(seconds * rate)
    return float(Fraction(frames, 1) / rate)


def _quantize_keys(item: dict[str, Any], keys: tuple[str, ...], rate: Fraction) -> int:
    changed = 0
    for key in keys:
        if key not in item:
            continue
        quantized = quantize_seconds(item.get(key), rate)
        if quantized is None:
            continue
        if quantized != item[key]:
            changed += 1
        item[key] = quantized
    return changed


def quantize_plan_times(
    plan: dict[str, Any],
    source_rate: Fraction,
    output_rate: Fraction = OUTPUT_FRAME_RATE,
) -> int:
    """Snap all plan times onto frame grids, in place.

    Times addressing the source video (source ranges, timeline source spans,
    manual-review ranges) snap to `source_rate`; times addressing the rendered
    clip (screen times, caption offsets, durations) snap to `output_rate`.
    Returns the number of values that moved.
    """
    changed = 0
    for clip in plan.get("clip_ads", []) or []:
        if not isinstance(clip, dict):
            continue
        changed += _quantize_keys(clip, ("target_duration_seconds",), output_rate)
        hook = clip.get("opening_hook")
        if isinstance(hook, dict):
            changed += _quantize_keys(hook, ("duration_seconds",), output_rate)
        for source_range in clip.get("source_ranges", []) or []:
            if isinstance(source_range, dict):
                changed += _quantize_keys(source_range, ("start", "end"), source_rate)
        for item in clip.get("timeline", []) or []:
            if isinstance(item, dict):
                changed += _quantize_keys(item, ("source_start", "source_end"), source_rate)
                changed += _quantize_keys(
                    item, ("screen_time_start", "screen_time_end"), output_rate
                )
        for caption in clip.get("caption_plan", []) or []:
            if isinstance(caption, dict):
                changed += _quantize_keys(
                    caption, ("start_offset", "end_offset", "start", "end"), output_rate
                )
    for entry in plan.get("manual_review_queue", []) or []:
        if isinstance(entry, dict):
            changed += _quantize_keys(entry, ("start", "end"), source_rate)
    return changed
