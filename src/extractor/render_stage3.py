from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

from extractor.preprocess_video import (
    DEFAULT_FFMPEG_BIN,
    DEFAULT_FFMPEG_HWACCEL,
    FFMPEG_HWACCEL_CHOICES,
    ffmpeg_hwaccel_input_args,
    resolve_ffmpeg_bin,
    resolve_ffmpeg_hwaccel,
)


DEFAULT_OUTPUT_DIR = Path("renders")
DEFAULT_WIDTH = 720
DEFAULT_HEIGHT = 1280
DEFAULT_FONT_FILE = Path("/System/Library/Fonts/SFNS.ttf")
TITLE_CARD_SWIFT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "render_title_card.swift"
DEFAULT_FOREGROUND_SCALE = 1.0
DEFAULT_INTERNAL_VISUAL_TRANSITION_DURATION = 0.0
DEFAULT_FOREGROUND_BRIGHTNESS = 0.0
DEFAULT_FOREGROUND_CONTRAST = 1.0
DEFAULT_FOREGROUND_SATURATION = 1.0
FINAL_VIDEO_FADE_OUT_DURATION = 0.6
MIN_TITLE_CARD_SECONDS = 4.0
FFMPEG_BIN = "ffmpeg"
FFMPEG_HWACCEL: str | None = None

TITLE_CARD_FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    "'": ["00100", "00100", "01000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found on PATH: {name}")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(message) from exc


def configure_ffmpeg(ffmpeg_bin: str, ffmpeg_hwaccel: str | None) -> None:
    global FFMPEG_BIN, FFMPEG_HWACCEL
    FFMPEG_BIN = ffmpeg_bin
    FFMPEG_HWACCEL = None if ffmpeg_hwaccel in {None, "none"} else ffmpeg_hwaccel


def ffmpeg_has_filter(name: str) -> bool:
    result = subprocess.run(
        [FFMPEG_BIN, "-hide_banner", "-filters"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return any(name in line.split() for line in result.stdout.splitlines())


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(path: Path, timeline: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for index, item in enumerate(timeline, start=1):
        text = str(item.get("caption_or_overlay", "")).strip()
        if not text:
            continue
        start = float(item["screen_time_start"])
        end = float(item["screen_time_end"])
        lines.extend(
            [
                str(index),
                f"{srt_timestamp(start)} --> {srt_timestamp(end)}",
                wrap_caption(text, width=34),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_srt_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        start = float(entry["start"])
        end = float(entry["end"])
        if end <= start:
            continue
        lines.extend(
            [
                str(index),
                f"{srt_timestamp(start)} --> {srt_timestamp(end)}",
                wrap_caption(text, width=34),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def mux_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> bool:
    try:
        run_command(
            [
                FFMPEG_BIN,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-i",
                str(srt_path),
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-c:s",
                "mov_text",
                "-metadata:s:s:0",
                "language=eng",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return True
    except Exception:
        return False


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def quote_concat_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def wrap_caption(text: str, width: int = 24) -> str:
    words = text.strip().split()
    if not words:
        return ""
    lines = textwrap.wrap(
        " ".join(words),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines[:4])


def title_card_lines(text: str, width: int, scale: int) -> list[str]:
    max_chars = max(8, int(width * 0.82 / ((5 + 1) * scale)))
    normalized = text.upper().replace("\u2019", "'")
    lines: list[str] = []
    for paragraph in normalized.splitlines() or [normalized]:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return lines or [""]


def title_line_columns(line: str) -> int:
    columns = 0
    for char in line:
        columns += 3 if char == " " else 6
    return max(0, columns - 1)


def choose_title_scale(text: str, width: int, height: int) -> tuple[int, list[str]]:
    for scale in range(14, 5, -1):
        lines = title_card_lines(text, width, scale)
        if len(lines) > 4:
            continue
        max_columns = max(title_line_columns(line) for line in lines)
        total_rows = 7 * len(lines) + 2 * (len(lines) - 1)
        if max_columns * scale <= width * 0.84 and total_rows * scale <= height * 0.45:
            return scale, lines
    return 6, title_card_lines(text, width, 6)[:4]


def draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_width: int,
    rect_height: int,
    color: tuple[int, int, int],
) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + rect_width)
    y1 = min(height, y + rect_height)
    if x0 >= x1 or y0 >= y1:
        return
    red, green, blue = color
    for row in range(y0, y1):
        offset = (row * width + x0) * 3
        for _ in range(x0, x1):
            pixels[offset : offset + 3] = bytes((red, green, blue))
            offset += 3


def draw_title_glyph(
    pixels: bytearray,
    width: int,
    height: int,
    char: str,
    x: int,
    y: int,
    scale: int,
    color: tuple[int, int, int],
) -> None:
    glyph = TITLE_CARD_FONT.get(char, TITLE_CARD_FONT["?"])
    for row, pattern in enumerate(glyph):
        for column, value in enumerate(pattern):
            if value == "1":
                draw_rect(
                    pixels,
                    width,
                    height,
                    x + column * scale,
                    y + row * scale,
                    scale,
                    scale,
                    color,
                )


def write_title_card_ppm(path: Path, text: str, width: int, height: int) -> None:
    scale, lines = choose_title_scale(text, width, height)
    pixels = bytearray([0, 0, 0] * width * height)
    color = (246, 246, 242)
    line_gap = 2 * scale
    line_height = 7 * scale
    total_height = len(lines) * line_height + (len(lines) - 1) * line_gap
    y = int((height - total_height) / 2)
    for line in lines:
        line_width = title_line_columns(line) * scale
        x = int((width - line_width) / 2)
        for char in line:
            if char == " ":
                x += 3 * scale
                continue
            draw_title_glyph(pixels, width, height, char, x, y, scale, color)
            x += 6 * scale
        y += line_height + line_gap

    with path.open("wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(pixels)


def write_title_card_image(path: Path, text: str, width: int, height: int) -> Path:
    if shutil.which("swift") and TITLE_CARD_SWIFT_SCRIPT.exists():
        png_path = path.with_suffix(".png")
        try:
            run_command(
                [
                    "swift",
                    str(TITLE_CARD_SWIFT_SCRIPT),
                    str(png_path),
                    str(width),
                    str(height),
                    text,
                ]
            )
            return png_path
        except Exception:
            pass

    ppm_path = path.with_suffix(".ppm")
    write_title_card_ppm(ppm_path, text, width, height)
    return ppm_path


def timeline_duration(item: dict[str, Any]) -> float:
    return float(item["screen_time_end"]) - float(item["screen_time_start"])


def source_duration(item: dict[str, Any]) -> float:
    return float(item["source_end"]) - float(item["source_start"])


def uses_dialogue(item: dict[str, Any]) -> bool:
    audio_source = str(item.get("audio_source", "")).lower()
    no_dialogue_markers = (
        "no dialogue",
        "no source dialogue",
        "no source audio",
        "no new dialogue",
        "no spoken audio",
        "room tone only",
        "roomtone only",
        "music hit only",
        "music bed only",
        "generated music bed only",
        "music sting",
        "music/roomtone tail",
        "silent generated",
        "silent black title",
        "silent title card",
        "generated black title card",
    )
    if any(marker in audio_source for marker in no_dialogue_markers):
        return False
    return True


def is_title_card_segment(item: dict[str, Any]) -> bool:
    text = " ".join(
        str(item.get(key, "")).lower()
        for key in ("edit_action", "visual_source", "audio_source", "purpose")
    )
    return any(
        marker in text
        for marker in (
            "title card",
            "title/tagline",
            "black title",
            "generated black",
            "cta card",
            "outro card",
            "text-only",
            "silent title",
            "watch the full",
        )
    )


def should_render_title_card(item: dict[str, Any], force_first_card: bool = False) -> bool:
    if force_first_card and not uses_dialogue(item):
        return True
    if not is_title_card_segment(item):
        return False
    return source_duration(item) <= 0.05 or not uses_dialogue(item)


def normalize_first_title_card_duration(
    clip: dict[str, Any],
    min_seconds: float = MIN_TITLE_CARD_SECONDS,
) -> tuple[dict[str, Any], bool, float]:
    timeline = sorted(clip.get("timeline", []), key=lambda item: int(item.get("order", 0)))
    if not timeline:
        return clip, False, 0.0
    first = timeline[0]
    if not should_render_title_card(first):
        return clip, False, 0.0

    duration = timeline_duration(first)
    if duration >= min_seconds:
        return clip, False, 0.0

    adjusted = deepcopy(clip)
    adjusted_timeline: list[dict[str, Any]] = []
    delta = min_seconds - duration
    original_start = float(first["screen_time_start"])
    original_end = float(first["screen_time_end"])
    for index, item in enumerate(timeline):
        copied = dict(item)
        if index == 0:
            copied["screen_time_end"] = round(float(copied["screen_time_end"]) + delta, 3)
        else:
            copied["screen_time_start"] = round(float(copied["screen_time_start"]) + delta, 3)
            copied["screen_time_end"] = round(float(copied["screen_time_end"]) + delta, 3)
        adjusted_timeline.append(copied)
    adjusted["timeline"] = adjusted_timeline

    adjusted_captions: list[dict[str, Any]] = []
    for item in adjusted.get("caption_plan", []) or []:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        raw_start = copied.get("start_offset", copied.get("start"))
        raw_end = copied.get("end_offset", copied.get("end"))
        if raw_start is not None and raw_end is not None:
            start = float(raw_start)
            end = float(raw_end)
            if start >= original_end - 0.001:
                start += delta
                end += delta
            elif start >= original_start - 0.001 and end <= original_end + 0.001:
                end += delta
            elif start < original_end < end:
                end += delta

            if "start_offset" in copied or "end_offset" in copied:
                copied["start_offset"] = round(start, 3)
                copied["end_offset"] = round(end, 3)
            else:
                copied["start"] = round(start, 3)
                copied["end"] = round(end, 3)
        adjusted_captions.append(copied)
    adjusted["caption_plan"] = adjusted_captions
    return adjusted, True, delta


def normalize_first_title_card_text(clip: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    concept_title = str(clip.get("concept_title", "")).strip()
    if not concept_title:
        return clip, False, ""

    timeline = sorted(clip.get("timeline", []), key=lambda item: int(item.get("order", 0)))
    if not timeline:
        return clip, False, ""

    first = timeline[0]
    if not should_render_title_card(first):
        return clip, False, ""

    original_text = str(first.get("caption_or_overlay", "")).strip()
    if original_text == concept_title:
        return clip, False, ""

    adjusted = deepcopy(clip)
    adjusted_timeline: list[dict[str, Any]] = []
    first_start = float(first["screen_time_start"])
    first_end = float(first["screen_time_end"])
    for index, item in enumerate(timeline):
        copied = dict(item)
        if index == 0:
            copied["caption_or_overlay"] = concept_title
        adjusted_timeline.append(copied)
    adjusted["timeline"] = adjusted_timeline

    adjusted_captions: list[dict[str, Any]] = []
    first_caption_updated = False
    for item in adjusted.get("caption_plan", []) or []:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        raw_start = copied.get("start_offset", copied.get("start"))
        raw_end = copied.get("end_offset", copied.get("end"))
        if raw_start is not None and raw_end is not None:
            start = float(raw_start)
            end = float(raw_end)
            inside_first_card = start >= first_start - 0.001 and end <= first_end + 0.001
            starts_first_card = not first_caption_updated and start <= first_start + 0.001
            if inside_first_card or starts_first_card:
                copied["text"] = concept_title
                first_caption_updated = True
        adjusted_captions.append(copied)
    adjusted["caption_plan"] = adjusted_captions
    return adjusted, True, original_text


def retime_timeline(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = 0.0
    retimed: list[dict[str, Any]] = []
    for item in timeline:
        duration = timeline_duration(item)
        if duration <= 0.04:
            continue
        copied = dict(item)
        copied["screen_time_start"] = round(current, 3)
        current += duration
        copied["screen_time_end"] = round(current, 3)
        retimed.append(copied)
    return retimed


def timeline_with_transition_timing(
    timeline: list[dict[str, Any]],
    transition_duration: float,
) -> list[dict[str, Any]]:
    if transition_duration <= 0 or len(timeline) < 2:
        return timeline

    adjusted: list[dict[str, Any]] = []
    current = 0.0
    for index, item in enumerate(timeline):
        copied = dict(item)
        duration = timeline_duration(item)
        if duration <= 0.04:
            continue
        display_duration = duration
        if index < len(timeline) - 1:
            display_duration = max(0.05, duration - transition_duration)
        copied["screen_time_start"] = round(current, 3)
        current += display_duration
        copied["screen_time_end"] = round(current, 3)
        adjusted.append(copied)
    return adjusted


def caption_plan_entries(clip: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(clip.get("caption_plan_source") or "").lower()
    if mode not in {"absolute_render_time", "render_time", "srt"} and not clip.get(
        "use_caption_plan_for_srt"
    ):
        return []

    entries: list[dict[str, Any]] = []
    for item in clip.get("caption_plan", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        raw_start = item.get("start_offset", item.get("start"))
        raw_end = item.get("end_offset", item.get("end"))
        if raw_start is None or raw_end is None:
            continue
        start = float(raw_start)
        end = float(raw_end)
        if end <= start:
            continue
        entries.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return sorted(entries, key=lambda item: (float(item["start"]), float(item["end"])))


def repair_silent_lead(timeline: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    if len(timeline) < 2:
        return timeline, False
    first = dict(timeline[0])
    second = dict(timeline[1])
    if uses_dialogue(first) or not uses_dialogue(second):
        return timeline, False

    lead_duration = timeline_duration(first)
    second_duration = timeline_duration(second)
    if lead_duration <= 0.04 or second_duration <= lead_duration + 0.25:
        return timeline, False

    second_source_start = float(second["source_start"])
    second_source_end = float(second["source_end"])
    first["source_start"] = round(second_source_start, 3)
    first["source_end"] = round(second_source_start + lead_duration, 3)
    first["audio_source"] = (
        "Use immediate dialogue borrowed from the following narration beat; "
        "keep the cold-open visual as B-roll."
    )

    second["source_start"] = round(second_source_start + lead_duration, 3)
    second["source_end"] = second_source_end
    repaired = [first, second, *[dict(item) for item in timeline[2:]]]
    current = 0.0
    retimed: list[dict[str, Any]] = []
    for index, item in enumerate(repaired):
        copied = dict(item)
        if index <= 1:
            duration = source_duration(copied)
        else:
            duration = timeline_duration(copied)
        if duration <= 0.04:
            continue
        copied["screen_time_start"] = round(current, 3)
        current += duration
        copied["screen_time_end"] = round(current, 3)
        retimed.append(copied)
    return retimed, True


def extract_time_ranges(text: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for match in re.finditer(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*[-–]\s*(\d{1,3}(?:\.\d+)?)", text):
        start = float(match.group(1))
        end = float(match.group(2))
        if 0 <= start < end <= 700 and end - start >= 0.25:
            ranges.append((start, end))
    return ranges


def extract_time_anchors(text: str) -> list[float]:
    anchors: list[float] = []
    for match in re.finditer(r"(?<![\d:])(\d{1,3}(?:\.\d+)?)(?![\d:])", text):
        value = float(match.group(1))
        end = match.end()
        context = text[end : end + 14].lower()
        before = text[max(0, match.start() - 12) : match.start()].lower()
        if "percent" in context or "%" in context:
            continue
        if "9:" in before or ":16" in context:
            continue
        if 0 <= value <= 700:
            anchors.append(value)
    deduped: list[float] = []
    for value in anchors:
        if not any(abs(value - existing) < 0.2 for existing in deduped):
            deduped.append(value)
    return deduped


def needs_visual_cover(item: dict[str, Any]) -> bool:
    text = " ".join(
        str(item.get(key, "")).lower()
        for key in ("edit_action", "visual_source", "audio_source", "purpose")
    )
    return any(
        marker in text
        for marker in (
            "b-roll",
            "cover",
            "j-cut",
            "l-cut",
            "avoid",
            "quick visual cuts",
            "cut each",
            "cold open",
            "underwater visuals",
            "reordered",
        )
    )


def build_visual_spans(item: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    source_start = float(item["source_start"])
    source_end = float(item["source_end"])
    if source_end <= source_start:
        raise RuntimeError(
            "Non-title-card segment has non-positive source duration. "
            "Generated cards must use visual_source='Generated black title card.' "
            "and audio_source='No source dialogue; silent generated title card.'"
        )
    visual_source = str(item.get("visual_source", ""))
    cover = needs_visual_cover(item)
    ranges = extract_time_ranges(visual_source)
    anchors = extract_time_anchors(visual_source)

    spans: list[dict[str, Any]] = []
    if cover and ranges:
        for start, end in ranges:
            spans.append(
                {
                    "source_start": start,
                    "available_duration": max(0.25, end - start),
                    "reason": "parsed_visual_range",
                }
            )
    elif cover and len(anchors) >= 2:
        for start in anchors:
            spans.append(
                {
                    "source_start": start,
                    "available_duration": 3.0,
                    "reason": "parsed_visual_anchor",
                }
            )
    elif cover and len(anchors) == 1 and abs(anchors[0] - source_start) > 0.2:
        spans.append(
            {
                "source_start": anchors[0],
                "available_duration": duration,
                "reason": "parsed_single_visual_anchor",
            }
        )

    if not spans:
        spans.append(
            {
                "source_start": source_start,
                "available_duration": max(0.25, source_end - source_start),
                "reason": "timeline_source",
            }
        )

    # Allocate visual chunks until the timeline beat is filled. If a listed
    # B-roll source is shorter than the beat, cycle through the candidates.
    allocated: list[dict[str, Any]] = []
    remaining = duration
    cursor = 0
    while remaining > 0.04:
        span = spans[cursor % len(spans)]
        chunk_duration = min(float(span["available_duration"]), remaining)
        allocated.append(
            {
                "source_start": float(span["source_start"]),
                "duration": chunk_duration,
                "reason": span["reason"],
            }
        )
        remaining = round(remaining - chunk_duration, 6)
        cursor += 1
        if cursor > 200:
            raise RuntimeError("Unable to allocate visual spans.")
    return allocated


def drawtext_filter(caption_file: Path, font_file: Path, first_segment: bool) -> str:
    y_expr = "(h-text_h)/2" if first_segment else "h-text_h-160"
    return (
        "drawtext="
        f"fontfile={font_file}:"
        f"textfile={caption_file}:"
        "fontcolor=white:"
        "fontsize=44:"
        "line_spacing=8:"
        "x=(w-text_w)/2:"
        f"y={y_expr}:"
        "box=1:"
        "boxcolor=black@0.62:"
        "boxborderw=18"
    )


def video_filter(
    caption_file: Path,
    font_file: Path,
    width: int,
    height: int,
    first_segment: bool,
    fade_out: bool,
    duration: float,
    burn_captions: bool,
    foreground_scale: float,
    tail_pad_duration: float,
    foreground_brightness: float,
    foreground_contrast: float,
    foreground_saturation: float,
) -> str:
    foreground_width = max(width, int(round(width * max(1.0, foreground_scale) / 2) * 2))
    foreground_filter = f"[fgsrc]scale={foreground_width}:-2"
    if (
        abs(foreground_brightness - DEFAULT_FOREGROUND_BRIGHTNESS) > 0.0001
        or abs(foreground_contrast - DEFAULT_FOREGROUND_CONTRAST) > 0.0001
        or abs(foreground_saturation - DEFAULT_FOREGROUND_SATURATION) > 0.0001
    ):
        foreground_filter += (
            f",eq=brightness={foreground_brightness:.3f}:"
            f"contrast={foreground_contrast:.3f}:"
            f"saturation={foreground_saturation:.3f}"
        )
    foreground_filter += "[fg]"
    filters = [
        "[0:v]setpts=PTS-STARTPTS,split=2[bgsrc][fgsrc]",
        (
            f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=24,eq=brightness=-0.07:saturation=0.9[bg]"
        ),
        foreground_filter,
    ]
    composed = "[bg][fg]overlay=(W-w)/2:(H-h)/2"
    if burn_captions:
        composed += f",{drawtext_filter(caption_file, font_file, first_segment)}"
    if fade_out:
        fade_start = max(0.0, duration - 0.2)
        composed += f",fade=t=out:st={fade_start:.3f}:d=0.2"
    if tail_pad_duration > 0:
        composed += f",tpad=stop_mode=clone:stop_duration={tail_pad_duration:.3f}"
    composed += ",format=yuv420p[v]"
    filters.append(composed)
    return ";".join(filters)


def audio_filter(input_label: str, duration: float, fade_out: bool) -> str:
    fade_in_duration = min(0.05, max(0.0, duration / 3))
    fade_out_duration = min(0.08 if fade_out else 0.05, max(0.0, duration / 3))
    fade_out_start = max(0.0, duration - fade_out_duration)
    return (
        f"[{input_label}:a]asetpts=PTS-STARTPTS,"
        "aresample=48000,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"afade=t=in:st=0:d={fade_in_duration:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_duration:.3f}[a]"
    )


def render_video_chunk(
    source_video: Path,
    clip_dir: Path,
    caption_file: Path,
    segment_index: int,
    chunk_index: int,
    source_start: float,
    duration: float,
    font_file: Path,
    width: int,
    height: int,
    first_segment: bool,
    fade_out: bool,
    burn_captions: bool,
    foreground_scale: float,
    tail_pad_duration: float,
    foreground_brightness: float,
    foreground_contrast: float,
    foreground_saturation: float,
) -> Path:
    chunk_path = clip_dir / f"segment-{segment_index:03d}.visual-{chunk_index:03d}.mp4"
    args = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{source_start:.3f}",
        "-t",
        f"{duration:.3f}",
        *ffmpeg_hwaccel_input_args(FFMPEG_HWACCEL),
        "-i",
        str(source_video),
        "-filter_complex",
        video_filter(
            caption_file=caption_file,
            font_file=font_file,
            width=width,
            height=height,
            first_segment=first_segment,
            fade_out=fade_out,
            duration=duration,
            burn_captions=burn_captions,
            foreground_scale=foreground_scale,
            tail_pad_duration=tail_pad_duration,
            foreground_brightness=foreground_brightness,
            foreground_contrast=foreground_contrast,
            foreground_saturation=foreground_saturation,
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(chunk_path),
    ]
    run_command(args)
    return chunk_path


def render_title_card_video(
    clip_dir: Path,
    segment_index: int,
    text: str,
    duration: float,
    width: int,
    height: int,
    fade_out: bool,
) -> Path:
    image_path = clip_dir / f"segment-{segment_index:03d}.title-card"
    video_path = clip_dir / f"segment-{segment_index:03d}.title-card.mp4"
    image_path = write_title_card_image(image_path, text, width, height)
    filters = [
        f"scale={width}:{height},setsar=1",
        "format=yuv420p",
        "fade=t=in:st=0:d=0.100",
    ]
    if fade_out:
        filters.append(f"fade=t=out:st={max(0.0, duration - 0.120):.3f}:d=0.120")
    run_command(
        [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            "30",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(image_path),
            "-vf",
            ",".join(filters),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-video_track_timescale",
            "30000",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
    )
    return video_path


def concat_media(paths: list[Path], output_path: Path) -> None:
    concat_file = output_path.with_suffix(".concat.txt")
    concat_file.write_text(
        "\n".join(f"file '{quote_concat_path(path)}'" for path in paths) + "\n",
        encoding="utf-8",
    )
    run_command(
        [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def concat_video_chunks_with_crossfade(
    paths: list[Path],
    output_path: Path,
    transition_duration: float,
    width: int,
    height: int,
) -> None:
    if len(paths) < 2 or transition_duration <= 0:
        concat_media(paths, output_path)
        return

    summaries = [ffprobe_summary(path) for path in paths]
    durations = [
        stream_duration(summary, "video", fallback_format=True)
        for summary in summaries
    ]
    usable_transition = min(
        transition_duration,
        *[max(0.0, (duration - 0.05) / 2) for duration in durations],
    )
    if usable_transition <= 0.01:
        concat_media(paths, output_path)
        return

    args = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"]
    for path in paths:
        args.extend(["-i", str(path)])

    filters: list[str] = []
    for index in range(len(paths)):
        filters.append(
            f"[{index}:v]fps=30,scale={width}:{height},setsar=1,"
            f"format=yuv420p,setpts=PTS-STARTPTS[v{index}]"
        )

    current_video = "v0"
    current_duration = durations[0]
    for index in range(1, len(paths)):
        video_out = f"vchunkx{index}"
        offset = max(0.0, current_duration - usable_transition)
        filters.append(
            f"[{current_video}][v{index}]"
            f"xfade=transition=fade:duration={usable_transition:.3f}:offset={offset:.3f}"
            f"[{video_out}]"
        )
        current_video = video_out
        current_duration = current_duration + durations[index] - usable_transition

    args.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{current_video}]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    run_command(args)


def render_audio_track(
    source_video: Path,
    clip_dir: Path,
    item: dict[str, Any],
    index: int,
    duration: float,
    fade_out: bool,
    force_silent: bool = False,
) -> Path:
    audio_path = clip_dir / f"segment-{index:03d}.audio.m4a"
    args = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if uses_dialogue(item) and not force_silent:
        args.extend(
            [
                "-ss",
                f"{float(item['source_start']):.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(source_video),
                "-filter_complex",
                audio_filter("0", duration, fade_out),
            ]
        )
    else:
        args.extend(
            [
                "-f",
                "lavfi",
                "-t",
                f"{duration:.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-filter_complex",
                audio_filter("0", duration, fade_out),
            ]
        )
    args.extend(
        [
            "-map",
            "[a]",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(audio_path),
        ]
    )
    run_command(args)
    return audio_path


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    run_command(
        [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def render_segment(
    source_video: Path,
    clip_dir: Path,
    item: dict[str, Any],
    index: int,
    total_segments: int,
    font_file: Path,
    width: int,
    height: int,
    burn_captions: bool,
    black_title_card: bool,
    foreground_scale: float,
    internal_visual_transition_duration: float,
    foreground_brightness: float,
    foreground_contrast: float,
    foreground_saturation: float,
) -> dict[str, Any]:
    duration = timeline_duration(item)
    if duration <= 0:
        raise RuntimeError(f"Non-positive timeline duration for segment {index}")

    caption_file = clip_dir / f"caption-{index:03d}.txt"
    caption_file.write_text(
        wrap_caption(str(item.get("caption_or_overlay", ""))),
        encoding="utf-8",
    )
    first_segment = index == 1
    last_segment = index == total_segments
    if black_title_card:
        visual_spans = [
            {
                "source_start": None,
                "duration": duration,
                "reason": "black_title_card",
            }
        ]
        video_chunks = [
            render_title_card_video(
                clip_dir=clip_dir,
                segment_index=index,
                text=str(item.get("caption_or_overlay", "")),
                duration=duration,
                width=width,
                height=height,
                fade_out=False,
            )
        ]
    else:
        visual_spans = build_visual_spans(item, duration)
        video_chunks: list[Path] = []
        for chunk_index, span in enumerate(visual_spans, start=1):
            tail_pad_duration = (
                internal_visual_transition_duration
                if internal_visual_transition_duration > 0 and chunk_index < len(visual_spans)
                else 0.0
            )
            video_chunks.append(
                render_video_chunk(
                    source_video=source_video,
                    clip_dir=clip_dir,
                    caption_file=caption_file,
                    segment_index=index,
                    chunk_index=chunk_index,
                    source_start=float(span["source_start"]),
                    duration=float(span["duration"]),
                    font_file=font_file,
                    width=width,
                    height=height,
                    first_segment=first_segment and chunk_index == 1,
                    fade_out=last_segment and chunk_index == len(visual_spans),
                    burn_captions=burn_captions,
                    foreground_scale=foreground_scale,
                    tail_pad_duration=tail_pad_duration,
                    foreground_brightness=foreground_brightness,
                    foreground_contrast=foreground_contrast,
                    foreground_saturation=foreground_saturation,
                )
            )
    video_track = clip_dir / f"segment-{index:03d}.video.mp4"
    if internal_visual_transition_duration > 0 and len(video_chunks) > 1:
        concat_video_chunks_with_crossfade(
            video_chunks,
            video_track,
            transition_duration=internal_visual_transition_duration,
            width=width,
            height=height,
        )
    else:
        concat_media(video_chunks, video_track)
    audio_track = render_audio_track(
        source_video=source_video,
        clip_dir=clip_dir,
        item=item,
        index=index,
        duration=duration,
        fade_out=last_segment,
        force_silent=black_title_card,
    )
    segment_path = clip_dir / f"segment-{index:03d}.mp4"
    mux_video_audio(video_track, audio_track, segment_path)
    uses_source_dialogue = uses_dialogue(item) and not black_title_card
    audio_start = None if not uses_source_dialogue else float(item["source_start"])
    separated = any(
        audio_start is None
        or span["source_start"] is None
        or abs(float(span["source_start"]) - audio_start) > 0.2
        for span in visual_spans
    )
    return {
        "path": segment_path,
        "audio_source_start": audio_start,
        "audio_source_end": None if audio_start is None else audio_start + duration,
        "visual_spans": visual_spans,
        "separated_audio_video": separated,
        "silent_audio": not uses_source_dialogue,
        "multiple_visual_cuts": len(visual_spans) > 1,
        "black_title_card": black_title_card,
        "foreground_scale": foreground_scale,
        "foreground_brightness": foreground_brightness,
        "foreground_contrast": foreground_contrast,
        "foreground_saturation": foreground_saturation,
        "internal_visual_transition_duration": (
            internal_visual_transition_duration if len(visual_spans) > 1 else 0.0
        ),
    }


def concat_segments(segment_paths: list[Path], output_path: Path, width: int, height: int) -> None:
    if not segment_paths:
        raise RuntimeError("Cannot concatenate an empty segment list.")

    args = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"]
    for path in segment_paths:
        args.extend(["-i", str(path)])

    filters: list[str] = []
    for index in range(len(segment_paths)):
        filters.append(
            f"[{index}:v]fps=30,scale={width}:{height},setsar=1,"
            f"format=yuv420p,setpts=PTS-STARTPTS[v{index}]"
        )
        filters.append(
            f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )

    if len(segment_paths) == 1:
        video_out = "v0"
        audio_out = "a0"
    else:
        concat_inputs = "".join(f"[v{index}][a{index}]" for index in range(len(segment_paths)))
        filters.append(f"{concat_inputs}concat=n={len(segment_paths)}:v=1:a=1[vout][aout]")
        video_out = "vout"
        audio_out = "aout"

    args.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{video_out}]",
            "-map",
            f"[{audio_out}]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    run_command(args)
    validate_av_stream_lengths(output_path)


def concat_segments_with_crossfade(
    segment_paths: list[Path],
    output_path: Path,
    transition_duration: float,
    width: int,
    height: int,
) -> None:
    if len(segment_paths) < 2 or transition_duration <= 0:
        concat_segments(segment_paths, output_path, width=width, height=height)
        return

    summaries = [ffprobe_summary(path) for path in segment_paths]
    video_durations = [
        stream_duration(summary, "video", fallback_format=True)
        for summary in summaries
    ]
    audio_durations = [
        stream_duration(summary, "audio", fallback_format=True)
        for summary in summaries
    ]
    all_durations = video_durations + audio_durations
    usable_transition = min(
        transition_duration,
        *[max(0.0, (duration - 0.05) / 2) for duration in all_durations],
    )
    if usable_transition <= 0.01:
        concat_segments(segment_paths, output_path, width=width, height=height)
        return

    args = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"]
    for path in segment_paths:
        args.extend(["-i", str(path)])

    filters: list[str] = []
    for index in range(len(segment_paths)):
        filters.append(
            f"[{index}:v]fps=30,scale={width}:{height},setsar=1,"
            f"format=yuv420p,setpts=PTS-STARTPTS[v{index}]"
        )
        filters.append(
            f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )

    current_video = "v0"
    current_audio = "a0"
    current_video_duration = video_durations[0]
    current_audio_duration = audio_durations[0]
    for index in range(1, len(segment_paths)):
        video_out = f"vx{index}"
        audio_out = f"ax{index}"
        offset = max(0.0, current_video_duration - usable_transition)
        filters.append(
            f"[{current_video}][v{index}]"
            f"xfade=transition=fade:duration={usable_transition:.3f}:offset={offset:.3f}"
            f"[{video_out}]"
        )
        filters.append(
            f"[{current_audio}][a{index}]"
            f"acrossfade=d={usable_transition:.3f}:c1=tri:c2=tri"
            f"[{audio_out}]"
        )
        current_video = video_out
        current_audio = audio_out
        current_video_duration = (
            current_video_duration + video_durations[index] - usable_transition
        )
        current_audio_duration = (
            current_audio_duration + audio_durations[index] - usable_transition
        )

    duration_gap = current_audio_duration - current_video_duration
    if duration_gap > 0.04:
        video_out = "vpad"
        filters.append(
            f"[{current_video}]tpad=stop_mode=clone:stop_duration={duration_gap:.3f}"
            f"[{video_out}]"
        )
        current_video = video_out
    elif duration_gap < -0.04:
        audio_out = "apadout"
        filters.append(
            f"[{current_audio}]apad=pad_dur={-duration_gap:.3f}[{audio_out}]"
        )
        current_audio = audio_out

    args.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{current_video}]",
            "-map",
            f"[{current_audio}]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    run_command(args)
    validate_av_stream_lengths(output_path)


def apply_final_video_fade_to_black(input_path: Path, output_path: Path, fade_duration: float) -> float:
    summary = ffprobe_summary(input_path)
    video_stream = next(
        (
            stream
            for stream in summary.get("streams", [])
            if stream.get("codec_type") == "video" and stream.get("duration")
        ),
        None,
    )
    duration = float(
        (video_stream or {}).get("duration")
        or summary.get("format", {}).get("duration")
        or 0
    )
    if duration <= 0:
        raise RuntimeError(f"Could not determine duration for {input_path}")
    usable_fade = min(fade_duration, max(0.0, duration / 2))
    if usable_fade <= 0.01:
        shutil.move(str(input_path), str(output_path))
        return 0.0
    fade_start = max(0.0, duration - usable_fade)
    run_command(
        [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"fade=t=out:st={fade_start:.3f}:d={usable_fade:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    input_path.unlink(missing_ok=True)
    validate_av_stream_lengths(output_path)
    return usable_fade


def ffprobe_summary(path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def stream_duration(
    summary: dict[str, Any],
    codec_type: str,
    *,
    fallback_format: bool,
) -> float:
    for stream in summary.get("streams", []):
        if stream.get("codec_type") == codec_type and stream.get("duration"):
            return float(stream["duration"])
    if fallback_format:
        return float(summary.get("format", {}).get("duration") or 0)
    return 0.0


def validate_av_stream_lengths(path: Path, tolerance: float = 0.5) -> dict[str, Any]:
    summary = ffprobe_summary(path)
    video_duration = stream_duration(summary, "video", fallback_format=False)
    audio_duration = stream_duration(summary, "audio", fallback_format=False)
    if video_duration and audio_duration and abs(video_duration - audio_duration) > tolerance:
        raise RuntimeError(
            f"Rendered stream duration mismatch for {path}: "
            f"video={video_duration:.3f}s audio={audio_duration:.3f}s"
        )
    return summary


def render_clip(
    source_video: Path,
    output_dir: Path,
    clip: dict[str, Any],
    font_file: Path,
    width: int,
    height: int,
    burn_captions: bool,
    repair_silent_lead_clip_ids: set[str],
    black_title_card_clip_ids: set[str],
    transition_duration: float,
    foreground_scale: float,
    internal_visual_transition_duration: float,
    foreground_brightness: float,
    foreground_contrast: float,
    foreground_saturation: float,
) -> dict[str, Any]:
    clip, title_card_text_normalized, title_card_original_text = normalize_first_title_card_text(clip)
    clip, title_card_duration_normalized, title_card_duration_delta = normalize_first_title_card_duration(clip)
    clip_id = safe_name(str(clip["clip_id"]))
    clip_dir = output_dir / "_work" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    timeline = sorted(clip.get("timeline", []), key=lambda item: int(item.get("order", 0)))
    silent_lead_repaired = False
    if clip_id in repair_silent_lead_clip_ids:
        timeline, silent_lead_repaired = repair_silent_lead(timeline)
    use_black_title_card = clip_id in black_title_card_clip_ids
    segment_results = [
        render_segment(
            source_video=source_video,
            clip_dir=clip_dir,
            item=item,
            index=index,
            total_segments=len(timeline),
            font_file=font_file,
            width=width,
            height=height,
            burn_captions=burn_captions,
            black_title_card=should_render_title_card(
                item,
                force_first_card=index == 1 and use_black_title_card,
            ),
            foreground_scale=foreground_scale,
            internal_visual_transition_duration=internal_visual_transition_duration,
            foreground_brightness=foreground_brightness,
            foreground_contrast=foreground_contrast,
            foreground_saturation=foreground_saturation,
        )
        for index, item in enumerate(timeline, start=1)
    ]
    segment_paths = [result["path"] for result in segment_results]
    output_path = output_dir / f"{clip_id}.draft.mp4"
    pre_fade_output_path = output_dir / f"{clip_id}.draft.pre-video-fade.mp4"
    concat_segments_with_crossfade(
        segment_paths,
        pre_fade_output_path,
        transition_duration=transition_duration,
        width=width,
        height=height,
    )
    final_video_fade_out_duration = apply_final_video_fade_to_black(
        pre_fade_output_path,
        output_path,
        FINAL_VIDEO_FADE_OUT_DURATION,
    )
    srt_path = output_dir / f"{clip_id}.draft.srt"
    caption_entries = caption_plan_entries(clip)
    if caption_entries:
        write_srt_entries(srt_path, caption_entries)
        srt_timeline = caption_entries
    else:
        srt_timeline = timeline_with_transition_timing(timeline, transition_duration)
        write_srt(srt_path, srt_timeline)
    subtitled_path = output_dir / f"{clip_id}.draft.subtitled.mp4"
    subtitles_muxed = mux_subtitles(output_path, srt_path, subtitled_path)
    return {
        "clip_id": clip_id,
        "concept_title": clip.get("concept_title"),
        "output_path": str(output_path),
        "srt_path": str(srt_path),
        "subtitled_output_path": str(subtitled_path) if subtitles_muxed else None,
        "burned_in_captions": burn_captions,
        "embedded_subtitles": subtitles_muxed,
        "segments": [str(path) for path in segment_paths],
        "segment_plans": [
            {
                **{
                    key: value
                    for key, value in result.items()
                    if key != "path"
                },
                "path": str(result["path"]),
            }
            for result in segment_results
        ],
        "implemented_edit_features": sorted(
            {
                feature
                for result in segment_results
                for feature in (
                    ["separated_audio_video"] if result["separated_audio_video"] else []
                )
                + (["silent_cold_open"] if result["silent_audio"] else [])
                + (["multi_cut_broll"] if result["multiple_visual_cuts"] else [])
                + (["black_title_card"] if result["black_title_card"] else [])
                + (
                    ["internal_visual_crossfades"]
                    if result["internal_visual_transition_duration"] > 0
                    else []
                )
            }
            | {"audio_fades", "vertical_reframe", "sidecar_captions"}
            | ({"embedded_subtitles"} if subtitles_muxed else set())
            | ({"silent_lead_repaired"} if silent_lead_repaired else set())
            | ({"crossfade_transitions"} if transition_duration > 0 else set())
            | ({"foreground_punch_in"} if foreground_scale > 1.0 else set())
            | ({"foreground_exposure_lift"} if foreground_brightness > 0 else set())
            | ({"phrase_level_captions"} if caption_entries else set())
            | ({"video_fade_to_black"} if final_video_fade_out_duration > 0 else set())
        ),
        "silent_lead_repaired": silent_lead_repaired,
        "title_card_text_normalized": title_card_text_normalized,
        "title_card_original_text": title_card_original_text,
        "title_card_duration_normalized": title_card_duration_normalized,
        "title_card_duration_delta": round(title_card_duration_delta, 3),
        "transition_duration": transition_duration,
        "final_video_fade_out_duration": final_video_fade_out_duration,
        "internal_visual_transition_duration": internal_visual_transition_duration,
        "foreground_scale": foreground_scale,
        "foreground_brightness": foreground_brightness,
        "foreground_contrast": foreground_contrast,
        "foreground_saturation": foreground_saturation,
        "caption_plan_source": clip.get("caption_plan_source"),
        "caption_entries": srt_timeline,
        "probe": ffprobe_summary(output_path),
    }


def render_setting_value(
    args: argparse.Namespace,
    global_settings: dict[str, Any],
    clip_settings: dict[str, Any],
    name: str,
    default: float,
) -> float:
    cli_value = getattr(args, name)
    if cli_value is not None:
        return float(cli_value)
    if args.use_plan_render_settings:
        for settings in (clip_settings, global_settings):
            value = settings.get(name)
            if value is not None:
                return float(value)
    return default


def render_settings_for_clip(
    args: argparse.Namespace,
    plan: dict[str, Any],
    clip: dict[str, Any],
) -> dict[str, Any]:
    global_settings = plan.get("render_settings") if isinstance(plan.get("render_settings"), dict) else {}
    clip_settings = clip.get("render_settings") if isinstance(clip.get("render_settings"), dict) else {}
    burn_captions = bool(args.burn_captions)
    if not burn_captions and args.use_plan_render_settings:
        for settings in (clip_settings, global_settings):
            if settings.get("burn_captions") is not None:
                burn_captions = bool(settings["burn_captions"])
                break
    return {
        "transition_duration": render_setting_value(
            args,
            global_settings,
            clip_settings,
            "transition_duration",
            0.0,
        ),
        "foreground_scale": render_setting_value(
            args,
            global_settings,
            clip_settings,
            "foreground_scale",
            DEFAULT_FOREGROUND_SCALE,
        ),
        "internal_visual_transition_duration": render_setting_value(
            args,
            global_settings,
            clip_settings,
            "internal_visual_transition_duration",
            DEFAULT_INTERNAL_VISUAL_TRANSITION_DURATION,
        ),
        "foreground_brightness": render_setting_value(
            args,
            global_settings,
            clip_settings,
            "foreground_brightness",
            DEFAULT_FOREGROUND_BRIGHTNESS,
        ),
        "foreground_contrast": render_setting_value(
            args,
            global_settings,
            clip_settings,
            "foreground_contrast",
            DEFAULT_FOREGROUND_CONTRAST,
        ),
        "foreground_saturation": render_setting_value(
            args,
            global_settings,
            clip_settings,
            "foreground_saturation",
            DEFAULT_FOREGROUND_SATURATION,
        ),
        "burn_captions": burn_captions,
    }


def validate_render_settings(settings: dict[str, Any]) -> None:
    if float(settings["transition_duration"]) < 0:
        raise RuntimeError("transition_duration must be >= 0")
    if float(settings["foreground_scale"]) < 1.0:
        raise RuntimeError("foreground_scale must be >= 1.0")
    if float(settings["internal_visual_transition_duration"]) < 0:
        raise RuntimeError("internal_visual_transition_duration must be >= 0")
    if float(settings["foreground_contrast"]) <= 0:
        raise RuntimeError("foreground_contrast must be > 0")
    if float(settings["foreground_saturation"]) < 0:
        raise RuntimeError("foreground_saturation must be >= 0")


def render_stage3(args: argparse.Namespace) -> dict[str, Any]:
    source_video = args.source_video.expanduser().resolve()
    stage3_path = args.stage3_output.expanduser().resolve()
    font_file = args.font_file.expanduser().resolve()
    if not source_video.exists():
        raise FileNotFoundError(source_video)
    if not stage3_path.exists():
        raise FileNotFoundError(stage3_path)
    if not font_file.exists():
        raise FileNotFoundError(font_file)

    plan = read_json(stage3_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    drawtext_available = ffmpeg_has_filter("drawtext")
    clip_ads = plan.get("clip_ads", [])
    if args.clip_id:
        clip_ads = [clip for clip in clip_ads if clip.get("clip_id") == args.clip_id]
        if not clip_ads:
            raise RuntimeError(f"No clip found with clip_id={args.clip_id!r}")

    results = []
    warned_drawtext = False
    for clip in clip_ads:
        settings = render_settings_for_clip(args, plan, clip)
        validate_render_settings(settings)
        burn_captions = bool(settings["burn_captions"]) and drawtext_available
        if settings["burn_captions"] and not burn_captions and not warned_drawtext:
            print("warning: ffmpeg drawtext filter is unavailable; writing sidecar captions instead.")
            warned_drawtext = True
        results.append(
            render_clip(
                source_video=source_video,
                output_dir=output_dir,
                clip=clip,
                font_file=font_file,
                width=args.width,
                height=args.height,
                burn_captions=burn_captions,
                repair_silent_lead_clip_ids=set(args.repair_silent_lead_clip),
                black_title_card_clip_ids=set(args.black_title_card_clip),
                transition_duration=float(settings["transition_duration"]),
                foreground_scale=float(settings["foreground_scale"]),
                internal_visual_transition_duration=float(
                    settings["internal_visual_transition_duration"]
                ),
                foreground_brightness=float(settings["foreground_brightness"]),
                foreground_contrast=float(settings["foreground_contrast"]),
                foreground_saturation=float(settings["foreground_saturation"]),
            )
        )
    manifest = {
        "source_video": str(source_video),
        "stage3_output": str(stage3_path),
        "output_dir": str(output_dir),
        "resolution": {"width": args.width, "height": args.height},
        "renderer": {
            "type": "ffmpeg_draft_renderer",
            "ffmpeg_bin": FFMPEG_BIN,
            "ffmpeg_hwaccel": FFMPEG_HWACCEL or "none",
            "use_plan_render_settings": args.use_plan_render_settings,
            "sidecar_captions": True,
            "notes": [
                "Uses stage3 timeline source ranges in order.",
                "Separates dialogue audio from visual B-roll when visual_source provides usable time cues.",
                "Builds multi-cut B-roll segments from parsed visual ranges and anchors.",
                "Can repair planned silent leads by borrowing immediate narration from the following beat.",
                "Can render silent generated title, hook, and CTA beats as baked-in black title cards.",
                "Can render short audio/video crossfades between planned segments.",
                "Always applies a final video fade to black after assembling the full clip.",
                "Can render short visual dissolves between internal multi-cut B-roll spans.",
                "Can punch in the foreground source frame to improve vertical legibility.",
                "Can lift foreground exposure and contrast for dark source footage.",
                "Can use explicit caption_plan render-time entries for phrase-level sidecar captions.",
                "Uses blurred 9:16 background with centered original landscape frame.",
                "Adds simple overlay text from caption_or_overlay when ffmpeg drawtext is available.",
                "Writes sidecar SRT captions for every rendered clip.",
                "Does not add licensed music or advanced motion graphics.",
            ],
        },
        "clips": results,
    }
    write_json(output_dir / "render-manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render draft local clips from OpenAI stage 3 edit plans.")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--stage3-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--font-file", type=Path, default=DEFAULT_FONT_FILE)
    parser.add_argument(
        "--ffmpeg-bin",
        default=DEFAULT_FFMPEG_BIN,
        help=(
            "ffmpeg executable. Use auto to prefer "
            "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg when present."
        ),
    )
    parser.add_argument(
        "--ffmpeg-hwaccel",
        choices=FFMPEG_HWACCEL_CHOICES,
        default=DEFAULT_FFMPEG_HWACCEL,
        help="Hardware decode mode for source video rendering.",
    )
    parser.add_argument(
        "--burn-captions",
        action="store_true",
        help="Burn caption_or_overlay text into the video when ffmpeg has drawtext.",
    )
    parser.add_argument(
        "--use-plan-render-settings",
        action="store_true",
        help="Use render_settings from the stage plan unless a matching CLI flag overrides it.",
    )
    parser.add_argument(
        "--repair-silent-lead-clip",
        action="append",
        default=[],
        help="Clip id whose silent opening beat should borrow immediate narration from the following beat.",
    )
    parser.add_argument(
        "--black-title-card-clip",
        action="append",
        default=[],
        help="Clip id whose first silent opening beat should render as a baked-in black title card.",
    )
    parser.add_argument(
        "--transition-duration",
        type=float,
        default=None,
        help="Optional seconds of audio/video crossfade between planned segments.",
    )
    parser.add_argument(
        "--foreground-scale",
        type=float,
        default=None,
        help="Scale foreground source above 1.0 for a vertical punch-in while retaining blurred background.",
    )
    parser.add_argument(
        "--internal-visual-transition-duration",
        type=float,
        default=None,
        help="Optional seconds of visual dissolve between internal multi-cut B-roll spans.",
    )
    parser.add_argument(
        "--foreground-brightness",
        type=float,
        default=None,
        help="Optional ffmpeg eq brightness adjustment for the centered foreground frame.",
    )
    parser.add_argument(
        "--foreground-contrast",
        type=float,
        default=None,
        help="Optional ffmpeg eq contrast adjustment for the centered foreground frame.",
    )
    parser.add_argument(
        "--foreground-saturation",
        type=float,
        default=None,
        help="Optional ffmpeg eq saturation adjustment for the centered foreground frame.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
        args.ffmpeg_hwaccel = resolve_ffmpeg_hwaccel(
            args.ffmpeg_hwaccel, args.ffmpeg_bin
        ) or "none"
        configure_ffmpeg(args.ffmpeg_bin, args.ffmpeg_hwaccel)
        require_command("ffprobe")
        manifest = render_stage3(args)
        print(f"Wrote {manifest['output_dir']}")
        for clip in manifest["clips"]:
            probe = clip["probe"]
            duration = float(probe.get("format", {}).get("duration") or 0)
            size = int(probe.get("format", {}).get("size") or 0)
            print(f"  {clip['clip_id']}: {duration:.2f}s, {size / 1024 / 1024:.1f} MB")
            print(f"    {clip['output_path']}")
            print(f"    captions: {clip['srt_path']}")
            if clip.get("subtitled_output_path"):
                print(f"    subtitled: {clip['subtitled_output_path']}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
