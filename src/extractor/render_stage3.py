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
FINAL_VIDEO_FADE_OUT_DURATION = 1.0
MIN_TITLE_CARD_SECONDS = 2.5
CAPTION_FONT_SIZE = 44
CAPTION_LINE_SPACING = 8
CAPTION_VERTICAL_CENTER_RATIO = 0.66
TRANSCRIPT_CAPTION_MAX_WORDS = 7
TRANSCRIPT_CAPTION_MAX_CHARS = 48
TRANSCRIPT_CAPTION_MAX_SECONDS = 2.4
TRANSCRIPT_CAPTION_GAP_SECONDS = 0.55
CAPTION_PLAN_MAX_CHARS = 58
CAPTION_PLAN_MAX_SECONDS = 4.0
CAPTION_PLAN_MIN_SECONDS = 1.2
OPENING_DIALOGUE_LEAD_IN_SECONDS = 0.45
OPENING_DIALOGUE_MIN_LEAD_IN_SECONDS = 0.16
AUTO_FOREGROUND_SCALE_STRONG = 3.05
AUTO_FOREGROUND_SCALE_MODERATE = 2.35
AUTO_FOREGROUND_SCALE_CONSERVATIVE = 1.85
AUTO_FOREGROUND_SCALE_WIDE_MOTION = 1.35
MOTION_ANALYSIS_WIDTH = 160
MOTION_ANALYSIS_HEIGHT = 90
MOTION_ANALYSIS_FPS = 2.0
MOTION_ANALYSIS_MAX_WINDOWS = 10
MOTION_ANALYSIS_WINDOW_SECONDS = 4.0
MOTION_PIXEL_DELTA_THRESHOLD = 12
MOTION_MIN_TOTAL = 120_000.0
FOREGROUND_FOCUS_WIDTH = 160
FOREGROUND_FOCUS_HEIGHT = 90
FOREGROUND_FOCUS_FPS = 1.25
FOREGROUND_FOCUS_MAX_FRAMES = 6
FOREGROUND_FOCUS_MIN_TOTAL = 40_000.0
FOREGROUND_FOCUS_OFFCENTER_SCALE_GUARD = 0.05
FOREGROUND_FOCUS_VISIBLE_MARGIN = 0.08
PROJECTED_WIDE_OFFCENTER_FOCUS_GUARD = 0.17
PROJECTED_WIDE_MAX_FOREGROUND_SCALE = 1.45
FFMPEG_BIN = "ffmpeg"
FFMPEG_HWACCEL: str | None = None
AUGMENTATION_ASSETS: dict[str, dict[str, Any]] = {}
FOREGROUND_FOCUS_CACHE: dict[tuple[str, float, float], dict[str, Any]] = {}
SUBJECT_INDEX_MIN_CONFIDENCE = 0.34
SUBJECT_INDEX_CENTER_DAMPING = 0.4
SUBJECT_INDEX_MAX_CENTER_SHIFT = 0.08
SUBJECT_INDEX_CACHE: dict[Path, dict[str, Any]] = {}
ASSET_REF_RE = re.compile(
    r"asset://(?P<id>[A-Za-z0-9_.-]+)"
    r"(?:[#?](?:t|start)=(?P<start>\d+(?:\.\d+)?))?"
)
MEDIA_REQUEST_REF_RE = re.compile(r"media_request://(?P<id>[A-Za-z0-9_.-]+)")

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


def infer_asset_type(path: Path, raw_type: str | None = None) -> str:
    if raw_type:
        normalized = raw_type.strip().lower()
        if normalized in {"video", "image", "audio", "graphic"}:
            return "image" if normalized == "graphic" else normalized
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic"}:
        return "image"
    if suffix in {".wav", ".mp3", ".m4a", ".aac", ".aiff", ".flac", ".ogg"}:
        return "audio"
    return "video"


def load_augmentation_assets(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = read_json(path)
    raw_assets = data.get("assets", data)
    if isinstance(raw_assets, dict):
        items = [
            {"id": asset_id, **asset}
            for asset_id, asset in raw_assets.items()
            if isinstance(asset, dict)
        ]
    elif isinstance(raw_assets, list):
        items = [asset for asset in raw_assets if isinstance(asset, dict)]
    else:
        raise RuntimeError("Augmentation assets manifest must contain an assets list or object.")

    assets: dict[str, dict[str, Any]] = {}
    for asset in items:
        asset_id = str(asset.get("id", "")).strip()
        if not asset_id:
            raise RuntimeError("Augmentation asset is missing required id.")
        raw_path = asset.get("path")
        if not raw_path:
            raise RuntimeError(f"Augmentation asset {asset_id!r} is missing required path.")
        asset_path = Path(str(raw_path)).expanduser()
        if not asset_path.is_absolute():
            asset_path = path.parent / asset_path
        if not asset_path.exists():
            raise FileNotFoundError(asset_path)
        copied = dict(asset)
        copied["id"] = asset_id
        copied["path"] = str(asset_path.resolve())
        copied["type"] = infer_asset_type(asset_path, str(asset.get("type", "")) or None)
        assets[asset_id] = copied
    return assets


def configure_augmentation_assets(path: Path | None) -> dict[str, dict[str, Any]]:
    global AUGMENTATION_ASSETS
    AUGMENTATION_ASSETS = load_augmentation_assets(path)
    return AUGMENTATION_ASSETS


def asset_ref_from_text(text: str) -> dict[str, Any] | None:
    match = ASSET_REF_RE.search(text)
    if not match:
        return None
    asset_id = match.group("id").rstrip(".,;:)")
    start = float(match.group("start") or 0.0)
    return {"id": asset_id, "start": start}


def resolve_asset(ref: dict[str, Any]) -> dict[str, Any]:
    asset_id = str(ref["id"])
    asset = AUGMENTATION_ASSETS.get(asset_id)
    if asset is None:
        available = ", ".join(sorted(AUGMENTATION_ASSETS)) or "none"
        raise RuntimeError(f"Unknown augmentation asset {asset_id!r}. Available assets: {available}")
    return asset


def asset_ref_for_item(item: dict[str, Any], key: str) -> dict[str, Any] | None:
    return asset_ref_from_text(str(item.get(key, "")))


def media_request_ids_from_text(text: str) -> list[str]:
    ids: list[str] = []
    for match in MEDIA_REQUEST_REF_RE.finditer(text):
        media_id = match.group("id").rstrip(".,;:)")
        if media_id and media_id not in ids:
            ids.append(media_id)
    return ids


def media_request_ids_for_item(item: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("audio_source", "visual_source", "edit_action", "purpose"):
        for media_id in media_request_ids_from_text(str(item.get(key, ""))):
            if media_id not in ids:
                ids.append(media_id)
    return ids


def asset_volume(asset: dict[str, Any], default: float = 0.14) -> float:
    raw = asset.get("volume", asset.get("default_volume", default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


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


def ass_timestamp(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_text(text: str) -> str:
    escaped = text.replace("{", "").replace("}", "")
    return wrap_caption(escaped, width=28).replace("\n", r"\N")


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


def write_ass_entries(path: Path, entries: list[dict[str, Any]], width: int, height: int) -> None:
    margin_v = max(40, int(round(height * (1.0 - CAPTION_VERTICAL_CENTER_RATIO))))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: PhoneSafe,Arial,"
            f"{CAPTION_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H99000000,"
            "0,0,0,0,100,100,0,0,3,1,0,2,36,36,"
            f"{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for entry in entries:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        start = float(entry["start"])
        end = float(entry["end"])
        if end <= start:
            continue
        lines.append(
            f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},"
            f"PhoneSafe,,0,0,0,,{ass_text(text)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sidecar_timeline_to_caption_entries(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in timeline:
        text = str(item.get("text") or item.get("caption_or_overlay") or "").strip()
        if not text:
            continue
        raw_start = item.get("start", item.get("screen_time_start"))
        raw_end = item.get("end", item.get("screen_time_end"))
        if raw_start is None or raw_end is None:
            continue
        start = float(raw_start)
        end = float(raw_end)
        if end <= start:
            continue
        entries.append({"start": start, "end": end, "text": text})
    return entries


def burn_ass_captions(input_path: Path, ass_path: Path, output_path: Path) -> None:
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
            f"ass={ass_path}",
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


def is_generated_silent_card_segment(item: dict[str, Any]) -> bool:
    if source_duration(item) > 0.05 or uses_dialogue(item):
        return False
    text = " ".join(
        str(item.get(key, "")).lower()
        for key in ("edit_action", "visual_source", "audio_source", "purpose")
    )
    return any(
        marker in text
        for marker in (
            "generated",
            "media_request://",
            "graphic",
            "context card",
            "silent card",
        )
    )


def should_render_title_card(item: dict[str, Any], force_first_card: bool = False) -> bool:
    if force_first_card and not uses_dialogue(item):
        return True
    if is_generated_silent_card_segment(item):
        return True
    if not is_title_card_segment(item):
        return False
    return source_duration(item) <= 0.05 or not uses_dialogue(item)


def shift_caption_plan(caption_plan: list[Any], delta: float) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for item in caption_plan:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        raw_start = copied.get("start_offset", copied.get("start"))
        raw_end = copied.get("end_offset", copied.get("end"))
        if raw_start is not None and raw_end is not None:
            start = float(raw_start) + delta
            end = float(raw_end) + delta
            if "start_offset" in copied or "end_offset" in copied:
                copied["start_offset"] = round(start, 3)
                copied["end_offset"] = round(end, 3)
            else:
                copied["start"] = round(start, 3)
                copied["end"] = round(end, 3)
        shifted.append(copied)
    return shifted


def ensure_opening_title_card(
    clip: dict[str, Any],
    required: bool,
    duration: float = MIN_TITLE_CARD_SECONDS,
) -> tuple[dict[str, Any], bool]:
    if not required:
        return clip, False
    timeline = sorted(clip.get("timeline", []), key=lambda item: int(item.get("order", 0)))
    if not timeline or should_render_title_card(timeline[0]):
        return clip, False

    title = str(clip.get("concept_title") or clip.get("opening_hook", {}).get("text") or "").strip()
    if not title:
        title = "Excerpt"

    adjusted = deepcopy(clip)
    adjusted_timeline: list[dict[str, Any]] = [
        {
            "order": 1,
            "screen_time_start": 0.0,
            "screen_time_end": round(duration, 3),
            "source_start": 0.0,
            "source_end": 0.0,
            "audio_source": "No source dialogue; silent generated title card.",
            "visual_source": "Generated black title card.",
            "edit_action": "Opening title card.",
            "caption_or_overlay": title,
            "purpose": "Identify the excerpt topic before source dialogue begins.",
        }
    ]
    for index, item in enumerate(timeline, start=2):
        copied = dict(item)
        copied["order"] = index
        copied["screen_time_start"] = round(float(copied["screen_time_start"]) + duration, 3)
        copied["screen_time_end"] = round(float(copied["screen_time_end"]) + duration, 3)
        adjusted_timeline.append(copied)

    adjusted["timeline"] = adjusted_timeline
    adjusted["caption_plan"] = shift_caption_plan(adjusted.get("caption_plan", []) or [], duration)
    return adjusted, True


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


def opening_lead_in_available_seconds(
    source_start: float,
    transcript_words: list[dict[str, Any]],
    requested: float,
) -> float:
    if source_start <= 0:
        return 0.0
    if not transcript_words:
        return min(requested, source_start)
    previous_end = 0.0
    for word in transcript_words:
        try:
            word_end = float(word["end"])
            word_start = float(word["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if word_start >= source_start - 0.001:
            break
        previous_end = max(previous_end, word_end)
    available = source_start - previous_end - 0.04
    return max(0.0, min(requested, available, source_start))


def add_opening_dialogue_lead_in(
    clip: dict[str, Any],
    transcript_words: list[dict[str, Any]],
    requested_seconds: float = OPENING_DIALOGUE_LEAD_IN_SECONDS,
) -> tuple[dict[str, Any], float]:
    timeline = sorted(clip.get("timeline", []), key=lambda item: int(item.get("order", 0)))
    if len(timeline) < 2 or not should_render_title_card(timeline[0]):
        return clip, 0.0
    dialogue_index = None
    for index, item in enumerate(timeline[1:], start=1):
        if uses_dialogue(item) and not should_render_title_card(item):
            dialogue_index = index
            break
        if timeline_duration(item) > 0.05:
            return clip, 0.0
    if dialogue_index is None:
        return clip, 0.0

    first_dialogue = timeline[dialogue_index]
    try:
        source_start = float(first_dialogue["source_start"])
    except (KeyError, TypeError, ValueError):
        return clip, 0.0
    lead_in = opening_lead_in_available_seconds(
        source_start,
        transcript_words,
        requested_seconds,
    )
    if lead_in < OPENING_DIALOGUE_MIN_LEAD_IN_SECONDS:
        return clip, 0.0

    adjusted = deepcopy(clip)
    adjusted_timeline: list[dict[str, Any]] = []
    first_dialogue_start = float(first_dialogue["screen_time_start"])
    for index, item in enumerate(timeline):
        copied = dict(item)
        if index == dialogue_index:
            copied["source_start"] = round(source_start - lead_in, 3)
            copied["screen_time_end"] = round(float(copied["screen_time_end"]) + lead_in, 3)
            copied["opening_dialogue_lead_in_seconds"] = round(lead_in, 3)
        elif index > dialogue_index:
            copied["screen_time_start"] = round(float(copied["screen_time_start"]) + lead_in, 3)
            copied["screen_time_end"] = round(float(copied["screen_time_end"]) + lead_in, 3)
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
            if start >= first_dialogue_start - 0.001:
                start += lead_in
                end += lead_in
                if "start_offset" in copied or "end_offset" in copied:
                    copied["start_offset"] = round(start, 3)
                    copied["end_offset"] = round(end, 3)
                else:
                    copied["start"] = round(start, 3)
                    copied["end"] = round(end, 3)
        adjusted_captions.append(copied)
    adjusted["caption_plan"] = adjusted_captions
    return adjusted, round(lead_in, 3)


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


def timeline_actual_positions(
    timeline: list[dict[str, Any]],
    transition_duration: float,
) -> list[tuple[dict[str, Any], float, float]]:
    valid_items = [
        (item, timeline_duration(item))
        for item in timeline
        if timeline_duration(item) > 0.04
    ]
    current = 0.0
    positions: list[tuple[dict[str, Any], float, float]] = []
    for index, (item, duration) in enumerate(valid_items):
        positions.append((item, current, duration))
        current += duration
        if index < len(valid_items) - 1:
            current = max(0.0, current - transition_duration)
    return positions


def render_time_for_transition_timeline(
    render_time: float,
    positions: list[tuple[dict[str, Any], float, float]],
) -> float:
    if not positions:
        return render_time

    first_item, first_actual_start, _ = positions[0]
    first_plan_start = float(first_item["screen_time_start"])
    if render_time <= first_plan_start:
        return max(0.0, first_actual_start + render_time - first_plan_start)

    for index, (item, actual_start, duration) in enumerate(positions):
        plan_start = float(item["screen_time_start"])
        plan_end = float(item["screen_time_end"])
        is_last = index == len(positions) - 1
        if plan_start <= render_time < plan_end or (is_last and render_time <= plan_end):
            return max(0.0, actual_start + render_time - plan_start)
        if render_time < plan_start:
            return max(0.0, actual_start)

    last_item, last_actual_start, last_duration = positions[-1]
    last_plan_end = float(last_item["screen_time_end"])
    return max(0.0, last_actual_start + last_duration + render_time - last_plan_end)


def retime_caption_entries_for_transitions(
    entries: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    transition_duration: float,
) -> list[dict[str, Any]]:
    if transition_duration <= 0 or len(timeline) < 2:
        return entries

    positions = timeline_actual_positions(timeline, transition_duration)
    retimed: list[dict[str, Any]] = []
    for entry in entries:
        start = render_time_for_transition_timeline(float(entry["start"]), positions)
        end = render_time_for_transition_timeline(float(entry["end"]), positions)
        if end - start < 0.05:
            continue
        retimed.append({**entry, "start": round(start, 3), "end": round(end, 3)})
    return sorted(retimed, key=lambda item: (float(item["start"]), float(item["end"])))


def card_plan_time_ranges(
    timeline: list[dict[str, Any]],
    padding_seconds: float = 0.0,
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for item in timeline:
        if should_render_title_card(item):
            ranges.append(
                (
                    max(0.0, float(item["screen_time_start"]) - padding_seconds),
                    float(item["screen_time_end"]) + padding_seconds,
                )
            )
    return ranges


def caption_entries_for_burn(
    entries: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    card_ranges = card_plan_time_ranges(timeline)
    burned: list[dict[str, Any]] = []
    for entry in entries:
        start = float(entry["start"])
        end = float(entry["end"])
        discard = False
        for range_start, range_end in card_ranges:
            if end <= range_start or start >= range_end:
                continue
            if start >= range_start and end <= range_end:
                discard = True
                break
            if start < range_start < end:
                end = min(end, range_start)
            if start < range_end < end:
                start = max(start, range_end)
        if discard or end - start < 0.25:
            continue
        burned.append({**entry, "start": round(start, 3), "end": round(end, 3)})
    return burned


def suppress_entries_during_final_fade(
    entries: list[dict[str, Any]],
    duration: float,
    fade_duration: float,
    *,
    start_key: str,
    end_key: str,
    min_duration: float = 0.25,
) -> list[dict[str, Any]]:
    if duration <= 0 or fade_duration <= 0:
        return [dict(entry) for entry in entries]

    cutoff = max(0.0, duration - fade_duration)
    suppressed: list[dict[str, Any]] = []
    for entry in entries:
        if start_key not in entry or end_key not in entry:
            continue
        start = float(entry[start_key])
        end = float(entry[end_key])
        if start >= cutoff:
            continue
        if end > cutoff:
            end = cutoff
        if end - start < min_duration:
            continue
        copied = dict(entry)
        copied[start_key] = round(start, 3)
        copied[end_key] = round(end, 3)
        suppressed.append(copied)
    return suppressed


def split_caption_text_for_plan(text: str, max_chars: int = CAPTION_PLAN_MAX_CHARS) -> list[str]:
    words = text.strip().split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        hard_break = len(candidate) > max_chars and current
        soft_break = (
            current
            and current[-1].rstrip().endswith((".", "?", "!", ";", ":"))
            and len(" ".join(current)) >= max_chars * 0.45
        )
        if hard_break or soft_break:
            chunks.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def split_caption_text_evenly(text: str, chunk_count: int) -> list[str]:
    words = text.strip().split()
    if not words:
        return []
    chunk_count = max(1, min(chunk_count, len(words)))
    if chunk_count == 1:
        return [" ".join(words)]
    chunks: list[str] = []
    for index in range(chunk_count):
        start = round(index * len(words) / chunk_count)
        end = round((index + 1) * len(words) / chunk_count)
        if end <= start:
            end = min(len(words), start + 1)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def split_caption_plan_entry(start: float, end: float, text: str) -> list[dict[str, Any]]:
    duration = max(0.0, end - start)
    chunks = split_caption_text_for_plan(text)
    required_chunk_count = max(1, int(math.ceil(duration / CAPTION_PLAN_MAX_SECONDS)))
    if len(chunks) < required_chunk_count:
        chunks = split_caption_text_evenly(text, required_chunk_count)
    if (
        not chunks
        or duration <= CAPTION_PLAN_MAX_SECONDS + 0.2
        or (len(text) <= CAPTION_PLAN_MAX_CHARS and len(chunks) <= 1)
    ):
        return [{"start": round(start, 3), "end": round(end, 3), "text": text}]

    entries: list[dict[str, Any]] = []
    cursor = start
    chunk_duration = duration / max(1, len(chunks))
    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1:
            chunk_end = end
        else:
            chunk_end = cursor + chunk_duration
        if chunk_end - cursor >= 0.25:
            entries.append(
                {
                    "start": round(cursor, 3),
                    "end": round(chunk_end, 3),
                    "text": chunk,
                }
            )
        cursor = chunk_end
    return entries


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
        entries.extend(split_caption_plan_entry(start, end, text))
    return sorted(entries, key=lambda item: (float(item["start"]), float(item["end"])))


def normalize_transcript_word(word: dict[str, Any]) -> dict[str, Any] | None:
    if str(word.get("type", "word")).lower() == "spacing":
        return None
    text = str(word.get("text", word.get("word", ""))).strip()
    if not text:
        return None
    start = word.get("start")
    end = word.get("end")
    if start is None or end is None:
        return None
    start_float = float(start)
    end_float = float(end)
    if end_float <= start_float:
        return None
    return {
        "text": text,
        "start": round(start_float, 3),
        "end": round(end_float, 3),
    }


def transcript_words_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for word in data.get("words", []) or []:
        normalized = normalize_transcript_word(word)
        if normalized:
            words.append(normalized)

    if words:
        return sorted(words, key=lambda item: (float(item["start"]), float(item["end"])))

    for segment in data.get("segments", []) or []:
        for word in segment.get("words", []) or []:
            normalized = normalize_transcript_word(word)
            if normalized:
                words.append(normalized)
    return sorted(words, key=lambda item: (float(item["start"]), float(item["end"])))


def infer_transcript_path(source_video: Path, stage3_path: Path) -> Path | None:
    roots = []
    if len(stage3_path.parents) >= 3:
        roots.append(stage3_path.parents[2])
    roots.append(source_video.parent)

    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        transcript_dir = root / "transcripts"
        if transcript_dir in seen or not transcript_dir.exists():
            continue
        seen.add(transcript_dir)
        for path in sorted(transcript_dir.glob("*.json")):
            name = path.name
            if name.endswith(".available.json"):
                continue
            candidates.append(path)

    if not candidates:
        return None

    source_stem = source_video.stem
    matching = [path for path in candidates if path.name.startswith(source_stem)]
    if matching:
        return matching[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def phrase_text(words: list[dict[str, Any]]) -> str:
    ignored_bracketed_tokens = {
        "applause",
        "audience laughing",
        "clears throat",
        "cough",
        "coughs",
        "laugh",
        "laughs",
        "laughing",
        "laughter",
        "marker squeaking",
        "music",
        "sigh",
        "writing",
    }
    raw_tokens: list[str] = []
    for word in words:
        token = str(word["text"]).strip()
        if not token:
            continue
        bracketed = re.fullmatch(r"\[([^\]]+)\]", token)
        if bracketed and bracketed.group(1).strip().lower() in ignored_bracketed_tokens:
            continue
        raw_tokens.append(token)
    text = " ".join(raw_tokens)
    text = re.sub(
        r"(?i)\s*\[(?:applause|audience laughing|clears throat|coughs?|laughs?|laughing|laughter|marker squeaking|music|sigh|writing)\]\s*",
        " ",
        text,
    )
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r'([?.!])"?(?=[A-Z])', r"\1 ", text)
    text = re.sub(r"\.(?=[A-Z])", ". ", text)
    text = re.sub(r"(?i)(\d)uh\b,?", r"\1", text)
    text = re.sub(r"(?i)\b(uh|um|erm),?\s+", "", text)
    text = re.sub(r"(?i)\blimb the details\b", "limn the details", text)
    text = re.sub(r"(?i)\bgives couple\b", "gives a couple", text)
    text = re.sub(r"(?i)\bhad have gotten\b", "had gotten", text)
    text = re.sub(r"\b([A-Za-z]+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def should_break_transcript_phrase(
    current: list[dict[str, Any]],
    word: dict[str, Any],
) -> bool:
    if not current:
        return False
    previous = current[-1]
    gap = float(word["start"]) - float(previous["end"])
    candidate_words = [*current, word]
    candidate_text = phrase_text(candidate_words)
    candidate_duration = float(word["end"]) - float(current[0]["start"])
    previous_text = str(previous["text"]).rstrip()
    if gap > TRANSCRIPT_CAPTION_GAP_SECONDS:
        return True
    if len(current) >= TRANSCRIPT_CAPTION_MAX_WORDS:
        return True
    if (
        len(candidate_text) > TRANSCRIPT_CAPTION_MAX_CHARS
        and len(current) >= 2
    ):
        return True
    if (
        candidate_duration > TRANSCRIPT_CAPTION_MAX_SECONDS
        and len(current) >= 2
    ):
        return True
    if (
        previous_text.endswith((".", "?", "!"))
        and float(previous["end"]) - float(current[0]["start"]) >= 0.8
    ):
        return True
    return False


def timed_words_to_caption_entries(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if should_break_transcript_phrase(current, word):
            text = phrase_text(current)
            if text:
                phrases.append(
                    {
                        "start": round(float(current[0]["start"]), 3),
                        "end": round(float(current[-1]["end"]), 3),
                        "text": text,
                    }
                )
            current = []
        current.append(word)

    if current:
        text = phrase_text(current)
        if text:
            phrases.append(
                {
                    "start": round(float(current[0]["start"]), 3),
                    "end": round(float(current[-1]["end"]), 3),
                    "text": text,
                }
            )

    for index in range(len(phrases) - 1):
        next_start = float(phrases[index + 1]["start"])
        if float(phrases[index]["end"]) > next_start - 0.02:
            phrases[index]["end"] = round(max(float(phrases[index]["start"]) + 0.2, next_start - 0.02), 3)
    for phrase in phrases:
        start = float(phrase["start"])
        end = float(phrase["end"])
        if end - start > TRANSCRIPT_CAPTION_MAX_SECONDS:
            phrase["end"] = round(start + TRANSCRIPT_CAPTION_MAX_SECONDS, 3)
    return [
        phrase
        for phrase in phrases
        if float(phrase["end"]) > float(phrase["start"])
    ]


def transcript_caption_entries_for_clip(
    timeline: list[dict[str, Any]],
    transcript_words: list[dict[str, Any]],
    transition_duration: float,
) -> list[dict[str, Any]]:
    if not transcript_words:
        return []

    timed_words: list[dict[str, Any]] = []
    for item, actual_start, duration in timeline_actual_positions(
        timeline,
        transition_duration,
    ):
        if should_render_title_card(item) or not uses_dialogue(item):
            continue
        source_start = float(item["source_start"])
        source_end = source_start + duration
        for word in transcript_words:
            word_start = float(word["start"])
            word_end = float(word["end"])
            if word_end <= source_start + 0.02:
                continue
            if word_start >= source_end - 0.02:
                break
            overlap_start = max(word_start, source_start)
            overlap_end = min(word_end, source_end)
            if overlap_end - overlap_start <= 0.03:
                continue
            render_start = actual_start + max(0.0, overlap_start - source_start)
            render_end = actual_start + min(duration, overlap_end - source_start)
            if render_end <= render_start:
                render_end = render_start + 0.08
            timed_words.append(
                {
                    "text": word["text"],
                    "start": round(render_start, 3),
                    "end": round(render_end, 3),
                }
            )

    return timed_words_to_caption_entries(
        sorted(timed_words, key=lambda item: (float(item["start"]), float(item["end"])))
    )


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
            "avoid showing",
            "avoid talking head",
            "avoid static",
            "quick visual cuts",
            "cut each",
            "cold open",
            "underwater visuals",
            "reordered",
        )
    )


def build_visual_spans(item: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    asset_ref = asset_ref_for_item(item, "visual_source")
    if asset_ref is not None:
        asset = resolve_asset(asset_ref)
        asset_type = str(asset.get("type", "video"))
        if asset_type not in {"video", "image"}:
            raise RuntimeError(
                f"Augmentation asset {asset_ref['id']!r} is type {asset_type!r}; "
                "visual_source assets must be video or image."
            )
        return [
            {
                "asset_id": str(asset_ref["id"]),
                "asset_path": str(asset["path"]),
                "asset_type": asset_type,
                "source_start": float(asset_ref.get("start", 0.0)),
                "duration": duration,
                "reason": "augmentation_asset",
            }
        ]

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


def source_backed_timeline_items(clip: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    timeline = sorted(clip.get("timeline", []), key=lambda item: int(item.get("order", 0)))
    for item in timeline:
        try:
            has_source = source_duration(item) > 0.05
        except (KeyError, TypeError, ValueError):
            has_source = False
        if has_source and not should_render_title_card(item):
            items.append(item)
    return items


def foreground_crop_bounds(scale: float, width: int = MOTION_ANALYSIS_WIDTH) -> tuple[int, int]:
    visible_fraction = min(1.0, 1.0 / max(1.0, scale))
    crop_width = max(1, min(width, int(round(width * visible_fraction))))
    left = max(0, (width - crop_width) // 2)
    return left, min(width, left + crop_width)


def motion_fraction_outside_crop(columns: list[float], scale: float) -> float:
    total = sum(columns)
    if total <= 0:
        return 0.0
    left, right = foreground_crop_bounds(scale, len(columns))
    inside = sum(columns[left:right])
    return max(0.0, min(1.0, (total - inside) / total))


def weighted_motion_bounds(columns: list[float]) -> tuple[float, float, float]:
    total = sum(columns)
    if total <= 0:
        return 0.0, 1.0, 1.0

    lower_target = total * 0.05
    upper_target = total * 0.95
    lower = 0
    upper = len(columns) - 1
    cumulative = 0.0
    for index, value in enumerate(columns):
        cumulative += value
        if cumulative >= lower_target:
            lower = index
            break
    cumulative = 0.0
    for index, value in enumerate(columns):
        cumulative += value
        if cumulative >= upper_target:
            upper = index
            break
    divisor = max(1, len(columns) - 1)
    lower_norm = lower / divisor
    upper_norm = upper / divisor
    return lower_norm, upper_norm, max(0.0, upper_norm - lower_norm)


def weighted_column_center(columns: list[float]) -> float:
    total = sum(columns)
    if total <= 0 or len(columns) <= 1:
        return 0.5
    divisor = len(columns) - 1
    return max(
        0.0,
        min(
            1.0,
            sum((index / divisor) * value for index, value in enumerate(columns)) / total,
        ),
    )


def strongest_column_window(
    columns: list[float],
    window_fraction: float = 0.42,
) -> dict[str, float]:
    if not columns:
        return {
            "center": 0.5,
            "left": 0.0,
            "right": 1.0,
            "score": 0.0,
            "score_ratio": 0.0,
        }
    total = sum(columns)
    if total <= 0:
        return {
            "center": 0.5,
            "left": 0.0,
            "right": 1.0,
            "score": 0.0,
            "score_ratio": 0.0,
        }
    window_size = max(3, min(len(columns), int(round(len(columns) * window_fraction))))
    current = sum(columns[:window_size])
    best_score = current
    best_left = 0
    for left in range(1, len(columns) - window_size + 1):
        current += columns[left + window_size - 1] - columns[left - 1]
        if current > best_score:
            best_score = current
            best_left = left
    best_right = best_left + window_size
    window_columns = columns[best_left:best_right]
    window_total = sum(window_columns)
    if window_total <= 0:
        center_index = best_left + (window_size - 1) / 2
    else:
        center_index = sum(
            (best_left + index) * value for index, value in enumerate(window_columns)
        ) / window_total
    divisor = max(1, len(columns) - 1)
    return {
        "center": max(0.0, min(1.0, center_index / divisor)),
        "left": best_left / divisor,
        "right": min(1.0, (best_right - 1) / divisor),
        "score": best_score,
        "score_ratio": best_score / total,
    }


def visual_analysis_windows_for_clip(clip: dict[str, Any]) -> list[tuple[float, float, str]]:
    windows: list[tuple[float, float, str]] = []
    for item in source_backed_timeline_items(clip):
        try:
            spans = build_visual_spans(item, timeline_duration(item))
        except Exception:
            continue
        for span in spans:
            if span.get("asset_id"):
                continue
            source_start = float(span["source_start"])
            duration = float(span["duration"])
            if duration <= 0.2:
                continue
            reason = str(span.get("reason", "timeline_source"))
            if duration <= MOTION_ANALYSIS_WINDOW_SECONDS:
                windows.append((source_start, duration, reason))
            else:
                window_duration = min(MOTION_ANALYSIS_WINDOW_SECONDS, duration)
                starts = [
                    source_start,
                    source_start + max(0.0, (duration - window_duration) / 2),
                    source_start + max(0.0, duration - window_duration),
                ]
                for start in starts:
                    if not any(abs(start - existing[0]) < 0.4 for existing in windows):
                        windows.append((start, window_duration, reason))
            if len(windows) >= MOTION_ANALYSIS_MAX_WINDOWS:
                return windows[:MOTION_ANALYSIS_MAX_WINDOWS]
    return windows[:MOTION_ANALYSIS_MAX_WINDOWS]


def sample_motion_columns_for_window(
    source_video: Path,
    source_start: float,
    duration: float,
) -> tuple[list[float], int]:
    frame_size = MOTION_ANALYSIS_WIDTH * MOTION_ANALYSIS_HEIGHT
    frame_count = max(2, int(math.ceil(duration * MOTION_ANALYSIS_FPS)) + 1)
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
        str(source_video),
        "-vf",
        (
            f"fps={MOTION_ANALYSIS_FPS:.3f},"
            f"scale={MOTION_ANALYSIS_WIDTH}:{MOTION_ANALYSIS_HEIGHT},"
            "format=gray"
        ),
        "-frames:v",
        str(frame_count),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        return [0.0] * MOTION_ANALYSIS_WIDTH, 0

    payload = result.stdout
    frames = [
        payload[index : index + frame_size]
        for index in range(0, len(payload) - frame_size + 1, frame_size)
    ]
    if len(frames) < 2:
        return [0.0] * MOTION_ANALYSIS_WIDTH, len(frames)

    columns = [0.0] * MOTION_ANALYSIS_WIDTH
    previous = frames[0]
    for frame in frames[1:]:
        for offset, (current_value, previous_value) in enumerate(zip(frame, previous)):
            delta = abs(current_value - previous_value)
            if delta >= MOTION_PIXEL_DELTA_THRESHOLD:
                columns[offset % MOTION_ANALYSIS_WIDTH] += float(delta)
        previous = frame
    return columns, len(frames)


def smooth_columns(columns: list[float]) -> list[float]:
    if len(columns) < 3:
        return columns
    smoothed = columns[:]
    for index in range(1, len(columns) - 1):
        smoothed[index] = (
            columns[index - 1] * 0.25
            + columns[index] * 0.5
            + columns[index + 1] * 0.25
        )
    return smoothed


def sample_foreground_focus_columns_for_window(
    source_video: Path,
    source_start: float,
    duration: float,
) -> tuple[list[float], int]:
    width = FOREGROUND_FOCUS_WIDTH
    height = FOREGROUND_FOCUS_HEIGHT
    frame_size = width * height * 3
    frame_count = max(
        1,
        min(
            FOREGROUND_FOCUS_MAX_FRAMES,
            int(math.ceil(duration * FOREGROUND_FOCUS_FPS)) + 1,
        ),
    )
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
        str(source_video),
        "-vf",
        (
            f"fps={FOREGROUND_FOCUS_FPS:.3f},"
            f"scale={width}:{height},"
            "format=rgb24"
        ),
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
        result = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        return [0.0] * width, 0

    payload = result.stdout
    frames = [
        payload[index : index + frame_size]
        for index in range(0, len(payload) - frame_size + 1, frame_size)
    ]
    columns = [0.0] * width
    for frame in frames:
        luma_rows: list[list[int]] = []
        for y in range(height):
            row: list[int] = []
            row_start = y * width * 3
            for x in range(width):
                offset = row_start + x * 3
                red = frame[offset]
                green = frame[offset + 1]
                blue = frame[offset + 2]
                row.append(int(0.2126 * red + 0.7152 * green + 0.0722 * blue))
            luma_rows.append(row)

        for y, row in enumerate(luma_rows):
            y_norm = y / max(1, height - 1)
            if y_norm < 0.08 or y_norm > 0.95:
                continue
            row_median = sorted(row)[width // 2]
            vertical_weight = 0.35 + 0.65 * (
                1.0 - min(1.0, abs(y_norm - 0.55) / 0.45)
            )
            previous_row = luma_rows[y - 1] if y > 0 else row
            next_row = luma_rows[y + 1] if y < height - 1 else row
            for x, value in enumerate(row):
                left = row[x - 1] if x > 0 else value
                right = row[x + 1] if x < width - 1 else value
                dx = abs(right - left)
                dy = abs(next_row[x] - previous_row[x])
                contrast_score = max(0.0, float(abs(value - row_median) - 10))
                edge_score = max(0.0, float(dx - 8)) * 0.55 + max(0.0, float(dy - 8)) * 0.35
                columns[x] += (contrast_score + edge_score) * vertical_weight

    return smooth_columns(columns), len(frames)


def foreground_focus_for_window(
    source_video: Path,
    source_start: float,
    duration: float,
) -> dict[str, Any]:
    cache_key = (
        str(source_video),
        round(float(source_start), 2),
        round(float(duration), 2),
    )
    cached = FOREGROUND_FOCUS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    columns, frame_count = sample_foreground_focus_columns_for_window(
        source_video,
        source_start,
        duration,
    )
    total = sum(columns)
    left, right, width = weighted_motion_bounds(columns)
    full_center = weighted_column_center(columns)
    peak = strongest_column_window(columns)
    available = (
        frame_count > 0
        and total >= FOREGROUND_FOCUS_MIN_TOTAL
        and (
            width <= 0.78
            or float(peak["score_ratio"]) >= 0.38
        )
    )
    center = float(peak["center"]) if available else full_center
    result = {
        "available": available,
        "method": "visual_saliency",
        "center_x": round(center if available else 0.5, 3),
        "full_center_x": round(full_center, 3),
        "bounds_left": round(left, 3),
        "bounds_right": round(right, 3),
        "bounds_width": round(width, 3),
        "peak_left": round(float(peak["left"]), 3),
        "peak_right": round(float(peak["right"]), 3),
        "peak_score_ratio": round(float(peak["score_ratio"]), 3),
        "frame_count": frame_count,
        "total_score": round(total, 3),
    }
    FOREGROUND_FOCUS_CACHE[cache_key] = result
    return result


def focus_adjusted_foreground_scale(
    foreground_scale: float,
    foreground_focus: dict[str, Any] | None,
) -> tuple[float, str | None]:
    if foreground_scale <= 1.0 or not foreground_focus or not foreground_focus.get("available"):
        return foreground_scale, None

    try:
        center_x = float(foreground_focus.get("center_x", 0.5))
        peak_left = float(foreground_focus.get("peak_left", center_x))
        peak_right = float(foreground_focus.get("peak_right", center_x))
    except (TypeError, ValueError):
        return foreground_scale, None

    off_center = abs(center_x - 0.5)
    if off_center < FOREGROUND_FOCUS_OFFCENTER_SCALE_GUARD:
        return foreground_scale, None

    peak_width = max(0.0, min(1.0, peak_right) - max(0.0, peak_left))
    required_visible_width = max(
        0.38,
        min(0.72, peak_width + FOREGROUND_FOCUS_VISIBLE_MARGIN),
    )
    max_scale_for_focus = max(1.0, 1.0 / required_visible_width)
    adjusted_scale = min(foreground_scale, max_scale_for_focus)
    if adjusted_scale >= foreground_scale - 0.01:
        return foreground_scale, None

    reason = (
        "capped span punch-in for off-center focus "
        f"(center_x={center_x:.2f}, peak_width={peak_width:.2f})"
    )
    return adjusted_scale, reason


def item_prefers_projected_wide_framing(item: dict[str, Any]) -> bool:
    text = " ".join(
        str(item.get(key, ""))
        for key in ("visual_source", "edit_action", "purpose", "caption_or_overlay")
    ).lower()
    projected_markers = (
        "slide",
        "projected",
        "projection",
        "screen",
        "map",
        "plan",
        "diagram",
        "whiteboard",
        "blackboard",
        "chalkboard",
    )
    return any(marker in text for marker in projected_markers)


def projected_wide_focus_adjustment(
    item: dict[str, Any],
    foreground_scale: float,
    foreground_center_x: float,
) -> tuple[float, float, str | None]:
    if foreground_scale <= 1.0 or not item_prefers_projected_wide_framing(item):
        return foreground_scale, foreground_center_x, None
    if abs(foreground_center_x - 0.5) < PROJECTED_WIDE_OFFCENTER_FOCUS_GUARD:
        return foreground_scale, foreground_center_x, None

    adjusted_scale = min(foreground_scale, PROJECTED_WIDE_MAX_FOREGROUND_SCALE)
    reason = (
        "kept centered wide crop for projected slide/screen layout "
        f"(detected_focus_x={foreground_center_x:.2f})"
    )
    return adjusted_scale, 0.5, reason


def load_subject_index(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if resolved in SUBJECT_INDEX_CACHE:
        return SUBJECT_INDEX_CACHE[resolved]
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    manifest = read_json(resolved)
    SUBJECT_INDEX_CACHE[resolved] = manifest
    return manifest


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def subject_index_entry_for_span(
    subject_index: dict[str, Any] | None,
    clip_id: str,
    item: dict[str, Any],
    span_start: float,
    span_duration: float,
) -> dict[str, Any] | None:
    if not subject_index:
        return None
    span_end = span_start + span_duration
    try:
        order = int(item.get("order", 0) or 0)
    except (TypeError, ValueError):
        order = 0
    candidates: list[tuple[float, dict[str, Any]]] = []
    for entry in subject_index.get("entries", []) or []:
        if not entry.get("available"):
            continue
        entry_clip_id = str(entry.get("clip_id", ""))
        if entry_clip_id and entry_clip_id != clip_id:
            continue
        try:
            entry_start = float(entry.get("analysis_start", entry.get("source_start")))
            entry_end = float(entry.get("analysis_end", entry.get("source_end")))
        except (TypeError, ValueError):
            continue
        overlap = overlap_seconds(span_start, span_end, entry_start, entry_end)
        if overlap <= 0:
            continue
        overlap_ratio = overlap / max(0.01, min(span_duration, entry_end - entry_start))
        order_bonus = 0.25 if int(entry.get("timeline_order", -1) or -1) == order else 0.0
        confidence = float((entry.get("recommended_crop") or {}).get("confidence", 0.0) or 0.0)
        score = overlap_ratio + order_bonus + confidence * 0.15
        candidates.append((score, entry))
    if not candidates:
        return None

    crops = [
        (score, entry, entry.get("recommended_crop") or {})
        for score, entry in candidates
        if entry.get("recommended_crop")
    ]
    if len(crops) == 1:
        return crops[0][1]

    total_weight = sum(score for score, _entry, _crop in crops)
    if total_weight <= 0:
        return max(candidates, key=lambda item: item[0])[1]

    center_x = sum(
        score * float(crop.get("center_x", 0.5))
        for score, _entry, crop in crops
    ) / total_weight
    confidence = sum(
        score * float(crop.get("confidence", 0.0))
        for score, _entry, crop in crops
    ) / total_weight
    max_scale = min(float(crop.get("max_scale", 1.0)) for _score, _entry, crop in crops)
    visible_width = max(float(crop.get("visible_width", 1.0)) for _score, _entry, crop in crops)
    best_score, best_entry = max(candidates, key=lambda item: item[0])
    synthetic = dict(best_entry)
    synthetic["recommended_crop"] = {
        **(best_entry.get("recommended_crop") or {}),
        "center_x": round(center_x, 3),
        "max_scale": round(max_scale, 3),
        "visible_width": round(visible_width, 3),
        "confidence": round(confidence, 3),
        "reason": "weighted subject-index crop across overlapping analysis windows",
    }
    synthetic["matched_entry_count"] = len(crops)
    synthetic["match_score"] = round(best_score, 3)
    return synthetic


def apply_subject_crop_guard(
    foreground_scale: float,
    foreground_center_x: float,
    subject_entry: dict[str, Any] | None,
    item: dict[str, Any],
) -> tuple[float, float, str | None]:
    if foreground_scale <= 1.0 or not subject_entry:
        return foreground_scale, foreground_center_x, None
    crop = subject_entry.get("recommended_crop") or {}
    try:
        confidence = float(crop.get("confidence", 0.0))
        center_x = float(crop.get("center_x", foreground_center_x))
        max_scale = float(crop.get("max_scale", foreground_scale))
    except (TypeError, ValueError):
        return foreground_scale, foreground_center_x, None
    if confidence < SUBJECT_INDEX_MIN_CONFIDENCE:
        return min(foreground_scale, 1.35), 0.5, (
            "used wide centered crop because subject-index confidence was low"
        )

    protect_regions = {
        str(region)
        for region in (crop.get("protect_regions") or [])
        if str(region).strip()
    }
    if item_prefers_projected_wide_framing(item) and "secondary_content" in protect_regions:
        max_scale = min(max_scale, PROJECTED_WIDE_MAX_FOREGROUND_SCALE)

    adjusted_scale = min(foreground_scale, max(1.0, max_scale))
    damped_center = 0.5 + (center_x - 0.5) * SUBJECT_INDEX_CENTER_DAMPING
    adjusted_center = max(
        0.5 - SUBJECT_INDEX_MAX_CENTER_SHIFT,
        min(0.5 + SUBJECT_INDEX_MAX_CENTER_SHIFT, damped_center),
    )
    if abs(adjusted_center - foreground_center_x) < 0.01 and adjusted_scale >= foreground_scale - 0.01:
        return foreground_scale, foreground_center_x, None
    reason = (
        "subject-aware crop "
        f"(raw_center_x={center_x:.2f}, center_x={adjusted_center:.2f}, "
        f"max_scale={max_scale:.2f}, confidence={confidence:.2f})"
    )
    return adjusted_scale, adjusted_center, reason


def motion_analysis_for_clip(source_video: Path, clip: dict[str, Any]) -> dict[str, Any]:
    windows = visual_analysis_windows_for_clip(clip)
    columns = [0.0] * MOTION_ANALYSIS_WIDTH
    analyzed_windows = 0
    analyzed_frames = 0
    for source_start, duration, _reason in windows:
        sampled, frame_count = sample_motion_columns_for_window(
            source_video,
            source_start,
            duration,
        )
        if frame_count >= 2:
            analyzed_windows += 1
            analyzed_frames += frame_count
        for index, value in enumerate(sampled):
            columns[index] += value

    total_motion = sum(columns)
    left, right, width = weighted_motion_bounds(columns)
    return {
        "available": analyzed_windows > 0 and total_motion >= MOTION_MIN_TOTAL,
        "window_count": len(windows),
        "analyzed_window_count": analyzed_windows,
        "analyzed_frame_count": analyzed_frames,
        "total_motion": round(total_motion, 3),
        "motion_left": round(left, 3),
        "motion_right": round(right, 3),
        "motion_width": round(width, 3),
        "outside_strong": round(motion_fraction_outside_crop(columns, AUTO_FOREGROUND_SCALE_STRONG), 3),
        "outside_moderate": round(motion_fraction_outside_crop(columns, AUTO_FOREGROUND_SCALE_MODERATE), 3),
        "outside_conservative": round(
            motion_fraction_outside_crop(columns, AUTO_FOREGROUND_SCALE_CONSERVATIVE),
            3,
        ),
        "outside_wide_motion": round(
            motion_fraction_outside_crop(columns, AUTO_FOREGROUND_SCALE_WIDE_MOTION),
            3,
        ),
    }


def apply_motion_crop_guard(scale: float, reason: str, motion: dict[str, Any] | None) -> tuple[float, str]:
    if not motion or not motion.get("available"):
        return scale, reason

    motion_width = float(motion["motion_width"])
    outside_conservative = float(motion["outside_conservative"])
    outside_moderate = float(motion["outside_moderate"])
    outside_strong = float(motion["outside_strong"])

    guarded_scale = scale
    guard_reason = ""
    if motion_width >= 0.7 or outside_conservative >= 0.22:
        guarded_scale = min(guarded_scale, AUTO_FOREGROUND_SCALE_WIDE_MOTION)
        guard_reason = "wide off-center motion detected"
    elif motion_width >= 0.58 or outside_moderate >= 0.24:
        guarded_scale = min(guarded_scale, AUTO_FOREGROUND_SCALE_CONSERVATIVE)
        guard_reason = "motion extends beyond moderate crop"
    elif motion_width >= 0.46 or outside_strong >= 0.28:
        guarded_scale = min(guarded_scale, AUTO_FOREGROUND_SCALE_MODERATE)
        guard_reason = "motion extends beyond strong crop"

    if guarded_scale < scale:
        reason = (
            f"{reason}; capped to {guarded_scale:.2f} because {guard_reason} "
            f"(motion_width={motion_width:.2f}, outside_strong={outside_strong:.2f}, "
            f"outside_moderate={outside_moderate:.2f}, outside_conservative={outside_conservative:.2f})"
        )
    return guarded_scale, reason


def derive_foreground_scale(clip: dict[str, Any], source_video: Path | None = None) -> dict[str, Any]:
    source_items = source_backed_timeline_items(clip)
    source_segment_count = len(source_items)
    multi_visual_span_count = 0
    for item in source_items:
        try:
            spans = build_visual_spans(item, timeline_duration(item))
        except Exception:
            spans = []
        if len(spans) > 1:
            multi_visual_span_count += 1
    distinct_source_windows = {
        (
            round(float(item.get("source_start", 0.0)) / 5.0) * 5.0,
            round(float(item.get("source_end", 0.0)) / 5.0) * 5.0,
        )
        for item in source_items
    }
    distinct_source_count = len(distinct_source_windows)

    if multi_visual_span_count > 0 or source_segment_count >= 4 or distinct_source_count >= 4:
        scale = AUTO_FOREGROUND_SCALE_CONSERVATIVE
        reason = "conservative crop for complex or multi-span visual clip"
    elif source_segment_count >= 3 or distinct_source_count >= 3:
        scale = AUTO_FOREGROUND_SCALE_MODERATE
        reason = "moderate crop for multiple source windows"
    else:
        scale = AUTO_FOREGROUND_SCALE_STRONG
        reason = "strong crop for source-stable talking-head clip"

    motion: dict[str, Any] | None = None
    if source_video is not None:
        motion = motion_analysis_for_clip(source_video, clip)
        scale, reason = apply_motion_crop_guard(scale, reason, motion)

    return {
        "scale": scale,
        "reason": reason,
        "source_segment_count": source_segment_count,
        "distinct_source_count": distinct_source_count,
        "multi_visual_span_count": multi_visual_span_count,
        "motion_analysis": motion,
    }


def drawtext_filter(caption_file: Path, font_file: Path, first_segment: bool) -> str:
    y_expr = (
        "(h-text_h)/2"
        if first_segment
        else f"h*{CAPTION_VERTICAL_CENTER_RATIO:.2f}-text_h/2"
    )
    return (
        "drawtext="
        f"fontfile={font_file}:"
        f"textfile={caption_file}:"
        "fontcolor=white:"
        f"fontsize={CAPTION_FONT_SIZE}:"
        f"line_spacing={CAPTION_LINE_SPACING}:"
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
    foreground_center_x: float,
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
    if foreground_width > width:
        focus_x = max(0.0, min(1.0, foreground_center_x))
        overlay_x = max(
            float(width - foreground_width),
            min(0.0, width / 2.0 - foreground_width * focus_x),
        )
        overlay_x_expr = f"{overlay_x:.1f}"
    else:
        overlay_x_expr = "(W-w)/2"
    composed = f"[bg][fg]overlay={overlay_x_expr}:(H-h)/2"
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


def augmentation_video_filter(
    caption_file: Path,
    font_file: Path,
    width: int,
    height: int,
    first_segment: bool,
    fade_out: bool,
    duration: float,
    burn_captions: bool,
    tail_pad_duration: float,
) -> str:
    filters = [
        "[0:v]setpts=PTS-STARTPTS,split=2[bgsrc][fgsrc]",
        (
            f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=24,eq=brightness=-0.06:saturation=0.9[bg]"
        ),
        f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease[fg]",
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
    foreground_center_x: float,
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
            foreground_center_x=foreground_center_x,
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


def render_augmentation_visual_chunk(
    clip_dir: Path,
    caption_file: Path,
    segment_index: int,
    chunk_index: int,
    asset_path: Path,
    asset_type: str,
    asset_start: float,
    duration: float,
    font_file: Path,
    width: int,
    height: int,
    first_segment: bool,
    fade_out: bool,
    burn_captions: bool,
    tail_pad_duration: float,
) -> Path:
    chunk_path = clip_dir / f"segment-{segment_index:03d}.asset-{chunk_index:03d}.mp4"
    args = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"]
    if asset_type == "image":
        args.extend(
            [
                "-loop",
                "1",
                "-framerate",
                "30",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(asset_path),
            ]
        )
    else:
        args.extend(["-stream_loop", "-1"])
        if asset_start > 0:
            args.extend(["-ss", f"{asset_start:.3f}"])
        args.extend(["-t", f"{duration:.3f}", "-i", str(asset_path)])
    args.extend(
        [
            "-filter_complex",
            augmentation_video_filter(
                caption_file=caption_file,
                font_file=font_file,
                width=width,
                height=height,
                first_segment=first_segment,
                fade_out=fade_out,
                duration=duration,
                burn_captions=burn_captions,
                tail_pad_duration=tail_pad_duration,
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
            "-r",
            "30",
            "-movflags",
            "+faststart",
            str(chunk_path),
        ]
    )
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
    usable_transition = usable_transition_duration_from_durations(
        durations,
        transition_duration,
    )
    if usable_transition <= 0:
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
    audio_ref = None if force_silent else asset_ref_for_item(item, "audio_source")
    direct_asset_audio = audio_ref is not None and str(item.get("audio_source", "")).strip().startswith(
        f"asset://{audio_ref['id']}"
    )
    if audio_ref is not None and direct_asset_audio:
        asset = resolve_asset(audio_ref)
        asset_type = str(asset.get("type", "audio"))
        if asset_type not in {"audio", "video"}:
            raise RuntimeError(
                f"Augmentation asset {audio_ref['id']!r} is type {asset_type!r}; "
                "audio_source assets must be audio or video."
            )
        args.extend(["-stream_loop", "-1"])
        if float(audio_ref.get("start", 0.0)) > 0:
            args.extend(["-ss", f"{float(audio_ref['start']):.3f}"])
        args.extend(
            [
                "-t",
                f"{duration:.3f}",
                "-i",
                str(asset["path"]),
                "-filter_complex",
                audio_filter("0", duration, fade_out),
            ]
        )
    elif uses_dialogue(item) and not force_silent:
        background_ref = audio_ref
        if background_ref is not None:
            asset = resolve_asset(background_ref)
            asset_type = str(asset.get("type", "audio"))
            if asset_type not in {"audio", "video"}:
                raise RuntimeError(
                    f"Augmentation asset {background_ref['id']!r} is type {asset_type!r}; "
                    "background audio assets must be audio or video."
                )
            volume = asset_volume(asset)
            fade_in_duration = min(0.05, max(0.0, duration / 3))
            fade_out_duration = min(0.08 if fade_out else 0.05, max(0.0, duration / 3))
            fade_out_start = max(0.0, duration - fade_out_duration)
            args.extend(
                [
                    "-ss",
                    f"{float(item['source_start']):.3f}",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(source_video),
                    "-stream_loop",
                    "-1",
                ]
            )
            if float(background_ref.get("start", 0.0)) > 0:
                args.extend(["-ss", f"{float(background_ref['start']):.3f}"])
            args.extend(
                [
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(asset["path"]),
                    "-filter_complex",
                    (
                        "[0:a]asetpts=PTS-STARTPTS,aresample=48000,"
                        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                        f"afade=t=in:st=0:d={fade_in_duration:.3f},"
                        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_duration:.3f}[voice];"
                        "[1:a]asetpts=PTS-STARTPTS,aresample=48000,"
                        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                        f"volume={volume:.3f},"
                        f"afade=t=in:st=0:d={fade_in_duration:.3f},"
                        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_duration:.3f}[bed];"
                        "[voice][bed]amix=inputs=2:duration=first:dropout_transition=0,"
                        "alimiter=limit=0.95[a]"
                    ),
                ]
            )
        else:
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
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def render_segment(
    source_video: Path,
    clip_dir: Path,
    clip_id: str,
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
    subject_index: dict[str, Any] | None,
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
            if span.get("asset_id"):
                video_chunks.append(
                    render_augmentation_visual_chunk(
                        clip_dir=clip_dir,
                        caption_file=caption_file,
                        segment_index=index,
                        chunk_index=chunk_index,
                        asset_path=Path(str(span["asset_path"])),
                        asset_type=str(span["asset_type"]),
                        asset_start=float(span.get("source_start", 0.0)),
                        duration=float(span["duration"]),
                        font_file=font_file,
                        width=width,
                        height=height,
                        first_segment=first_segment and chunk_index == 1,
                        fade_out=last_segment and chunk_index == len(visual_spans),
                        burn_captions=burn_captions,
                        tail_pad_duration=tail_pad_duration,
                    )
                )
            else:
                foreground_focus: dict[str, Any] | None = None
                subject_focus: dict[str, Any] | None = None
                foreground_center_x = 0.5
                span_foreground_scale = foreground_scale
                if foreground_scale > 1.0:
                    span_source_start = float(span["source_start"])
                    span_duration = float(span["duration"])
                    foreground_focus = foreground_focus_for_window(
                        source_video,
                        span_source_start,
                        span_duration,
                    )
                    foreground_center_x = float(foreground_focus.get("center_x", 0.5))
                    span_foreground_scale, scale_adjustment_reason = (
                        focus_adjusted_foreground_scale(foreground_scale, foreground_focus)
                    )
                    subject_focus = subject_index_entry_for_span(
                        subject_index,
                        clip_id,
                        item,
                        span_source_start,
                        span_duration,
                    )
                    (
                        span_foreground_scale,
                        foreground_center_x,
                        subject_adjustment_reason,
                    ) = apply_subject_crop_guard(
                        span_foreground_scale,
                        foreground_center_x,
                        subject_focus,
                        item,
                    )
                    (
                        span_foreground_scale,
                        foreground_center_x,
                        projected_adjustment_reason,
                    ) = projected_wide_focus_adjustment(
                        item,
                        span_foreground_scale,
                        foreground_center_x,
                    )
                    span["foreground_focus"] = foreground_focus
                    if subject_focus:
                        span["subject_focus"] = {
                            key: value
                            for key, value in subject_focus.items()
                            if key
                            in {
                                "clip_id",
                                "timeline_order",
                                "analysis_start",
                                "analysis_end",
                                "primary_subject",
                                "secondary_content",
                                "recommended_crop",
                                "scores",
                            }
                        }
                    span["foreground_center_x"] = round(foreground_center_x, 3)
                    span["foreground_scale"] = round(span_foreground_scale, 3)
                    adjustment_reasons = [
                        reason
                        for reason in (
                            scale_adjustment_reason,
                            subject_adjustment_reason,
                            projected_adjustment_reason,
                        )
                        if reason
                    ]
                    if adjustment_reasons:
                        span["foreground_scale_adjustment"] = "; ".join(adjustment_reasons)
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
                        foreground_scale=span_foreground_scale,
                        foreground_center_x=foreground_center_x,
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
    validate_av_stream_lengths(segment_path, tolerance=0.08)
    uses_source_dialogue = uses_dialogue(item) and not black_title_card
    audio_start = None if not uses_source_dialogue else float(item["source_start"])
    separated = any(
        audio_start is None
        or span.get("asset_id")
        or span["source_start"] is None
        or abs(float(span["source_start"]) - audio_start) > 0.2
        for span in visual_spans
    )
    audio_ref = None if black_title_card else asset_ref_for_item(item, "audio_source")
    asset_ids = sorted(
        {
            *[
                str(span["asset_id"])
                for span in visual_spans
                if span.get("asset_id")
            ],
            *([str(audio_ref["id"])] if audio_ref is not None else []),
        }
    )
    media_request_ids = sorted(media_request_ids_for_item(item))
    return {
        "path": segment_path,
        "audio_source_start": audio_start,
        "audio_source_end": None if audio_start is None else audio_start + duration,
        "visual_spans": visual_spans,
        "separated_audio_video": separated,
        "silent_audio": not uses_source_dialogue,
        "multiple_visual_cuts": len(visual_spans) > 1,
        "augmentation_asset_ids": asset_ids,
        "media_request_ids": media_request_ids,
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
    usable_transition = usable_transition_duration_from_durations(
        all_durations,
        transition_duration,
    )
    if usable_transition <= 0:
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


def apply_final_video_fade_to_black(input_path: Path, output_path: Path, fade_duration: float) -> tuple[float, float]:
    summary = ffprobe_summary(input_path)
    video_stream = next(
        (
            stream
            for stream in summary.get("streams", [])
            if stream.get("codec_type") == "video" and stream.get("duration")
        ),
        None,
    )
    audio_stream = next(
        (
            stream
            for stream in summary.get("streams", [])
            if stream.get("codec_type") == "audio" and stream.get("duration")
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
    audio_duration = float(
        (audio_stream or {}).get("duration")
        or summary.get("format", {}).get("duration")
        or duration
    )
    usable_video_fade = min(fade_duration, max(0.0, duration / 2))
    usable_audio_fade = (
        min(fade_duration, max(0.0, audio_duration / 2))
        if audio_stream is not None and audio_duration > 0
        else 0.0
    )
    if usable_video_fade <= 0.01:
        shutil.move(str(input_path), str(output_path))
        return 0.0, 0.0
    video_fade_start = max(0.0, duration - usable_video_fade)
    if audio_stream is not None and usable_audio_fade > 0.01:
        audio_fade_start = max(0.0, audio_duration - usable_audio_fade)
        command = [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-filter_complex",
            (
                f"[0:v:0]fade=t=out:st={video_fade_start:.3f}:d={usable_video_fade:.3f}[v];"
                f"[0:a:0]afade=t=out:st={audio_fade_start:.3f}:d={usable_audio_fade:.3f}[a]"
            ),
            "-map",
            "[v]",
            "-map",
            "[a]",
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
            "160k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    else:
        command = [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"fade=t=out:st={video_fade_start:.3f}:d={usable_video_fade:.3f}",
            "-map",
            "0:v:0",
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
    run_command(command)
    input_path.unlink(missing_ok=True)
    validate_av_stream_lengths(output_path)
    return usable_video_fade, usable_audio_fade


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


def usable_transition_duration_from_durations(
    durations: list[float],
    transition_duration: float,
) -> float:
    if len(durations) < 2 or transition_duration <= 0:
        return 0.0
    usable_transition = min(
        transition_duration,
        *[max(0.0, (duration - 0.05) / 2) for duration in durations],
    )
    return usable_transition if usable_transition > 0.01 else 0.0


def usable_media_transition_duration(
    paths: list[Path],
    transition_duration: float,
    *,
    include_audio: bool,
) -> float:
    if len(paths) < 2 or transition_duration <= 0:
        return 0.0
    summaries = [ffprobe_summary(path) for path in paths]
    durations = [
        stream_duration(summary, "video", fallback_format=True)
        for summary in summaries
    ]
    if include_audio:
        durations.extend(
            stream_duration(summary, "audio", fallback_format=True)
            for summary in summaries
        )
    return usable_transition_duration_from_durations(durations, transition_duration)


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
    transcript_words: list[dict[str, Any]],
    caption_source_mode: str,
    font_file: Path,
    width: int,
    height: int,
    burn_captions: bool,
    repair_silent_lead_clip_ids: set[str],
    black_title_card_clip_ids: set[str],
    transition_duration: float,
    foreground_scale: float,
    foreground_scale_source: str,
    foreground_scale_reason: str,
    foreground_motion_analysis: dict[str, Any] | None,
    internal_visual_transition_duration: float,
    foreground_brightness: float,
    foreground_contrast: float,
    foreground_saturation: float,
    require_opening_title_card: bool,
    subject_index: dict[str, Any] | None,
) -> dict[str, Any]:
    clip, opening_title_card_inserted = ensure_opening_title_card(
        clip,
        required=require_opening_title_card,
    )
    clip, title_card_text_normalized, title_card_original_text = normalize_first_title_card_text(clip)
    clip, title_card_duration_normalized, title_card_duration_delta = normalize_first_title_card_duration(clip)
    clip, opening_dialogue_lead_in_seconds = add_opening_dialogue_lead_in(
        clip,
        transcript_words,
    )
    clip_id = safe_name(str(clip["clip_id"]))
    clip_dir = output_dir / "_work" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    timeline = sorted(clip.get("timeline", []), key=lambda item: int(item.get("order", 0)))
    silent_lead_repaired = False
    if clip_id in repair_silent_lead_clip_ids:
        timeline, silent_lead_repaired = repair_silent_lead(timeline)
    plan_caption_entries = caption_plan_entries(clip)
    suppress_segment_caption_overlays = burn_captions and (
        bool(plan_caption_entries)
        or (caption_source_mode in {"auto", "transcript"} and bool(transcript_words))
    )
    use_black_title_card = clip_id in black_title_card_clip_ids
    segment_results = [
        render_segment(
            source_video=source_video,
            clip_dir=clip_dir,
            clip_id=clip_id,
            item=item,
            index=index,
            total_segments=len(timeline),
            font_file=font_file,
            width=width,
            height=height,
            burn_captions=burn_captions and not suppress_segment_caption_overlays,
            black_title_card=should_render_title_card(
                item,
                force_first_card=index == 1 and use_black_title_card,
            ),
            foreground_scale=foreground_scale,
            internal_visual_transition_duration=internal_visual_transition_duration,
            foreground_brightness=foreground_brightness,
            foreground_contrast=foreground_contrast,
            foreground_saturation=foreground_saturation,
            subject_index=subject_index,
        )
        for index, item in enumerate(timeline, start=1)
    ]
    segment_paths = [result["path"] for result in segment_results]
    effective_transition_duration = usable_media_transition_duration(
        segment_paths,
        transition_duration,
        include_audio=True,
    )
    caption_entries = retime_caption_entries_for_transitions(
        plan_caption_entries,
        timeline,
        effective_transition_duration,
    )
    transcript_caption_entries = (
        transcript_caption_entries_for_clip(
            timeline,
            transcript_words,
            effective_transition_duration,
        )
        if caption_source_mode in {"auto", "transcript"}
        else []
    )
    if caption_source_mode == "transcript":
        caption_entries = transcript_caption_entries
        caption_source = "transcript_words" if caption_entries else "none"
    elif caption_source_mode == "auto" and transcript_caption_entries:
        caption_entries = transcript_caption_entries
        caption_source = "transcript_words"
    else:
        caption_source = "caption_plan" if caption_entries else "none"
    output_path = output_dir / f"{clip_id}.draft.mp4"
    pre_fade_output_path = output_dir / f"{clip_id}.draft.pre-video-fade.mp4"
    concat_segments_with_crossfade(
        segment_paths,
        pre_fade_output_path,
        transition_duration=transition_duration,
        width=width,
        height=height,
    )
    pre_fade_summary = ffprobe_summary(pre_fade_output_path)
    pre_fade_duration = stream_duration(pre_fade_summary, "video", fallback_format=True)
    final_caption_suppression_duration = min(
        FINAL_VIDEO_FADE_OUT_DURATION,
        max(0.0, pre_fade_duration / 2),
    )
    caption_entry_count_before_final_fade = len(caption_entries)
    transcript_caption_entry_count_before_final_fade = len(transcript_caption_entries)
    caption_entries = suppress_entries_during_final_fade(
        caption_entries,
        pre_fade_duration,
        final_caption_suppression_duration,
        start_key="start",
        end_key="end",
    )
    transcript_caption_entries = suppress_entries_during_final_fade(
        transcript_caption_entries,
        pre_fade_duration,
        final_caption_suppression_duration,
        start_key="start",
        end_key="end",
    )
    srt_path = output_dir / f"{clip_id}.draft.srt"
    if caption_entries:
        write_srt_entries(srt_path, caption_entries)
        srt_timeline = caption_entries
    else:
        srt_timeline = timeline_with_transition_timing(
            timeline,
            effective_transition_duration,
        )
        srt_timeline = suppress_entries_during_final_fade(
            srt_timeline,
            pre_fade_duration,
            final_caption_suppression_duration,
            start_key="screen_time_start",
            end_key="screen_time_end",
        )
        write_srt(srt_path, srt_timeline)

    burned_caption_source = "none"
    burned_caption_entries: list[dict[str, Any]] = []
    final_fade_input_path = pre_fade_output_path
    if burn_captions and caption_entries:
        if caption_source == "caption_plan":
            burn_source_caption_entries = caption_entries_for_burn(
                plan_caption_entries,
                timeline,
            )
            burned_caption_entries = retime_caption_entries_for_transitions(
                burn_source_caption_entries,
                timeline,
                effective_transition_duration,
            )
            burned_caption_entries = suppress_entries_during_final_fade(
                burned_caption_entries,
                pre_fade_duration,
                final_caption_suppression_duration,
                start_key="start",
                end_key="end",
            )
        else:
            burned_caption_entries = caption_entries
        if burned_caption_entries:
            ass_path = output_dir / f"{clip_id}.draft.burned.ass"
            captioned_path = output_dir / f"{clip_id}.draft.burned.pre-video-fade.mp4"
            write_ass_entries(ass_path, burned_caption_entries, width, height)
            burn_ass_captions(pre_fade_output_path, ass_path, captioned_path)
            final_fade_input_path = captioned_path
            burned_caption_source = caption_source
    elif burn_captions and not suppress_segment_caption_overlays:
        burned_caption_source = "segment_overlays"

    final_video_fade_out_duration, final_audio_fade_out_duration = apply_final_video_fade_to_black(
        final_fade_input_path,
        output_path,
        FINAL_VIDEO_FADE_OUT_DURATION,
    )
    if final_fade_input_path != pre_fade_output_path:
        pre_fade_output_path.unlink(missing_ok=True)

    subtitled_path = output_dir / f"{clip_id}.draft.subtitled.mp4"
    subtitles_burned = False
    if burn_captions:
        shutil.copy2(output_path, subtitled_path)
        subtitles_muxed = False
        subtitles_burned = True
    else:
        subtitle_entries = sidecar_timeline_to_caption_entries(srt_timeline)
        if subtitle_entries:
            subtitled_ass_path = output_dir / f"{clip_id}.draft.subtitled.ass"
            write_ass_entries(subtitled_ass_path, subtitle_entries, width, height)
            burn_ass_captions(output_path, subtitled_ass_path, subtitled_path)
            subtitles_muxed = False
            subtitles_burned = True
        else:
            subtitles_muxed = mux_subtitles(output_path, srt_path, subtitled_path)
    declared_media_request_ids = {
        str(media_id).strip()
        for media_id in (clip.get("media_request_ids", []) or [])
        if str(media_id).strip()
    }
    return {
        "clip_id": clip_id,
        "concept_title": clip.get("concept_title"),
        "output_path": str(output_path),
        "srt_path": str(srt_path),
        "subtitled_output_path": str(subtitled_path) if subtitled_path.exists() else None,
        "burned_in_captions": burn_captions,
        "caption_source_requested": caption_source_mode,
        "caption_source": caption_source,
        "burned_caption_source": burned_caption_source,
        "burned_caption_entry_count": len(burned_caption_entries),
        "transcript_caption_entry_count": len(transcript_caption_entries),
        "caption_entry_count_before_final_fade": caption_entry_count_before_final_fade,
        "transcript_caption_entry_count_before_final_fade": (
            transcript_caption_entry_count_before_final_fade
        ),
        "final_caption_suppression_duration": round(final_caption_suppression_duration, 3),
        "final_caption_suppression_start": round(
            max(0.0, pre_fade_duration - final_caption_suppression_duration),
            3,
        ),
        "embedded_subtitles": subtitles_muxed,
        "subtitled_output_burned_in": subtitles_burned,
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
        "augmentation_asset_ids": sorted(
            {
                asset_id
                for result in segment_results
                for asset_id in result.get("augmentation_asset_ids", [])
            }
        ),
        "media_request_ids": sorted(
            declared_media_request_ids
            | {
                media_id
                for result in segment_results
                for media_id in result.get("media_request_ids", [])
            }
        ),
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
            | ({"burned_subtitled_output"} if subtitles_burned else set())
            | ({"silent_lead_repaired"} if silent_lead_repaired else set())
            | ({"crossfade_transitions"} if transition_duration > 0 else set())
            | ({"foreground_punch_in"} if foreground_scale > 1.0 else set())
            | ({"foreground_exposure_lift"} if foreground_brightness > 0 else set())
            | ({"phrase_level_captions"} if caption_entries else set())
            | ({"transcript_aligned_captions"} if caption_source == "transcript_words" else set())
            | ({"video_fade_to_black"} if final_video_fade_out_duration > 0 else set())
            | ({"audio_fade_to_silence"} if final_audio_fade_out_duration > 0 else set())
        ),
        "silent_lead_repaired": silent_lead_repaired,
        "title_card_text_normalized": title_card_text_normalized,
        "title_card_original_text": title_card_original_text,
        "title_card_duration_normalized": title_card_duration_normalized,
        "title_card_duration_delta": round(title_card_duration_delta, 3),
        "opening_title_card_inserted": opening_title_card_inserted,
        "opening_title_card_required": require_opening_title_card,
        "opening_dialogue_lead_in_seconds": opening_dialogue_lead_in_seconds,
        "transition_duration": transition_duration,
        "effective_transition_duration": round(effective_transition_duration, 3),
        "final_video_fade_out_duration": final_video_fade_out_duration,
        "final_audio_fade_out_duration": final_audio_fade_out_duration,
        "internal_visual_transition_duration": internal_visual_transition_duration,
        "foreground_scale": foreground_scale,
        "foreground_scale_source": foreground_scale_source,
        "foreground_scale_reason": foreground_scale_reason,
        "foreground_motion_analysis": foreground_motion_analysis,
        "foreground_brightness": foreground_brightness,
        "foreground_contrast": foreground_contrast,
        "foreground_saturation": foreground_saturation,
        "caption_plan_source": clip.get("caption_plan_source"),
        "caption_timing_adjusted_for_transitions": bool(
            caption_entries and effective_transition_duration > 0
        ),
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
    source_video: Path,
) -> dict[str, Any]:
    global_settings = plan.get("render_settings") if isinstance(plan.get("render_settings"), dict) else {}
    clip_settings = clip.get("render_settings") if isinstance(clip.get("render_settings"), dict) else {}
    burn_captions = bool(args.burn_captions)
    if not burn_captions and args.use_plan_render_settings:
        for settings in (clip_settings, global_settings):
            if settings.get("burn_captions") is not None:
                burn_captions = bool(settings["burn_captions"])
                break
    foreground_scale_source = "default"
    foreground_scale_reason = ""
    foreground_motion_analysis = None
    foreground_scale = DEFAULT_FOREGROUND_SCALE
    if args.foreground_scale is not None:
        foreground_scale = float(args.foreground_scale)
        foreground_scale_source = "cli"
    elif args.auto_foreground_scale:
        derived = derive_foreground_scale(clip, source_video)
        foreground_scale = float(derived["scale"])
        foreground_scale_source = "auto"
        foreground_motion_analysis = derived.get("motion_analysis")
        foreground_scale_reason = (
            f"{derived['reason']}; source_segments={derived['source_segment_count']}, "
            f"distinct_source_windows={derived['distinct_source_count']}, "
            f"multi_visual_span_segments={derived['multi_visual_span_count']}"
        )
    elif args.use_plan_render_settings:
        for settings_name, settings in (("clip_plan", clip_settings), ("global_plan", global_settings)):
            value = settings.get("foreground_scale")
            if value is not None:
                foreground_scale = float(value)
                foreground_scale_source = settings_name
                break
    return {
        "transition_duration": render_setting_value(
            args,
            global_settings,
            clip_settings,
            "transition_duration",
            0.0,
        ),
        "foreground_scale": foreground_scale,
        "foreground_scale_source": foreground_scale_source,
        "foreground_scale_reason": foreground_scale_reason,
        "foreground_motion_analysis": foreground_motion_analysis,
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
    augmentation_assets_path = (
        args.augmentation_assets.expanduser().resolve()
        if args.augmentation_assets
        else None
    )
    augmentation_assets = configure_augmentation_assets(augmentation_assets_path)
    subject_index_path = (
        args.subject_index_manifest.expanduser().resolve()
        if args.subject_index_manifest
        else None
    )
    subject_index = load_subject_index(subject_index_path)

    plan = read_json(stage3_path)
    transcript_path = args.transcript.expanduser().resolve() if args.transcript else None
    if transcript_path is None:
        transcript_path = infer_transcript_path(source_video, stage3_path)
    transcript_words: list[dict[str, Any]] = []
    if transcript_path is not None:
        if not transcript_path.exists():
            raise FileNotFoundError(transcript_path)
        transcript_words = transcript_words_from_data(read_json(transcript_path))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    drawtext_available = ffmpeg_has_filter("drawtext")
    clip_ads = plan.get("clip_ads", [])
    content_mode = str(plan.get("content_mode", "ad"))
    if args.clip_id:
        clip_ads = [clip for clip in clip_ads if clip.get("clip_id") == args.clip_id]
        if not clip_ads:
            raise RuntimeError(f"No clip found with clip_id={args.clip_id!r}")

    results = []
    warned_drawtext = False
    for clip in clip_ads:
        settings = render_settings_for_clip(args, plan, clip, source_video)
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
                transcript_words=transcript_words,
                caption_source_mode=args.caption_source,
                font_file=font_file,
                width=args.width,
                height=args.height,
                burn_captions=burn_captions,
                repair_silent_lead_clip_ids=set(args.repair_silent_lead_clip),
                black_title_card_clip_ids=set(args.black_title_card_clip),
                transition_duration=float(settings["transition_duration"]),
                foreground_scale=float(settings["foreground_scale"]),
                foreground_scale_source=str(settings["foreground_scale_source"]),
                foreground_scale_reason=str(settings["foreground_scale_reason"]),
                foreground_motion_analysis=settings.get("foreground_motion_analysis"),
                internal_visual_transition_duration=float(
                    settings["internal_visual_transition_duration"]
                ),
                foreground_brightness=float(settings["foreground_brightness"]),
                foreground_contrast=float(settings["foreground_contrast"]),
                foreground_saturation=float(settings["foreground_saturation"]),
                require_opening_title_card=content_mode == "excerpt",
                subject_index=subject_index,
            )
        )
    manifest = {
        "source_video": str(source_video),
        "stage3_output": str(stage3_path),
        "transcript": str(transcript_path) if transcript_path else None,
        "content_mode": content_mode,
        "output_dir": str(output_dir),
        "resolution": {"width": args.width, "height": args.height},
        "renderer": {
            "type": "ffmpeg_draft_renderer",
            "ffmpeg_bin": FFMPEG_BIN,
            "ffmpeg_hwaccel": FFMPEG_HWACCEL or "none",
            "use_plan_render_settings": args.use_plan_render_settings,
            "auto_foreground_scale": args.auto_foreground_scale,
            "augmentation_assets_manifest": (
                str(augmentation_assets_path) if augmentation_assets_path else None
            ),
            "subject_index_manifest": str(subject_index_path) if subject_index_path else None,
            "subject_index_entry_count": (
                len(subject_index.get("entries", [])) if subject_index else 0
            ),
            "augmentation_asset_ids": sorted(augmentation_assets),
            "caption_source": args.caption_source,
            "transcript_caption_word_count": len(transcript_words),
            "sidecar_captions": True,
            "notes": [
                "Uses stage3 timeline source ranges in order.",
                "Separates dialogue audio from visual B-roll when visual_source provides usable time cues.",
                "Builds multi-cut B-roll segments from parsed visual ranges and anchors.",
                "Can repair planned silent leads by borrowing immediate narration from the following beat.",
                "Can render silent generated title, hook, and CTA beats as baked-in black title cards.",
                "Adds a short transcript-safe source lead-in after opening title cards to avoid abrupt first words.",
                "Can render short audio/video crossfades between planned segments.",
                "Always applies a final video fade to black after assembling the full clip.",
                "Can render short visual dissolves between internal multi-cut B-roll spans.",
                "Can punch in the foreground source frame to improve vertical legibility.",
                "Can derive a per-clip foreground punch-in from source-window complexity.",
                "Can shift punched-in talking-head crops horizontally from per-segment visual focus analysis.",
                "Can protect detected primary subjects and damp crop-center shifts using a local subject index.",
                "Can lift foreground exposure and contrast for dark source footage.",
                "Prefers transcript word timestamps for dialogue-tracking phrase captions.",
                "Can use explicit caption_plan render-time entries when transcript words are unavailable or requested.",
                "Uses blurred 9:16 background with centered original landscape frame.",
                "Burns captions in a phone-safe lower-third band above common short-form app controls.",
                "Creates a burned-caption .draft.subtitled.mp4 so subtitle placement is deterministic.",
                "Adds simple overlay text from caption_or_overlay when ffmpeg drawtext is available.",
                "Writes sidecar SRT captions for every rendered clip.",
                "Does not add licensed music or advanced motion graphics.",
            ],
            "caption_layout": {
                "font_size": CAPTION_FONT_SIZE,
                "line_spacing": CAPTION_LINE_SPACING,
                "vertical_center_ratio": CAPTION_VERTICAL_CENTER_RATIO,
                "title_card_position": "center",
                "spoken_caption_position": "phone_safe_lower_third",
            },
        },
        "clips": results,
    }
    write_json(output_dir / "render-manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render draft local clips from OpenAI stage 3 edit plans.")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--stage3-output", type=Path, required=True)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help=(
            "Optional normalized transcript JSON with word timestamps. "
            "If omitted, the renderer tries to infer it from the run directory."
        ),
    )
    parser.add_argument(
        "--caption-source",
        choices=("auto", "caption-plan", "transcript"),
        default="auto",
        help=(
            "Caption timing source. auto prefers transcript word timestamps when available "
            "so subtitles track spoken dialogue, and falls back to caption_plan entries."
        ),
    )
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
        "--augmentation-assets",
        type=Path,
        default=None,
        help=(
            "Optional JSON asset catalog. Timeline visual_source/audio_source fields may "
            "reference local catalog entries with asset://id or asset://id#t=12.5."
        ),
    )
    parser.add_argument(
        "--subject-index-manifest",
        type=Path,
        default=None,
        help=(
            "Optional local subject-index manifest. When present, speaker/content "
            "detections constrain per-span vertical punch-in crops."
        ),
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
        "--auto-foreground-scale",
        action="store_true",
        help=(
            "Derive a foreground punch-in scale per clip from source-window "
            "complexity. Ignored when --foreground-scale is provided."
        ),
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
