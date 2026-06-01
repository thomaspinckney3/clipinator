from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("video-evidence")
DEFAULT_FRAME_INTERVAL_SECONDS = 1.0
DEFAULT_FRAME_WIDTH = 480
DEFAULT_CONTACT_SHEET_FRAME_WIDTH = 180
DEFAULT_CONTACT_SHEET_COLUMNS = 12
DEFAULT_WINDOW_SHEET_SECONDS = 60.0
DEFAULT_WINDOW_SHEET_FRAME_WIDTH = 240
DEFAULT_WINDOW_SHEET_COLUMNS = 10
SHEET_PADDING_PIXELS = 8
SHEET_MARGIN_PIXELS = 8
MAX_CONTACT_SHEET_DIMENSION_PIXELS = 60_000
DEFAULT_FFMPEG_BIN = "auto"
HOMEBREW_FFMPEG_FULL = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
DEFAULT_FFMPEG_HWACCEL = "auto"
FFMPEG_HWACCEL_CHOICES = ("auto", "none", "videotoolbox")
DEFAULT_SCENE_THRESHOLD = 0.35
DEFAULT_MAX_SCENES = 80
DEFAULT_SILENCE_NOISE_DB = -35
DEFAULT_SILENCE_MIN_DURATION = 0.7


@dataclass(frozen=True)
class FrameSample:
    index: int
    timestamp: float
    path: Path

    def to_dict(self, root: Path) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": round(self.timestamp, 3),
            "display_time": format_time(self.timestamp),
            "path": relative_to_root(self.path, root),
        }


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found on PATH: {name}")


def resolve_ffmpeg_bin(value: str | Path | None) -> str:
    raw_value = str(value or DEFAULT_FFMPEG_BIN)
    if raw_value == "auto":
        if HOMEBREW_FFMPEG_FULL.exists():
            return str(HOMEBREW_FFMPEG_FULL)
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise RuntimeError("Required command not found on PATH: ffmpeg")

    expanded = Path(raw_value).expanduser()
    if expanded.parent != Path(".") or "/" in raw_value:
        if not expanded.exists():
            raise RuntimeError(f"Required ffmpeg binary not found: {expanded}")
        return str(expanded)

    found = shutil.which(raw_value)
    if found is None:
        raise RuntimeError(f"Required command not found on PATH: {raw_value}")
    return found


def ffmpeg_supports_filter(ffmpeg_bin: str, name: str) -> bool:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-filters"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return re.search(rf"\b{name}\b", f"{result.stdout}\n{result.stderr}") is not None


def ffmpeg_supports_hwaccel(ffmpeg_bin: str, name: str) -> bool:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-hwaccels"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return name in f"{result.stdout}\n{result.stderr}".split()


def resolve_ffmpeg_hwaccel(value: str, ffmpeg_bin: str | None = None) -> str | None:
    if value == "none":
        return None
    if value == "auto":
        resolved_bin = ffmpeg_bin or resolve_ffmpeg_bin("auto")
        if platform.system() == "Darwin" and ffmpeg_supports_hwaccel(
            resolved_bin, "videotoolbox"
        ):
            return "videotoolbox"
        return None
    if value not in FFMPEG_HWACCEL_CHOICES:
        choices = ", ".join(FFMPEG_HWACCEL_CHOICES)
        raise ValueError(f"Unknown ffmpeg hwaccel {value!r}; choose one of: {choices}.")
    return value


def ffmpeg_hwaccel_input_args(value: str | None) -> list[str]:
    resolved = resolve_ffmpeg_hwaccel(value or "none")
    return ["-hwaccel", resolved] if resolved else []


def run_command(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def format_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def ffprobe_json(path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def media_summary(probe: dict[str, Any]) -> dict[str, Any]:
    video_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), {})
    format_info = probe.get("format", {})
    duration = float(format_info.get("duration") or video_stream.get("duration") or 0)
    return {
        "duration_seconds": round(duration, 3),
        "container": format_info.get("format_name"),
        "size_bytes": int(format_info.get("size") or 0),
        "bit_rate": int(format_info.get("bit_rate") or 0),
        "video": {
            "codec": video_stream.get("codec_name"),
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "duration_seconds": float(video_stream.get("duration") or duration or 0),
            "avg_frame_rate": video_stream.get("avg_frame_rate"),
            "pix_fmt": video_stream.get("pix_fmt"),
        },
        "audio": {
            "codec": audio_stream.get("codec_name"),
            "channels": audio_stream.get("channels"),
            "sample_rate": audio_stream.get("sample_rate"),
            "duration_seconds": float(audio_stream.get("duration") or duration or 0),
        },
    }


def frame_available_duration(
    frame_samples: list[FrameSample],
    interval_seconds: float,
    container_duration: float,
) -> float | None:
    if not frame_samples:
        return None
    decoded_duration = frame_samples[-1].timestamp + interval_seconds
    if container_duration > 0:
        decoded_duration = min(container_duration, decoded_duration)
    return max(0.0, decoded_duration)


def transcript_available_duration(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    source = data.get("source", {}) if isinstance(data, dict) else {}
    for key in ("effective_duration_seconds", "duration_seconds"):
        value = source.get(key)
        if value is None:
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    return None


def media_availability(
    container_duration: float,
    frame_samples: list[FrameSample],
    interval_seconds: float,
    transcript_path: Path | None,
) -> dict[str, Any]:
    tolerance = max(2.0, interval_seconds * 2)
    effective_duration = container_duration
    truncated = False
    sources: dict[str, float] = {"container": round(container_duration, 3)}

    visual_duration = frame_available_duration(
        frame_samples=frame_samples,
        interval_seconds=interval_seconds,
        container_duration=container_duration,
    )
    if visual_duration is not None:
        sources["decoded_video"] = round(visual_duration, 3)
        if container_duration - visual_duration > tolerance:
            truncated = True
            effective_duration = min(effective_duration, visual_duration)

    transcript_duration = transcript_available_duration(transcript_path)
    if transcript_duration is not None:
        sources["transcript"] = round(transcript_duration, 3)
        if container_duration - transcript_duration > tolerance:
            truncated = True
            effective_duration = min(effective_duration, transcript_duration)

    if effective_duration <= 0 and visual_duration is not None:
        effective_duration = visual_duration
    return {
        "effective_duration_seconds": round(effective_duration, 3),
        "container_duration_seconds": round(container_duration, 3),
        "truncated": truncated,
        "duration_sources": sources,
    }


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-")


def frame_timestamps(duration: float, interval_seconds: float) -> list[float]:
    if interval_seconds <= 0:
        raise RuntimeError("--frame-interval-seconds must be > 0")
    if duration <= 0:
        return []
    timestamps = [0.0]
    current = interval_seconds
    while current < duration:
        timestamps.append(round(current, 3))
        current += interval_seconds
    if duration - timestamps[-1] > interval_seconds * 0.4:
        timestamps.append(round(max(0.0, duration - 0.5), 3))
    return timestamps


def fps_expression(frame_interval_seconds: float) -> str:
    fps = 1.0 / frame_interval_seconds
    if abs(fps - round(fps)) < 0.000001:
        return str(int(round(fps)))
    return f"{fps:.6f}"


def sampled_frame_path(frame_dir: Path, index: int, timestamp: float) -> Path:
    return frame_dir / f"frame-{index:04d}-{timestamp:09.3f}s.jpg"


def existing_frame_samples(frame_dir: Path, interval_seconds: float) -> list[FrameSample]:
    samples: list[FrameSample] = []
    for frame_path in sorted(frame_dir.glob("frame-*.jpg")):
        match = re.match(r"frame-(\d+)-([0-9.]+)s\.jpg$", frame_path.name)
        if not match:
            continue
        index = int(match.group(1))
        timestamp = float(match.group(2))
        if abs(timestamp - index * interval_seconds) > max(0.01, interval_seconds * 0.01):
            return []
        samples.append(FrameSample(index=index, timestamp=timestamp, path=frame_path))
    if not samples:
        return []
    if [sample.index for sample in samples] != list(range(len(samples))):
        return []
    return samples


def extract_frame_samples(
    video_path: Path,
    frame_dir: Path,
    duration: float,
    interval_seconds: float,
    frame_width: int,
    ffmpeg_bin: str,
    ffmpeg_hwaccel: str | None,
) -> list[FrameSample]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    reusable_samples = existing_frame_samples(frame_dir, interval_seconds)
    if reusable_samples:
        return reusable_samples

    for stale_frame in frame_dir.glob("frame-*.jpg"):
        stale_frame.unlink()

    if abs(interval_seconds - 1.0) < 0.000001:
        result = run_command(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                *ffmpeg_hwaccel_input_args(ffmpeg_hwaccel),
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps_expression(interval_seconds)},scale={frame_width}:-2",
                "-q:v",
                "3",
                "-pix_fmt",
                "yuvj420p",
                "-strict",
                "unofficial",
                "-start_number",
                "0",
                str(frame_dir / "frame-tmp-%06d.jpg"),
            ],
            check=False,
        )
        temp_frames = sorted(frame_dir.glob("frame-tmp-*.jpg"))
        if result.returncode != 0 and not temp_frames:
            raise RuntimeError(result.stderr.strip() or "ffmpeg failed to extract frames")
        if result.returncode != 0:
            print(
                "warning: ffmpeg returned an error during frame extraction; "
                "continuing with decoded frames."
            )
        samples: list[FrameSample] = []
        for index, temp_path in enumerate(temp_frames):
            timestamp = min(duration, index * interval_seconds)
            frame_path = sampled_frame_path(frame_dir, index, timestamp)
            temp_path.rename(frame_path)
            samples.append(FrameSample(index=index, timestamp=timestamp, path=frame_path))
        return samples

    samples: list[FrameSample] = []
    for index, timestamp in enumerate(frame_timestamps(duration, interval_seconds)):
        frame_path = sampled_frame_path(frame_dir, index, timestamp)
        args = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            *ffmpeg_hwaccel_input_args(ffmpeg_hwaccel),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={frame_width}:-2",
            "-q:v",
            "3",
            str(frame_path),
        ]
        try:
            run_command(args)
        except subprocess.CalledProcessError:
            if samples:
                print(
                    f"warning: stopping frame extraction at {timestamp:.3f}s; "
                    "source appears truncated or unreadable."
                )
                break
            raise
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            if samples:
                print(
                    f"warning: stopping frame extraction at {timestamp:.3f}s; "
                    "ffmpeg produced no frame for the next timestamp."
                )
                break
            raise RuntimeError(f"ffmpeg produced no frame: {frame_path}")
        samples.append(FrameSample(index=index, timestamp=timestamp, path=frame_path))
    return samples


def quote_drawtext_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def label_text(sample: FrameSample) -> str:
    return f"{sample.index:04d}  {format_time(sample.timestamp)}"


def create_labeled_frame(
    ffmpeg_bin: str,
    sample: FrameSample,
    output_path: Path,
    frame_width: int,
    label_file: Path,
) -> None:
    label_file.write_text(label_text(sample), encoding="utf-8")
    font_size = max(12, min(26, round(frame_width * 0.07)))
    label_height = max(24, font_size + 12)
    filter_expr = (
        f"scale={frame_width}:-2,"
        f"drawbox=x=0:y=ih-{label_height}:w=iw:h={label_height}:"
        "color=black@0.68:t=fill,"
        f"drawtext=textfile='{quote_drawtext_path(label_file)}':"
        f"x=8:y=h-th-6:fontsize={font_size}:fontcolor=white:"
        "borderw=1:bordercolor=black"
    )
    run_command(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(sample.path),
            "-vf",
            filter_expr,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(output_path),
        ]
    )


def sheet_frame_entries(
    frame_samples: list[FrameSample],
    root: Path,
    columns: int,
) -> list[dict[str, Any]]:
    return [
        {
            **sample.to_dict(root),
            "sheet_row": index // columns,
            "sheet_column": index % columns,
            "label": label_text(sample),
        }
        for index, sample in enumerate(frame_samples)
    ]


def image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        probe = ffprobe_json(path)
    except Exception:
        return None
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if not isinstance(stream, dict):
        return None
    try:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def adjusted_sheet_columns(
    frame_samples: list[FrameSample],
    requested_columns: int,
    frame_width: int,
) -> tuple[int, bool]:
    if requested_columns <= 0:
        raise RuntimeError("contact sheet columns must be > 0")
    if not frame_samples:
        return requested_columns, False

    dimensions = image_dimensions(frame_samples[0].path)
    if dimensions is None:
        # Most lecture sources are 16:9; use a conservative fallback when the
        # probe is unavailable.
        scaled_height = math.ceil(frame_width * 9 / 16)
    else:
        source_width, source_height = dimensions
        scaled_height = math.ceil(source_height * frame_width / source_width)

    max_rows = max(
        1,
        math.floor(
            (
                MAX_CONTACT_SHEET_DIMENSION_PIXELS
                - 2 * SHEET_MARGIN_PIXELS
                + SHEET_PADDING_PIXELS
            )
            / (scaled_height + SHEET_PADDING_PIXELS)
        ),
    )
    max_columns = max(
        1,
        math.floor(
            (
                MAX_CONTACT_SHEET_DIMENSION_PIXELS
                - 2 * SHEET_MARGIN_PIXELS
                + SHEET_PADDING_PIXELS
            )
            / (frame_width + SHEET_PADDING_PIXELS)
        ),
    )
    needed_columns = max(requested_columns, math.ceil(len(frame_samples) / max_rows))
    if needed_columns > max_columns:
        raise RuntimeError(
            "contact sheet would exceed the safe image dimensions even after "
            f"auto-adjusting columns: frames={len(frame_samples)}, "
            f"frame_width={frame_width}, needed_columns={needed_columns}, "
            f"max_columns={max_columns}. Increase --frame-interval-seconds or "
            "lower --contact-sheet-frame-width."
        )
    return needed_columns, needed_columns != requested_columns


def create_contact_sheet(
    ffmpeg_bin: str,
    frame_samples: list[FrameSample],
    output_path: Path,
    columns: int,
    frame_width: int,
    root: Path,
    sheet_id: str,
    title: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not frame_samples:
        return None
    requested_columns = columns
    columns, columns_adjusted = adjusted_sheet_columns(
        frame_samples,
        requested_columns=columns,
        frame_width=frame_width,
    )
    rows = math.ceil(len(frame_samples) / columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(frame_samples) > 1:
        label_interval = max(0.001, frame_samples[1].timestamp - frame_samples[0].timestamp)
    else:
        label_interval = 1.0
    label_start = frame_samples[0].timestamp
    label_index_offset = frame_samples[0].index
    font_size = max(12, min(26, round(frame_width * 0.07)))
    label_height = max(24, font_size + 12)
    filter_expr = (
        f"scale={frame_width}:-2,"
        f"setpts=PTS+{label_start:.6f}/TB,"
        f"drawbox=x=0:y=ih-{label_height}:w=iw:h={label_height}:"
        "color=black@0.68:t=fill,"
        f"drawtext=text='%{{eif\\:n+{label_index_offset}\\:d}} %{{pts\\:hms}}':"
        f"x=8:y=h-th-6:fontsize={font_size}:fontcolor=white:"
        "borderw=1:bordercolor=black,"
        f"tile={columns}x{rows}:"
        f"padding={SHEET_PADDING_PIXELS}:margin={SHEET_MARGIN_PIXELS}"
    )
    with tempfile.TemporaryDirectory(prefix="extractor-sheet-") as temp_name:
        temp_dir = Path(temp_name)
        input_dir = temp_dir / "frames"
        input_dir.mkdir()
        for index, sample in enumerate(frame_samples):
            (input_dir / f"frame-{index:06d}.jpg").symlink_to(sample.path)
        run_command(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                fps_expression(label_interval),
                "-start_number",
                "0",
                "-i",
                str(input_dir / "frame-%06d.jpg"),
                "-vf",
                filter_expr,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(output_path),
            ]
        )
    payload: dict[str, Any] = {
        "sheet_id": sheet_id,
        "title": title,
        "path": relative_to_root(output_path, root),
        "columns": columns,
        "rows": rows,
        "requested_columns": requested_columns,
        "columns_auto_adjusted": columns_adjusted,
        "cell_width": frame_width,
        "frame_count": len(frame_samples),
        "labeled": True,
        "read_order": (
            "Frames read left-to-right, top-to-bottom. Each cell is labeled "
            "with frame index and source timecode."
        ),
        "frames": sheet_frame_entries(frame_samples, root, columns),
    }
    if extra:
        payload.update(extra)
    return payload


def create_window_sheets(
    ffmpeg_bin: str,
    frame_samples: list[FrameSample],
    output_root: Path,
    video_root: Path,
    duration: float,
    sheet_seconds: float,
    columns: int,
    frame_width: int,
) -> list[dict[str, Any]]:
    if sheet_seconds <= 0:
        raise RuntimeError("--window-sheet-seconds must be > 0")
    sheets_dir = video_root / "sheets"
    if sheets_dir.exists():
        shutil.rmtree(sheets_dir)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    sheets: list[dict[str, Any]] = []
    start = 0.0
    window_index = 0
    while start < duration:
        end = min(duration, start + sheet_seconds)
        window_frames = [
            sample for sample in frame_samples if start <= sample.timestamp < end
        ]
        if window_frames:
            display_range = f"{format_time(start)}-{format_time(end)}"
            sheet_id = f"window-{window_index:03d}-{start:06.1f}s-{end:06.1f}s"
            output_path = sheets_dir / f"{sheet_id}.jpg"
            sheet = create_contact_sheet(
                ffmpeg_bin=ffmpeg_bin,
                frame_samples=window_frames,
                output_path=output_path,
                columns=columns,
                frame_width=frame_width,
                root=output_root,
                sheet_id=sheet_id,
                title=f"Window {window_index:03d}: {display_range}",
                extra={
                    "kind": "window",
                    "window_index": window_index,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "display_range": display_range,
                },
            )
            if sheet:
                sheets.append(sheet)
        start += sheet_seconds
        window_index += 1
    return sheets


def detect_scene_changes(
    video_path: Path,
    threshold: float,
    max_scenes: int,
    ffmpeg_bin: str,
    ffmpeg_hwaccel: str | None,
    duration_limit: float | None,
) -> list[dict[str, Any]]:
    limit_args = ["-t", f"{max(0.001, duration_limit):.3f}"] if duration_limit else []
    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "info",
        *ffmpeg_hwaccel_input_args(ffmpeg_hwaccel),
        *limit_args,
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    result = run_command(args, check=True)
    scene_times: list[float] = []
    for line in result.stderr.splitlines():
        match = re.search(r"pts_time:([0-9.]+)", line)
        if match:
            scene_times.append(float(match.group(1)))
    deduped: list[float] = []
    for value in scene_times:
        if not deduped or value - deduped[-1] > 0.25:
            deduped.append(value)
    return [
        {"timestamp": round(value, 3), "display_time": format_time(value)}
        for value in deduped[:max_scenes]
    ]


def detect_silence(
    video_path: Path,
    noise_db: int,
    min_duration: float,
    ffmpeg_bin: str,
    duration_limit: float | None,
) -> list[dict[str, Any]]:
    limit_args = ["-t", f"{max(0.001, duration_limit):.3f}"] if duration_limit else []
    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "info",
        *limit_args,
        "-i",
        str(video_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    result = run_command(args, check=True)
    starts: list[float] = []
    silences: list[dict[str, Any]] = []
    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start: ([0-9.]+)", line)
        if start_match:
            starts.append(float(start_match.group(1)))
            continue
        end_match = re.search(r"silence_end: ([0-9.]+) \| silence_duration: ([0-9.]+)", line)
        if end_match:
            end = float(end_match.group(1))
            duration = float(end_match.group(2))
            start = starts.pop(0) if starts else max(0.0, end - duration)
            silences.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(duration, 3),
                    "display_range": f"{format_time(start)}-{format_time(end)}",
                }
            )
    return silences


def silence_summary(silences: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    total = sum(float(item["duration"]) for item in silences)
    long_items = [item for item in silences if float(item["duration"]) >= 3.0]
    return {
        "count": len(silences),
        "total_seconds": round(total, 3),
        "ratio": round(total / duration, 4) if duration else 0.0,
        "long_silences": long_items[:30],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {manifest['video']['filename']}",
        "",
        "## Media",
        "",
        f"- Duration: `{format_time(manifest['media']['duration_seconds'])}`",
        f"- Container duration: `{format_time(manifest['media'].get('container_duration_seconds', manifest['media']['duration_seconds']))}`",
        f"- Truncated: `{manifest['media'].get('truncated', False)}`",
        f"- Video: `{manifest['media']['video']['width']}x{manifest['media']['video']['height']}` `{manifest['media']['video']['codec']}`",
        f"- Audio: `{manifest['media']['audio']['codec']}` `{manifest['media']['audio']['channels']}` channel(s)",
        "",
        "## Evidence Files",
        "",
        f"- Contact sheet: `{manifest['visual']['contact_sheet']}`",
        f"- Frame directory: `{manifest['visual']['frame_directory']}`",
        f"- Frame interval: `{manifest['visual']['frame_interval_seconds']}s`",
        f"- Frame width: `{manifest['visual']['frame_width']}px`",
        f"- Contact sheet cell width: `{manifest['visual']['contact_sheet_frame_width']}px`",
        f"- Window sheets: `{len(manifest['visual']['window_sheets'])}`",
        f"- Window sheet seconds: `{manifest['visual']['window_sheet_seconds']}s`",
        f"- Window sheet cell width: `{manifest['visual']['window_sheet_frame_width']}px`",
        f"- FFmpeg binary: `{manifest['visual']['ffmpeg_bin']}`",
        f"- FFmpeg hardware decode: `{manifest['visual']['ffmpeg_hwaccel']}`",
        "",
        "## LLM Sheet Guidance",
        "",
        "- Use the overview sheet for broad visual scanning.",
        "- Use the labeled window sheets for precise timestamp grounding.",
        "- Each sheet cell shows `frame_index timecode`; manifest sheet metadata maps cells to timestamps.",
        "",
        "## Window Sheets",
        "",
    ]
    for sheet in manifest["visual"]["window_sheets"][:20]:
        lines.append(
            f"- `{sheet['display_range']}` -> `{sheet['path']}` "
            f"({sheet['frame_count']} frames)"
        )
    if len(manifest["visual"]["window_sheets"]) > 20:
        lines.append(f"- ... {len(manifest['visual']['window_sheets']) - 20} more")
    lines.extend([
        "",
        "## Scene Changes",
        "",
    ])
    for scene in manifest["visual"]["scene_changes"][:20]:
        lines.append(f"- `{scene['display_time']}`")
    if len(manifest["visual"]["scene_changes"]) > 20:
        lines.append(f"- ... {len(manifest['visual']['scene_changes']) - 20} more")
    lines.extend(["", "## Audio Silence", ""])
    summary = manifest["audio"]["silence_summary"]
    lines.append(
        f"- {summary['count']} silence regions, {summary['total_seconds']:.1f}s total "
        f"({summary['ratio']:.1%} of runtime)"
    )
    for item in summary["long_silences"][:20]:
        lines.append(f"- `{item['display_range']}` ({item['duration']:.1f}s)")
    if manifest.get("transcript"):
        lines.extend(
            [
                "",
                "## Transcript",
                "",
                f"- Path: `{manifest['transcript']['path']}`",
                f"- QA score: `{manifest['transcript']['qa']['score']}/100`",
                f"- Findings: `{manifest['transcript']['qa']['finding_counts']}`",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def transcript_summary(transcript_path: Path | None) -> dict[str, Any] | None:
    if transcript_path is None:
        return None
    from extractor.plan_clip_ads import default_qa_args
    from extractor.qa_transcript import build_report, read_transcript

    data = read_transcript(transcript_path)
    report = build_report(transcript_path, default_qa_args())
    return {
        "path": str(transcript_path),
        "source_duration_seconds": data.get("source", {}).get("duration_seconds"),
        "segment_count": len(data.get("segments", [])),
        "word_count": len(data.get("words", [])),
        "qa": {
            "score": report["score"],
            "finding_counts": report["finding_counts"],
            "top_findings": report["findings"][:10],
        },
    }


def preprocess_video(video_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_root = args.output_dir.expanduser().resolve()
    video_root = output_root / safe_stem(video_path)
    frame_dir = video_root / "frames"
    video_root.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    if not ffmpeg_supports_filter(ffmpeg_bin, "drawtext"):
        raise RuntimeError(
            f"{ffmpeg_bin} does not support the drawtext filter. "
            "Install ffmpeg-full or pass --ffmpeg-bin to a drawtext-enabled ffmpeg."
        )
    resolved_hwaccel = resolve_ffmpeg_hwaccel(args.ffmpeg_hwaccel, ffmpeg_bin) or "none"

    probe = ffprobe_json(video_path)
    media = media_summary(probe)
    container_duration = float(media["duration_seconds"])
    frame_samples = extract_frame_samples(
        video_path=video_path,
        frame_dir=frame_dir,
        duration=container_duration,
        interval_seconds=args.frame_interval_seconds,
        frame_width=args.frame_width,
        ffmpeg_bin=ffmpeg_bin,
        ffmpeg_hwaccel=resolved_hwaccel,
    )
    availability = media_availability(
        container_duration=container_duration,
        frame_samples=frame_samples,
        interval_seconds=args.frame_interval_seconds,
        transcript_path=args.transcript,
    )
    duration = float(availability["effective_duration_seconds"])
    media["container_duration_seconds"] = availability["container_duration_seconds"]
    media["effective_duration_seconds"] = availability["effective_duration_seconds"]
    media["duration_seconds"] = availability["effective_duration_seconds"]
    media["truncated"] = availability["truncated"]
    media["duration_sources"] = availability["duration_sources"]
    media["video"]["effective_duration_seconds"] = availability["effective_duration_seconds"]
    media["audio"]["effective_duration_seconds"] = availability["effective_duration_seconds"]
    contact_sheet_path = video_root / "contact-sheet.jpg"
    overview_sheet = create_contact_sheet(
        ffmpeg_bin=ffmpeg_bin,
        frame_samples=frame_samples,
        output_path=contact_sheet_path,
        columns=args.contact_sheet_columns,
        frame_width=args.contact_sheet_frame_width,
        root=output_root,
        sheet_id="overview",
        title="Full video overview",
        extra={
            "kind": "overview",
            "start": 0.0,
            "end": round(duration, 3),
            "display_range": f"{format_time(0)}-{format_time(duration)}",
            "container_end": round(container_duration, 3),
        },
    )
    window_sheets = create_window_sheets(
        ffmpeg_bin=ffmpeg_bin,
        frame_samples=frame_samples,
        output_root=output_root,
        video_root=video_root,
        duration=duration,
        sheet_seconds=args.window_sheet_seconds,
        columns=args.window_sheet_columns,
        frame_width=args.window_sheet_frame_width,
    )

    scene_changes = [] if args.skip_scene_detection else detect_scene_changes(
        video_path=video_path,
        threshold=args.scene_threshold,
        max_scenes=args.max_scenes,
        ffmpeg_bin=ffmpeg_bin,
        ffmpeg_hwaccel=resolved_hwaccel,
        duration_limit=duration,
    )
    silences = [] if args.skip_audio_analysis else detect_silence(
        video_path=video_path,
        noise_db=args.silence_noise_db,
        min_duration=args.silence_min_duration,
        ffmpeg_bin=ffmpeg_bin,
        duration_limit=duration,
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "api_calls": False,
        "video": {
            "path": str(video_path),
            "filename": video_path.name,
            "stem": safe_stem(video_path),
        },
        "media": media,
        "visual": {
            "frame_interval_seconds": args.frame_interval_seconds,
            "frame_width": args.frame_width,
            "contact_sheet_frame_width": args.contact_sheet_frame_width,
            "contact_sheet_columns": args.contact_sheet_columns,
            "window_sheet_seconds": args.window_sheet_seconds,
            "window_sheet_frame_width": args.window_sheet_frame_width,
            "window_sheet_columns": args.window_sheet_columns,
            "ffmpeg_bin": ffmpeg_bin,
            "ffmpeg_hwaccel": resolved_hwaccel,
            "container_duration_seconds": round(container_duration, 3),
            "effective_duration_seconds": round(duration, 3),
            "truncated": bool(availability["truncated"]),
            "frame_directory": relative_to_root(frame_dir, output_root),
            "contact_sheet": relative_to_root(contact_sheet_path, output_root),
            "overview_sheet": overview_sheet,
            "window_sheets": window_sheets,
            "sampled_frames": [sample.to_dict(output_root) for sample in frame_samples],
            "scene_detection": {
                "threshold": args.scene_threshold,
                "max_scenes": args.max_scenes,
                "skipped": args.skip_scene_detection,
            },
            "scene_changes": scene_changes,
        },
        "audio": {
            "silence_detection": {
                "noise_db": args.silence_noise_db,
                "min_duration_seconds": args.silence_min_duration,
                "skipped": args.skip_audio_analysis,
            },
            "silences": silences,
            "silence_summary": silence_summary(silences, duration),
        },
        "transcript": transcript_summary(args.transcript) if args.transcript else None,
        "prompt_cache_guidance": {
            "recommended_prefix_items": [
                "stable system instructions",
                "this manifest JSON",
                "transcript JSON or compact transcript timeline",
                "QA report",
                "labeled overview and window sheet metadata",
            ],
            "variable_suffix_items": [
                "stage-specific task",
                "prior stage output",
            ],
        },
    }
    write_json(video_root / "manifest.json", manifest)
    if args.write_markdown:
        write_markdown(video_root / "manifest.md", manifest)
    return manifest


def infer_transcript_for_video(video_path: Path, transcript_dir: Path | None) -> Path | None:
    if transcript_dir is None:
        return None
    candidates = sorted(transcript_dir.expanduser().glob(f"{video_path.stem}*.json"))
    return candidates[0] if candidates else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess videos into local evidence bundles for clip-ad planning."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transcript", type=Path, default=None)
    parser.add_argument("--transcript-dir", type=Path, default=None)
    parser.add_argument(
        "--frame-interval-seconds",
        type=float,
        default=DEFAULT_FRAME_INTERVAL_SECONDS,
        help="Seconds between sampled evidence frames.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="Downscaled sampled frame width in pixels.",
    )
    parser.add_argument(
        "--contact-sheet-frame-width",
        type=int,
        default=DEFAULT_CONTACT_SHEET_FRAME_WIDTH,
        help="Downscaled width for each cell in the overview contact sheet.",
    )
    parser.add_argument(
        "--contact-sheet-columns",
        type=int,
        default=DEFAULT_CONTACT_SHEET_COLUMNS,
        help="Number of columns in the overview contact sheet.",
    )
    parser.add_argument(
        "--window-sheet-seconds",
        type=float,
        default=DEFAULT_WINDOW_SHEET_SECONDS,
        help="Seconds covered by each labeled LLM window sheet.",
    )
    parser.add_argument(
        "--window-sheet-frame-width",
        type=int,
        default=DEFAULT_WINDOW_SHEET_FRAME_WIDTH,
        help="Downscaled width for each cell in labeled LLM window sheets.",
    )
    parser.add_argument(
        "--window-sheet-columns",
        type=int,
        default=DEFAULT_WINDOW_SHEET_COLUMNS,
        help="Number of columns in labeled LLM window sheets.",
    )
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
        help="Hardware decode mode for video frame extraction and scene detection.",
    )
    parser.add_argument("--scene-threshold", type=float, default=DEFAULT_SCENE_THRESHOLD)
    parser.add_argument("--max-scenes", type=int, default=DEFAULT_MAX_SCENES)
    parser.add_argument("--silence-noise-db", type=int, default=DEFAULT_SILENCE_NOISE_DB)
    parser.add_argument("--silence-min-duration", type=float, default=DEFAULT_SILENCE_MIN_DURATION)
    parser.add_argument("--skip-scene-detection", action="store_true")
    parser.add_argument("--skip-audio-analysis", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        resolve_ffmpeg_bin(args.ffmpeg_bin)
        require_command("ffprobe")
        if args.transcript and len(args.videos) > 1:
            raise RuntimeError("--transcript can only be used with one video. Use --transcript-dir for batches.")
        for index, video in enumerate(args.videos):
            if args.transcript_dir:
                args.transcript = infer_transcript_for_video(video, args.transcript_dir)
            manifest = preprocess_video(video, args)
            print(manifest["video"]["filename"])
            print(f"  duration: {format_time(manifest['media']['duration_seconds'])}")
            print(f"  frames: {len(manifest['visual']['sampled_frames'])}")
            print(f"  scene changes: {len(manifest['visual']['scene_changes'])}")
            print(f"  silences: {manifest['audio']['silence_summary']['count']}")
            if manifest.get("transcript"):
                print(f"  transcript QA: {manifest['transcript']['qa']['score']}/100")
            output_path = args.output_dir.expanduser().resolve() / manifest["video"]["stem"] / "manifest.json"
            print(f"  wrote: {output_path}")
            if index != len(args.videos) - 1:
                print()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
