from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.preprocess_video import (
    DEFAULT_FFMPEG_BIN,
    DEFAULT_FFMPEG_HWACCEL,
    FFMPEG_HWACCEL_CHOICES,
    ffmpeg_hwaccel_input_args,
    format_time,
    resolve_ffmpeg_bin,
    resolve_ffmpeg_hwaccel,
)


DEFAULT_OUTPUT_DIR = Path("visual-index")
DEFAULT_FRAME_INTERVAL_SECONDS = 1.0
DEFAULT_FRAME_WIDTH = 480
DEFAULT_SHEET_SECONDS = 60.0
DEFAULT_SHEET_COLUMNS = 10
DEFAULT_CANDIDATE_PAD_SECONDS = 10.0
DEFAULT_CANDIDATE_SHEET_COLUMNS = 10


@dataclass(frozen=True)
class IndexedFrame:
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


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def ffprobe_duration(path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def fps_expression(frame_interval_seconds: float) -> str:
    if frame_interval_seconds <= 0:
        raise RuntimeError("--frame-interval-seconds must be > 0")
    fps = 1.0 / frame_interval_seconds
    if abs(fps - round(fps)) < 0.000001:
        return str(int(round(fps)))
    return f"{fps:.6f}"


def frames_available_duration(
    frames: list[IndexedFrame],
    frame_interval_seconds: float,
    container_duration: float,
) -> tuple[float, bool]:
    if not frames:
        return 0.0, container_duration > 0
    decoded_duration = frames[-1].timestamp + frame_interval_seconds
    if container_duration > 0:
        decoded_duration = min(container_duration, decoded_duration)
    tolerance = max(2.0, frame_interval_seconds * 2)
    truncated = container_duration - decoded_duration > tolerance
    return (
        round(decoded_duration if truncated else container_duration, 3),
        truncated,
    )


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    duration: float,
    frame_interval_seconds: float,
    frame_width: int,
    ffmpeg_bin: str,
    ffmpeg_hwaccel: str | None,
    force: bool,
) -> list[IndexedFrame]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(frames_dir.glob("frame-*.jpg"))
    if force:
        for frame in existing_frames:
            frame.unlink()
        existing_frames = []

    if not existing_frames:
        result = subprocess.run(
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
                f"fps={fps_expression(frame_interval_seconds)},scale={frame_width}:-2",
                "-q:v",
                "3",
                "-pix_fmt",
                "yuvj420p",
                "-strict",
                "unofficial",
                "-start_number",
                "0",
                str(frames_dir / "frame-%06d.jpg"),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frame_paths = sorted(frames_dir.glob("frame-*.jpg"))
        if result.returncode != 0 and not frame_paths:
            raise RuntimeError(result.stderr.strip() or "ffmpeg failed to extract frames")
        if result.returncode != 0:
            print(
                "warning: ffmpeg returned an error during frame extraction; "
                "continuing with decoded frames."
            )

    frame_paths = sorted(frames_dir.glob("frame-*.jpg"))
    frames: list[IndexedFrame] = []
    for index, frame_path in enumerate(frame_paths):
        timestamp = min(duration, index * frame_interval_seconds)
        frames.append(IndexedFrame(index=index, timestamp=timestamp, path=frame_path))
    return frames


def selected_frames(
    frames: list[IndexedFrame],
    start: float,
    end: float,
) -> list[IndexedFrame]:
    return [frame for frame in frames if start <= frame.timestamp <= end]


def make_sheet_inputs(sheet_input_dir: Path, frames: list[IndexedFrame]) -> None:
    sheet_input_dir.mkdir(parents=True, exist_ok=True)
    for old_link in sheet_input_dir.glob("frame-*.jpg"):
        old_link.unlink()
    for index, frame in enumerate(frames):
        link = sheet_input_dir / f"frame-{index:06d}.jpg"
        link.symlink_to(frame.path)


def make_contact_sheet(
    sheet_input_dir: Path,
    frame_count: int,
    columns: int,
    output_path: Path,
) -> None:
    if frame_count <= 0:
        raise RuntimeError(f"No frames selected for contact sheet: {output_path}")
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
            "-start_number",
            "0",
            "-i",
            str(sheet_input_dir / "frame-%06d.jpg"),
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


def sheet_frame_entries(
    frames: list[IndexedFrame],
    root: Path,
    columns: int,
) -> list[dict[str, Any]]:
    return [
        {
            **frame.to_dict(root),
            "sheet_row": index // columns,
            "sheet_column": index % columns,
        }
        for index, frame in enumerate(frames)
    ]


def create_sheet(
    frames: list[IndexedFrame],
    root: Path,
    sheets_dir: Path,
    sheet_id: str,
    columns: int,
    title: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_id = safe_name(sheet_id)
    input_dir = sheets_dir / f"{safe_id}-inputs"
    output_path = sheets_dir / f"{safe_id}.jpg"
    make_sheet_inputs(input_dir, frames)
    make_contact_sheet(
        sheet_input_dir=input_dir,
        frame_count=len(frames),
        columns=columns,
        output_path=output_path,
    )
    payload: dict[str, Any] = {
        "sheet_id": safe_id,
        "title": title,
        "path": relative_to_root(output_path, root),
        "columns": columns,
        "frame_count": len(frames),
        "read_order": "Frames read left-to-right, top-to-bottom. Use sheet_row and sheet_column to map cells back to timestamps.",
        "frames": sheet_frame_entries(frames, root, columns),
    }
    if extra:
        payload.update(extra)
    return payload


def create_periodic_sheets(
    frames: list[IndexedFrame],
    root: Path,
    sheets_dir: Path,
    duration: float,
    sheet_seconds: float,
    columns: int,
) -> list[dict[str, Any]]:
    sheets: list[dict[str, Any]] = []
    window_index = 0
    start = 0.0
    while start < duration:
        end = min(duration, start + sheet_seconds - 0.001)
        window_frames = selected_frames(frames, start, end)
        if window_frames:
            sheets.append(
                create_sheet(
                    frames=window_frames,
                    root=root,
                    sheets_dir=sheets_dir,
                    sheet_id=f"window-{window_index:03d}-{start:06.1f}s-{end:06.1f}s",
                    columns=columns,
                    title=f"{format_time(start)}-{format_time(end)}",
                    extra={
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "display_range": f"{format_time(start)}-{format_time(end)}",
                    },
                )
            )
        start += sheet_seconds
        window_index += 1
    return sheets


def iter_stage2_candidates(stage2_output: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for angle_set in stage2_output.get("angle_moment_sets", []) or []:
        angle = angle_set.get("angle")
        for candidate in angle_set.get("candidate_windows", []) or []:
            if not isinstance(candidate, dict):
                continue
            copied = dict(candidate)
            copied["angle"] = angle
            copied["stage1_rank"] = angle_set.get("stage1_rank")
            candidates.append(copied)
    return candidates


def create_candidate_sheets(
    frames: list[IndexedFrame],
    root: Path,
    sheets_dir: Path,
    stage2_output_path: Path | None,
    duration: float,
    pad_seconds: float,
    columns: int,
) -> list[dict[str, Any]]:
    if stage2_output_path is None:
        return []
    stage2_output = read_json(stage2_output_path)
    candidate_sheets: list[dict[str, Any]] = []
    for candidate in iter_stage2_candidates(stage2_output):
        candidate_id = str(candidate.get("candidate_id") or "candidate")
        start = max(0.0, float(candidate.get("start") or 0) - pad_seconds)
        end = min(duration, float(candidate.get("end") or 0) + pad_seconds)
        window_frames = selected_frames(frames, start, end)
        if not window_frames:
            continue
        candidate_sheets.append(
            create_sheet(
                frames=window_frames,
                root=root,
                sheets_dir=sheets_dir,
                sheet_id=f"candidate-{candidate_id}-{start:06.1f}s-{end:06.1f}s",
                columns=columns,
                title=f"{candidate_id}: {candidate.get('working_title') or candidate.get('opening_hook') or ''}",
                extra={
                    "candidate_id": candidate_id,
                    "angle": candidate.get("angle"),
                    "stage1_rank": candidate.get("stage1_rank"),
                    "candidate_window": {
                        "start": candidate.get("start"),
                        "end": candidate.get("end"),
                        "display_range": (
                            f"{format_time(float(candidate.get('start') or 0))}-"
                            f"{format_time(float(candidate.get('end') or 0))}"
                        ),
                        "working_title": candidate.get("working_title"),
                        "opening_hook": candidate.get("opening_hook"),
                        "payoff": candidate.get("payoff"),
                        "visual_expectation": candidate.get("visual_expectation"),
                        "recommended_use": candidate.get("recommended_use"),
                        "confidence": candidate.get("confidence"),
                    },
                    "sheet_start": round(start, 3),
                    "sheet_end": round(end, 3),
                    "sheet_display_range": f"{format_time(start)}-{format_time(end)}",
                },
            )
        )
    return candidate_sheets


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# Visual Index: {manifest['video']['filename']}",
        "",
        f"- Frame interval: `{manifest['frame_interval_seconds']}s`",
        f"- Frames: `{manifest['frame_count']}`",
        f"- Periodic sheets: `{len(manifest['periodic_sheets'])}`",
        f"- Candidate sheets: `{len(manifest['candidate_sheets'])}`",
        "",
    ]
    if manifest.get("candidate_sheets"):
        lines.append("## Candidate Sheets")
        lines.append("")
        for sheet in manifest["candidate_sheets"]:
            window = sheet.get("candidate_window") or {}
            lines.append(
                f"- `{sheet.get('candidate_id')}` {window.get('working_title', '')}: "
                f"`{sheet.get('sheet_display_range')}` -> `{sheet.get('path')}`"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_visual_index(args: argparse.Namespace) -> dict[str, Any]:
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    stage2_output = args.stage2_output.expanduser().resolve() if args.stage2_output else None
    if stage2_output is not None and not stage2_output.exists():
        raise FileNotFoundError(stage2_output)

    output_root = args.output_dir.expanduser().resolve()
    video_root = output_root / safe_name(video_path.stem)
    frames_dir = video_root / "frames"
    sheets_dir = video_root / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    container_duration = ffprobe_duration(video_path)
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    resolved_hwaccel = resolve_ffmpeg_hwaccel(args.ffmpeg_hwaccel, ffmpeg_bin) or "none"
    frames = extract_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        duration=container_duration,
        frame_interval_seconds=args.frame_interval_seconds,
        frame_width=args.frame_width,
        ffmpeg_bin=ffmpeg_bin,
        ffmpeg_hwaccel=resolved_hwaccel,
        force=args.force,
    )
    duration, truncated = frames_available_duration(
        frames=frames,
        frame_interval_seconds=args.frame_interval_seconds,
        container_duration=container_duration,
    )
    periodic_sheets = create_periodic_sheets(
        frames=frames,
        root=output_root,
        sheets_dir=sheets_dir,
        duration=duration,
        sheet_seconds=args.sheet_seconds,
        columns=args.sheet_columns,
    )
    candidate_sheets = create_candidate_sheets(
        frames=frames,
        root=output_root,
        sheets_dir=sheets_dir,
        stage2_output_path=stage2_output,
        duration=duration,
        pad_seconds=args.candidate_pad_seconds,
        columns=args.candidate_sheet_columns,
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "api_calls": False,
        "video": {
            "path": str(video_path),
            "filename": video_path.name,
            "stem": safe_name(video_path.stem),
            "duration_seconds": round(duration, 3),
            "effective_duration_seconds": round(duration, 3),
            "container_duration_seconds": round(container_duration, 3),
            "truncated": truncated,
            "display_duration": format_time(duration),
        },
        "frame_interval_seconds": args.frame_interval_seconds,
        "frame_width": args.frame_width,
        "ffmpeg_bin": ffmpeg_bin,
        "ffmpeg_hwaccel": resolved_hwaccel,
        "frame_count": len(frames),
        "frames_dir": relative_to_root(frames_dir, output_root),
        "frames": [frame.to_dict(output_root) for frame in frames],
        "periodic_sheets": periodic_sheets,
        "candidate_sheets": candidate_sheets,
        "stage2_output": str(stage2_output) if stage2_output else None,
        "usage_note": (
            "Use candidate_sheets as visual proof packages for concrete-object/action clip "
            "planning. The model should cite exact timestamps from the sheet before rendering."
        ),
    }
    write_json(video_root / "manifest.json", manifest)
    if args.write_markdown:
        write_markdown(video_root / "manifest.md", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a dense local visual index and contact sheets for video clip planning."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage2-output", type=Path, default=None)
    parser.add_argument("--frame-interval-seconds", type=float, default=DEFAULT_FRAME_INTERVAL_SECONDS)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
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
        help="Hardware decode mode for frame extraction.",
    )
    parser.add_argument("--sheet-seconds", type=float, default=DEFAULT_SHEET_SECONDS)
    parser.add_argument("--sheet-columns", type=int, default=DEFAULT_SHEET_COLUMNS)
    parser.add_argument("--candidate-pad-seconds", type=float, default=DEFAULT_CANDIDATE_PAD_SECONDS)
    parser.add_argument("--candidate-sheet-columns", type=int, default=DEFAULT_CANDIDATE_SHEET_COLUMNS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        resolve_ffmpeg_bin(args.ffmpeg_bin)
        require_command("ffprobe")
        manifest = build_visual_index(args)
        print(manifest["video"]["filename"])
        print(f"  duration: {manifest['video']['display_duration']}")
        print(f"  frames: {manifest['frame_count']} at every {manifest['frame_interval_seconds']}s")
        print(f"  periodic sheets: {len(manifest['periodic_sheets'])}")
        print(f"  candidate sheets: {len(manifest['candidate_sheets'])}")
        output_path = args.output_dir.expanduser().resolve() / manifest["video"]["stem"] / "manifest.json"
        print(f"  wrote: {output_path}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
