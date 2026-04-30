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


DEFAULT_OUTPUT_DIR = Path("video-evidence")
DEFAULT_FRAME_INTERVAL_SECONDS = 10.0
DEFAULT_FRAME_WIDTH = 480
DEFAULT_CONTACT_SHEET_COLUMNS = 5
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


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-")


def frame_timestamps(duration: float, interval_seconds: float) -> list[float]:
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


def extract_frame_samples(
    video_path: Path,
    frame_dir: Path,
    duration: float,
    interval_seconds: float,
    frame_width: int,
) -> list[FrameSample]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    samples: list[FrameSample] = []
    for index, timestamp in enumerate(frame_timestamps(duration, interval_seconds)):
        frame_path = frame_dir / f"frame-{index:04d}-{timestamp:09.3f}s.jpg"
        args = [
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
            "-vf",
            f"scale={frame_width}:-2",
            "-q:v",
            "3",
            str(frame_path),
        ]
        run_command(args)
        samples.append(FrameSample(index=index, timestamp=timestamp, path=frame_path))
    return samples


def create_contact_sheet(
    frame_samples: list[FrameSample],
    output_path: Path,
    columns: int,
    frame_width: int,
) -> Path | None:
    if not frame_samples:
        return None
    rows = math.ceil(len(frame_samples) / columns)
    # Re-read the sampled frames at a steady cadence and tile them into one sheet.
    pattern = frame_samples[0].path.parent / "frame-*.jpg"
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-pattern_type",
        "glob",
        "-i",
        str(pattern),
        "-vf",
        f"scale={frame_width}:-2,tile={columns}x{rows}:padding=8:margin=8",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(output_path),
    ]
    run_command(args)
    return output_path


def detect_scene_changes(video_path: Path, threshold: float, max_scenes: int) -> list[dict[str, Any]]:
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
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
) -> list[dict[str, Any]]:
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(video_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_duration}",
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
        f"- Video: `{manifest['media']['video']['width']}x{manifest['media']['video']['height']}` `{manifest['media']['video']['codec']}`",
        f"- Audio: `{manifest['media']['audio']['codec']}` `{manifest['media']['audio']['channels']}` channel(s)",
        "",
        "## Evidence Files",
        "",
        f"- Contact sheet: `{manifest['visual']['contact_sheet']}`",
        f"- Frame directory: `{manifest['visual']['frame_directory']}`",
        "",
        "## Scene Changes",
        "",
    ]
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

    probe = ffprobe_json(video_path)
    media = media_summary(probe)
    duration = float(media["duration_seconds"])
    frame_samples = extract_frame_samples(
        video_path=video_path,
        frame_dir=frame_dir,
        duration=duration,
        interval_seconds=args.frame_interval_seconds,
        frame_width=args.frame_width,
    )
    contact_sheet_path = video_root / "contact-sheet.jpg"
    create_contact_sheet(
        frame_samples=frame_samples,
        output_path=contact_sheet_path,
        columns=args.contact_sheet_columns,
        frame_width=args.contact_sheet_frame_width,
    )

    scene_changes = [] if args.skip_scene_detection else detect_scene_changes(
        video_path=video_path,
        threshold=args.scene_threshold,
        max_scenes=args.max_scenes,
    )
    silences = [] if args.skip_audio_analysis else detect_silence(
        video_path=video_path,
        noise_db=args.silence_noise_db,
        min_duration=args.silence_min_duration,
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
            "frame_directory": relative_to_root(frame_dir, output_root),
            "contact_sheet": relative_to_root(contact_sheet_path, output_root),
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
                "sampled frame references with identical detail setting",
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
    parser.add_argument("--frame-interval-seconds", type=float, default=DEFAULT_FRAME_INTERVAL_SECONDS)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--contact-sheet-frame-width", type=int, default=320)
    parser.add_argument("--contact-sheet-columns", type=int, default=DEFAULT_CONTACT_SHEET_COLUMNS)
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
        require_command("ffmpeg")
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
