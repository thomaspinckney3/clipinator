from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "whisper-1"
DEFAULT_AUDIO_BITRATE_KBPS = 64
DEFAULT_MAX_UPLOAD_MB = 24
DEFAULT_ENV_FILE = ".env"
DEFAULT_WHISPER_1_USD_PER_MINUTE = 0.006


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    duration: float
    path: Path


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found on PATH: {name}")


def env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def env_int(name: str, default: int | None = None) -> int | None:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


def env_float(name: str, default: float | None = None) -> float | None:
    value = env_value(name)
    if value is None:
        return default
    return float(value)


def env_path(name: str, default: Path) -> Path:
    value = env_value(name)
    if value is None:
        return default
    return Path(value)


def load_env_file(path: Path | None) -> None:
    if path is None:
        return
    path = path.expanduser()
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


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


def computed_chunk_seconds(max_upload_mb: int, audio_bitrate_kbps: int) -> int:
    max_bytes = max_upload_mb * 1024 * 1024
    bitrate_bytes_per_second = audio_bitrate_kbps * 1000 / 8
    # Leave margin for MP3/container overhead and imperfect bitrate control.
    return max(30, math.floor(max_bytes / bitrate_bytes_per_second * 0.85))


def plan_chunks(
    duration: float,
    max_upload_mb: int,
    audio_bitrate_kbps: int,
    requested_chunk_seconds: int | None,
) -> list[tuple[int, float, float]]:
    chunk_seconds = requested_chunk_seconds or computed_chunk_seconds(
        max_upload_mb=max_upload_mb,
        audio_bitrate_kbps=audio_bitrate_kbps,
    )
    chunks: list[tuple[int, float, float]] = []
    start = 0.0
    index = 0
    while start < duration:
        chunk_duration = min(float(chunk_seconds), duration - start)
        chunks.append((index, start, chunk_duration))
        start += chunk_duration
        index += 1
    return chunks


def estimate_api_cost(duration_seconds: float, model: str, whisper_1_usd_per_minute: float) -> float:
    if model != "whisper-1":
        raise RuntimeError(
            f"No built-in cost estimator for model {model!r}. "
            "Use --model whisper-1 or add pricing support before making API calls."
        )
    return duration_seconds / 60 * whisper_1_usd_per_minute


def print_cost_estimate(
    videos: list[Path],
    model: str,
    whisper_1_usd_per_minute: float,
) -> float:
    total_duration = 0.0
    total_cost = 0.0
    print("API cost estimate:")
    for video in videos:
        duration = ffprobe_duration(video.expanduser().resolve())
        cost = estimate_api_cost(duration, model, whisper_1_usd_per_minute)
        total_duration += duration
        total_cost += cost
        print(
            f"  {video}: {duration / 60:.2f} audio minutes "
            f"x ${whisper_1_usd_per_minute:.4f}/minute = ${cost:.4f}"
        )
    print(f"  Total: {total_duration / 60:.2f} audio minutes = ${total_cost:.4f}")
    return total_cost


def extract_audio_chunks(
    video_path: Path,
    work_dir: Path,
    chunks: list[tuple[int, float, float]],
    audio_bitrate_kbps: int,
    max_upload_mb: int,
) -> list[AudioChunk]:
    max_upload_bytes = max_upload_mb * 1024 * 1024
    audio_chunks: list[AudioChunk] = []
    for index, start, duration in chunks:
        chunk_path = work_dir / f"{video_path.stem}.chunk-{index:03d}.mp3"
        args = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            f"{audio_bitrate_kbps}k",
            str(chunk_path),
        ]
        run_command(args)
        size = chunk_path.stat().st_size
        if size > max_upload_bytes:
            raise RuntimeError(
                f"Audio chunk exceeds {max_upload_mb} MB after extraction: "
                f"{chunk_path} ({size / 1024 / 1024:.1f} MB). "
                "Lower --audio-bitrate-kbps or --chunk-seconds."
            )
        audio_chunks.append(
            AudioChunk(index=index, start=start, duration=duration, path=chunk_path)
        )
    return audio_chunks


def transcription_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "json"):
        return json.loads(value.json())
    raise TypeError(f"Unsupported transcription response type: {type(value)!r}")


def transcribe_chunk(
    chunk: AudioChunk,
    model: str,
    language: str | None,
    prompt: str | None,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    request: dict[str, Any] = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
        "temperature": 0,
    }
    if language:
        request["language"] = language
    if prompt:
        request["prompt"] = prompt

    with chunk.path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(file=audio_file, **request)
    return transcription_to_dict(response)


def offset_number(value: Any, offset: float) -> float | None:
    if value is None:
        return None
    return round(float(value) + offset, 3)


def normalize_word(word: dict[str, Any], offset: float, chunk_index: int) -> dict[str, Any]:
    normalized = {
        "chunk_index": chunk_index,
        "text": word.get("word", word.get("text", "")),
        "start": offset_number(word.get("start"), offset),
        "end": offset_number(word.get("end"), offset),
    }
    for key in ("confidence", "probability"):
        if key in word:
            normalized[key] = word[key]
    return normalized


def normalize_segment(
    segment: dict[str, Any],
    offset: float,
    chunk_index: int,
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    start = offset_number(segment.get("start"), offset)
    end = offset_number(segment.get("end"), offset)
    normalized: dict[str, Any] = {
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "text": segment.get("text", "").strip(),
    }
    for key in (
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
        "temperature",
    ):
        if key in segment:
            normalized[key] = segment[key]

    if start is not None and end is not None:
        normalized["words"] = [
            word
            for word in words
            if word.get("start") is not None
            and word.get("end") is not None
            and float(word["start"]) >= start - 0.05
            and float(word["end"]) <= end + 0.05
        ]
    else:
        normalized["words"] = []
    return normalized


def normalize_transcription(
    source_video: Path,
    source_duration: float,
    model: str,
    language: str | None,
    audio_bitrate_kbps: int,
    chunks: list[AudioChunk],
    raw_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    all_words: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    full_text: list[str] = []

    for chunk, payload in zip(chunks, raw_payloads, strict=True):
        full_text.append(payload.get("text", "").strip())
        chunk_words = [
            normalize_word(word, chunk.start, chunk.index)
            for word in payload.get("words", [])
        ]
        all_words.extend(chunk_words)
        all_segments.extend(
            normalize_segment(segment, chunk.start, chunk.index, chunk_words)
            for segment in payload.get("segments", [])
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "provider": "openai",
        "model": model,
        "language": language,
        "source": {
            "path": str(source_video),
            "filename": source_video.name,
            "duration_seconds": round(source_duration, 3),
        },
        "audio_extraction": {
            "codec": "mp3",
            "channels": 1,
            "sample_rate_hz": 16000,
            "bitrate_kbps": audio_bitrate_kbps,
        },
        "chunks": [
            {
                "index": chunk.index,
                "start": round(chunk.start, 3),
                "end": round(chunk.start + chunk.duration, 3),
                "duration": round(chunk.duration, 3),
            }
            for chunk in chunks
        ],
        "text": " ".join(text for text in full_text if text),
        "segments": all_segments,
        "words": all_words,
    }


def srt_timestamp(seconds: float | None) -> str:
    if seconds is None:
        seconds = 0.0
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    index = 1
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{srt_timestamp(segment.get('start'))} --> {srt_timestamp(segment.get('end'))}",
                text,
                "",
            ]
        )
        index += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_prompt(args: argparse.Namespace) -> str | None:
    if args.prompt and args.prompt_file:
        raise ValueError("Use either --prompt or --prompt-file, not both.")
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return args.prompt


def transcribe_video(video_path: Path, args: argparse.Namespace) -> None:
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(video_path)
    chunk_plan = plan_chunks(
        duration=duration,
        max_upload_mb=args.max_upload_mb,
        audio_bitrate_kbps=args.audio_bitrate_kbps,
        requested_chunk_seconds=args.chunk_seconds,
    )

    base_name = f"{video_path.stem}.{args.model}"
    normalized_path = args.output_dir / f"{base_name}.json"
    srt_path = args.output_dir / f"{base_name}.srt"

    if normalized_path.exists() and not args.force and not args.dry_run:
        print(f"Skipping existing transcript: {normalized_path}")
        return

    if args.dry_run:
        print(f"{video_path}")
        print(f"  duration: {duration:.3f}s")
        print(f"  planned chunks: {len(chunk_plan)}")
        for index, start, chunk_duration in chunk_plan:
            print(
                f"  chunk {index:03d}: start={start:.3f}s "
                f"duration={chunk_duration:.3f}s"
            )
        print(f"  output: {normalized_path}")
        return

    prompt = read_prompt(args)
    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if args.audio_dir:
        work_dir = Path(args.audio_dir).expanduser().resolve()
    elif args.keep_audio:
        work_dir = Path(tempfile.mkdtemp(prefix="extractor-audio-"))
        print(f"Keeping audio chunks in {work_dir}")
    else:
        temporary_context = tempfile.TemporaryDirectory()
        work_dir = Path(temporary_context.name)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        chunks = extract_audio_chunks(
            video_path=video_path,
            work_dir=work_dir,
            chunks=chunk_plan,
            audio_bitrate_kbps=args.audio_bitrate_kbps,
            max_upload_mb=args.max_upload_mb,
        )

        raw_payloads: list[dict[str, Any]] = []
        for chunk in chunks:
            print(f"Transcribing {video_path.name} chunk {chunk.index + 1}/{len(chunks)}")
            payload = transcribe_chunk(
                chunk=chunk,
                model=args.model,
                language=args.language,
                prompt=prompt,
            )
            raw_payloads.append(payload)
            raw_path = raw_dir / f"{video_path.stem}.chunk-{chunk.index:03d}.{args.model}.json"
            write_json(
                raw_path,
                {
                    "chunk": {
                        "index": chunk.index,
                        "start": round(chunk.start, 3),
                        "duration": round(chunk.duration, 3),
                    },
                    "response": payload,
                },
            )

        normalized = normalize_transcription(
            source_video=video_path,
            source_duration=duration,
            model=args.model,
            language=args.language,
            audio_bitrate_kbps=args.audio_bitrate_kbps,
            chunks=chunks,
            raw_payloads=raw_payloads,
        )
        write_json(normalized_path, normalized)
        write_srt(srt_path, normalized["segments"])
        print(f"Wrote {normalized_path}")
        print(f"Wrote {srt_path}")
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe videos with OpenAI whisper-1 and normalized timestamps."
    )
    parser.add_argument("videos", nargs="+", type=Path, help="Video files to transcribe.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(DEFAULT_ENV_FILE),
        help="Optional .env file to load before reading config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for normalized transcripts and raw provider responses.",
    )
    parser.add_argument("--model", default=None, help="OpenAI transcription model.")
    parser.add_argument("--language", default=None, help="Optional ISO language code, e.g. en.")
    parser.add_argument("--prompt", default=None, help="Optional transcription prompt.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional path containing a transcription prompt.",
    )
    parser.add_argument(
        "--audio-bitrate-kbps",
        type=int,
        default=None,
        help="MP3 bitrate for uploaded audio chunks.",
    )
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=None,
        help="Maximum audio chunk size target in MB.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=None,
        help="Override automatic chunk duration.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Optional directory for extracted audio chunks.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep temporary audio chunks when using the default temp directory.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing transcripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print chunk plan without API calls.")
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Print API cost estimate without making API calls.",
    )
    parser.add_argument(
        "--approve-api-cost",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Explicitly approve API calls up to this estimated USD amount. "
            "Required for non-dry-run API calls."
        ),
    )
    parser.add_argument(
        "--whisper-1-usd-per-minute",
        type=float,
        default=None,
        help="Pricing used for whisper-1 cost estimates.",
    )
    return parser


def apply_env_defaults(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = env_path("TRANSCRIBE_OUTPUT_DIR", Path("transcripts"))
    if args.model is None:
        args.model = env_value("TRANSCRIBE_MODEL") or DEFAULT_MODEL
    if args.language is None:
        args.language = env_value("TRANSCRIBE_LANGUAGE")
    if args.prompt is None and args.prompt_file is None:
        args.prompt = env_value("TRANSCRIBE_PROMPT")
    if args.audio_bitrate_kbps is None:
        args.audio_bitrate_kbps = env_int(
            "TRANSCRIBE_AUDIO_BITRATE_KBPS", DEFAULT_AUDIO_BITRATE_KBPS
        )
    if args.max_upload_mb is None:
        args.max_upload_mb = env_int("TRANSCRIBE_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)
    if args.chunk_seconds is None:
        args.chunk_seconds = env_int("TRANSCRIBE_CHUNK_SECONDS")
    if args.whisper_1_usd_per_minute is None:
        args.whisper_1_usd_per_minute = env_float(
            "TRANSCRIBE_WHISPER_1_USD_PER_MINUTE", DEFAULT_WHISPER_1_USD_PER_MINUTE
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        load_env_file(args.env_file)
        apply_env_defaults(args)
        require_command("ffmpeg")
        require_command("ffprobe")

        should_call_api = not args.dry_run and not args.estimate_cost
        if args.estimate_cost or should_call_api:
            estimated_cost = print_cost_estimate(
                videos=args.videos,
                model=args.model,
                whisper_1_usd_per_minute=args.whisper_1_usd_per_minute,
            )
            if args.estimate_cost:
                return 0
            if args.approve_api_cost is None:
                raise RuntimeError(
                    "API calls require explicit approval. Re-run with "
                    f"--approve-api-cost {estimated_cost:.4f} or a higher USD cap."
                )
            if estimated_cost > args.approve_api_cost:
                raise RuntimeError(
                    f"Estimated API cost ${estimated_cost:.4f} exceeds approved cap "
                    f"${args.approve_api_cost:.4f}."
                )
            print(
                f"Approved API cost cap: ${args.approve_api_cost:.4f}; "
                f"estimated cost: ${estimated_cost:.4f}"
            )

        if should_call_api and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")
        for video in args.videos:
            transcribe_video(video, args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
