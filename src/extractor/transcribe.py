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


DEFAULT_PROVIDER = "elevenlabs"
DEFAULT_OPENAI_MODEL = "whisper-1"
DEFAULT_ELEVENLABS_MODEL = "scribe_v2"
TRANSCRIBE_PROVIDER_CHOICES = ("openai", "elevenlabs")
DEFAULT_AUDIO_BITRATE_KBPS = 64
DEFAULT_MAX_UPLOAD_MB = 24
DEFAULT_ENV_FILE = ".env"
DEFAULT_WHISPER_1_USD_PER_MINUTE = 0.006
DEFAULT_ELEVENLABS_USD_PER_HOUR = 0.40
ELEVENLABS_TRANSCRIPTION_URL = "https://api.elevenlabs.io/v1/speech-to-text"


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    duration: float
    path: Path
    requested_duration: float | None = None
    extraction_returncode: int = 0
    truncated: bool = False


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


def env_bool(name: str, default: bool | None = None) -> bool | None:
    value = env_value(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


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


def resolve_transcribe_provider(value: str | None) -> str:
    provider = value or env_value("TRANSCRIBE_PROVIDER") or DEFAULT_PROVIDER
    provider = provider.strip().lower()
    if provider not in TRANSCRIBE_PROVIDER_CHOICES:
        choices = ", ".join(TRANSCRIBE_PROVIDER_CHOICES)
        raise ValueError(
            f"Unknown transcription provider {provider!r}; choose one of: {choices}."
        )
    return provider


def default_model_for_provider(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    if provider == "elevenlabs":
        return DEFAULT_ELEVENLABS_MODEL
    raise ValueError(f"Unknown transcription provider: {provider}")


def resolve_transcribe_model(provider: str, value: str | None) -> str:
    if value:
        return value
    if provider == "openai":
        return (
            env_value("TRANSCRIBE_OPENAI_MODEL")
            or env_value("TRANSCRIBE_MODEL")
            or default_model_for_provider(provider)
        )
    if provider == "elevenlabs":
        return env_value("TRANSCRIBE_ELEVENLABS_MODEL") or default_model_for_provider(provider)
    raise ValueError(f"Unknown transcription provider: {provider}")


def safe_name(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value
    )
    return cleaned.strip(".-") or "default"


def transcript_file_stem(video_stem: str, provider: str, model: str) -> str:
    model_name = safe_name(model)
    if provider == "openai":
        return f"{video_stem}.{model_name}"
    return f"{video_stem}.{safe_name(provider)}-{model_name}"


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


def ffprobe_duration_or_none(path: Path) -> float | None:
    try:
        return ffprobe_duration(path)
    except Exception:
        return None


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


def estimate_api_cost(
    duration_seconds: float,
    provider: str,
    model: str,
    whisper_1_usd_per_minute: float,
    elevenlabs_usd_per_hour: float,
) -> float:
    if provider == "openai" and model != "whisper-1":
        raise RuntimeError(
            f"No built-in cost estimator for model {model!r}. "
            "Use --model whisper-1 or add pricing support before making API calls."
        )
    if provider == "openai":
        return duration_seconds / 60 * whisper_1_usd_per_minute
    if provider == "elevenlabs":
        if model not in {"scribe_v1", "scribe_v2"}:
            raise RuntimeError(
                f"No built-in cost estimator for ElevenLabs model {model!r}. "
                "Use --model scribe_v2, --model scribe_v1, or add pricing support."
            )
        return duration_seconds / 3600 * elevenlabs_usd_per_hour
    raise ValueError(f"Unknown transcription provider: {provider}")


def print_cost_estimate(
    videos: list[Path],
    provider: str,
    model: str,
    whisper_1_usd_per_minute: float,
    elevenlabs_usd_per_hour: float,
) -> float:
    total_duration = 0.0
    total_cost = 0.0
    print("API cost estimate:")
    for video in videos:
        duration = ffprobe_duration(video.expanduser().resolve())
        cost = estimate_api_cost(
            duration,
            provider=provider,
            model=model,
            whisper_1_usd_per_minute=whisper_1_usd_per_minute,
            elevenlabs_usd_per_hour=elevenlabs_usd_per_hour,
        )
        total_duration += duration
        total_cost += cost
        if provider == "openai":
            print(
                f"  {video}: {duration / 60:.2f} audio minutes "
                f"x ${whisper_1_usd_per_minute:.4f}/minute = ${cost:.4f}"
            )
        elif provider == "elevenlabs":
            print(
                f"  {video}: {duration / 60:.2f} audio minutes "
                f"x ${elevenlabs_usd_per_hour:.4f}/hour = ${cost:.4f}"
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
        result = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0 and (
            not chunk_path.exists() or chunk_path.stat().st_size == 0
        ):
            if audio_chunks:
                print(
                    f"warning: stopping audio extraction at {start:.3f}s; "
                    "source appears truncated or unreadable."
                )
                break
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(message or f"ffmpeg failed extracting {chunk_path}")
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            if audio_chunks:
                print(
                    f"warning: stopping audio extraction at {start:.3f}s; "
                    "ffmpeg produced no audio for the next chunk."
                )
                break
            raise RuntimeError(f"ffmpeg produced no audio chunk: {chunk_path}")

        size = chunk_path.stat().st_size
        if size > max_upload_bytes:
            raise RuntimeError(
                f"Audio chunk exceeds {max_upload_mb} MB after extraction: "
                f"{chunk_path} ({size / 1024 / 1024:.1f} MB). "
                "Lower --audio-bitrate-kbps or --chunk-seconds."
            )
        actual_duration = ffprobe_duration_or_none(chunk_path)
        if actual_duration is None or actual_duration <= 0:
            if audio_chunks:
                print(
                    f"warning: stopping audio extraction at {start:.3f}s; "
                    f"could not read duration for {chunk_path}."
                )
                break
            raise RuntimeError(f"Could not read duration for extracted audio chunk: {chunk_path}")
        effective_duration = min(duration, actual_duration)
        truncated = result.returncode != 0 or effective_duration < duration - 0.5
        audio_chunks.append(
            AudioChunk(
                index=index,
                start=start,
                duration=effective_duration,
                path=chunk_path,
                requested_duration=duration,
                extraction_returncode=result.returncode,
                truncated=truncated,
            )
        )
        if truncated:
            print(
                f"warning: extracted audio ended at {start + effective_duration:.3f}s "
                f"before requested chunk end {start + duration:.3f}s; treating source "
                "as truncated and continuing with available audio."
            )
            break
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


def transcribe_chunk_openai(
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


def transcribe_chunk_elevenlabs(
    chunk: AudioChunk,
    model: str,
    language: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import httpx

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs transcription.")

    fields: dict[str, str] = {
        "model_id": model,
        "timestamps_granularity": "word",
        "tag_audio_events": str(args.elevenlabs_tag_audio_events).lower(),
        "diarize": str(args.elevenlabs_diarize).lower(),
        "file_format": "other",
    }
    if language:
        fields["language_code"] = language
    if args.elevenlabs_num_speakers is not None:
        fields["num_speakers"] = str(args.elevenlabs_num_speakers)
    if args.elevenlabs_no_verbatim:
        fields["no_verbatim"] = "true"
    if args.elevenlabs_temperature is not None:
        fields["temperature"] = str(args.elevenlabs_temperature)
    if args.elevenlabs_seed is not None:
        fields["seed"] = str(args.elevenlabs_seed)

    timeout = env_float("ELEVENLABS_CLIENT_TIMEOUT_SECONDS", 1800.0)
    headers = {"xi-api-key": api_key}
    with chunk.path.open("rb") as audio_file:
        files = {"file": (chunk.path.name, audio_file, "audio/mpeg")}
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                ELEVENLABS_TRANSCRIPTION_URL,
                headers=headers,
                data=fields,
                files=files,
            )
    if response.status_code >= 400:
        raise RuntimeError(
            f"ElevenLabs transcription failed with HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )
    return response.json()


def transcribe_chunk(
    chunk: AudioChunk,
    args: argparse.Namespace,
    language: str | None,
    prompt: str | None,
) -> dict[str, Any]:
    if args.provider == "openai":
        return transcribe_chunk_openai(
            chunk=chunk,
            model=args.model,
            language=language,
            prompt=prompt,
        )
    if args.provider == "elevenlabs":
        return transcribe_chunk_elevenlabs(
            chunk=chunk,
            model=args.model,
            language=language,
            args=args,
        )
    raise ValueError(f"Unknown transcription provider: {args.provider}")


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


def normalize_elevenlabs_word(
    word: dict[str, Any],
    offset: float,
    chunk_index: int,
) -> dict[str, Any] | None:
    text = str(word.get("text", ""))
    if not text.strip():
        return None
    start = offset_number(word.get("start"), offset)
    end = offset_number(word.get("end"), offset)
    if start is None or end is None:
        return None

    normalized: dict[str, Any] = {
        "chunk_index": chunk_index,
        "text": text,
        "start": start,
        "end": end,
    }
    for key in ("type", "speaker_id", "logprob", "channel_index"):
        if key in word:
            normalized[key] = word[key]
    return normalized


def join_word_texts(words: list[dict[str, Any]]) -> str:
    text = " ".join(
        str(word.get("text", "")).strip()
        for word in words
        if str(word.get("text", "")).strip()
    )
    for mark in (".", ",", "?", "!", ":", ";", ")", "]", "}"):
        text = text.replace(f" {mark}", mark)
    for mark in ("(", "[", "{"):
        text = text.replace(f"{mark} ", mark)
    return text.strip()


def average_logprob(words: list[dict[str, Any]]) -> float | None:
    values = [float(word["logprob"]) for word in words if word.get("logprob") is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def segment_from_words(
    words: list[dict[str, Any]],
    chunk_index: int,
) -> dict[str, Any] | None:
    if not words:
        return None
    start = words[0].get("start")
    end = words[-1].get("end")
    if start is None or end is None:
        return None
    segment: dict[str, Any] = {
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "text": join_word_texts(words),
        "words": words,
    }
    avg_logprob = average_logprob(words)
    if avg_logprob is not None:
        segment["avg_logprob"] = avg_logprob
    speaker_ids = sorted({str(word["speaker_id"]) for word in words if word.get("speaker_id")})
    if speaker_ids:
        segment["speaker_ids"] = speaker_ids
    return segment


def build_elevenlabs_segments(
    words: list[dict[str, Any]],
    chunk_index: int,
    fallback_text: str,
    fallback_start: float,
    fallback_end: float,
) -> list[dict[str, Any]]:
    if not words:
        text = fallback_text.strip()
        if not text:
            return []
        return [
            {
                "chunk_index": chunk_index,
                "start": round(fallback_start, 3),
                "end": round(fallback_end, 3),
                "text": text,
                "words": [],
            }
        ]

    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    previous_end: float | None = None
    previous_speaker: str | None = None

    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        speaker = str(word.get("speaker_id")) if word.get("speaker_id") else None
        text = str(word.get("text", "")).strip()

        should_break = False
        if current:
            segment_start = float(current[0]["start"])
            segment_duration = max(0.0, end - segment_start)
            word_gap = start - previous_end if previous_end is not None else 0.0
            if previous_speaker and speaker and speaker != previous_speaker:
                should_break = True
            elif word_gap > 1.25:
                should_break = True
            elif segment_duration >= 12.0:
                should_break = True
            elif len(current) >= 45:
                should_break = True
            elif (
                segment_duration >= 3.0
                and str(current[-1].get("text", "")).strip().endswith((".", "?", "!"))
            ):
                should_break = True

        if should_break:
            segment = segment_from_words(current, chunk_index)
            if segment:
                segments.append(segment)
            current = []

        current.append(word)
        previous_end = end
        if speaker:
            previous_speaker = speaker
        elif text:
            previous_speaker = None

    segment = segment_from_words(current, chunk_index)
    if segment:
        segments.append(segment)
    return segments


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
    provider: str,
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
        if provider == "openai":
            chunk_words = [
                normalize_word(word, chunk.start, chunk.index)
                for word in payload.get("words", [])
            ]
            all_words.extend(chunk_words)
            all_segments.extend(
                normalize_segment(segment, chunk.start, chunk.index, chunk_words)
                for segment in payload.get("segments", [])
            )
        elif provider == "elevenlabs":
            chunk_words = []
            for word in payload.get("words", []):
                normalized = normalize_elevenlabs_word(word, chunk.start, chunk.index)
                if normalized is not None:
                    chunk_words.append(normalized)
            all_words.extend(chunk_words)
            all_segments.extend(
                build_elevenlabs_segments(
                    words=chunk_words,
                    chunk_index=chunk.index,
                    fallback_text=str(payload.get("text", "")),
                    fallback_start=chunk.start,
                    fallback_end=chunk.start + chunk.duration,
                )
            )
        else:
            raise ValueError(f"Unknown transcription provider: {provider}")

    detected_language = language
    language_probabilities: list[float] = []
    for payload in raw_payloads:
        if detected_language is None and payload.get("language_code"):
            detected_language = str(payload["language_code"])
        if payload.get("language_probability") is not None:
            language_probabilities.append(float(payload["language_probability"]))

    available_duration = (
        max(chunk.start + chunk.duration for chunk in chunks)
        if chunks
        else source_duration
    )
    truncated = available_duration < source_duration - 0.5 or any(
        chunk.truncated for chunk in chunks
    )

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "model": model,
        "language": detected_language,
        "language_probability": (
            round(sum(language_probabilities) / len(language_probabilities), 4)
            if language_probabilities
            else None
        ),
        "source": {
            "path": str(source_video),
            "filename": source_video.name,
            "duration_seconds": round(available_duration, 3),
            "effective_duration_seconds": round(available_duration, 3),
            "container_duration_seconds": round(source_duration, 3),
            "truncated": truncated,
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
                "requested_duration": round(
                    chunk.requested_duration
                    if chunk.requested_duration is not None
                    else chunk.duration,
                    3,
                ),
                "truncated": chunk.truncated,
                "extraction_returncode": chunk.extraction_returncode,
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

    base_name = transcript_file_stem(video_path.stem, args.provider, args.model)
    normalized_path = args.output_dir / f"{base_name}.json"
    srt_path = args.output_dir / f"{base_name}.srt"

    if normalized_path.exists() and not args.force and not args.dry_run:
        print(f"Skipping existing transcript: {normalized_path}")
        return

    if args.dry_run:
        print(f"{video_path}")
        print(f"  provider: {args.provider}")
        print(f"  model: {args.model}")
        print(f"  duration: {duration:.3f}s")
        print(f"  planned chunks: {len(chunk_plan)}")
        for index, start, chunk_duration in chunk_plan:
            print(
                f"  chunk {index:03d}: start={start:.3f}s "
                f"duration={chunk_duration:.3f}s"
            )
        print(f"  output: {normalized_path}")
        return

    prompt = read_prompt(args) if args.provider == "openai" else None
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
                args=args,
                language=args.language,
                prompt=prompt,
            )
            raw_payloads.append(payload)
            if args.provider == "openai":
                raw_path = (
                    raw_dir / f"{video_path.stem}.chunk-{chunk.index:03d}.{args.model}.json"
                )
            else:
                raw_path = raw_dir / f"{base_name}.chunk-{chunk.index:03d}.json"
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
            provider=args.provider,
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
        description=(
            "Transcribe videos with OpenAI Whisper or ElevenLabs Scribe by uploading "
            "audio-only chunks and writing normalized timestamps."
        )
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
    parser.add_argument(
        "--provider",
        choices=TRANSCRIBE_PROVIDER_CHOICES,
        default=None,
        help="Transcription provider. Defaults to TRANSCRIBE_PROVIDER or elevenlabs.",
    )
    parser.add_argument("--model", default=None, help="Provider transcription model.")
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
    parser.add_argument(
        "--elevenlabs-usd-per-hour",
        type=float,
        default=None,
        help="Pricing used for ElevenLabs Scribe cost estimates.",
    )
    parser.add_argument(
        "--elevenlabs-diarize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ask ElevenLabs to annotate speaker IDs.",
    )
    parser.add_argument(
        "--elevenlabs-tag-audio-events",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ask ElevenLabs to include non-speech audio event tags.",
    )
    parser.add_argument(
        "--elevenlabs-no-verbatim",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ask ElevenLabs Scribe v2 to remove fillers, false starts, and non-speech sounds.",
    )
    parser.add_argument(
        "--elevenlabs-num-speakers",
        type=int,
        default=None,
        help="Optional maximum speaker count for ElevenLabs diarization.",
    )
    parser.add_argument(
        "--elevenlabs-temperature",
        type=float,
        default=None,
        help="Optional ElevenLabs transcription temperature.",
    )
    parser.add_argument(
        "--elevenlabs-seed",
        type=int,
        default=None,
        help="Optional ElevenLabs deterministic sampling seed.",
    )
    return parser


def apply_env_defaults(args: argparse.Namespace) -> None:
    args.provider = resolve_transcribe_provider(args.provider)
    args.model = resolve_transcribe_model(args.provider, args.model)
    if args.output_dir is None:
        args.output_dir = env_path("TRANSCRIBE_OUTPUT_DIR", Path("transcripts"))
    if args.language is None:
        args.language = env_value("TRANSCRIBE_LANGUAGE")
    if args.prompt is None and args.prompt_file is None:
        args.prompt = (
            env_value("TRANSCRIBE_PROMPT") if args.provider == "openai" else None
        )
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
    if args.elevenlabs_usd_per_hour is None:
        args.elevenlabs_usd_per_hour = env_float(
            "TRANSCRIBE_ELEVENLABS_USD_PER_HOUR", DEFAULT_ELEVENLABS_USD_PER_HOUR
        )
    if args.elevenlabs_diarize is None:
        args.elevenlabs_diarize = env_bool("TRANSCRIBE_ELEVENLABS_DIARIZE", False)
    if args.elevenlabs_tag_audio_events is None:
        args.elevenlabs_tag_audio_events = env_bool(
            "TRANSCRIBE_ELEVENLABS_TAG_AUDIO_EVENTS", True
        )
    if args.elevenlabs_no_verbatim is None:
        args.elevenlabs_no_verbatim = env_bool("TRANSCRIBE_ELEVENLABS_NO_VERBATIM", False)
    if args.elevenlabs_num_speakers is None:
        args.elevenlabs_num_speakers = env_int("TRANSCRIBE_ELEVENLABS_NUM_SPEAKERS")
    if args.elevenlabs_temperature is None:
        args.elevenlabs_temperature = env_float("TRANSCRIBE_ELEVENLABS_TEMPERATURE")
    if args.elevenlabs_seed is None:
        args.elevenlabs_seed = env_int("TRANSCRIBE_ELEVENLABS_SEED")


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
                provider=args.provider,
                model=args.model,
                whisper_1_usd_per_minute=args.whisper_1_usd_per_minute,
                elevenlabs_usd_per_hour=args.elevenlabs_usd_per_hour,
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

        if (
            should_call_api
            and args.provider == "openai"
            and not os.environ.get("OPENAI_API_KEY")
        ):
            raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")
        if (
            should_call_api
            and args.provider == "elevenlabs"
            and not os.environ.get("ELEVENLABS_API_KEY")
        ):
            raise RuntimeError("ELEVENLABS_API_KEY is required unless --dry-run is used.")
        if args.provider == "elevenlabs" and (args.prompt or args.prompt_file):
            print(
                "warning: --prompt/TRANSCRIBE_PROMPT is ignored by ElevenLabs transcription",
                file=sys.stderr,
            )
        for video in args.videos:
            transcribe_video(video, args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
