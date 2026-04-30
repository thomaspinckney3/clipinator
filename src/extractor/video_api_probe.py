from __future__ import annotations

import argparse
import base64
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.openai_stage1 import (
    Pricing,
    cost_from_counts,
    estimate_text_tokens,
    response_output_text,
    sha256_file,
    sha256_text,
    usage_dict,
    write_json,
)
from extractor.render_stage3 import ffprobe_summary, require_command, run_command, safe_name
from extractor.transcribe import env_float, env_int, env_value, load_env_file


DEFAULT_ENV_FILE = ".env"
DEFAULT_OUTPUT_DIR = Path("/Volumes/t9/extract/api-probes")
DEFAULT_VIDEO = Path(
    "/Volumes/t9/extract/renders/space-school-underwater-fixed/renders-tight/"
    "clip_ad_01_underwater_spacewalk_payoff_first_tight.draft.mp4"
)

DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OPENAI_REASONING_EFFORT = "low"
DEFAULT_OPENAI_INPUT_USD_PER_MILLION = 5.00
DEFAULT_OPENAI_OUTPUT_USD_PER_MILLION = 30.00

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_ANTHROPIC_INPUT_USD_PER_MILLION = 3.00
DEFAULT_ANTHROPIC_OUTPUT_USD_PER_MILLION = 15.00

DEFAULT_MAX_OUTPUT_TOKENS = 300
DEFAULT_SAMPLE_START_SECONDS = 1.8
DEFAULT_SAMPLE_SECONDS = 4.0
DEFAULT_SAMPLE_HEIGHT = 360
DEFAULT_SAMPLE_FPS = 10
DEFAULT_SAMPLE_CRF = 36
DEFAULT_SAMPLE_AUDIO_KBPS = 32


PROBE_PROMPT = """\
This is a capability probe for short-form clip QA.

I attached an MP4 video file. If you can inspect the actual video and audio,
reply with compact JSON:

{
  "accepted_video_input": true,
  "could_inspect_motion": true|false,
  "could_inspect_audio": true|false,
  "visible_summary": "...",
  "audio_summary": "...",
  "limitations": ["..."]
}

If you cannot inspect the MP4, say so plainly in the same JSON shape with
accepted_video_input=false and list the limitation.
"""


@dataclass(frozen=True)
class ProbePricing:
    input_usd_per_million: float
    output_usd_per_million: float


def data_url_for_mp4(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def estimate_base64_file_tokens(path: Path) -> int:
    # Conservative fallback estimate if an API treats file_data as text-like input.
    encoded_chars = 4 * math.ceil(path.stat().st_size / 3)
    return estimate_text_tokens("data:video/mp4;base64,") + estimate_text_tokens("a" * encoded_chars)


def estimate_probe_cost(
    sample_path: Path,
    prompt: str,
    max_output_tokens: int,
    pricing: ProbePricing,
) -> dict[str, Any]:
    text_tokens = estimate_text_tokens(prompt)
    file_tokens = estimate_base64_file_tokens(sample_path)
    input_tokens = text_tokens + file_tokens
    estimated_usd = (
        input_tokens / 1_000_000 * pricing.input_usd_per_million
        + max_output_tokens / 1_000_000 * pricing.output_usd_per_million
    )
    return {
        "method": (
            "conservative upper-bound using chars/4 for the base64 MP4 payload; "
            "if the provider rejects video before model execution, billable cost should be near $0"
        ),
        "text_tokens_estimate": text_tokens,
        "base64_file_tokens_estimate": file_tokens,
        "input_tokens_estimate": input_tokens,
        "max_output_tokens": max_output_tokens,
        "estimated_usd": round(estimated_usd, 6),
    }


def actual_openai_cost(usage: dict[str, Any], pricing: ProbePricing) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens") or 0)
    cached_tokens = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    stage_pricing = Pricing(
        input_usd_per_million=pricing.input_usd_per_million,
        cached_input_usd_per_million=pricing.input_usd_per_million,
        output_usd_per_million=pricing.output_usd_per_million,
    )
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "estimated_usd": round(
            cost_from_counts(
                input_tokens=input_tokens,
                cached_input_tokens=cached_tokens,
                output_tokens=output_tokens,
                pricing=stage_pricing,
            ),
            6,
        ),
    }


def actual_anthropic_cost(usage: dict[str, Any], pricing: ProbePricing) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_creation_tokens = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    billed_input_tokens = input_tokens + cache_creation_tokens + cache_read_tokens
    estimated_usd = (
        billed_input_tokens / 1_000_000 * pricing.input_usd_per_million
        + output_tokens / 1_000_000 * pricing.output_usd_per_million
    )
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "estimated_usd": round(estimated_usd, 6),
    }


def build_sample_video(args: argparse.Namespace) -> Path:
    source = args.video.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if args.use_original:
        return source

    require_command("ffmpeg")
    asset_dir = args.output_dir / "_video-probe-assets" / safe_name(source.stem)
    asset_dir.mkdir(parents=True, exist_ok=True)
    sample_path = asset_dir / (
        f"{safe_name(source.stem)}."
        f"{args.sample_start:.3f}s-{args.sample_seconds:.3f}s."
        f"h{args.sample_height}.fps{args.sample_fps}.crf{args.sample_crf}.mp4"
    )
    if sample_path.exists() and sample_path.stat().st_size > 0:
        return sample_path

    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{args.sample_start:.3f}",
            "-t",
            f"{args.sample_seconds:.3f}",
            "-i",
            str(source),
            "-vf",
            f"scale=-2:{args.sample_height},fps={args.sample_fps}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(args.sample_crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            f"{args.sample_audio_kbps}k",
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            str(sample_path),
        ]
    )
    return sample_path


def response_error_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    for attr in ("status_code", "code", "type", "param"):
        value = getattr(exc, attr, None)
        if value is not None:
            payload[attr] = value
    body = getattr(exc, "body", None)
    if body is not None:
        payload["body"] = body
    return payload


def openai_reasoning_config(model: str, effort: str | None) -> dict[str, str] | None:
    if not effort:
        return None
    lowered = model.lower()
    if lowered.startswith("gpt-5") or lowered.startswith("o"):
        return {"effort": effort}
    return None


def run_openai_probe(args: argparse.Namespace, sample_path: Path) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    content: list[dict[str, Any]] = [{"type": "input_text", "text": args.prompt}]
    uploaded_file_id: str | None = None

    try:
        if args.openai_mode == "direct-input-file":
            content.append(
                {
                    "type": "input_file",
                    "filename": sample_path.name,
                    "file_data": data_url_for_mp4(sample_path),
                }
            )
        elif args.openai_mode == "files-input-file":
            with sample_path.open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="user_data")
            uploaded_file_id = uploaded.id
            content.append({"type": "input_file", "file_id": uploaded_file_id})
        else:
            raise ValueError(f"Unknown OpenAI probe mode: {args.openai_mode}")

        kwargs: dict[str, Any] = {
            "model": args.model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": args.max_output_tokens,
            "store": False,
        }
        reasoning = openai_reasoning_config(args.model, args.reasoning_effort)
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
        response = client.responses.create(**kwargs)
        response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
        return {
            "ok": True,
            "uploaded_file_id": uploaded_file_id,
            "response": response_dict,
            "output_text": response_output_text(response, response_dict),
            "usage": usage_dict(response_dict),
        }
    except Exception as exc:  # noqa: BLE001 - probe should preserve provider errors.
        return {
            "ok": False,
            "uploaded_file_id": uploaded_file_id,
            "error": response_error_payload(exc),
            "usage": {},
        }
    finally:
        if uploaded_file_id and args.cleanup_uploaded_files:
            try:
                client.files.delete(uploaded_file_id)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to delete OpenAI uploaded file {uploaded_file_id}: {exc}", file=os.sys.stderr)


def anthropic_headers(args: argparse.Namespace, *, beta: bool) -> dict[str, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic probes.")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": args.anthropic_version,
    }
    if beta:
        headers["anthropic-beta"] = args.anthropic_files_beta
    return headers


def run_anthropic_probe(args: argparse.Namespace, sample_path: Path) -> dict[str, Any]:
    import httpx

    base_url = args.anthropic_base_url.rstrip("/")
    uploaded_file_id: str | None = None
    client = httpx.Client(timeout=args.http_timeout)
    try:
        if args.anthropic_mode == "base64-document":
            content: list[dict[str, Any]] = [
                {"type": "text", "text": args.prompt},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "video/mp4",
                        "data": base64.b64encode(sample_path.read_bytes()).decode("ascii"),
                    },
                },
            ]
            response = client.post(
                f"{base_url}/v1/messages",
                headers={
                    **anthropic_headers(args, beta=False),
                    "content-type": "application/json",
                },
                json={
                    "model": args.model,
                    "max_tokens": args.max_output_tokens,
                    "messages": [{"role": "user", "content": content}],
                },
            )
        elif args.anthropic_mode == "files-document":
            with sample_path.open("rb") as handle:
                upload_response = client.post(
                    f"{base_url}/v1/files",
                    headers=anthropic_headers(args, beta=True),
                    files={"file": (sample_path.name, handle, "video/mp4")},
                )
            upload_json = safe_response_json(upload_response)
            if upload_response.status_code >= 400:
                return {
                    "ok": False,
                    "uploaded_file_id": None,
                    "upload_status_code": upload_response.status_code,
                    "upload_response": upload_json,
                    "usage": {},
                }
            uploaded_file_id = upload_json.get("id")
            response = client.post(
                f"{base_url}/v1/messages",
                headers={
                    **anthropic_headers(args, beta=True),
                    "content-type": "application/json",
                },
                json={
                    "model": args.model,
                    "max_tokens": args.max_output_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": args.prompt},
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "file",
                                        "file_id": uploaded_file_id,
                                    },
                                },
                            ],
                        }
                    ],
                },
            )
        else:
            raise ValueError(f"Unknown Anthropic probe mode: {args.anthropic_mode}")

        response_json = safe_response_json(response)
        return {
            "ok": response.status_code < 400,
            "uploaded_file_id": uploaded_file_id,
            "status_code": response.status_code,
            "response": response_json,
            "output_text": anthropic_output_text(response_json),
            "usage": response_json.get("usage") if isinstance(response_json, dict) else {},
        }
    except Exception as exc:  # noqa: BLE001 - probe should preserve provider errors.
        return {
            "ok": False,
            "uploaded_file_id": uploaded_file_id,
            "error": response_error_payload(exc),
            "usage": {},
        }
    finally:
        if uploaded_file_id and args.cleanup_uploaded_files:
            try:
                client.delete(
                    f"{base_url}/v1/files/{uploaded_file_id}",
                    headers=anthropic_headers(args, beta=True),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to delete Anthropic uploaded file {uploaded_file_id}: {exc}", file=os.sys.stderr)
        client.close()


def safe_response_json(response: Any) -> dict[str, Any]:
    try:
        value = response.json()
    except Exception:  # noqa: BLE001
        return {"raw_text": response.text}
    return value if isinstance(value, dict) else {"value": value}


def anthropic_output_text(response_json: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response_json.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(part for part in parts if part).strip()


def write_summary(
    run_dir: Path,
    provider: str,
    mode: str,
    sample_path: Path,
    estimate: dict[str, Any],
    result: dict[str, Any] | None,
    actual_cost: dict[str, Any] | None,
) -> None:
    lines = [
        "# Video API Probe",
        "",
        f"- Provider: `{provider}`",
        f"- Mode: `{mode}`",
        f"- Sample: `{sample_path}`",
        f"- Sample bytes: `{sample_path.stat().st_size}`",
        f"- Estimated upper-bound cost: `${estimate['estimated_usd']:.6f}`",
    ]
    if result is None:
        lines.append("- API call made: `false`")
    else:
        lines.append("- API call made: `true`")
        lines.append(f"- Accepted: `{str(result.get('ok')).lower()}`")
        if result.get("status_code") is not None:
            lines.append(f"- HTTP status: `{result['status_code']}`")
        if actual_cost:
            lines.append(f"- Actual estimated billed cost: `${actual_cost['estimated_usd']:.6f}`")
        if result.get("error"):
            lines.extend(["", "## Error", "", "```json", json.dumps(result["error"], indent=2), "```"])
        elif result.get("upload_response") and not result.get("ok"):
            lines.extend(["", "## Upload Response", "", "```json", json.dumps(result["upload_response"], indent=2), "```"])
        elif result.get("output_text"):
            lines.extend(["", "## Output Text", "", str(result["output_text"])])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = build_sample_video(args)
    provider = args.provider
    mode = args.openai_mode if provider == "openai" else args.anthropic_mode
    pricing = ProbePricing(
        input_usd_per_million=args.input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,
    )
    estimate = estimate_probe_cost(
        sample_path=sample_path,
        prompt=args.prompt,
        max_output_tokens=args.max_output_tokens,
        pricing=pricing,
    )
    run_id = (
        f"{safe_name(sample_path.stem)}."
        f"{provider}.{safe_name(mode)}.{safe_name(args.model)}."
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "mode": mode,
        "model": args.model,
        "source_video": str(args.video.expanduser().resolve()),
        "sample_video": str(sample_path),
        "sample_sha256": sha256_file(sample_path),
        "sample_bytes": sample_path.stat().st_size,
        "sample_probe": ffprobe_summary(sample_path),
        "prompt_sha256": sha256_text(args.prompt),
        "prompt": args.prompt,
        "estimate": estimate,
    }
    write_json(run_dir / "request-manifest.json", manifest)

    print(f"Video probe sample: {sample_path}")
    print(f"Sample bytes: {sample_path.stat().st_size}")
    print(f"Provider/mode/model: {provider} / {mode} / {args.model}")
    print(f"Estimated upper-bound cost: ${estimate['estimated_usd']:.6f}")

    if args.dry_run:
        print("Dry run only. No API call made.")
        write_summary(run_dir, provider, mode, sample_path, estimate, None, None)
        return {"run_dir": str(run_dir), "api_call_made": False, "estimate": estimate}

    if args.approve_api_cost is None:
        raise RuntimeError(
            "API calls require explicit approval. "
            f"Re-run with --approve-api-cost {estimate['estimated_usd']:.6f} or a higher USD cap."
        )
    if args.approve_api_cost < estimate["estimated_usd"]:
        raise RuntimeError(
            f"Approved cap ${args.approve_api_cost:.6f} is below estimate "
            f"${estimate['estimated_usd']:.6f}."
        )
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI probes.")
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic probes.")

    print(
        f"Approved API cost cap: ${args.approve_api_cost:.6f}; "
        f"estimated cost: ${estimate['estimated_usd']:.6f}"
    )
    if provider == "openai":
        result = run_openai_probe(args, sample_path)
        actual_cost = actual_openai_cost(result.get("usage") or {}, pricing)
    elif provider == "anthropic":
        result = run_anthropic_probe(args, sample_path)
        actual_cost = actual_anthropic_cost(result.get("usage") or {}, pricing)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    write_json(run_dir / "provider-result.json", result)
    write_json(
        run_dir / "usage-and-cost.json",
        {
            "api_call_made": True,
            "preflight_estimate": estimate,
            "usage": result.get("usage") or {},
            "cost": actual_cost,
        },
    )
    write_summary(run_dir, provider, mode, sample_path, estimate, result, actual_cost)
    print(f"Accepted: {result.get('ok')}")
    print(f"Actual estimated billed cost from usage: ${actual_cost['estimated_usd']:.6f}")
    print(f"Wrote {run_dir / 'provider-result.json'}")
    print(f"Wrote {run_dir / 'summary.md'}")
    return {
        "run_dir": str(run_dir),
        "api_call_made": True,
        "estimate": estimate,
        "result": result,
        "cost": actual_cost,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether provider APIs accept MP4 files as model inputs."
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--approve-api-cost",
        type=float,
        default=None,
        help="Explicitly approve this API call up to the preflight estimate.",
    )
    parser.add_argument("--use-original", action="store_true", help="Upload the original video instead of a small local sample.")
    parser.add_argument("--sample-start", type=float, default=None)
    parser.add_argument("--sample-seconds", type=float, default=None)
    parser.add_argument("--sample-height", type=int, default=None)
    parser.add_argument("--sample-fps", type=int, default=None)
    parser.add_argument("--sample-crf", type=int, default=None)
    parser.add_argument("--sample-audio-kbps", type=int, default=None)
    parser.add_argument("--prompt", default=PROBE_PROMPT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--input-usd-per-million", type=float, default=None)
    parser.add_argument("--output-usd-per-million", type=float, default=None)
    parser.add_argument(
        "--openai-mode",
        choices=["direct-input-file", "files-input-file"],
        default=None,
    )
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument(
        "--anthropic-mode",
        choices=["files-document", "base64-document"],
        default=None,
    )
    parser.add_argument("--anthropic-base-url", default="https://api.anthropic.com")
    parser.add_argument("--anthropic-version", default="2023-06-01")
    parser.add_argument("--anthropic-files-beta", default="files-api-2025-04-14")
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument(
        "--keep-uploaded-files",
        action="store_false",
        dest="cleanup_uploaded_files",
        help="Leave provider-side uploaded files in place after the probe.",
    )
    parser.set_defaults(cleanup_uploaded_files=True)
    args = parser.parse_args(argv)

    if args.env_file.exists():
        load_env_file(args.env_file)

    args.output_dir = Path(env_value("VIDEO_PROBE_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR) if args.output_dir is None else args.output_dir
    args.sample_start = env_float("VIDEO_PROBE_SAMPLE_START_SECONDS", DEFAULT_SAMPLE_START_SECONDS) if args.sample_start is None else args.sample_start
    args.sample_seconds = env_float("VIDEO_PROBE_SAMPLE_SECONDS", DEFAULT_SAMPLE_SECONDS) if args.sample_seconds is None else args.sample_seconds
    args.sample_height = env_int("VIDEO_PROBE_SAMPLE_HEIGHT", DEFAULT_SAMPLE_HEIGHT) if args.sample_height is None else args.sample_height
    args.sample_fps = env_int("VIDEO_PROBE_SAMPLE_FPS", DEFAULT_SAMPLE_FPS) if args.sample_fps is None else args.sample_fps
    args.sample_crf = env_int("VIDEO_PROBE_SAMPLE_CRF", DEFAULT_SAMPLE_CRF) if args.sample_crf is None else args.sample_crf
    args.sample_audio_kbps = env_int("VIDEO_PROBE_SAMPLE_AUDIO_KBPS", DEFAULT_SAMPLE_AUDIO_KBPS) if args.sample_audio_kbps is None else args.sample_audio_kbps
    args.max_output_tokens = env_int("VIDEO_PROBE_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS) if args.max_output_tokens is None else args.max_output_tokens

    if args.provider == "openai":
        if args.model is None:
            args.model = env_value("VIDEO_PROBE_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        if args.openai_mode is None:
            args.openai_mode = env_value("VIDEO_PROBE_OPENAI_MODE") or "direct-input-file"
        if args.reasoning_effort is None:
            args.reasoning_effort = env_value("VIDEO_PROBE_OPENAI_REASONING_EFFORT") or DEFAULT_OPENAI_REASONING_EFFORT
        args.input_usd_per_million = env_float("VIDEO_PROBE_OPENAI_INPUT_USD_PER_MILLION", DEFAULT_OPENAI_INPUT_USD_PER_MILLION) if args.input_usd_per_million is None else args.input_usd_per_million
        args.output_usd_per_million = env_float("VIDEO_PROBE_OPENAI_OUTPUT_USD_PER_MILLION", DEFAULT_OPENAI_OUTPUT_USD_PER_MILLION) if args.output_usd_per_million is None else args.output_usd_per_million
    else:
        if args.model is None:
            args.model = env_value("VIDEO_PROBE_ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
        if args.anthropic_mode is None:
            args.anthropic_mode = env_value("VIDEO_PROBE_ANTHROPIC_MODE") or "files-document"
        args.input_usd_per_million = env_float("VIDEO_PROBE_ANTHROPIC_INPUT_USD_PER_MILLION", DEFAULT_ANTHROPIC_INPUT_USD_PER_MILLION) if args.input_usd_per_million is None else args.input_usd_per_million
        args.output_usd_per_million = env_float("VIDEO_PROBE_ANTHROPIC_OUTPUT_USD_PER_MILLION", DEFAULT_ANTHROPIC_OUTPUT_USD_PER_MILLION) if args.output_usd_per_million is None else args.output_usd_per_million

    if args.sample_seconds <= 0:
        raise ValueError("--sample-seconds must be positive.")
    if args.sample_height <= 0:
        raise ValueError("--sample-height must be positive.")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive.")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run_probe(args)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
