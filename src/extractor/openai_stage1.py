from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor import stage_cache
from extractor.preprocess_video import format_time
from extractor.prompt_templates import load_prompt_template, prompt_template_path, render_prompt_template
from extractor.transcribe import env_float, env_int, env_value, load_env_file


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_ENV_FILE = ".env"
DEFAULT_OUTPUT_DIR = Path("openai-runs")
DEFAULT_CACHE_SUBDIR = "_response-cache"
DEFAULT_MAX_OUTPUT_TOKENS = 12_000
DEFAULT_IMAGE_DETAIL = "high"
DEFAULT_IMAGE_TOKEN_ALLOWANCE = 50_000
CONTENT_MODE_CHOICES = ("ad", "excerpt")

DEFAULT_GPT55_INPUT_USD_PER_MILLION = 5.00
DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION = 0.50
DEFAULT_GPT55_OUTPUT_USD_PER_MILLION = 30.00


@dataclass(frozen=True)
class Pricing:
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float


STAGE1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "content_mode",
        "video_title",
        "core_idea",
        "why_watch",
        "viewer_promise",
        "target_viewers",
        "source_quality",
        "visual_inventory",
        "ad_angles",
        "avoid_or_review",
        "notes_for_next_stage",
    ],
    "properties": {
        "content_mode": {"type": "string", "enum": ["ad", "excerpt"]},
        "video_title": {"type": "string"},
        "core_idea": {"type": "string"},
        "why_watch": {"type": "string"},
        "viewer_promise": {"type": "string"},
        "target_viewers": {"type": "array", "items": {"type": "string"}},
        "source_quality": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "transcript_risks", "visual_risks"],
            "properties": {
                "summary": {"type": "string"},
                "transcript_risks": {"type": "array", "items": {"type": "string"}},
                "visual_risks": {"type": "array", "items": {"type": "string"}},
            },
        },
        "visual_inventory": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "evidence", "timestamps", "clip_value"],
                "properties": {
                    "label": {"type": "string"},
                    "evidence": {"type": "string"},
                    "timestamps": {"type": "array", "items": {"type": "number"}},
                    "clip_value": {"type": "string"},
                },
            },
        },
        "ad_angles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "rank",
                    "angle",
                    "viewer_hook",
                    "why_it_sells_full_video",
                    "evidence_ranges",
                    "visual_requirements",
                    "recommended_clip_form",
                    "risks",
                    "confidence",
                ],
                "properties": {
                    "rank": {"type": "integer"},
                    "angle": {"type": "string"},
                    "viewer_hook": {"type": "string"},
                    "why_it_sells_full_video": {"type": "string"},
                    "evidence_ranges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["start", "end", "reason"],
                            "properties": {
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                    "visual_requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "recommended_clip_form": {"type": "string"},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
            },
        },
        "avoid_or_review": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end", "reason"],
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
        "notes_for_next_stage": {"type": "array", "items": {"type": "string"}},
    },
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def estimate_text_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def cost_from_counts(
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    pricing: Pricing,
) -> float:
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    return (
        uncached_input_tokens / 1_000_000 * pricing.input_usd_per_million
        + cached_input_tokens / 1_000_000 * pricing.cached_input_usd_per_million
        + output_tokens / 1_000_000 * pricing.output_usd_per_million
    )


def usage_dict(response_dict: dict[str, Any]) -> dict[str, Any]:
    usage = response_dict.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    return usage


def estimated_request_cost(
    source_materials: str,
    task_prompt: str,
    include_image: bool,
    max_output_tokens: int,
    image_token_allowance: int,
    pricing: Pricing,
) -> dict[str, Any]:
    text_tokens = estimate_text_tokens(source_materials) + estimate_text_tokens(task_prompt)
    image_tokens = image_token_allowance if include_image else 0
    input_tokens = text_tokens + image_tokens
    return {
        "method": "rough upper-bound estimate using chars/4 for text, configured image allowance, and max_output_tokens",
        "text_tokens_estimate": text_tokens,
        "image_tokens_allowance": image_tokens,
        "input_tokens_estimate": input_tokens,
        "max_output_tokens": max_output_tokens,
        "estimated_usd": round(
            cost_from_counts(
                input_tokens=input_tokens,
                cached_input_tokens=0,
                output_tokens=max_output_tokens,
                pricing=pricing,
            ),
            6,
        ),
    }


def actual_cost(usage: dict[str, Any], pricing: Pricing) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens") or 0)
    input_details = usage.get("input_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "estimated_usd": round(
            cost_from_counts(
                input_tokens=input_tokens,
                cached_input_tokens=cached_tokens,
                output_tokens=output_tokens,
                pricing=pricing,
            ),
            6,
        ),
    }


def compact_transcript(transcript: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": transcript.get("provider"),
        "model": transcript.get("model"),
        "language": transcript.get("language"),
        "source": transcript.get("source"),
        "segments": [
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
                "avg_logprob": segment.get("avg_logprob"),
                "compression_ratio": segment.get("compression_ratio"),
                "no_speech_prob": segment.get("no_speech_prob"),
            }
            for segment in transcript.get("segments", [])
        ],
    }


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    visual = manifest.get("visual") or {}
    audio = manifest.get("audio") or {}
    transcript = manifest.get("transcript") or {}
    qa = transcript.get("qa") or {}
    return {
        "video": manifest.get("video"),
        "media": manifest.get("media"),
        "visual": {
            "frame_interval_seconds": visual.get("frame_interval_seconds"),
            "sampled_frame_count": len(visual.get("sampled_frames") or []),
            "sampled_frames": [
                {
                    "index": frame.get("index"),
                    "timestamp": frame.get("timestamp"),
                    "display_time": frame.get("display_time"),
                }
                for frame in visual.get("sampled_frames", [])
            ],
            "scene_changes": visual.get("scene_changes"),
        },
        "audio": {
            "silence_summary": audio.get("silence_summary"),
            "silences": audio.get("silences"),
        },
        "transcript": {
            "path": transcript.get("path"),
            "segment_count": transcript.get("segment_count"),
            "word_count": transcript.get("word_count"),
            "qa": {
                "score": qa.get("score"),
                "finding_counts": qa.get("finding_counts"),
                "top_findings": qa.get("top_findings"),
            },
        },
    }


def build_source_materials(
    manifest_path: Path,
    transcript_path: Path,
    contact_sheet_path: Path | None,
) -> str:
    manifest = compact_manifest(read_json(manifest_path))
    transcript = compact_transcript(read_json(transcript_path))
    contact_sheet_note = "No contact sheet provided."
    if contact_sheet_path is not None:
        contact_sheet_note = (
            "A contact sheet image is attached. It samples frames at the manifest cadence "
            "and is arranged left-to-right, top-to-bottom. Treat timestamps as approximate "
            "visual evidence, not exact shot boundaries."
        )
    return render_prompt_template(
        "source_materials.md",
        contact_sheet_note=contact_sheet_note,
        manifest_json=json.dumps(manifest, ensure_ascii=False, indent=2),
        transcript_json=json.dumps(transcript, ensure_ascii=False, indent=2),
    )


def build_task_prompt(content_mode: str) -> str:
    return render_prompt_template(f"stage1_{content_mode}.md")


DEVELOPER_PROMPT_TEMPLATE = "stage1_developer.md"


def developer_instructions() -> str:
    return load_prompt_template(DEVELOPER_PROMPT_TEMPLATE)


def response_text_config() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "stage1_video_brief",
            "schema": STAGE1_SCHEMA,
            "strict": False,
        },
        "verbosity": "medium",
    }


def request_fingerprint_payload(
    args: argparse.Namespace,
    source_materials: str,
    task_prompt: str,
    contact_sheet_hash: str | None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "sha256": sha256_text(source_materials),
        }
    ]
    if contact_sheet_hash is not None:
        content.append(
            {
                "type": "input_image",
                "sha256": contact_sheet_hash,
                "detail": args.image_detail,
            }
        )
    return {
        "cache_schema_version": 1,
        "api": "openai.responses.create",
        "model": args.model,
        "input": [
            {
                "role": "developer",
                "content": developer_instructions(),
            },
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "sha256": sha256_text(task_prompt),
                    }
                ],
            },
        ],
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": args.max_output_tokens,
        "store": False,
        "text": response_text_config(),
    }


def request_cache_key(fingerprint_payload: dict[str, Any]) -> str:
    return stage_cache.request_cache_key(fingerprint_payload)


def cache_entry_dir(cache_dir: Path, cache_key: str) -> Path:
    return stage_cache.cache_entry_dir(cache_dir, cache_key)


def cache_entry_exists(entry_dir: Path) -> bool:
    return stage_cache.cache_entry_exists(entry_dir)


def cache_source_files() -> list[Path]:
    root = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        root / "prompt_templates.py",
        prompt_template_path("source_materials.md"),
        prompt_template_path(DEVELOPER_PROMPT_TEMPLATE),
        prompt_template_path("stage1_ad.md"),
        prompt_template_path("stage1_excerpt.md"),
        root / "stage_cache.py",
        root / "preprocess_video.py",
        root / "transcribe.py",
    ]


def cache_manifest_payload(
    cache_key: str,
    fingerprint_payload: dict[str, Any],
    source: str,
    cache_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return stage_cache.cache_manifest_payload(
        cache_key=cache_key,
        query_fingerprint_payload=fingerprint_payload,
        source=source,
        cache_identity=cache_identity,
        created_at=datetime.now(UTC).isoformat(),
    )


def write_cache_entry(
    entry_dir: Path,
    cache_key: str,
    fingerprint_payload: dict[str, Any],
    response_dict: dict[str, Any],
    output_text: str,
    parsed_output: dict[str, Any],
    usage_and_cost: dict[str, Any],
    source: str,
    cache_identity: dict[str, Any] | None = None,
) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        entry_dir / "cache-manifest.json",
        cache_manifest_payload(cache_key, fingerprint_payload, source, cache_identity),
    )
    write_json(entry_dir / "raw-response.json", response_dict)
    (entry_dir / "output-text.json").write_text(output_text.rstrip() + "\n", encoding="utf-8")
    write_json(entry_dir / "stage1-output.json", parsed_output)
    write_json(entry_dir / "usage-and-cost.json", usage_and_cost)


def load_cached_response(entry_dir: Path) -> dict[str, Any]:
    response_dict = read_json(entry_dir / "raw-response.json")
    output_text_path = entry_dir / "output-text.json"
    if output_text_path.exists():
        output_text = output_text_path.read_text(encoding="utf-8")
    else:
        output_text = response_output_text(None, response_dict)
    parsed_path = entry_dir / "stage1-output.json"
    if parsed_path.exists():
        parsed_output = read_json(parsed_path)
    else:
        parsed_output = json.loads(output_text)
    cached_usage_and_cost = (
        read_json(entry_dir / "usage-and-cost.json")
        if (entry_dir / "usage-and-cost.json").exists()
        else {}
    )
    return {
        "response_dict": response_dict,
        "output_text": output_text,
        "parsed_output": parsed_output,
        "cached_usage_and_cost": cached_usage_and_cost,
    }


def write_response_artifacts(
    run_dir: Path,
    response_dict: dict[str, Any],
    output_text: str,
    parsed_output: dict[str, Any],
    usage_and_cost: dict[str, Any],
    summary_cost: dict[str, Any],
    cache_info: dict[str, Any] | None,
) -> None:
    write_json(run_dir / "raw-response.json", response_dict)
    (run_dir / "output-text.json").write_text(output_text.rstrip() + "\n", encoding="utf-8")
    write_json(run_dir / "stage1-output.json", parsed_output)
    write_json(run_dir / "usage-and-cost.json", usage_and_cost)
    write_markdown(run_dir / "summary.md", parsed_output, summary_cost, cache_info=cache_info)


def seed_cache_from_run(
    source_run_dir: Path,
    entry_dir: Path,
    cache_key: str,
    fingerprint_payload: dict[str, Any],
    pricing: Pricing,
    cache_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not source_run_dir.exists():
        raise FileNotFoundError(source_run_dir)
    raw_response_path = source_run_dir / "raw-response.json"
    if not raw_response_path.exists():
        raise RuntimeError(f"Existing run has no raw-response.json: {source_run_dir}")
    manifest_path = source_run_dir / "request-manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        source_content = fingerprint_payload["input"][1]["content"]
        expected_contact_hash = None
        if len(source_content) > 1:
            expected_contact_hash = source_content[1].get("sha256")
        expected = {
            "model": fingerprint_payload["model"],
            "reasoning_effort": fingerprint_payload["reasoning"]["effort"],
            "max_output_tokens": fingerprint_payload["max_output_tokens"],
            "image_detail": source_content[1].get("detail") if len(source_content) > 1 else None,
            "source_materials_sha256": source_content[0]["sha256"],
            "task_prompt_sha256": fingerprint_payload["input"][2]["content"][0]["sha256"],
            "contact_sheet_sha256": expected_contact_hash,
        }
        actual = {
            "model": manifest.get("model"),
            "reasoning_effort": manifest.get("reasoning_effort"),
            "max_output_tokens": manifest.get("max_output_tokens"),
            "image_detail": manifest.get("image_detail"),
            "source_materials_sha256": (manifest.get("hashes") or {}).get("source_materials_sha256"),
            "task_prompt_sha256": (manifest.get("hashes") or {}).get("task_prompt_sha256"),
            "contact_sheet_sha256": (manifest.get("hashes") or {}).get("contact_sheet_sha256"),
        }
        mismatches = [
            key
            for key, expected_value in expected.items()
            if expected_value != actual.get(key)
        ]
        if mismatches:
            mismatch_list = ", ".join(mismatches)
            raise RuntimeError(
                f"Existing run does not match the current cache fingerprint: {mismatch_list}"
            )
    response_dict = read_json(raw_response_path)
    output_text_path = source_run_dir / "output-text.json"
    output_text = (
        output_text_path.read_text(encoding="utf-8")
        if output_text_path.exists()
        else response_output_text(None, response_dict)
    )
    parsed_output = (
        read_json(source_run_dir / "stage1-output.json")
        if (source_run_dir / "stage1-output.json").exists()
        else json.loads(output_text)
    )
    usage = usage_dict(response_dict)
    cost = actual_cost(usage, pricing)
    usage_and_cost = {
        "api_call_made": True,
        "cache": {
            "hit": False,
            "seeded_from_run": str(source_run_dir),
            "key": cache_key,
        },
        "usage": usage,
        "cost": cost,
        "pricing": {
            "input_usd_per_million": pricing.input_usd_per_million,
            "cached_input_usd_per_million": pricing.cached_input_usd_per_million,
            "output_usd_per_million": pricing.output_usd_per_million,
        },
    }
    write_cache_entry(
        entry_dir=entry_dir,
        cache_key=cache_key,
        fingerprint_payload=fingerprint_payload,
        response_dict=response_dict,
        output_text=output_text,
        parsed_output=parsed_output,
        usage_and_cost=usage_and_cost,
        source=f"seeded_from_run:{source_run_dir}",
        cache_identity=cache_identity,
    )
    return usage_and_cost


def response_output_text(response: Any, response_dict: dict[str, Any]) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    parts: list[str] = []
    for item in response_dict.get("output", []) or []:
        for content in item.get("content", []) or []:
            value = content.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def write_markdown(
    path: Path,
    output: dict[str, Any],
    cost: dict[str, Any],
    cache_info: dict[str, Any] | None = None,
) -> None:
    lines = [
        f"# {output.get('video_title', 'Stage 1 Video Brief')}",
        "",
        f"- Content mode: `{output.get('content_mode', 'ad')}`",
        "",
        "## Core Idea",
        "",
        str(output.get("core_idea", "")).strip(),
        "",
        "## Why Watch",
        "",
        str(output.get("why_watch", "")).strip(),
        "",
        "## Viewer Promise",
        "",
        str(output.get("viewer_promise", "")).strip(),
        "",
        "## Ad Angles" if output.get("content_mode") != "excerpt" else "## Excerpt Candidates",
        "",
    ]
    for angle in output.get("ad_angles", []) or []:
        rank = angle.get("rank", "?")
        title = angle.get("angle", "")
        lines.append(f"{rank}. {title}")
        lines.append(f"   - Hook: {angle.get('viewer_hook', '')}")
        lines.append(f"   - Sells full video: {angle.get('why_it_sells_full_video', '')}")
        ranges = []
        for item in angle.get("evidence_ranges", []) or []:
            start = item.get("start")
            end = item.get("end")
            if isinstance(start, int | float) and isinstance(end, int | float):
                ranges.append(f"{format_time(float(start))}-{format_time(float(end))}")
        if ranges:
            lines.append(f"   - Evidence: {', '.join(ranges)}")
        lines.append("")
    if output.get("avoid_or_review"):
        lines.extend(["## Avoid Or Review", ""])
        for item in output.get("avoid_or_review", []) or []:
            start = item.get("start")
            end = item.get("end")
            if isinstance(start, int | float) and isinstance(end, int | float):
                lines.append(
                    f"- `{format_time(float(start))}-{format_time(float(end))}`: "
                    f"{item.get('reason', '')}"
                )
            else:
                lines.append(f"- {item.get('reason', '')}")
        lines.append("")
    lines.extend(
        [
            "## Cost",
            "",
        ]
    )
    if cache_info and cache_info.get("hit"):
        original_cost = cache_info.get("original_cost") or {}
        lines.extend(
            [
                "- Local response cache: `hit`",
                "- API call made for this run: `false`",
                f"- Original response estimated USD: `${original_cost.get('estimated_usd', 0):.6f}`",
            ]
        )
    elif cache_info:
        lines.extend(
            [
                "- Local response cache: `miss`",
                f"- Stored cache key: `{cache_info.get('key', '')}`",
            ]
        )
    lines.extend(
        [
            f"- Input tokens: `{cost.get('input_tokens', 0)}`",
            f"- Cached input tokens: `{cost.get('cached_input_tokens', 0)}`",
            f"- Output tokens: `{cost.get('output_tokens', 0)}`",
            f"- Estimated USD: `${cost.get('estimated_usd', 0):.6f}`",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    transcript_path = args.transcript.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    contact_sheet_path = args.contact_sheet.expanduser().resolve() if args.contact_sheet else None
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if contact_sheet_path is not None and not contact_sheet_path.exists():
        raise FileNotFoundError(contact_sheet_path)

    pricing = Pricing(
        input_usd_per_million=args.input_usd_per_million,
        cached_input_usd_per_million=args.cached_input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,
    )
    source_materials = build_source_materials(
        manifest_path=manifest_path,
        transcript_path=transcript_path,
        contact_sheet_path=contact_sheet_path,
    )
    task_prompt = build_task_prompt(args.content_mode)
    preflight_cost = estimated_request_cost(
        source_materials=source_materials,
        task_prompt=task_prompt,
        include_image=contact_sheet_path is not None,
        max_output_tokens=args.max_output_tokens,
        image_token_allowance=args.image_token_allowance,
        pricing=pricing,
    )

    print("API cost estimate:")
    print(
        f"  model: {args.model}; reasoning: {args.reasoning_effort}; "
        f"max output tokens: {args.max_output_tokens}"
    )
    print(
        f"  estimated input tokens: {preflight_cost['input_tokens_estimate']} "
        f"({preflight_cost['text_tokens_estimate']} text + "
        f"{preflight_cost['image_tokens_allowance']} image allowance)"
    )
    print(f"  estimated upper-bound cost: ${preflight_cost['estimated_usd']:.6f}")

    contact_sheet_hash = sha256_file(contact_sheet_path) if contact_sheet_path else None
    cache_material = "\n".join(
        [
            source_materials,
            args.model,
            args.reasoning_effort,
            args.content_mode,
            contact_sheet_hash or "",
        ]
    )
    prompt_cache_key = args.prompt_cache_key or f"stage1-{sha256_text(cache_material)[:48]}"
    fingerprint_payload = request_fingerprint_payload(
        args=args,
        source_materials=source_materials,
        task_prompt=task_prompt,
        contact_sheet_hash=contact_sheet_hash,
    )
    cache_mode = "off" if args.no_local_cache else args.cache_mode
    cache_identity = stage_cache.build_cache_identity(
        query_fingerprint_payload=fingerprint_payload,
        source_files=cache_source_files(),
        cache_mode=cache_mode,
    )
    local_cache_key = cache_identity["cache_key"]
    cache_dir = None if cache_identity["mode"] == "off" else args.cache_dir.expanduser().resolve()
    entry_dir = cache_entry_dir(cache_dir, local_cache_key) if cache_dir else None
    cache_hit = bool(entry_dir and cache_entry_exists(entry_dir) and not args.refresh_cache)
    if cache_dir:
        print(f"  local response cache mode: {cache_identity['mode']}")
        print(f"  local response cache key: {local_cache_key}")
        print(f"  query hash: {cache_identity['query_hash']}")
        print(f"  implementation hash: {cache_identity['implementation_hash']}")
        print(f"  local response cache: {'hit' if cache_hit else 'miss'}")
    else:
        print(f"  local response cache: disabled ({cache_identity['mode']})")

    if args.dry_run:
        return {
            "preflight_cost": preflight_cost,
            "source_materials_sha256": sha256_text(source_materials),
            "task_prompt_sha256": sha256_text(task_prompt),
            "local_response_cache": {
                **stage_cache.response_cache_status_payload(
                    enabled=cache_dir is not None,
                    hit=cache_hit,
                    cache_identity=cache_identity,
                    entry_dir=entry_dir,
                ),
            },
        }
    if args.seed_cache_from_run:
        if entry_dir is None:
            raise RuntimeError("--seed-cache-from-run requires local response caching to be enabled.")
        seeded = seed_cache_from_run(
            source_run_dir=args.seed_cache_from_run.expanduser().resolve(),
            entry_dir=entry_dir,
            cache_key=local_cache_key,
            fingerprint_payload=fingerprint_payload,
            pricing=pricing,
            cache_identity=cache_identity,
        )
        print(f"Seeded local response cache from {args.seed_cache_from_run}")
        print(f"  cache entry: {entry_dir}")
        print(f"  original estimated API cost: ${seeded['cost']['estimated_usd']:.6f}")
        return {
            "local_response_cache": {
                "enabled": True,
                "seeded": True,
                "key": local_cache_key,
                "entry_dir": str(entry_dir),
                "mode": cache_identity["mode"],
                "query_hash": cache_identity["query_hash"],
                "implementation_hash": cache_identity["implementation_hash"],
            },
            "usage_and_cost": seeded,
        }

    if not cache_hit:
        if args.approve_api_cost is None:
            raise RuntimeError(
                "API calls require explicit approval. Re-run with "
                f"--approve-api-cost {preflight_cost['estimated_usd']:.4f} or a higher USD cap."
            )
        if preflight_cost["estimated_usd"] > args.approve_api_cost:
            raise RuntimeError(
                f"Estimated API cost ${preflight_cost['estimated_usd']:.6f} exceeds approved cap "
                f"${args.approve_api_cost:.6f}."
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or (
        f"{transcript_path.stem}.stage1.{args.model}.{args.reasoning_effort}."
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(
        run_dir / "request-manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "image_detail": args.image_detail if contact_sheet_path else None,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": args.prompt_cache_retention,
            "paths": {
                "transcript": str(transcript_path),
                "manifest": str(manifest_path),
                "contact_sheet": str(contact_sheet_path) if contact_sheet_path else None,
            },
            "hashes": {
                "source_materials_sha256": sha256_text(source_materials),
                "task_prompt_sha256": sha256_text(task_prompt),
                "contact_sheet_sha256": contact_sheet_hash,
            },
            "preflight_cost": preflight_cost,
            "pricing": {
                "input_usd_per_million": pricing.input_usd_per_million,
                "cached_input_usd_per_million": pricing.cached_input_usd_per_million,
                "output_usd_per_million": pricing.output_usd_per_million,
            },
            "local_response_cache": {
                "enabled": cache_dir is not None,
                "hit": cache_hit,
                "refresh_requested": args.refresh_cache,
                "mode": cache_identity["mode"],
                "cache_key": local_cache_key if cache_dir else None,
                "entry_dir": str(entry_dir) if entry_dir else None,
                "query_hash": cache_identity["query_hash"],
                "implementation_hash": cache_identity["implementation_hash"],
                "source_files": cache_identity["source_files"],
                "query_fingerprint_payload": fingerprint_payload if cache_dir else None,
                "cache_identity": cache_identity if cache_dir else None,
            },
        },
    )
    (run_dir / "source-materials.txt").write_text(source_materials, encoding="utf-8")
    (run_dir / "task-prompt.txt").write_text(task_prompt, encoding="utf-8")

    if cache_hit:
        assert entry_dir is not None
        cached = load_cached_response(entry_dir)
        usage = usage_dict(cached["response_dict"])
        original_cost = actual_cost(usage, pricing)
        zero_cost = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "estimated_usd": 0.0,
        }
        usage_and_cost = {
            "api_call_made": False,
            "cache": {
                "hit": True,
                "key": local_cache_key,
                "entry_dir": str(entry_dir),
                "mode": cache_identity["mode"],
                "query_hash": cache_identity["query_hash"],
                "implementation_hash": cache_identity["implementation_hash"],
            },
            "usage": {},
            "cost": zero_cost,
            "original_api_usage": usage,
            "original_api_cost": original_cost,
            "pricing": {
                "input_usd_per_million": pricing.input_usd_per_million,
                "cached_input_usd_per_million": pricing.cached_input_usd_per_million,
                "output_usd_per_million": pricing.output_usd_per_million,
            },
        }
        cache_info = {
            "hit": True,
            "key": local_cache_key,
            "entry_dir": str(entry_dir),
            "mode": cache_identity["mode"],
            "query_hash": cache_identity["query_hash"],
            "implementation_hash": cache_identity["implementation_hash"],
            "original_cost": original_cost,
        }
        write_response_artifacts(
            run_dir=run_dir,
            response_dict=cached["response_dict"],
            output_text=cached["output_text"],
            parsed_output=cached["parsed_output"],
            usage_and_cost=usage_and_cost,
            summary_cost=zero_cost,
            cache_info=cache_info,
        )
        print(f"Local response cache hit: {entry_dir}")
        print("No OpenAI API call made.")
        print(f"Wrote {run_dir / 'stage1-output.json'}")
        print(f"Wrote {run_dir / 'summary.md'}")
        return {
            "run_dir": str(run_dir),
            "output": cached["parsed_output"],
            "cost": zero_cost,
            "usage": {},
            "local_response_cache": cache_info,
        }

    from openai import OpenAI

    client = OpenAI(timeout=env_float("OPENAI_CLIENT_TIMEOUT_SECONDS", 1800.0))
    content: list[dict[str, Any]] = [{"type": "input_text", "text": source_materials}]
    if contact_sheet_path:
        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url(contact_sheet_path),
                "detail": args.image_detail,
            }
        )
    input_messages: list[dict[str, Any]] = [
        {
            "role": "developer",
            "content": developer_instructions(),
        },
        {"role": "user", "content": content},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": task_prompt}],
        },
    ]

    print(
        f"Approved API cost cap: ${args.approve_api_cost:.6f}; "
        f"estimated cost: ${preflight_cost['estimated_usd']:.6f}"
    )
    print(f"Writing run artifacts to {run_dir}")
    response = client.responses.create(
        model=args.model,
        input=input_messages,
        reasoning={"effort": args.reasoning_effort},
        max_output_tokens=args.max_output_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=args.prompt_cache_retention,
        store=False,
        text=response_text_config(),
    )

    response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
    output_text = response_output_text(response, response_dict)
    try:
        parsed_output = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model output was not valid JSON. Raw response saved to {run_dir}") from exc

    usage = usage_dict(response_dict)
    cost = actual_cost(usage, pricing)
    usage_and_cost = {
        "api_call_made": True,
        "cache": {
            "hit": False,
            "stored": entry_dir is not None,
            "key": local_cache_key if entry_dir else None,
            "entry_dir": str(entry_dir) if entry_dir else None,
            "mode": cache_identity["mode"],
            "query_hash": cache_identity["query_hash"],
            "implementation_hash": cache_identity["implementation_hash"],
        },
        "usage": usage,
        "cost": cost,
        "pricing": {
            "input_usd_per_million": pricing.input_usd_per_million,
            "cached_input_usd_per_million": pricing.cached_input_usd_per_million,
            "output_usd_per_million": pricing.output_usd_per_million,
        },
    }
    cache_info = {
        "hit": False,
        "key": local_cache_key if entry_dir else None,
        "entry_dir": str(entry_dir) if entry_dir else None,
        "mode": cache_identity["mode"],
        "query_hash": cache_identity["query_hash"],
        "implementation_hash": cache_identity["implementation_hash"],
    }
    write_response_artifacts(
        run_dir=run_dir,
        response_dict=response_dict,
        output_text=output_text,
        parsed_output=parsed_output,
        usage_and_cost=usage_and_cost,
        summary_cost=cost,
        cache_info=cache_info if entry_dir else None,
    )
    if entry_dir is not None:
        write_cache_entry(
            entry_dir=entry_dir,
            cache_key=local_cache_key,
            fingerprint_payload=fingerprint_payload,
            response_dict=response_dict,
            output_text=output_text,
            parsed_output=parsed_output,
            usage_and_cost=usage_and_cost,
            source=f"api_run:{run_dir}",
            cache_identity=cache_identity,
        )
        print(f"Stored local response cache entry: {entry_dir}")
    print(
        f"Actual token usage: input={cost['input_tokens']}, "
        f"cached_input={cost['cached_input_tokens']}, output={cost['output_tokens']}"
    )
    print(f"Estimated billed cost from usage: ${cost['estimated_usd']:.6f}")
    print(f"Wrote {run_dir / 'stage1-output.json'}")
    print(f"Wrote {run_dir / 'summary.md'}")
    return {
        "run_dir": str(run_dir),
        "output": parsed_output,
        "cost": cost,
        "usage": usage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenAI stage 1: full-video brief and clip-ad angle hypotheses."
    )
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--content-mode", choices=CONTENT_MODE_CHOICES, default="ad")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--image-detail", choices=["low", "high", "auto", "original"], default=None)
    parser.add_argument("--image-token-allowance", type=int, default=None)
    parser.add_argument("--prompt-cache-key", default=None)
    parser.add_argument("--prompt-cache-retention", choices=["in_memory", "24h"], default="24h")
    parser.add_argument(
        "--cache-mode",
        choices=stage_cache.CACHE_MODE_CHOICES,
        default=None,
        help=(
            "Local response cache identity mode. strict includes LLM query and "
            "stage source-file hashes; query-only ignores source-file hash drift; "
            "off disables the local response cache."
        ),
    )
    parser.add_argument("--no-local-cache", action="store_true")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass any local cached response and make a fresh approved API call.",
    )
    parser.add_argument(
        "--seed-cache-from-run",
        type=Path,
        default=None,
        help="Populate the local response cache from an existing run directory, then exit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--approve-api-cost",
        type=float,
        default=None,
        metavar="USD",
        help="Explicitly approve this OpenAI API call up to the preflight estimate.",
    )
    parser.add_argument("--input-usd-per-million", type=float, default=None)
    parser.add_argument("--cached-input-usd-per-million", type=float, default=None)
    parser.add_argument("--output-usd-per-million", type=float, default=None)
    return parser


def apply_env_defaults(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = Path(env_value("STAGE1_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    if args.cache_dir is None:
        args.cache_dir = Path(
            env_value("STAGE1_CACHE_DIR")
            or env_value("OPENAI_RESPONSE_CACHE_DIR")
            or (args.output_dir / DEFAULT_CACHE_SUBDIR)
        )
    if args.model is None:
        args.model = env_value("STAGE1_MODEL") or DEFAULT_MODEL
    if args.reasoning_effort is None:
        args.reasoning_effort = env_value("STAGE1_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT
    if args.max_output_tokens is None:
        args.max_output_tokens = env_int("STAGE1_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
    if args.image_detail is None:
        args.image_detail = env_value("STAGE1_IMAGE_DETAIL") or DEFAULT_IMAGE_DETAIL
    if args.image_token_allowance is None:
        args.image_token_allowance = env_int(
            "STAGE1_IMAGE_TOKEN_ALLOWANCE",
            DEFAULT_IMAGE_TOKEN_ALLOWANCE,
        )
    if args.input_usd_per_million is None:
        args.input_usd_per_million = env_float(
            "STAGE1_INPUT_USD_PER_MILLION",
            DEFAULT_GPT55_INPUT_USD_PER_MILLION,
        )
    if args.cached_input_usd_per_million is None:
        args.cached_input_usd_per_million = env_float(
            "STAGE1_CACHED_INPUT_USD_PER_MILLION",
            DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION,
        )
    if args.output_usd_per_million is None:
        args.output_usd_per_million = env_float(
            "STAGE1_OUTPUT_USD_PER_MILLION",
            DEFAULT_GPT55_OUTPUT_USD_PER_MILLION,
        )
    if args.cache_mode is None:
        args.cache_mode = (
            env_value("STAGE1_CACHE_MODE")
            or env_value("OPENAI_RESPONSE_CACHE_MODE")
            or stage_cache.DEFAULT_CACHE_MODE
        )
    args.cache_mode = stage_cache.normalize_cache_mode(args.cache_mode)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        load_env_file(args.env_file)
        apply_env_defaults(args)
        run_stage1(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
