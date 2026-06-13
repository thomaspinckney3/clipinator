from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor import stage_cache
from extractor.openai_strategy import (
    CONTENT_MODE_CHOICES,
    DEFAULT_CACHE_SUBDIR,
    DEFAULT_ENV_FILE,
    DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION,
    DEFAULT_GPT55_INPUT_USD_PER_MILLION,
    DEFAULT_GPT55_OUTPUT_USD_PER_MILLION,
    DEFAULT_IMAGE_DETAIL,
    DEFAULT_IMAGE_TOKEN_ALLOWANCE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REASONING_EFFORT,
    Pricing,
    actual_cost,
    build_source_materials,
    cache_entry_dir,
    cache_entry_exists,
    cost_from_counts,
    estimate_text_tokens,
    image_data_url,
    read_json,
    response_output_text,
    sha256_file,
    sha256_text,
    stable_json,
    usage_dict,
    write_json,
)
from extractor.preprocess_video import format_time
from extractor.prompt_templates import load_prompt_template, prompt_template_path, render_prompt_template
from extractor.openai_responses import create_response_streaming
from extractor.transcribe import env_float, env_int, env_value, load_env_file


DEFAULT_MAX_OUTPUT_TOKENS = 18_000
DEFAULT_CLIP_COUNT = 3
DEFAULT_TARGET_ASPECT_RATIO = "9:16"
DEFAULT_TARGET_SECONDS = 45
DEFAULT_VISUAL_INDEX_MAX_IMAGES = 24


def cache_source_files() -> list[Path]:
    root = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        root / "prompt_templates.py",
        prompt_template_path(DEVELOPER_PROMPT_TEMPLATE),
        prompt_template_path("plan_ad.md"),
        prompt_template_path("plan_excerpt.md"),
        root / "stage_cache.py",
        root / "openai_strategy.py",
        root / "openai_moments.py",
        root / "preprocess_video.py",
        root / "transcribe.py",
    ]


DEVELOPER_PROMPT_TEMPLATE = "plan_developer.md"


def developer_instructions() -> str:
    return load_prompt_template(DEVELOPER_PROMPT_TEMPLATE)


MEDIA_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "request_type",
        "desired_asset_type",
        "purpose",
        "placement_notes",
        "license_requirements",
        "rights_risk",
    ],
    "properties": {
        "id": {"type": "string"},
        "request_type": {
            "type": "string",
            "enum": [
                "third_party_media",
                "generated_video",
                "generated_image",
                "generated_graphic",
                "generated_audio",
            ],
        },
        "desired_asset_type": {
            "type": "string",
            "enum": ["video", "image", "graphic", "audio"],
        },
        "purpose": {"type": "string"},
        "placement_notes": {"type": "string"},
        "search_query": {"type": "string"},
        "suggested_sources": {"type": "array", "items": {"type": "string"}},
        "generation_prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "duration_seconds": {"type": "number"},
        "aspect_ratio": {"type": "string"},
        "style_notes": {"type": "string"},
        "license_requirements": {"type": "string"},
        "rights_risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "review_notes": {"type": "array", "items": {"type": "string"}},
    },
}


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "content_mode",
        "video_title",
        "plan_summary",
        "global_editing_rules",
        "clip_ads",
        "media_requests",
        "manual_review_queue",
        "deferred_candidates",
        "notes_for_rendering",
    ],
    "properties": {
        "content_mode": {"type": "string", "enum": ["ad", "excerpt"]},
        "video_title": {"type": "string"},
        "plan_summary": {"type": "string"},
        "global_editing_rules": {"type": "array", "items": {"type": "string"}},
        "clip_ads": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "clip_id",
                    "source_candidate_id",
                    "concept_title",
                    "target_platform",
                    "target_aspect_ratio",
                    "target_duration_seconds",
                    "editorial_objective",
                    "why_this_ad_sells_the_full_video",
                    "source_ranges",
                    "opening_hook",
                    "timeline",
                    "caption_plan",
                    "audio_plan",
                    "visual_plan",
                    "media_request_ids",
                    "verification_steps",
                    "risk_notes",
                    "success_criteria",
                ],
                "properties": {
                    "clip_id": {"type": "string"},
                    "source_candidate_id": {"type": "string"},
                    "concept_title": {"type": "string"},
                    "target_platform": {"type": "string"},
                    "target_aspect_ratio": {"type": "string"},
                    "target_duration_seconds": {"type": "number"},
                    "editorial_objective": {"type": "string"},
                    "why_this_ad_sells_the_full_video": {"type": "string"},
                    "excerpt_type": {"type": "string"},
                    "excerpt_scope": {
                        "type": "string",
                        "enum": ["atomic_excerpt", "chapter_extract", "multi_part_extract"],
                    },
                    "coherence_unit": {"type": "string"},
                    "merge_or_split_rationale": {"type": "string"},
                    "depends_on_prior_excerpt": {"type": "boolean"},
                    "self_contained_score": {"type": "number"},
                    "context_needed": {"type": "string"},
                    "source_order_preserved": {"type": "boolean"},
                    "cta_policy": {"type": "string"},
                    "source_ranges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["start", "end", "role", "reason"],
                            "properties": {
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "role": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                    "opening_hook": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "visual", "audio", "duration_seconds"],
                        "properties": {
                            "text": {"type": "string"},
                            "visual": {"type": "string"},
                            "audio": {"type": "string"},
                            "duration_seconds": {"type": "number"},
                        },
                    },
                    "timeline": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "order",
                                "screen_time_start",
                                "screen_time_end",
                                "source_start",
                                "source_end",
                                "audio_source",
                                "visual_source",
                                "edit_action",
                                "caption_or_overlay",
                                "purpose",
                            ],
                            "properties": {
                                "order": {"type": "integer"},
                                "screen_time_start": {"type": "number"},
                                "screen_time_end": {"type": "number"},
                                "source_start": {"type": "number"},
                                "source_end": {"type": "number"},
                                "audio_source": {"type": "string"},
                                "visual_source": {"type": "string"},
                                "edit_action": {"type": "string"},
                                "caption_or_overlay": {"type": "string"},
                                "purpose": {"type": "string"},
                            },
                        },
                    },
                    "caption_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["start_offset", "end_offset", "text", "style_note"],
                            "properties": {
                                "start_offset": {"type": "number"},
                                "end_offset": {"type": "number"},
                                "text": {"type": "string"},
                                "style_note": {"type": "string"},
                            },
                        },
                    },
                    "audio_plan": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["primary_audio", "sync_strategy", "music_or_roomtone", "cleanup_notes"],
                        "properties": {
                            "primary_audio": {"type": "string"},
                            "sync_strategy": {"type": "string"},
                            "music_or_roomtone": {"type": "string"},
                            "cleanup_notes": {"type": "string"},
                        },
                    },
                    "visual_plan": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["first_frame", "b_roll_strategy", "crop_strategy", "needed_visual_checks"],
                        "properties": {
                            "first_frame": {"type": "string"},
                            "b_roll_strategy": {"type": "string"},
                            "crop_strategy": {"type": "string"},
                            "needed_visual_checks": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "media_request_ids": {"type": "array", "items": {"type": "string"}},
                    "verification_steps": {"type": "array", "items": {"type": "string"}},
                    "risk_notes": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "media_requests": {"type": "array", "items": MEDIA_REQUEST_SCHEMA},
        "manual_review_queue": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end", "reason", "required_action"],
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "reason": {"type": "string"},
                    "required_action": {"type": "string"},
                },
            },
        },
        "deferred_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "reason"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "notes_for_rendering": {"type": "array", "items": {"type": "string"}},
    },
}


def response_text_config() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "stage3_edit_plans",
            "schema": PLAN_SCHEMA,
            "strict": False,
        },
        "verbosity": "medium",
    }


def compact_visual_index_manifest(manifest_path: Path, max_images: int) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    candidate_sheets = manifest.get("candidate_sheets", []) or []
    compact_sheets: list[dict[str, Any]] = []
    for sheet in candidate_sheets[:max_images]:
        compact_sheets.append(
            {
                "candidate_id": sheet.get("candidate_id"),
                "title": sheet.get("title"),
                "path": sheet.get("path"),
                "columns": sheet.get("columns"),
                "frame_count": sheet.get("frame_count"),
                "read_order": sheet.get("read_order"),
                "angle": sheet.get("angle"),
                "candidate_window": sheet.get("candidate_window"),
                "sheet_start": sheet.get("sheet_start"),
                "sheet_end": sheet.get("sheet_end"),
                "sheet_display_range": sheet.get("sheet_display_range"),
                "frames": [
                    {
                        "timestamp": frame.get("timestamp"),
                        "display_time": frame.get("display_time"),
                        "sheet_row": frame.get("sheet_row"),
                        "sheet_column": frame.get("sheet_column"),
                    }
                    for frame in sheet.get("frames", []) or []
                ],
            }
        )
    return {
        "schema_version": manifest.get("schema_version"),
        "video": manifest.get("video"),
        "frame_interval_seconds": manifest.get("frame_interval_seconds"),
        "frame_count": manifest.get("frame_count"),
        "included_candidate_sheet_count": len(compact_sheets),
        "total_candidate_sheet_count": len(candidate_sheets),
        "candidate_sheets": compact_sheets,
        "usage_note": manifest.get("usage_note"),
    }


def visual_index_root(manifest_path: Path) -> Path:
    return manifest_path.parent.parent


def visual_index_image_paths(manifest_path: Path, max_images: int) -> list[Path]:
    manifest = read_json(manifest_path)
    root = visual_index_root(manifest_path)
    paths: list[Path] = []
    for sheet in (manifest.get("candidate_sheets", []) or [])[:max_images]:
        sheet_path = sheet.get("path")
        if not sheet_path:
            continue
        path = Path(str(sheet_path))
        if not path.is_absolute():
            path = root / path
        if path.exists():
            paths.append(path)
    return paths


def compact_augmentation_assets(path: Path) -> dict[str, Any]:
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

    compact: list[dict[str, Any]] = []
    for asset in items:
        asset_id = str(asset.get("id", "")).strip()
        if not asset_id:
            continue
        compact.append(
            {
                "id": asset_id,
                "type": asset.get("type"),
                "description": asset.get("description"),
                "tags": asset.get("tags", []),
                "license": asset.get("license"),
                "license_url": asset.get("license_url"),
                "source_url": asset.get("source_url"),
                "creator": asset.get("creator"),
                "usage_notes": asset.get("usage_notes"),
                "default_volume": asset.get("volume", asset.get("default_volume")),
                "has_local_path": bool(asset.get("path")),
            }
        )
    return {
        "asset_count": len(compact),
        "usage_contract": (
            "Reference a local video/image/graphic asset in timeline.visual_source as "
            "asset://id or asset://id#t=SECONDS. Reference an audio asset in "
            "timeline.audio_source only as an explicit quiet bed or standalone silent-card "
            "audio source. Preserve source dialogue sync."
        ),
        "assets": compact,
    }


def estimated_plan_cost(
    source_materials: str,
    task_prompt: str,
    image_count: int,
    max_output_tokens: int,
    image_token_allowance: int,
    pricing: Pricing,
) -> dict[str, Any]:
    text_tokens = estimate_text_tokens(source_materials) + estimate_text_tokens(task_prompt)
    image_tokens = image_count * image_token_allowance
    input_tokens = text_tokens + image_tokens
    return {
        "method": "rough upper-bound estimate using chars/4 for text, configured per-image allowance, and max_output_tokens",
        "text_tokens_estimate": text_tokens,
        "image_count": image_count,
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


def build_task_prompt(
    strategy_output: dict[str, Any],
    moments_output: dict[str, Any],
    clip_count: int,
    target_aspect_ratio: str,
    target_seconds: int,
    content_mode: str,
) -> str:
    return render_prompt_template(
        f"plan_{content_mode}.md",
        clip_count=clip_count,
        target_seconds=target_seconds,
        target_aspect_ratio=target_aspect_ratio,
        stage1_output_json=json.dumps(strategy_output, ensure_ascii=False, indent=2),
        stage2_output_json=json.dumps(moments_output, ensure_ascii=False, indent=2),
    )


def request_fingerprint_payload(
    args: argparse.Namespace,
    source_materials: str,
    task_prompt: str,
    contact_sheet_hash: str | None,
    visual_index_image_hashes: list[str],
    strategy_output_hash: str,
    moments_output_hash: str,
    visual_index_hash: str | None,
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
    for image_hash in visual_index_image_hashes:
        content.append(
            {
                "type": "input_image",
                "sha256": image_hash,
                "detail": args.image_detail,
                "source": "candidate_visual_index_sheet",
            }
        )
    return {
        "cache_schema_version": 2,
        "stage": "plan_edits",
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
                        "strategy_output_sha256": strategy_output_hash,
                        "moments_output_sha256": moments_output_hash,
                        "visual_index_sha256": visual_index_hash,
                    }
                ],
            },
        ],
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": args.max_output_tokens,
        "store": False,
        "text": response_text_config(),
    }


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
        stage_cache.cache_manifest_payload(
            cache_key=cache_key,
            query_fingerprint_payload=fingerprint_payload,
            source=source,
            cache_identity=cache_identity,
            created_at=datetime.now(UTC).isoformat(),
        ),
    )
    write_json(entry_dir / "raw-response.json", response_dict)
    (entry_dir / "output-text.json").write_text(output_text.rstrip() + "\n", encoding="utf-8")
    write_json(entry_dir / "plan-output.json", parsed_output)
    write_json(entry_dir / "usage-and-cost.json", usage_and_cost)


def load_cached_response(entry_dir: Path) -> dict[str, Any]:
    response_dict = read_json(entry_dir / "raw-response.json")
    output_text_path = entry_dir / "output-text.json"
    output_text = (
        output_text_path.read_text(encoding="utf-8")
        if output_text_path.exists()
        else response_output_text(None, response_dict)
    )
    parsed_path = entry_dir / "plan-output.json"
    parsed_output = read_json(parsed_path) if parsed_path.exists() else json.loads(output_text)
    return {
        "response_dict": response_dict,
        "output_text": output_text,
        "parsed_output": parsed_output,
    }


def write_markdown(
    path: Path,
    output: dict[str, Any],
    cost: dict[str, Any],
    cache_info: dict[str, Any] | None = None,
) -> None:
    lines = [
        f"# {output.get('video_title', 'Stage 3 Edit Plans')}",
        "",
        f"- Content mode: `{output.get('content_mode', 'ad')}`",
        "",
        "## Summary",
        "",
        str(output.get("plan_summary", "")).strip(),
        "",
        "## Clip Ads" if output.get("content_mode") != "excerpt" else "## Excerpt Plans",
        "",
    ]
    for clip in output.get("clip_ads", []) or []:
        lines.append(f"### {clip.get('clip_id', '')}: {clip.get('concept_title', '')}")
        lines.append(f"- Candidate: `{clip.get('source_candidate_id', '')}`")
        lines.append(f"- Target: `{clip.get('target_duration_seconds', '')}s` `{clip.get('target_aspect_ratio', '')}`")
        lines.append(f"- Objective: {clip.get('editorial_objective', '')}")
        hook = clip.get("opening_hook") or {}
        lines.append(f"- Hook: {hook.get('text', '')}")
        lines.append("")
        lines.append("Timeline:")
        for item in clip.get("timeline", []) or []:
            screen_start = item.get("screen_time_start")
            screen_end = item.get("screen_time_end")
            source_start = item.get("source_start")
            source_end = item.get("source_end")
            screen = f"{screen_start}-{screen_end}s"
            source = ""
            if isinstance(source_start, int | float) and isinstance(source_end, int | float):
                source = f"`{format_time(float(source_start))}-{format_time(float(source_end))}`"
            lines.append(
                f"- `{screen}` {source}: {item.get('edit_action', '')} "
                f"({item.get('purpose', '')})"
            )
        lines.append("")
        if clip.get("verification_steps"):
            lines.append("Verification:")
            for step in clip.get("verification_steps", []) or []:
                lines.append(f"- {step}")
            lines.append("")
        if clip.get("media_request_ids"):
            lines.append(f"Media requests: `{', '.join(clip.get('media_request_ids', []))}`")
            lines.append("")
    if output.get("media_requests"):
        lines.extend(["## Media Requests", ""])
        for item in output.get("media_requests", []) or []:
            lines.append(
                f"- `{item.get('id', '')}` `{item.get('request_type', '')}` "
                f"`{item.get('desired_asset_type', '')}` risk `{item.get('rights_risk', '')}`: "
                f"{item.get('purpose', '')}"
            )
            if item.get("search_query"):
                lines.append(f"  Search: {item.get('search_query')}")
            if item.get("generation_prompt"):
                lines.append(f"  Generate: {item.get('generation_prompt')}")
        lines.append("")
    if output.get("manual_review_queue"):
        lines.extend(["## Manual Review Queue", ""])
        for item in output.get("manual_review_queue", []) or []:
            start = item.get("start")
            end = item.get("end")
            if isinstance(start, int | float) and isinstance(end, int | float):
                timing = f"`{format_time(float(start))}-{format_time(float(end))}`"
            else:
                timing = "`unknown`"
            lines.append(f"- {timing}: {item.get('reason', '')} -> {item.get('required_action', '')}")
        lines.append("")
    lines.extend(["## Cost", ""])
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
    write_json(run_dir / "plan-output.json", parsed_output)
    write_json(run_dir / "usage-and-cost.json", usage_and_cost)
    write_markdown(run_dir / "summary.md", parsed_output, summary_cost, cache_info=cache_info)


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    transcript_path = args.transcript.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    strategy_path = args.strategy_output.expanduser().resolve()
    moments_path = args.moments_output.expanduser().resolve()
    contact_sheet_path = args.contact_sheet.expanduser().resolve() if args.contact_sheet else None
    visual_index_path = args.visual_index_manifest.expanduser().resolve() if args.visual_index_manifest else None
    augmentation_assets_path = (
        args.augmentation_assets.expanduser().resolve()
        if args.augmentation_assets
        else None
    )
    for path in [transcript_path, manifest_path, strategy_path, moments_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if contact_sheet_path is not None and not contact_sheet_path.exists():
        raise FileNotFoundError(contact_sheet_path)
    if visual_index_path is not None and not visual_index_path.exists():
        raise FileNotFoundError(visual_index_path)
    if augmentation_assets_path is not None and not augmentation_assets_path.exists():
        raise FileNotFoundError(augmentation_assets_path)

    pricing = Pricing(
        input_usd_per_million=args.input_usd_per_million,
        cached_input_usd_per_million=args.cached_input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,
    )
    strategy_output = read_json(strategy_path)
    moments_output = read_json(moments_path)
    content_mode = args.content_mode or str(
        moments_output.get("content_mode") or strategy_output.get("content_mode") or "ad"
    )
    strategy_output_text = stable_json(strategy_output)
    moments_output_text = stable_json(moments_output)
    source_materials = build_source_materials(
        manifest_path=manifest_path,
        transcript_path=transcript_path,
        contact_sheet_path=contact_sheet_path,
    )
    visual_index_images: list[Path] = []
    visual_index_text = ""
    if visual_index_path is not None:
        visual_index = compact_visual_index_manifest(
            manifest_path=visual_index_path,
            max_images=args.visual_index_max_images,
        )
        visual_index_images = [] if args.no_visual_index_images else visual_index_image_paths(
            manifest_path=visual_index_path,
            max_images=args.visual_index_max_images,
        )
        visual_index_text = stable_json(visual_index)
        source_materials = "\n\n".join(
            [
                source_materials,
                "DENSE CANDIDATE VISUAL INDEX",
                (
                    "Candidate visual-index contact sheets are attached after the main contact sheet "
                    "in the same order as candidate_sheets below. Use these 1 fps sheets to verify "
                    "concrete objects/actions before creating a render plan."
                ),
                "COMPACT VISUAL INDEX JSON:",
                json.dumps(visual_index, ensure_ascii=False, indent=2),
            ]
        )
    augmentation_assets_text = ""
    if augmentation_assets_path is not None:
        augmentation_assets = compact_augmentation_assets(augmentation_assets_path)
        augmentation_assets_text = stable_json(augmentation_assets)
        source_materials = "\n\n".join(
            [
                source_materials,
                "OPTIONAL AUGMENTATION ASSET CATALOG",
                (
                    "These local, rights-reviewed assets may be used as related cutaway, "
                    "graphic, or quiet audio-bed material. Use only asset ids listed here, "
                    "and only when the usage notes and license metadata make the asset "
                    "appropriate for the excerpt."
                ),
                json.dumps(augmentation_assets, ensure_ascii=False, indent=2),
            ]
        )
    task_prompt = build_task_prompt(
        strategy_output=strategy_output,
        moments_output=moments_output,
        clip_count=args.clip_count,
        target_aspect_ratio=args.target_aspect_ratio,
        target_seconds=args.target_seconds,
        content_mode=content_mode,
    )
    image_count = (1 if contact_sheet_path is not None else 0) + len(visual_index_images)
    preflight_cost = estimated_plan_cost(
        source_materials=source_materials,
        task_prompt=task_prompt,
        image_count=image_count,
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
        f"{preflight_cost['image_tokens_allowance']} image allowance across "
        f"{preflight_cost['image_count']} images)"
    )
    print(f"  estimated upper-bound cost: ${preflight_cost['estimated_usd']:.6f}")

    contact_sheet_hash = sha256_file(contact_sheet_path) if contact_sheet_path else None
    visual_index_image_hashes = [sha256_file(path) for path in visual_index_images]
    visual_index_hash = sha256_text(visual_index_text) if visual_index_text else None
    stable_prefix_cache_material = "\n".join(
        [
            source_materials,
            args.model,
            args.reasoning_effort,
            contact_sheet_hash or "",
            *visual_index_image_hashes,
        ]
    )
    prompt_cache_key = args.prompt_cache_key or f"strategy-{sha256_text(stable_prefix_cache_material)[:48]}"
    fingerprint_payload = request_fingerprint_payload(
        args=args,
        source_materials=source_materials,
        task_prompt=task_prompt,
        contact_sheet_hash=contact_sheet_hash,
        visual_index_image_hashes=visual_index_image_hashes,
        strategy_output_hash=sha256_text(strategy_output_text),
        moments_output_hash=sha256_text(moments_output_text),
        visual_index_hash=visual_index_hash,
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
            "strategy_output_sha256": sha256_text(strategy_output_text),
            "moments_output_sha256": sha256_text(moments_output_text),
            "visual_index_sha256": visual_index_hash,
            "prompt_cache_key": prompt_cache_key,
            "local_response_cache": stage_cache.response_cache_status_payload(
                enabled=cache_dir is not None,
                hit=cache_hit,
                cache_identity=cache_identity,
                entry_dir=entry_dir,
            ),
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
        f"{transcript_path.stem}.plan.{args.model}.{args.reasoning_effort}."
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(
        run_dir / "request-manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "stage": "plan_edits",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "clip_count": args.clip_count,
            "target_aspect_ratio": args.target_aspect_ratio,
            "target_seconds": args.target_seconds,
            "image_detail": args.image_detail if image_count else None,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": args.prompt_cache_retention,
            "paths": {
                "transcript": str(transcript_path),
                "manifest": str(manifest_path),
                "contact_sheet": str(contact_sheet_path) if contact_sheet_path else None,
                "strategy_output": str(strategy_path),
                "moments_output": str(moments_path),
            "visual_index_manifest": str(visual_index_path) if visual_index_path else None,
            "visual_index_images": [str(path) for path in visual_index_images],
            "augmentation_assets": str(augmentation_assets_path) if augmentation_assets_path else None,
        },
            "hashes": {
                "source_materials_sha256": sha256_text(source_materials),
                "task_prompt_sha256": sha256_text(task_prompt),
                "strategy_output_sha256": sha256_text(strategy_output_text),
                "moments_output_sha256": sha256_text(moments_output_text),
                "contact_sheet_sha256": contact_sheet_hash,
            "visual_index_sha256": visual_index_hash,
            "visual_index_image_sha256s": visual_index_image_hashes,
            "augmentation_assets_sha256": (
                sha256_text(augmentation_assets_text) if augmentation_assets_text else None
            ),
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
        print(f"Wrote {run_dir / 'plan-output.json'}")
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
    for path in visual_index_images:
        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url(path),
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
    response = create_response_streaming(
        client,
        label="plan",
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
    usage = usage_dict(response_dict)
    cost = actual_cost(usage, pricing)
    try:
        parsed_output = json.loads(output_text)
    except json.JSONDecodeError as exc:
        write_json(run_dir / "raw-response.json", response_dict)
        (run_dir / "output-text.json").write_text(output_text.rstrip() + "\n", encoding="utf-8")
        write_json(
            run_dir / "usage-and-cost.json",
            {
                "api_call_made": True,
                "cache": {
                    "hit": False,
                    "stored": False,
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
                "parse_error": str(exc),
            },
        )
        print(
            f"Actual token usage: input={cost['input_tokens']}, "
            f"cached_input={cost['cached_input_tokens']}, output={cost['output_tokens']}"
        )
        print(f"Estimated billed cost from usage: ${cost['estimated_usd']:.6f}")
        raise RuntimeError(f"Model output was not valid JSON. Raw response saved to {run_dir}") from exc

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
    print(f"Wrote {run_dir / 'plan-output.json'}")
    print(f"Wrote {run_dir / 'summary.md'}")
    return {
        "run_dir": str(run_dir),
        "output": parsed_output,
        "cost": cost,
        "usage": usage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenAI stage 3: concrete clip-ad edit plans."
    )
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strategy-output", type=Path, required=True)
    parser.add_argument("--moments-output", type=Path, required=True)
    parser.add_argument("--content-mode", choices=CONTENT_MODE_CHOICES, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--visual-index-manifest", type=Path, default=None)
    parser.add_argument(
        "--augmentation-assets",
        type=Path,
        default=None,
        help=(
            "Optional JSON catalog of local, rights-reviewed assets that Stage 3 "
            "may reference as asset://id in visual_source/audio_source."
        ),
    )
    parser.add_argument("--visual-index-max-images", type=int, default=None)
    parser.add_argument(
        "--no-visual-index-images",
        action="store_true",
        help="Include visual-index metadata but do not attach candidate contact sheet images.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--clip-count", type=int, default=None)
    parser.add_argument("--target-aspect-ratio", default=None)
    parser.add_argument("--target-seconds", type=int, default=None)
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
        args.output_dir = Path(env_value("PLAN_OUTPUT_DIR") or env_value("STRATEGY_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    if args.cache_dir is None:
        args.cache_dir = Path(
            env_value("PLAN_CACHE_DIR")
            or env_value("MOMENTS_CACHE_DIR")
            or env_value("STRATEGY_CACHE_DIR")
            or env_value("OPENAI_RESPONSE_CACHE_DIR")
            or (args.output_dir / DEFAULT_CACHE_SUBDIR)
        )
    if args.model is None:
        args.model = env_value("PLAN_MODEL") or env_value("STRATEGY_MODEL") or DEFAULT_MODEL
    if args.reasoning_effort is None:
        args.reasoning_effort = (
            env_value("PLAN_REASONING_EFFORT")
            or env_value("STRATEGY_REASONING_EFFORT")
            or DEFAULT_REASONING_EFFORT
        )
    if args.max_output_tokens is None:
        args.max_output_tokens = env_int("PLAN_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
    if args.clip_count is None:
        args.clip_count = env_int("PLAN_CLIP_COUNT", DEFAULT_CLIP_COUNT)
    if args.target_aspect_ratio is None:
        args.target_aspect_ratio = env_value("PLAN_TARGET_ASPECT_RATIO") or DEFAULT_TARGET_ASPECT_RATIO
    if args.target_seconds is None:
        args.target_seconds = env_int("PLAN_TARGET_SECONDS", DEFAULT_TARGET_SECONDS)
    if args.visual_index_max_images is None:
        args.visual_index_max_images = env_int(
            "PLAN_VISUAL_INDEX_MAX_IMAGES",
            DEFAULT_VISUAL_INDEX_MAX_IMAGES,
        )
    if args.image_detail is None:
        args.image_detail = env_value("PLAN_IMAGE_DETAIL") or env_value("STRATEGY_IMAGE_DETAIL") or DEFAULT_IMAGE_DETAIL
    if args.image_token_allowance is None:
        args.image_token_allowance = env_int(
            "PLAN_IMAGE_TOKEN_ALLOWANCE",
            env_int("STRATEGY_IMAGE_TOKEN_ALLOWANCE", DEFAULT_IMAGE_TOKEN_ALLOWANCE),
        )
    if args.input_usd_per_million is None:
        args.input_usd_per_million = env_float(
            "PLAN_INPUT_USD_PER_MILLION",
            env_float("STRATEGY_INPUT_USD_PER_MILLION", DEFAULT_GPT55_INPUT_USD_PER_MILLION),
        )
    if args.cached_input_usd_per_million is None:
        args.cached_input_usd_per_million = env_float(
            "PLAN_CACHED_INPUT_USD_PER_MILLION",
            env_float("STRATEGY_CACHED_INPUT_USD_PER_MILLION", DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION),
        )
    if args.output_usd_per_million is None:
        args.output_usd_per_million = env_float(
            "PLAN_OUTPUT_USD_PER_MILLION",
            env_float("STRATEGY_OUTPUT_USD_PER_MILLION", DEFAULT_GPT55_OUTPUT_USD_PER_MILLION),
        )
    if args.cache_mode is None:
        args.cache_mode = (
            env_value("PLAN_CACHE_MODE")
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
        run_plan(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
