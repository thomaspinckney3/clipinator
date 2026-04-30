from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.openai_stage1 import (
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
    build_source_materials as build_video_source_materials,
    cache_entry_dir,
    cache_entry_exists,
    estimated_request_cost,
    image_data_url,
    read_json,
    request_cache_key,
    response_output_text,
    sha256_file,
    sha256_text,
    stable_json,
    usage_dict,
    write_json,
)
from extractor.openai_stage3 import STAGE3_SCHEMA
from extractor.transcribe import env_float, env_int, env_value, load_env_file


DEFAULT_MAX_OUTPUT_TOKENS = 18_000


DEVELOPER_INSTRUCTIONS = (
    "You are a senior short-form documentary editor and pipeline designer. "
    "Convert post-render critique into a revised, renderable edit plan. "
    "Treat the work as a general video-to-clip system, not a one-off manual fix. "
    "Return only valid JSON for the requested schema."
)


RENDER_SETTINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "transition_duration": {"type": "number"},
        "internal_visual_transition_duration": {"type": "number"},
        "foreground_scale": {"type": "number"},
        "foreground_brightness": {"type": "number"},
        "foreground_contrast": {"type": "number"},
        "foreground_saturation": {"type": "number"},
        "burn_captions": {"type": "boolean"},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}


def stage3_schema_for_stage5() -> dict[str, Any]:
    schema = deepcopy(STAGE3_SCHEMA)
    schema["properties"]["render_settings"] = deepcopy(RENDER_SETTINGS_SCHEMA)
    clip_schema = schema["properties"]["clip_ads"]["items"]
    clip_schema["properties"]["caption_plan_source"] = {
        "type": "string",
        "enum": ["absolute_render_time", "render_time", "srt"],
    }
    clip_schema["properties"]["use_caption_plan_for_srt"] = {"type": "boolean"}
    clip_schema["properties"]["render_settings"] = deepcopy(RENDER_SETTINGS_SCHEMA)
    clip_schema["properties"]["stage5_revision_notes"] = {
        "type": "array",
        "items": {"type": "string"},
    }
    return schema


STAGE5_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "stage5_summary",
        "revision_policy",
        "clip_revision_reports",
        "render_queue",
        "revised_stage3_output",
    ],
    "properties": {
        "stage5_summary": {"type": "string"},
        "revision_policy": {"type": "array", "items": {"type": "string"}},
        "clip_revision_reports": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "clip_id",
                    "prior_clip_id",
                    "action",
                    "issues_addressed",
                    "applied_changes",
                    "rejected_or_unresolved_feedback",
                    "remaining_risks",
                    "should_render",
                    "should_review_again",
                ],
                "properties": {
                    "clip_id": {"type": "string"},
                    "prior_clip_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["keep_as_is", "revise", "reject", "defer"],
                    },
                    "issues_addressed": {"type": "array", "items": {"type": "string"}},
                    "applied_changes": {"type": "array", "items": {"type": "string"}},
                    "rejected_or_unresolved_feedback": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "remaining_risks": {"type": "array", "items": {"type": "string"}},
                    "render_settings": deepcopy(RENDER_SETTINGS_SCHEMA),
                    "should_render": {"type": "boolean"},
                    "should_review_again": {"type": "boolean"},
                },
            },
        },
        "render_queue": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["clip_id", "priority", "reason", "render_settings"],
                "properties": {
                    "clip_id": {"type": "string"},
                    "priority": {"type": "integer"},
                    "reason": {"type": "string"},
                    "render_settings": deepcopy(RENDER_SETTINGS_SCHEMA),
                },
            },
        },
        "revised_stage3_output": stage3_schema_for_stage5(),
    },
}


def response_text_config() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "stage5_reedit_plan",
            "schema": STAGE5_SCHEMA,
            "strict": False,
        },
        "verbosity": "medium",
    }


def filter_stage3_output(stage3_output: dict[str, Any], clip_ids: set[str]) -> dict[str, Any]:
    if not clip_ids:
        return stage3_output
    filtered = deepcopy(stage3_output)
    filtered["clip_ads"] = [
        clip for clip in filtered.get("clip_ads", []) or [] if str(clip.get("clip_id")) in clip_ids
    ]
    return filtered


def filter_stage4_output(stage4_output: dict[str, Any], clip_ids: set[str]) -> dict[str, Any]:
    if not clip_ids:
        return stage4_output
    filtered = deepcopy(stage4_output)
    filtered["clip_reviews"] = [
        review
        for review in filtered.get("clip_reviews", []) or []
        if str(review.get("clip_id")) in clip_ids
    ]
    filtered["recommended_rerender_order"] = [
        item
        for item in filtered.get("recommended_rerender_order", []) or []
        if str(item.get("clip_id")) in clip_ids
    ]
    return filtered


def compact_render_manifest(render_manifest: dict[str, Any], clip_ids: set[str]) -> dict[str, Any]:
    clips = render_manifest.get("clips", []) or []
    if clip_ids:
        clips = [clip for clip in clips if str(clip.get("clip_id")) in clip_ids]
    compact_clips: list[dict[str, Any]] = []
    for clip in clips:
        probe = clip.get("probe") or {}
        media_format = probe.get("format") or {}
        compact_clips.append(
            {
                "clip_id": clip.get("clip_id"),
                "concept_title": clip.get("concept_title"),
                "output_path": clip.get("output_path"),
                "srt_path": clip.get("srt_path"),
                "duration_seconds": media_format.get("duration"),
                "implemented_edit_features": clip.get("implemented_edit_features"),
                "transition_duration": clip.get("transition_duration"),
                "internal_visual_transition_duration": clip.get(
                    "internal_visual_transition_duration"
                ),
                "foreground_scale": clip.get("foreground_scale"),
                "foreground_brightness": clip.get("foreground_brightness"),
                "foreground_contrast": clip.get("foreground_contrast"),
                "foreground_saturation": clip.get("foreground_saturation"),
                "caption_plan_source": clip.get("caption_plan_source"),
                "caption_entries": clip.get("caption_entries"),
                "segment_plans": clip.get("segment_plans"),
            }
        )
    return {
        "source_video": render_manifest.get("source_video"),
        "stage3_output": render_manifest.get("stage3_output"),
        "resolution": render_manifest.get("resolution"),
        "renderer": render_manifest.get("renderer"),
        "clips": compact_clips,
    }


def compact_transition_sheet(sheet: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(sheet, dict):
        return None
    return {
        "columns": sheet.get("columns"),
        "read_order": sheet.get("read_order"),
        "sampled_frames": [
            {
                "transition_index": frame.get("transition_index"),
                "transition_type": frame.get("transition_type"),
                "from_segment_order": frame.get("from_segment_order"),
                "to_segment_order": frame.get("to_segment_order"),
                "from_visual_span_index": frame.get("from_visual_span_index"),
                "to_visual_span_index": frame.get("to_visual_span_index"),
                "render_time_seconds": frame.get("render_time_seconds"),
                "offset_from_transition_mid_seconds": frame.get(
                    "offset_from_transition_mid_seconds"
                ),
            }
            for frame in sheet.get("sampled_frames", []) or []
        ],
    }


def compact_review_package(package: dict[str, Any], clip_ids: set[str]) -> dict[str, Any]:
    clips = package.get("clips", []) or []
    if clip_ids:
        clips = [clip for clip in clips if str(clip.get("clip_id")) in clip_ids]
    compact_clips: list[dict[str, Any]] = []
    for clip in clips:
        compact_clips.append(
            {
                "clip_id": clip.get("clip_id"),
                "rendered_video_path": clip.get("rendered_video_path"),
                "local_timing_evidence": clip.get("local_timing_evidence"),
                "srt_entries": clip.get("srt_entries"),
                "sampled_frames": [
                    {
                        "index": frame.get("index"),
                        "render_time_seconds": frame.get("render_time_seconds"),
                    }
                    for frame in clip.get("sampled_frames", []) or []
                ],
                "transition_sheet": compact_transition_sheet(clip.get("transition_sheet")),
            }
        )
    return {
        "schema_version": package.get("schema_version"),
        "human_feedback_rubric": package.get("human_feedback_rubric"),
        "review_evidence_note": package.get("review_evidence_note"),
        "clips": compact_clips,
    }


def build_task_prompt(
    stage3_output: dict[str, Any],
    stage4_output: dict[str, Any],
    render_manifest: dict[str, Any],
    review_package: dict[str, Any] | None,
    include_video_source_materials: bool,
) -> str:
    evidence_scope = (
        "The stable video transcript and preprocessing evidence are included, so you may "
        "use them to choose better sentence boundaries or source ranges."
        if include_video_source_materials
        else "No full transcript/source prefix is included. Do not invent new source ranges; "
        "revise only from the prior plan, render manifest, and Stage 4 critique."
    )
    return "\n\n".join(
        [
            "Create stage 5 of a video-to-clip-ad pipeline: re-edit after post-render critique.",
            evidence_scope,
            "The output must include a full revised Stage 3 plan under revised_stage3_output. It must be directly renderable by extractor.render_stage3.",
            "Do not produce a one-off patch for the current example. Express fixes as general edit decisions driven by critique categories and evidence.",
            "Hard editorial rules:",
            "- If there is no clear payoff, reject or defer the clip instead of trying to rescue it with captions or faster cuts.",
            "- Every revised clip ad should open with a short, intentional title/tagline card unless the visual action itself is the payoff.",
            "- For any generated title, hook, or CTA card, set source_start and source_end to 0, visual_source exactly to 'Generated black title card.', and audio_source to 'No source dialogue; silent generated title card.' This is the renderer contract; do not use source_start=0 footage as a fake card.",
            "- The payoff or surprising line should be the first spoken beat after the title/tagline card whenever it can open cleanly.",
            "- Do not put a payoff line first if it begins with a connective such as 'but', 'and', or 'so', depends on prior context, or sounds like a mid-sentence start. Include the setup before it or reject/defer the clip.",
            "- Title, overlay, and caption wording must preserve the source claim. Do not turn a narrator interpretation into a literal factual claim; for example, 'the next words were about science' is not the same as 'the next words were science'.",
            "- Prefer an intentional black title card over silent source video before dialogue.",
            "- If Stage 4 flags a broken title, repair the title card text/timing; do not remove the title card and start directly on interview or narration footage.",
            "- Do not start or end spoken audio mid-sentence or mid-word. Leave a small tail when needed, but trim before topic drift.",
            "- Avoid showing a talking head while unrelated narration or a different speaker is heard; use B-roll, title cards, or matching-source visuals instead.",
            "- If B-roll repeats to stretch the clip, shorten the clip unless verified alternate visuals are available.",
            "- If Stage 4 says the concrete object/action named by the clip is not visually legible, either find explicit visual proof from the supplied source evidence or mark the clip reject/defer with should_render=false.",
            "- Do not turn a hard visual-proof gap into a soft note for later review. If a hammer clip cannot show a hammer, hammer-like tool, or unambiguous gloved-hand tool use, do not include it in revised_stage3_output.clip_ads.",
            "- Smooth abrupt cuts with crossfades, J/L cuts, cut-on-action choices, or shorter ranges, but do not hide story problems with transitions.",
            "- The ending must stay on the same story and should land after the thought resolves.",
            "- Phrase-level captions should be explicit render-time SRT entries. When doing that, set caption_plan_source to absolute_render_time on the revised clip.",
            "- Use render_settings when the critique calls for crossfades, internal visual dissolves, foreground punch-in, or exposure/contrast fixes.",
            "For each clip, explain which Stage 4 issues you addressed and which feedback remains unresolved. If a requested fix needs source evidence that is not present, say so and choose the conservative edit.",
            "PREVIOUS STAGE 3 OUTPUT JSON:",
            json.dumps(stage3_output, ensure_ascii=False, indent=2),
            "RENDER MANIFEST JSON:",
            json.dumps(render_manifest, ensure_ascii=False, indent=2),
            "STAGE 4 CRITIQUE JSON:",
            json.dumps(stage4_output, ensure_ascii=False, indent=2),
            "OPTIONAL LOCAL REVIEW PACKAGE JSON:",
            json.dumps(review_package, ensure_ascii=False, indent=2)
            if review_package is not None
            else "No review package JSON supplied.",
        ]
    )


def request_fingerprint_payload(
    args: argparse.Namespace,
    source_materials: str,
    task_prompt: str,
    contact_sheet_hash: str | None,
    stage3_output_hash: str,
    stage4_output_hash: str,
    render_manifest_hash: str,
    review_package_hash: str | None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "sha256": sha256_text(source_materials)}]
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
        "stage": "stage5_reedit_plan",
        "api": "openai.responses.create",
        "model": args.model,
        "input": [
            {"role": "developer", "content": DEVELOPER_INSTRUCTIONS},
            {"role": "user", "content": content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "sha256": sha256_text(task_prompt),
                        "stage3_output_sha256": stage3_output_hash,
                        "stage4_output_sha256": stage4_output_hash,
                        "render_manifest_sha256": render_manifest_hash,
                        "review_package_sha256": review_package_hash,
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
) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        entry_dir / "cache-manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "cache_key": cache_key,
            "source": source,
            "fingerprint_payload": fingerprint_payload,
        },
    )
    write_json(entry_dir / "raw-response.json", response_dict)
    (entry_dir / "output-text.json").write_text(output_text.rstrip() + "\n", encoding="utf-8")
    write_json(entry_dir / "stage5-output.json", parsed_output)
    write_json(entry_dir / "stage3-output.json", parsed_output["revised_stage3_output"])
    write_json(entry_dir / "usage-and-cost.json", usage_and_cost)


def load_cached_response(entry_dir: Path) -> dict[str, Any]:
    response_dict = read_json(entry_dir / "raw-response.json")
    output_text_path = entry_dir / "output-text.json"
    output_text = (
        output_text_path.read_text(encoding="utf-8")
        if output_text_path.exists()
        else response_output_text(None, response_dict)
    )
    parsed_path = entry_dir / "stage5-output.json"
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
        "# Stage 5 Re-Edit Plan",
        "",
        "## Summary",
        "",
        str(output.get("stage5_summary", "")).strip(),
        "",
        "## Clip Revisions",
        "",
    ]
    for report in output.get("clip_revision_reports", []) or []:
        lines.append(f"### {report.get('clip_id', '')}")
        lines.append(f"- Prior clip: `{report.get('prior_clip_id', '')}`")
        lines.append(f"- Action: `{report.get('action', '')}`")
        lines.append(f"- Render again: `{str(report.get('should_render', False)).lower()}`")
        if report.get("issues_addressed"):
            lines.append("- Issues addressed:")
            for issue in report.get("issues_addressed", []) or []:
                lines.append(f"  - {issue}")
        if report.get("applied_changes"):
            lines.append("- Applied changes:")
            for change in report.get("applied_changes", []) or []:
                lines.append(f"  - {change}")
        if report.get("remaining_risks"):
            lines.append("- Remaining risks:")
            for risk in report.get("remaining_risks", []) or []:
                lines.append(f"  - {risk}")
        lines.append("")
    if output.get("render_queue"):
        lines.extend(["## Render Queue", ""])
        for item in output.get("render_queue", []) or []:
            lines.append(
                f"- `{item.get('priority', '')}` `{item.get('clip_id', '')}`: "
                f"{item.get('reason', '')}"
            )
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
    write_json(run_dir / "stage5-output.json", parsed_output)
    write_json(run_dir / "stage3-output.json", parsed_output["revised_stage3_output"])
    write_json(run_dir / "usage-and-cost.json", usage_and_cost)
    write_markdown(run_dir / "summary.md", parsed_output, summary_cost, cache_info=cache_info)


def run_stage5(args: argparse.Namespace) -> dict[str, Any]:
    stage3_path = args.stage3_output.expanduser().resolve()
    stage4_path = args.stage4_output.expanduser().resolve()
    render_manifest_path = args.render_manifest.expanduser().resolve()
    review_package_path = args.review_package.expanduser().resolve() if args.review_package else None
    transcript_path = args.transcript.expanduser().resolve() if args.transcript else None
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else None
    contact_sheet_path = args.contact_sheet.expanduser().resolve() if args.contact_sheet else None

    for path in [stage3_path, stage4_path, render_manifest_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    for optional_path in [review_package_path, transcript_path, manifest_path, contact_sheet_path]:
        if optional_path is not None and not optional_path.exists():
            raise FileNotFoundError(optional_path)
    if (transcript_path is None) != (manifest_path is None):
        raise RuntimeError("--transcript and --manifest must be supplied together.")
    if contact_sheet_path is not None and transcript_path is None:
        raise RuntimeError("--contact-sheet requires --transcript and --manifest.")

    clip_ids = set(args.clip_id)
    pricing = Pricing(
        input_usd_per_million=args.input_usd_per_million,
        cached_input_usd_per_million=args.cached_input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,
    )

    source_materials = (
        build_video_source_materials(
            manifest_path=manifest_path,
            transcript_path=transcript_path,
            contact_sheet_path=contact_sheet_path,
        )
        if transcript_path is not None and manifest_path is not None
        else "No optional full-video transcript/preprocessing prefix supplied for Stage 5."
    )
    stage3_output = filter_stage3_output(read_json(stage3_path), clip_ids)
    stage4_output = filter_stage4_output(read_json(stage4_path), clip_ids)
    render_manifest = compact_render_manifest(read_json(render_manifest_path), clip_ids)
    review_package = (
        compact_review_package(read_json(review_package_path), clip_ids)
        if review_package_path is not None
        else None
    )
    task_prompt = build_task_prompt(
        stage3_output=stage3_output,
        stage4_output=stage4_output,
        render_manifest=render_manifest,
        review_package=review_package,
        include_video_source_materials=transcript_path is not None,
    )
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
    stage3_output_text = stable_json(stage3_output)
    stage4_output_text = stable_json(stage4_output)
    render_manifest_text = stable_json(render_manifest)
    review_package_text = stable_json(review_package) if review_package is not None else ""
    stable_prefix_cache_material = "\n".join(
        [
            source_materials,
            args.model,
            args.reasoning_effort,
            contact_sheet_hash or "",
        ]
    )
    prompt_cache_key = args.prompt_cache_key or f"stage5-{sha256_text(stable_prefix_cache_material)[:48]}"
    fingerprint_payload = request_fingerprint_payload(
        args=args,
        source_materials=source_materials,
        task_prompt=task_prompt,
        contact_sheet_hash=contact_sheet_hash,
        stage3_output_hash=sha256_text(stage3_output_text),
        stage4_output_hash=sha256_text(stage4_output_text),
        render_manifest_hash=sha256_text(render_manifest_text),
        review_package_hash=sha256_text(review_package_text) if review_package_text else None,
    )
    local_cache_key = request_cache_key(fingerprint_payload)
    cache_dir = None if args.no_local_cache else args.cache_dir.expanduser().resolve()
    entry_dir = cache_entry_dir(cache_dir, local_cache_key) if cache_dir else None
    cache_hit = bool(entry_dir and cache_entry_exists(entry_dir) and not args.refresh_cache)
    if cache_dir:
        print(f"  local response cache key: {local_cache_key}")
        print(f"  local response cache: {'hit' if cache_hit else 'miss'}")
    else:
        print("  local response cache: disabled")

    if args.dry_run:
        return {
            "preflight_cost": preflight_cost,
            "source_materials_sha256": sha256_text(source_materials),
            "task_prompt_sha256": sha256_text(task_prompt),
            "stage3_output_sha256": sha256_text(stage3_output_text),
            "stage4_output_sha256": sha256_text(stage4_output_text),
            "render_manifest_sha256": sha256_text(render_manifest_text),
            "review_package_sha256": sha256_text(review_package_text)
            if review_package_text
            else None,
            "prompt_cache_key": prompt_cache_key,
            "local_response_cache": {
                "enabled": cache_dir is not None,
                "hit": cache_hit,
                "key": local_cache_key if cache_dir else None,
                "entry_dir": str(entry_dir) if entry_dir else None,
            },
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
        f"{stage4_path.parent.name}.stage5.{args.model}.{args.reasoning_effort}."
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(
        run_dir / "request-manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "stage": "stage5_reedit_plan",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "image_detail": args.image_detail if contact_sheet_path else None,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": args.prompt_cache_retention,
            "clip_ids": sorted(clip_ids),
            "paths": {
                "stage3_output": str(stage3_path),
                "stage4_output": str(stage4_path),
                "render_manifest": str(render_manifest_path),
                "review_package": str(review_package_path) if review_package_path else None,
                "transcript": str(transcript_path) if transcript_path else None,
                "manifest": str(manifest_path) if manifest_path else None,
                "contact_sheet": str(contact_sheet_path) if contact_sheet_path else None,
            },
            "hashes": {
                "source_materials_sha256": sha256_text(source_materials),
                "task_prompt_sha256": sha256_text(task_prompt),
                "stage3_output_sha256": sha256_text(stage3_output_text),
                "stage4_output_sha256": sha256_text(stage4_output_text),
                "render_manifest_sha256": sha256_text(render_manifest_text),
                "review_package_sha256": sha256_text(review_package_text)
                if review_package_text
                else None,
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
                "cache_key": local_cache_key if cache_dir else None,
                "entry_dir": str(entry_dir) if entry_dir else None,
                "fingerprint_payload": fingerprint_payload if cache_dir else None,
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
        print(f"Wrote {run_dir / 'stage5-output.json'}")
        print(f"Wrote {run_dir / 'stage3-output.json'}")
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
        {"role": "developer", "content": DEVELOPER_INSTRUCTIONS},
        {"role": "user", "content": content},
        {"role": "user", "content": [{"type": "input_text", "text": task_prompt}]},
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
        )
        print(f"Stored local response cache entry: {entry_dir}")
    print(
        f"Actual token usage: input={cost['input_tokens']}, "
        f"cached_input={cost['cached_input_tokens']}, output={cost['output_tokens']}"
    )
    print(f"Estimated billed cost from usage: ${cost['estimated_usd']:.6f}")
    print(f"Wrote {run_dir / 'stage5-output.json'}")
    print(f"Wrote {run_dir / 'stage3-output.json'}")
    print(f"Wrote {run_dir / 'summary.md'}")
    return {
        "run_dir": str(run_dir),
        "output": parsed_output,
        "cost": cost,
        "usage": usage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenAI stage 5: revise edit plans from Stage 4 critique."
    )
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--stage3-output", type=Path, required=True)
    parser.add_argument("--stage4-output", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--review-package", type=Path, default=None)
    parser.add_argument("--transcript", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--clip-id", action="append", default=[])
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
        args.output_dir = Path(env_value("STAGE5_OUTPUT_DIR") or env_value("STAGE1_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    if args.cache_dir is None:
        args.cache_dir = Path(
            env_value("STAGE5_CACHE_DIR")
            or env_value("STAGE4_CACHE_DIR")
            or env_value("STAGE1_CACHE_DIR")
            or env_value("OPENAI_RESPONSE_CACHE_DIR")
            or (args.output_dir / DEFAULT_CACHE_SUBDIR)
        )
    if args.model is None:
        args.model = env_value("STAGE5_MODEL") or env_value("STAGE1_MODEL") or DEFAULT_MODEL
    if args.reasoning_effort is None:
        args.reasoning_effort = (
            env_value("STAGE5_REASONING_EFFORT")
            or env_value("STAGE1_REASONING_EFFORT")
            or DEFAULT_REASONING_EFFORT
        )
    if args.max_output_tokens is None:
        args.max_output_tokens = env_int("STAGE5_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
    if args.image_detail is None:
        args.image_detail = env_value("STAGE5_IMAGE_DETAIL") or env_value("STAGE1_IMAGE_DETAIL") or DEFAULT_IMAGE_DETAIL
    if args.image_token_allowance is None:
        args.image_token_allowance = env_int(
            "STAGE5_IMAGE_TOKEN_ALLOWANCE",
            env_int("STAGE1_IMAGE_TOKEN_ALLOWANCE", DEFAULT_IMAGE_TOKEN_ALLOWANCE),
        )
    if args.input_usd_per_million is None:
        args.input_usd_per_million = env_float(
            "STAGE5_INPUT_USD_PER_MILLION",
            env_float("STAGE1_INPUT_USD_PER_MILLION", DEFAULT_GPT55_INPUT_USD_PER_MILLION),
        )
    if args.cached_input_usd_per_million is None:
        args.cached_input_usd_per_million = env_float(
            "STAGE5_CACHED_INPUT_USD_PER_MILLION",
            env_float("STAGE1_CACHED_INPUT_USD_PER_MILLION", DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION),
        )
    if args.output_usd_per_million is None:
        args.output_usd_per_million = env_float(
            "STAGE5_OUTPUT_USD_PER_MILLION",
            env_float("STAGE1_OUTPUT_USD_PER_MILLION", DEFAULT_GPT55_OUTPUT_USD_PER_MILLION),
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        load_env_file(args.env_file)
        apply_env_defaults(args)
        run_stage5(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
