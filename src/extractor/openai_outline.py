"""OpenAI outline stage: label the source's content units for a human editor.

Turns the transcript into a scannable outline — one short title and a one-line
plain-language gist per logical content unit — so an editor can see what
building blocks the source offers without reading the transcript or scrubbing
the video. It runs early in the pipeline (right after transcription) and is a
prerequisite for the storyboard review UX.

The unit list is derived by `extractor.transcript_units`, the same code the
review server uses, so the outline lines up 1:1 with the storyboard timeline.
Labels are produced in batched LLM calls and cached to disk keyed by the
transcript, so a repeat run is free. Like the other openai_* stages, it prints a
preflight cost estimate, refuses to call the API without `--approve-api-cost`,
and honors the local response cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.openai_strategy import (
    Pricing,
    actual_cost,
    cost_from_counts,
    estimate_text_tokens,
    read_json,
    response_output_text,
    sha256_text,
    usage_dict,
    write_json,
)
from extractor.openai_responses import create_response_streaming
from extractor.preprocess_video import format_time
from extractor.prompt_templates import load_prompt_template, prompt_template_path
from extractor.transcribe import env_float, env_int, env_value, load_env_file
from extractor.transcript_units import transcript_units_from_data

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "low"  # a labeling task; keep it cheap
DEFAULT_ENV_FILE = ".env"
DEFAULT_OUTPUT_DIR = Path("openai-runs")
DEFAULT_CACHE_SUBDIR = "_response-cache"
DEFAULT_BATCH_UNITS = 80
OUTPUT_TOKENS_PER_UNIT = 60  # title + one-line gist + json scaffolding, upper bound
DEVELOPER_PROMPT_TEMPLATE = "outline_developer.md"
SCHEMA_VERSION = 1

DEFAULT_GPT55_INPUT_USD_PER_MILLION = 5.00
DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION = 0.50
DEFAULT_GPT55_OUTPUT_USD_PER_MILLION = 30.00

OUTLINE_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["units"],
    "properties": {
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "title", "summary"],
                "properties": {
                    "index": {"type": "integer"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
        }
    },
}


def developer_instructions() -> str:
    return load_prompt_template(DEVELOPER_PROMPT_TEMPLATE)


def numbered_units_text(units: list[dict[str, Any]], offset: int = 0) -> str:
    """Render a batch of units as a numbered, timecoded list for the model."""
    lines = []
    for i, u in enumerate(units):
        tc = f"{format_time(float(u['start']))}-{format_time(float(u['end']))}"
        lines.append(f"{i + offset}. [{tc}] {str(u.get('text', '')).strip()}")
    return "\n".join(lines)


def outline_cache_key(
    transcript_path: str, units: list[dict[str, Any]], model: str, effort: str
) -> str:
    sig = "|".join([
        "outline_v" + str(SCHEMA_VERSION),
        transcript_path,
        model,
        effort,
        sha256_text(developer_instructions()),
        str(len(units)),
        ";".join(f"{u['start']:.2f}-{u['end']:.2f}" for u in units[:6000]),
    ])
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


def outline_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "outline" / f"{key}.json"


def preflight_cost(units: list[dict[str, Any]], batch: int, pricing: Pricing) -> dict[str, Any]:
    """Rough upper-bound: all unit text in, plus the developer prompt once per
    batch, and OUTPUT_TOKENS_PER_UNIT out for every unit."""
    n = len(units)
    num_batches = max(1, (n + batch - 1) // batch)
    text_tokens = estimate_text_tokens(numbered_units_text(units))
    dev_tokens = estimate_text_tokens(developer_instructions()) * num_batches
    input_tokens = text_tokens + dev_tokens
    output_tokens = n * OUTPUT_TOKENS_PER_UNIT
    return {
        "method": "chars/4 for unit text + developer prompt per batch; output = units * per-unit budget",
        "unit_count": n,
        "batches": num_batches,
        "input_tokens_estimate": input_tokens,
        "output_tokens_estimate": output_tokens,
        "estimated_usd": round(
            cost_from_counts(input_tokens, 0, output_tokens, pricing), 6
        ),
    }


def label_units(
    units: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Batched LLM call. Returns (labeled units [{index,start,end,title,summary}],
    per-batch raw responses, summed usage/cost)."""
    from openai import OpenAI

    client = OpenAI(timeout=env_float("OPENAI_CLIENT_TIMEOUT_SECONDS", 1800.0))
    developer = developer_instructions()
    labeled: list[dict[str, Any]] = [
        {"index": i, "start": u["start"], "end": u["end"],
         "title": u.get("summary", ""), "summary": ""}
        for i, u in enumerate(units)
    ]
    raw_responses: list[dict[str, Any]] = []
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0}
    pricing = Pricing(args.input_usd_per_million, args.cached_input_usd_per_million,
                      args.output_usd_per_million)

    for start in range(0, len(units), args.batch):
        chunk = units[start:start + args.batch]
        user = (
            f"Segments {start} to {start + len(chunk) - 1}. For EACH, return its "
            f"index, a short title, and a one-sentence summary.\n\n"
            + numbered_units_text(chunk, offset=start)
        )
        resp = create_response_streaming(
            client, label="outline",
            model=args.model,
            input=[{"role": "developer", "content": developer},
                   {"role": "user", "content": user}],
            reasoning={"effort": args.reasoning_effort},
            max_output_tokens=OUTPUT_TOKENS_PER_UNIT * len(chunk) + 500,
            store=False,
            text={"format": {"type": "json_schema", "name": "outline_batch",
                             "schema": OUTLINE_BATCH_SCHEMA, "strict": True}},
        )
        response_dict = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {}
        raw_responses.append(response_dict)
        try:
            parsed = json.loads(response_output_text(resp, response_dict))
        except json.JSONDecodeError:
            parsed = {"units": []}
        for item in parsed.get("units") or []:
            try:
                gi = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= gi < len(labeled):
                title = str(item.get("title") or "").strip()
                summary = str(item.get("summary") or "").strip()
                if title:
                    labeled[gi]["title"] = title
                labeled[gi]["summary"] = summary
        cost = actual_cost(usage_dict(response_dict), pricing)
        totals["input_tokens"] += cost["input_tokens"]
        totals["cached_input_tokens"] += cost["cached_input_tokens"]
        totals["output_tokens"] += cost["output_tokens"]
        totals["estimated_usd"] = round(totals["estimated_usd"] + cost["estimated_usd"], 6)

    return labeled, raw_responses, totals


def write_outline_markdown(path: Path, title: str, units: list[dict[str, Any]],
                           cost: dict[str, Any]) -> None:
    lines = [f"# Source outline — {title}", "",
             f"{len(units)} content units. Scan these to see the building blocks in the "
             "source without watching the video or reading the transcript.", ""]
    for u in units:
        tc = f"{format_time(float(u['start']))}–{format_time(float(u['end']))}"
        lines.append(f"- **[{tc}] {u['title']}** — {u['summary']}".rstrip(" —"))
    lines += ["", "## Cost", "",
              f"- Estimated USD: `${cost.get('estimated_usd', 0):.6f}`",
              f"- Input tokens: `{cost.get('input_tokens', 0)}` · "
              f"Output tokens: `{cost.get('output_tokens', 0)}`"]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def outline_output_payload(transcript_path: Path, video: str | None,
                           units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "transcript": str(transcript_path),
        "source_video": video,
        "unit_count": len(units),
        "units": [
            {"index": u["index"], "start": round(float(u["start"]), 3),
             "end": round(float(u["end"]), 3), "title": u["title"], "summary": u["summary"]}
            for u in units
        ],
    }


def run_outline(args: argparse.Namespace) -> dict[str, Any]:
    transcript_path = args.transcript.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)
    tdata = read_json(transcript_path)
    units = transcript_units_from_data(tdata)
    if not units:
        raise RuntimeError("The transcript has no content units to outline.")
    video = (tdata.get("source") or {}).get("path")
    title = (tdata.get("source") or {}).get("filename") or transcript_path.stem

    pricing = Pricing(args.input_usd_per_million, args.cached_input_usd_per_million,
                      args.output_usd_per_million)
    est = preflight_cost(units, args.batch, pricing)

    cache_dir = None if args.no_local_cache else args.cache_dir.expanduser().resolve()
    key = outline_cache_key(str(transcript_path), units, args.model, args.reasoning_effort)
    cache_path = outline_cache_path(cache_dir, key) if cache_dir else None
    cache_hit = bool(cache_path and cache_path.exists() and not args.refresh_cache)

    print("API cost estimate:")
    print(f"  model: {args.model}; reasoning: {args.reasoning_effort}; "
          f"units: {est['unit_count']} in {est['batches']} batch(es)")
    print(f"  estimated input tokens: {est['input_tokens_estimate']}, "
          f"output tokens: {est['output_tokens_estimate']}")
    print(f"  estimated upper-bound cost: ${est['estimated_usd']:.6f}")
    if cache_dir:
        print(f"  local response cache key: {key}")
        print(f"  local response cache: {'hit' if cache_hit else 'miss'}")
    else:
        print("  local response cache: disabled")

    if args.dry_run:
        return {"preflight_cost": est, "cache_hit": cache_hit}

    if not cache_hit:
        if args.approve_api_cost is None:
            raise RuntimeError(
                "API calls require explicit approval. Re-run with "
                f"--approve-api-cost {est['estimated_usd']:.4f} or a higher USD cap.")
        if est["estimated_usd"] > args.approve_api_cost:
            raise RuntimeError(
                f"Estimated API cost ${est['estimated_usd']:.6f} exceeds approved cap "
                f"${args.approve_api_cost:.6f}.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or (
        f"{transcript_path.stem}.outline.{args.model}.{args.reasoning_effort}."
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "source-materials.txt").write_text(numbered_units_text(units), encoding="utf-8")
    (run_dir / "task-prompt.txt").write_text(developer_instructions(), encoding="utf-8")

    if cache_hit:
        assert cache_path is not None
        cached = read_json(cache_path)
        labeled = cached["units"]
        raw_responses = cached.get("raw_responses", [])
        original_cost = cached.get("cost", {})
        zero = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0}
        usage_and_cost = {"api_call_made": False,
                          "cache": {"hit": True, "key": key, "path": str(cache_path)},
                          "cost": zero, "original_api_cost": original_cost}
        summary_cost = original_cost or zero
        print(f"Local response cache hit: {cache_path}")
        print("No OpenAI API call made.")
    else:
        labeled, raw_responses, summary_cost = label_units(units, args)
        usage_and_cost = {"api_call_made": True,
                          "cache": {"hit": False, "key": key,
                                    "path": str(cache_path) if cache_path else None},
                          "cost": summary_cost}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(cache_path, {"units": labeled, "raw_responses": raw_responses,
                                    "cost": summary_cost})
        print(f"Estimated billed cost from usage: ${summary_cost['estimated_usd']:.6f}")

    payload = outline_output_payload(transcript_path, video, labeled)
    write_json(run_dir / "request-manifest.json", {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model, "reasoning_effort": args.reasoning_effort, "batch": args.batch,
        "transcript": str(transcript_path), "unit_count": len(units),
        "preflight_cost": est,
        "local_response_cache": {"enabled": cache_dir is not None, "hit": cache_hit,
                                 "key": key if cache_dir else None,
                                 "path": str(cache_path) if cache_path else None},
    })
    write_json(run_dir / "raw-response.json", {"batches": raw_responses})
    write_json(run_dir / "outline-output.json", payload)
    write_json(run_dir / "usage-and-cost.json", usage_and_cost)
    write_outline_markdown(run_dir / "outline.md", title, labeled, summary_cost)
    print(f"Wrote {run_dir / 'outline-output.json'}")
    print(f"Wrote {run_dir / 'outline.md'}")
    return {"run_dir": str(run_dir), "output": payload, "cost": summary_cost}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outline the source's content units (title + one-line gist per unit) for editors.")
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--no-local-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Bypass any cached outline and make a fresh approved API call.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve-api-cost", type=float, default=None, metavar="USD",
                        help="Explicitly approve this OpenAI API call up to the preflight estimate.")
    parser.add_argument("--input-usd-per-million", type=float, default=None)
    parser.add_argument("--cached-input-usd-per-million", type=float, default=None)
    parser.add_argument("--output-usd-per-million", type=float, default=None)
    return parser


def apply_env_defaults(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = Path(env_value("OUTLINE_OUTPUT_DIR")
                               or env_value("STRATEGY_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    if args.cache_dir is None:
        args.cache_dir = Path(env_value("OUTLINE_CACHE_DIR")
                              or env_value("OPENAI_RESPONSE_CACHE_DIR")
                              or (args.output_dir / DEFAULT_CACHE_SUBDIR))
    if args.model is None:
        args.model = env_value("OUTLINE_MODEL") or env_value("STRATEGY_MODEL") or DEFAULT_MODEL
    if args.reasoning_effort is None:
        args.reasoning_effort = env_value("OUTLINE_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT
    if args.batch is None:
        args.batch = env_int("OUTLINE_BATCH_UNITS", DEFAULT_BATCH_UNITS)
    if args.input_usd_per_million is None:
        args.input_usd_per_million = env_float(
            "OUTLINE_INPUT_USD_PER_MILLION",
            env_float("STRATEGY_INPUT_USD_PER_MILLION", DEFAULT_GPT55_INPUT_USD_PER_MILLION))
    if args.cached_input_usd_per_million is None:
        args.cached_input_usd_per_million = env_float(
            "OUTLINE_CACHED_INPUT_USD_PER_MILLION",
            env_float("STRATEGY_CACHED_INPUT_USD_PER_MILLION", DEFAULT_GPT55_CACHED_INPUT_USD_PER_MILLION))
    if args.output_usd_per_million is None:
        args.output_usd_per_million = env_float(
            "OUTLINE_OUTPUT_USD_PER_MILLION",
            env_float("STRATEGY_OUTPUT_USD_PER_MILLION", DEFAULT_GPT55_OUTPUT_USD_PER_MILLION))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        load_env_file(args.env_file)
        apply_env_defaults(args)
        run_outline(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
