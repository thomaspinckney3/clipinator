from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor import stage_cache
from extractor.project_layout import establish_media, generate_layout, source_slug
from extractor.preprocess_video import (
    DEFAULT_FFMPEG_BIN,
    DEFAULT_FFMPEG_HWACCEL,
    DEFAULT_WINDOW_SHEET_COLUMNS,
    DEFAULT_WINDOW_SHEET_FRAME_WIDTH,
    DEFAULT_WINDOW_SHEET_SECONDS,
    FFMPEG_HWACCEL_CHOICES,
    resolve_ffmpeg_bin,
    resolve_ffmpeg_hwaccel,
    safe_stem,
)
from extractor.transcribe import (
    TRANSCRIBE_PROVIDER_CHOICES,
    load_env_file,
    resolve_transcribe_model,
    resolve_transcribe_provider,
    transcript_file_stem,
    write_srt,
)


DEFAULT_ENV_FILE = Path(".env")
DEFAULT_FRAME_INTERVAL_SECONDS = 1.0
DEFAULT_FRAME_WIDTH = 480
DEFAULT_CONTACT_SHEET_FRAME_WIDTH = 180
DEFAULT_CONTACT_SHEET_COLUMNS = 12
CONTENT_MODE_CHOICES = ("ad", "excerpt")
EXCERPT_SECONDS_PER_CLIP = 10 * 60
PLAN_DEFAULT_MAX_OUTPUT_TOKENS = 18_000
PLAN_MAX_OUTPUT_TOKENS = 32_000
PLAN_OUTPUT_TOKENS_PER_EXTRACT = 6_000


@dataclass
class CostRecord:
    stage: str
    estimate_usd: float
    actual_or_reserved_usd: float
    cache_hit: bool = False


@dataclass
class PipelineState:
    approved_cap_usd: float | None
    cost_records: list[CostRecord] = field(default_factory=list)

    @property
    def spent_or_reserved_usd(self) -> float:
        return sum(record.actual_or_reserved_usd for record in self.cost_records)

    def remaining_cap_usd(self) -> float | None:
        if self.approved_cap_usd is None:
            return None
        return self.approved_cap_usd - self.spent_or_reserved_usd

    def require_approval(self, stage: str, estimate_usd: float, cache_hit: bool) -> None:
        if cache_hit or estimate_usd <= 0:
            return
        if self.approved_cap_usd is None:
            raise RuntimeError(
                f"{stage} would make an API call. Estimated upper-bound cost is "
                f"${estimate_usd:.6f}. Re-run with --approve-api-cost {estimate_usd:.4f} "
                "or a higher total pipeline cap."
            )
        remaining = self.remaining_cap_usd()
        if remaining is None or estimate_usd > remaining + 1e-9:
            raise RuntimeError(
                f"{stage} estimate ${estimate_usd:.6f} exceeds remaining approved "
                f"pipeline cap ${max(0.0, remaining or 0.0):.6f}."
            )

    def record(self, stage: str, estimate_usd: float, actual_usd: float | None, cache_hit: bool) -> None:
        charged = 0.0 if cache_hit else (actual_usd if actual_usd is not None else estimate_usd)
        self.cost_records.append(
            CostRecord(
                stage=stage,
                estimate_usd=estimate_usd,
                actual_or_reserved_usd=charged,
                cache_hit=cache_hit,
            )
        )

    def has_stage(self, stage: str) -> bool:
        return any(record.stage == stage for record in self.cost_records)


@dataclass(frozen=True)
class PipelinePaths:
    video: Path
    run_root: Path
    stem: str
    run_id_base: str
    content_mode: str
    transcribe_provider: str
    transcribe_model: str
    render_variant: str | None = None

    @property
    def transcript_dir(self) -> Path:
        return self.run_root / "transcripts"

    @property
    def transcript_json(self) -> Path:
        stem = transcript_file_stem(
            self.stem,
            self.transcribe_provider,
            self.transcribe_model,
        )
        return self.transcript_dir / f"{stem}.json"

    @property
    def available_transcript_json(self) -> Path:
        stem = transcript_file_stem(
            self.stem,
            self.transcribe_provider,
            self.transcribe_model,
        )
        return self.transcript_dir / f"{stem}.available.json"

    @property
    def video_evidence_dir(self) -> Path:
        return self.run_root / "video-evidence"

    @property
    def evidence_manifest(self) -> Path:
        return self.video_evidence_dir / self.stem / "manifest.json"

    @property
    def contact_sheet(self) -> Path:
        return self.video_evidence_dir / self.stem / "contact-sheet.jpg"

    @property
    def openai_runs_dir(self) -> Path:
        return self.run_root / "openai-runs"

    @property
    def response_cache_dir(self) -> Path:
        return self.openai_runs_dir / "_response-cache"

    @property
    def visual_index_dir(self) -> Path:
        if self.render_variant:
            return self.run_root / "visual-index" / self.render_variant
        return self.run_root / "visual-index"

    @property
    def visual_index_manifest(self) -> Path:
        return self.visual_index_dir / self.stem / "manifest.json"

    @property
    def subject_index_dir(self) -> Path:
        return self.run_root / "subject-index"

    @property
    def draft_subject_index_manifest(self) -> Path:
        return self.subject_index_dir / self.plan_run_id / "manifest.json"

    @property
    def final_subject_index_manifest(self) -> Path:
        return self.subject_index_dir / self.revise_run_id / "manifest.json"

    @property
    def outline_run_id(self) -> str:
        return f"{self.run_id_base}-outline"

    @property
    def outline_output(self) -> Path:
        return self.openai_runs_dir / self.outline_run_id / "outline-output.json"

    @property
    def strategy_run_id(self) -> str:
        return f"{self.run_id_base}-strategy"

    @property
    def moments_run_id(self) -> str:
        return f"{self.run_id_base}-moments"

    @property
    def plan_run_id(self) -> str:
        return f"{self.run_id_base}-plan"

    @property
    def critique_run_id(self) -> str:
        return f"{self.run_id_base}-critique"

    @property
    def revise_run_id(self) -> str:
        return f"{self.run_id_base}-revise"

    @property
    def strategy_output(self) -> Path:
        return self.openai_runs_dir / self.strategy_run_id / "strategy-output.json"

    @property
    def moments_output(self) -> Path:
        return self.openai_runs_dir / self.moments_run_id / "moments-output.json"

    @property
    def plan_output(self) -> Path:
        return self.openai_runs_dir / self.plan_run_id / "plan-output.json"

    @property
    def critique_output(self) -> Path:
        return self.openai_runs_dir / self.critique_run_id / "critique-output.json"

    @property
    def critique_package(self) -> Path:
        return self.openai_runs_dir / self.critique_run_id / "critique-package.json"

    @property
    def revised_plan_output(self) -> Path:
        return self.openai_runs_dir / self.revise_run_id / "plan-output.json"

    @property
    def draft_render_dir(self) -> Path:
        stage_dir = f"{self.render_variant}-draft" if self.render_variant else "draft"
        return self.run_root / "_renders" / self.content_mode / stage_dir

    @property
    def draft_render_manifest(self) -> Path:
        return self.draft_render_dir / "render-manifest.json"

    @property
    def final_render_dir(self) -> Path:
        stage_dir = f"{self.render_variant}-final" if self.render_variant else "final"
        return self.run_root / "_renders" / self.content_mode / stage_dir

    @property
    def final_render_manifest(self) -> Path:
        return self.final_render_dir / "render-manifest.json"


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip(".-") or "run"


def command_text(args: list[str]) -> str:
    return " ".join(args)


def run_command(
    label: str,
    args: list[str],
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"\n== {label} ==")
    print(command_text(args))
    result = subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    if capture:
        return result
    return result


def parse_estimated_cost(output: str) -> float:
    patterns = [
        r"estimated upper-bound cost:\s*\$([0-9.]+)",
        r"Total:.*=\s*\$([0-9.]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return float(matches[-1])
    return 0.0


def parse_actual_cost(output: str) -> float | None:
    matches = re.findall(r"Estimated billed cost from usage:\s*\$([0-9.]+)", output)
    if matches:
        return float(matches[-1])
    return None


def parse_cache_hit(output: str) -> bool:
    return "local response cache: hit" in output


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} did not produce expected file: {path}")


def python_module(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def approved_cap_for_stage(estimate_usd: float) -> str:
    return f"{estimate_usd + 0.01:.6f}"


def cache_mode_args(args: argparse.Namespace) -> list[str]:
    return ["--cache-mode", args.cache_mode] if args.cache_mode else []


def response_cache_args(paths: PipelinePaths) -> list[str]:
    return ["--cache-dir", str(paths.response_cache_dir)]


def run_api_step(
    state: PipelineState,
    label: str,
    dry_run_args: list[str],
    actual_args: list[str],
    expected_output: Path,
    *,
    dry_run_flag: str = "--dry-run",
) -> None:
    if expected_output.exists():
        print(f"\n== {label} ==")
        print(f"Skipping existing output: {expected_output}")
        if not state.has_stage(label):
            state.record(label, estimate_usd=0.0, actual_usd=0.0, cache_hit=True)
        return

    dry = run_command(label=f"{label} estimate", args=[*dry_run_args, dry_run_flag], capture=True)
    combined = f"{dry.stdout}\n{dry.stderr}"
    estimate_usd = parse_estimated_cost(combined)
    cache_hit = parse_cache_hit(combined)
    state.require_approval(label, estimate_usd, cache_hit)

    command = list(actual_args)
    if not cache_hit and estimate_usd > 0:
        command.extend(["--approve-api-cost", approved_cap_for_stage(estimate_usd)])
    archive_incomplete_stage_dir(expected_output)
    actual = run_command(label=label, args=command, capture=True)
    actual_combined = f"{actual.stdout}\n{actual.stderr}"
    state.record(
        label,
        estimate_usd=estimate_usd,
        actual_usd=parse_actual_cost(actual_combined),
        cache_hit=cache_hit,
    )
    ensure_file(expected_output, label)


def run_transcription(state: PipelineState, paths: PipelinePaths, args: argparse.Namespace) -> None:
    if paths.transcript_json.exists() and not args.force:
        print(f"\n== transcribe ==")
        print(f"Skipping existing transcript: {paths.transcript_json}")
        if not state.has_stage("transcribe"):
            state.record("transcribe", estimate_usd=0.0, actual_usd=0.0, cache_hit=True)
        return

    base = [
        *python_module("extractor.transcribe"),
        str(paths.video),
        "--env-file",
        str(args.env_file),
        "--output-dir",
        str(paths.transcript_dir),
        "--provider",
        paths.transcribe_provider,
        "--model",
        paths.transcribe_model,
    ]
    actual_base = [*base, *(["--force"] if args.force else [])]
    estimate = run_command("transcribe estimate", [*base, "--estimate-cost"], capture=True)
    combined = f"{estimate.stdout}\n{estimate.stderr}"
    estimate_usd = parse_estimated_cost(combined)
    state.require_approval("transcribe", estimate_usd, cache_hit=False)
    actual = run_command(
        "transcribe",
        [*actual_base, "--approve-api-cost", approved_cap_for_stage(estimate_usd)],
        capture=True,
    )
    state.record("transcribe", estimate_usd=estimate_usd, actual_usd=None, cache_hit=False)
    ensure_file(paths.transcript_json, "transcribe")


def run_local_step(label: str, command: list[str], expected_output: Path, force: bool = False) -> None:
    if expected_output.exists() and not force:
        print(f"\n== {label} ==")
        print(f"Skipping existing output: {expected_output}")
        return
    run_command(label, command)
    ensure_file(expected_output, label)


def paths_match(left: str | None, right: Path) -> bool:
    if not left:
        return False
    try:
        return Path(left).expanduser().resolve() == right.expanduser().resolve()
    except (OSError, TypeError):
        return False


def numbers_match(left: Any, right: float, tolerance: float = 0.000001) -> bool:
    try:
        return abs(float(left) - right) <= tolerance
    except (TypeError, ValueError):
        return False


def manifest_sheet_paths_exist(manifest_path: Path, manifest: dict[str, Any]) -> bool:
    root = manifest_path.parent.parent
    for collection_name in ("periodic_sheets", "candidate_sheets"):
        for sheet in manifest.get(collection_name, []) or []:
            sheet_path = sheet.get("path") if isinstance(sheet, dict) else None
            if not sheet_path:
                continue
            path = Path(str(sheet_path))
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                return False
    return True


def visual_index_rebuild_decision(
    paths: PipelinePaths,
    args: argparse.Namespace,
) -> tuple[bool, bool, str | None]:
    """Return (rebuild, force_frame_extraction, reason)."""
    if args.force:
        return True, True, "--force requested"

    manifest_path = paths.visual_index_manifest
    if not manifest_path.exists():
        return True, True, "missing visual-index manifest"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True, True, "invalid visual-index manifest"

    video_info = manifest.get("video", {}) if isinstance(manifest, dict) else {}
    if not isinstance(video_info, dict) or not paths_match(video_info.get("path"), paths.video):
        return True, True, "source video changed"

    if not paths_match(manifest.get("moments_output"), paths.moments_output):
        return True, False, "moments output path changed"

    if not numbers_match(manifest.get("frame_interval_seconds"), args.frame_interval_seconds):
        return True, True, "frame interval changed"

    if not numbers_match(manifest.get("frame_width"), float(args.frame_width)):
        return True, True, "frame width changed"

    resolved_ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    resolved_hwaccel = resolve_ffmpeg_hwaccel(args.ffmpeg_hwaccel, resolved_ffmpeg_bin) or "none"
    if manifest.get("ffmpeg_bin") != resolved_ffmpeg_bin:
        return True, True, "ffmpeg binary changed"
    if manifest.get("ffmpeg_hwaccel") != resolved_hwaccel:
        return True, True, "ffmpeg hardware decode changed"

    manifest_mtime = manifest_path.stat().st_mtime
    if paths.video.stat().st_mtime > manifest_mtime + 1.0:
        return True, True, "source video is newer than visual-index manifest"
    if paths.moments_output.stat().st_mtime > manifest_mtime + 1.0:
        return True, False, "moments output is newer than visual-index manifest"

    if not manifest_sheet_paths_exist(manifest_path, manifest):
        return True, False, "visual-index sheet image is missing"

    return False, False, None


def write_pipeline_manifest(paths: PipelinePaths, state: PipelineState) -> None:
    # Partitioned by content mode under "runs": a run rewrites only its own mode's
    # entry and preserves the others, so independent ad/excerpt runs in any order
    # never clobber each other's state or cost history. The file stays named
    # pipeline-manifest.json (run-root detection keys off its existence).
    run_payload: dict[str, Any] = {
        "content_mode": paths.content_mode,
        "created_at": datetime.now(UTC).isoformat(),
        "video": str(paths.video),
        "run_root": str(paths.run_root),
        "transcript": str(paths.transcript_json),
        "available_transcript": (
            str(paths.available_transcript_json)
            if paths.available_transcript_json.exists()
            else None
        ),
        "response_cache_dir": str(paths.response_cache_dir),
        "transcription": {
            "provider": paths.transcribe_provider,
            "model": paths.transcribe_model,
        },
        "video_evidence_manifest": str(paths.evidence_manifest),
        "visual_index_manifest": str(paths.visual_index_manifest),
        "subject_index_manifests": {
            "draft": str(paths.draft_subject_index_manifest),
            "final": str(paths.final_subject_index_manifest),
        },
        "stage_outputs": {
            "outline": str(paths.outline_output),
            "strategy": str(paths.strategy_output),
            "moments": str(paths.moments_output),
            "plan": str(paths.plan_output),
            "critique": str(paths.critique_output),
            "revised_plan": str(paths.revised_plan_output),
        },
        "renders": {
            "draft": str(paths.draft_render_manifest),
            "final": str(paths.final_render_manifest),
        },
        "approved_cap_usd": state.approved_cap_usd,
        "costs": [
            {
                "stage": record.stage,
                "estimate_usd": record.estimate_usd,
                "actual_or_reserved_usd": record.actual_or_reserved_usd,
                "cache_hit": record.cache_hit,
            }
            for record in state.cost_records
        ],
        "total_actual_or_reserved_usd": state.spent_or_reserved_usd,
    }
    paths.run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.run_root / "pipeline-manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    runs = existing.get("runs") if isinstance(existing.get("runs"), dict) else {}
    runs[paths.content_mode] = run_payload
    manifest = {
        "schema_version": 2,
        "created_at": existing.get("created_at") or run_payload["created_at"],
        "updated_at": datetime.now(UTC).isoformat(),
        "run_root": str(paths.run_root),
        "runs": runs,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_pipeline_state(paths: PipelinePaths, approved_cap_usd: float | None) -> PipelineState:
    state = PipelineState(approved_cap_usd=approved_cap_usd)
    manifest_path = paths.run_root / "pipeline-manifest.json"
    if not manifest_path.exists():
        return state
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs")
    if isinstance(runs, dict):
        run = runs.get(paths.content_mode) or {}
    else:
        # legacy flat manifest: only adopt its costs if it recorded this same mode.
        run = manifest if manifest.get("content_mode") == paths.content_mode else {}
    for item in run.get("costs", []) or []:
        state.cost_records.append(
            CostRecord(
                stage=str(item.get("stage", "")),
                estimate_usd=float(item.get("estimate_usd") or 0.0),
                actual_or_reserved_usd=float(item.get("actual_or_reserved_usd") or 0.0),
                cache_hit=bool(item.get("cache_hit")),
            )
        )
    return state


def archive_incomplete_stage_dir(expected_output: Path) -> None:
    run_dir = expected_output.parent
    if not run_dir.exists() or expected_output.exists():
        return
    archive_dir = run_dir.with_name(f"{run_dir.name}.incomplete-{utc_timestamp()}")
    shutil.move(str(run_dir), str(archive_dir))
    print(f"Archived incomplete stage directory: {archive_dir}")


def effective_duration_from_manifest(path: Path) -> float | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    media = manifest.get("media", {}) if isinstance(manifest, dict) else {}
    value = media.get("effective_duration_seconds") or media.get("duration_seconds")
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def transcript_source_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("source", {}) if isinstance(data, dict) else {}
    value = source.get("effective_duration_seconds") or source.get("duration_seconds")
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def clip_transcript_for_duration(source_path: Path, output_path: Path, duration: float) -> Path:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []
    for segment in data.get("segments", []) or []:
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or 0)
        if start >= duration:
            continue
        copied = dict(segment)
        copied["end"] = round(min(end, duration), 3)
        if isinstance(copied.get("words"), list):
            copied["words"] = [
                dict(word)
                for word in copied["words"]
                if word.get("start") is not None
                and word.get("end") is not None
                and float(word["start"]) < duration
                and float(word["end"]) <= duration + 0.05
            ]
        if copied["end"] > start:
            segments.append(copied)

    words: list[dict[str, Any]] = []
    for word in data.get("words", []) or []:
        start_value = word.get("start")
        end_value = word.get("end")
        if start_value is None or end_value is None:
            continue
        start = float(start_value)
        end = float(end_value)
        if start >= duration or end > duration + 0.05:
            continue
        words.append(dict(word))

    clipped = dict(data)
    source = dict(clipped.get("source", {}) or {})
    original_duration = source.get("effective_duration_seconds") or source.get("duration_seconds")
    source["original_duration_seconds"] = original_duration
    source["duration_seconds"] = round(duration, 3)
    source["effective_duration_seconds"] = round(duration, 3)
    source["clipped_to_video_duration"] = True
    clipped["source"] = source
    clipped["segments"] = segments
    clipped["words"] = words
    clipped["text"] = " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()
    output_path.write_text(json.dumps(clipped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_srt(output_path.with_suffix(".srt"), segments)
    return output_path


def transcript_for_api_stages(paths: PipelinePaths) -> Path:
    effective_duration = effective_duration_from_manifest(paths.evidence_manifest)
    source_duration = transcript_source_duration(paths.transcript_json)
    if (
        effective_duration is None
        or source_duration is None
        or source_duration <= effective_duration + 0.5
    ):
        return paths.transcript_json

    clipped_path = paths.available_transcript_json
    if not clipped_path.exists() or clipped_path.stat().st_mtime < paths.transcript_json.stat().st_mtime:
        clipped_path.parent.mkdir(parents=True, exist_ok=True)
        clip_transcript_for_duration(paths.transcript_json, clipped_path, effective_duration)
        print(
            f"Clipped transcript for API stages to {effective_duration:.3f}s: "
            f"{clipped_path}"
        )
    return clipped_path


def excerpt_clip_count_for_duration(duration_seconds: float | None) -> int:
    if duration_seconds is None or duration_seconds <= 0:
        return 1
    return max(1, int(math.ceil(duration_seconds / EXCERPT_SECONDS_PER_CLIP)))


def plan_clip_count(paths: PipelinePaths) -> int:
    if paths.content_mode != "excerpt":
        return 3
    return excerpt_clip_count_for_duration(effective_duration_from_manifest(paths.evidence_manifest))


def plan_max_output_tokens(clip_count: int) -> int:
    return min(
        PLAN_MAX_OUTPUT_TOKENS,
        max(PLAN_DEFAULT_MAX_OUTPUT_TOKENS, clip_count * PLAN_OUTPUT_TOKENS_PER_EXTRACT),
    )


def run_pipeline(args: argparse.Namespace) -> PipelinePaths:
    input_video = args.video.expanduser().resolve()
    if not input_video.exists():
        raise FileNotFoundError(input_video)

    load_env_file(args.env_file)
    transcribe_provider = resolve_transcribe_provider(args.transcribe_provider)
    transcribe_model = resolve_transcribe_model(transcribe_provider, args.transcribe_model)

    # Source-centric project layout: the run root is a project dir named for the
    # source, and the master is pulled into <project>/media/source.mp4 so the
    # project is self-contained on its own disk. Re-running on an already-
    # established project (media/source.mp4) or with --output-root reuses it.
    original_master = input_video.name
    if args.output_root:
        project_dir = args.output_root.expanduser().resolve()
        video = establish_media(project_dir, input_video)
    elif input_video.name == "source.mp4" and input_video.parent.name == "media":
        project_dir = input_video.parent.parent
        video = input_video
    else:
        base = args.library.expanduser().resolve() if args.library else input_video.parent
        project_dir = base / source_slug(input_video)
        video = establish_media(project_dir, input_video)
    # run_id naming uses the human project name; but the per-stage output subdirs
    # (video-evidence/<stem>/, visual-index/<stem>/, transcripts/<stem>.*) must use
    # the *video's* stem, because preprocess/visual_index/transcribe derive their
    # output paths from the media filename. Under the source-centric layout the
    # master is always media/source.mp4, so that stem is "source" — keeping the two
    # separate is what lets the pipeline find the files the tools actually write.
    stem = project_dir.name
    path_stem = safe_stem(video)

    # Every mode names itself, in run ids and under _renders/ alike: "ad" is a
    # content mode like any other, not the unnamed default. Two modes of the same
    # source therefore sit side by side under identical rules.
    run_id_parts = [stem, args.content_mode]
    if args.run_id is not None:
        run_id_parts.append(safe_run_id(args.run_id))
    run_id_base = "-".join(run_id_parts)
    run_root = project_dir
    paths = PipelinePaths(
        video=video,
        run_root=run_root,
        stem=path_stem,
        run_id_base=run_id_base,
        content_mode=args.content_mode,
        transcribe_provider=transcribe_provider,
        transcribe_model=transcribe_model,
        render_variant=safe_run_id(args.run_id) if args.run_id else None,
    )
    paths.run_root.mkdir(parents=True, exist_ok=True)
    state = load_pipeline_state(paths, approved_cap_usd=args.approve_api_cost)

    try:
        run_transcription(state, paths, args)

        run_local_step(
            "preprocess",
            [
                *python_module("extractor.preprocess_video"),
                str(paths.video),
                "--transcript",
                str(paths.transcript_json),
                "--output-dir",
                str(paths.video_evidence_dir),
                "--frame-interval-seconds",
                str(args.frame_interval_seconds),
                "--frame-width",
                str(args.frame_width),
                "--contact-sheet-frame-width",
                str(args.contact_sheet_frame_width),
                "--contact-sheet-columns",
                str(args.contact_sheet_columns),
                "--window-sheet-seconds",
                str(args.window_sheet_seconds),
                "--window-sheet-frame-width",
                str(args.window_sheet_frame_width),
                "--window-sheet-columns",
                str(args.window_sheet_columns),
                "--ffmpeg-bin",
                args.ffmpeg_bin,
                "--ffmpeg-hwaccel",
                args.ffmpeg_hwaccel,
                "--write-markdown",
            ],
            paths.evidence_manifest,
            force=args.force,
        )
        stage_transcript = transcript_for_api_stages(paths)

        # Content outline: label every source unit (title + one-line gist) so a
        # human editor can see the building blocks without reading the transcript
        # or watching the video. Runs early; the storyboard review UX reads it.
        outline = [
            *python_module("extractor.openai_outline"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(stage_transcript),
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.outline_run_id,
            *response_cache_args(paths),
        ]
        run_api_step(state, "outline", outline, outline, paths.outline_output)

        plan_count = plan_clip_count(paths)
        plan_count_args = (
            [
                "--clip-count",
                str(plan_count),
                "--max-output-tokens",
                str(plan_max_output_tokens(plan_count)),
            ]
            if paths.content_mode == "excerpt"
            else []
        )
        augmentation_assets_args = (
            ["--augmentation-assets", str(args.augmentation_assets.expanduser().resolve())]
            if args.augmentation_assets
            else []
        )

        strategy = [
            *python_module("extractor.openai_strategy"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(stage_transcript),
            "--manifest",
            str(paths.evidence_manifest),
            "--contact-sheet",
            str(paths.contact_sheet),
            "--content-mode",
            args.content_mode,
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.strategy_run_id,
            *response_cache_args(paths),
            *cache_mode_args(args),
        ]
        run_api_step(state, "strategy", strategy, strategy, paths.strategy_output)

        moments = [
            *python_module("extractor.openai_moments"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(stage_transcript),
            "--manifest",
            str(paths.evidence_manifest),
            "--contact-sheet",
            str(paths.contact_sheet),
            "--strategy-output",
            str(paths.strategy_output),
            "--content-mode",
            args.content_mode,
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.moments_run_id,
            "--max-output-tokens",
            "24000",
            *response_cache_args(paths),
            *cache_mode_args(args),
        ]
        run_api_step(state, "moments", moments, moments, paths.moments_output)

        visual_index_command = [
            *python_module("extractor.visual_index"),
            str(paths.video),
            "--output-dir",
            str(paths.visual_index_dir),
            "--moments-output",
            str(paths.moments_output),
            "--frame-interval-seconds",
            str(args.frame_interval_seconds),
            "--frame-width",
            str(args.frame_width),
            "--ffmpeg-bin",
            args.ffmpeg_bin,
            "--ffmpeg-hwaccel",
            args.ffmpeg_hwaccel,
            "--write-markdown",
        ]
        rebuild_visual_index, force_visual_index_frames, rebuild_reason = visual_index_rebuild_decision(
            paths,
            args,
        )
        if force_visual_index_frames:
            visual_index_command.append("--force")
        if rebuild_reason:
            print(f"\n== visual-index ==")
            print(f"Rebuilding visual index: {rebuild_reason}")
        run_local_step(
            "visual-index",
            visual_index_command,
            paths.visual_index_manifest,
            force=rebuild_visual_index,
        )

        plan = [
            *python_module("extractor.openai_plan"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(stage_transcript),
            "--manifest",
            str(paths.evidence_manifest),
            "--contact-sheet",
            str(paths.contact_sheet),
            "--visual-index-manifest",
            str(paths.visual_index_manifest),
            "--strategy-output",
            str(paths.strategy_output),
            "--moments-output",
            str(paths.moments_output),
            "--content-mode",
            args.content_mode,
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.plan_run_id,
            *plan_count_args,
            *augmentation_assets_args,
            *response_cache_args(paths),
            *cache_mode_args(args),
        ]
        run_api_step(state, "plan", plan, plan, paths.plan_output)

        run_local_step(
            "subject-index-draft",
            [
                *python_module("extractor.subject_index"),
                str(paths.video),
                "--plan-output",
                str(paths.plan_output),
                "--output-dir",
                str(paths.subject_index_dir),
                "--run-id",
                paths.plan_run_id,
                "--ffmpeg-bin",
                args.ffmpeg_bin,
                "--ffmpeg-hwaccel",
                args.ffmpeg_hwaccel,
            ],
            paths.draft_subject_index_manifest,
            force=args.force,
        )

        run_local_step(
            "draft-render",
            [
                *python_module("extractor.render_plan"),
                "--source-video",
                str(paths.video),
                "--plan-output",
                str(paths.plan_output),
                "--transcript",
                str(stage_transcript),
                "--output-dir",
                str(paths.draft_render_dir),
                "--use-plan-render-settings",
                "--auto-foreground-scale",
                "--subject-index-manifest",
                str(paths.draft_subject_index_manifest),
                *augmentation_assets_args,
                "--ffmpeg-bin",
                args.ffmpeg_bin,
                "--ffmpeg-hwaccel",
                args.ffmpeg_hwaccel,
            ],
            paths.draft_render_manifest,
            force=args.force,
        )

        critique = [
            *python_module("extractor.openai_critique"),
            "--env-file",
            str(args.env_file),
            "--render-manifest",
            str(paths.draft_render_manifest),
            "--plan-output",
            str(paths.plan_output),
            "--content-mode",
            args.content_mode,
            "--output-dir",
            str(paths.openai_runs_dir),
            "--assets-dir",
            str(paths.openai_runs_dir / "_critique-assets" / paths.critique_run_id),
            "--run-id",
            paths.critique_run_id,
            *response_cache_args(paths),
            *cache_mode_args(args),
        ]
        run_api_step(state, "critique", critique, critique, paths.critique_output)

        revise = [
            *python_module("extractor.openai_revise"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(stage_transcript),
            "--manifest",
            str(paths.evidence_manifest),
            "--contact-sheet",
            str(paths.contact_sheet),
            "--plan-output",
            str(paths.plan_output),
            "--render-manifest",
            str(paths.draft_render_manifest),
            "--critique-output",
            str(paths.critique_output),
            "--critique-package",
            str(paths.critique_package),
            "--content-mode",
            args.content_mode,
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.revise_run_id,
            *response_cache_args(paths),
            *cache_mode_args(args),
        ]
        run_api_step(state, "revise", revise, revise, paths.revised_plan_output)

        run_local_step(
            "subject-index-final",
            [
                *python_module("extractor.subject_index"),
                str(paths.video),
                "--plan-output",
                str(paths.revised_plan_output),
                "--output-dir",
                str(paths.subject_index_dir),
                "--run-id",
                paths.revise_run_id,
                "--ffmpeg-bin",
                args.ffmpeg_bin,
                "--ffmpeg-hwaccel",
                args.ffmpeg_hwaccel,
            ],
            paths.final_subject_index_manifest,
            force=args.force,
        )

        run_local_step(
            "render-final",
            [
                *python_module("extractor.render_plan"),
                "--source-video",
                str(paths.video),
                "--plan-output",
                str(paths.revised_plan_output),
                "--transcript",
                str(stage_transcript),
                "--output-dir",
                str(paths.final_render_dir),
                "--output-label",
                "",
                "--use-plan-render-settings",
                "--auto-foreground-scale",
                "--subject-index-manifest",
                str(paths.final_subject_index_manifest),
                *augmentation_assets_args,
                "--ffmpeg-bin",
                args.ffmpeg_bin,
                "--ffmpeg-hwaccel",
                args.ffmpeg_hwaccel,
            ],
            paths.final_render_manifest,
            force=args.force,
        )
    finally:
        write_pipeline_manifest(paths, state)

    # Human-facing overlay: <excerpts|ads>/NN-title-slug/ + project.json, identical
    # to what the archive migrator produces, so new and migrated projects match.
    try:
        rm = (paths.final_render_manifest if paths.final_render_manifest.exists()
              else paths.draft_render_manifest)
        plan_path = (paths.revised_plan_output if paths.revised_plan_output.exists()
                     else paths.plan_output)
        if plan_path.exists():
            clips = generate_layout(
                paths.run_root, rm if rm.exists() else None,
                json.loads(plan_path.read_text()), paths.run_root.name,
                content_mode=args.content_mode,
                original_master=original_master, promoted_from=paths.run_root.name)
            clips_dir = paths.run_root / f"{args.content_mode}s"
            print(f"Clips: {len(clips)} -> {clips_dir}")
    except Exception as exc:
        print(f"note: project layout overlay skipped ({type(exc).__name__}: {exc})")

    print("\n== pipeline complete ==")
    print(f"Run root: {paths.run_root}")
    print(f"Final render manifest: {paths.final_render_manifest}")
    print(f"Total actual/reserved API cost: ${state.spent_or_reserved_usd:.6f}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the self-contained video-to-clips pipeline from one input video."
    )
    parser.add_argument("video", type=Path, help="Input video file. This is the only required input.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--content-mode",
        choices=CONTENT_MODE_CHOICES,
        default="ad",
        help=(
            "Pipeline creative mode. ad sells the longer video; excerpt creates "
            "self-contained extracts from the longer video. Excerpts may be atomic "
            "moments or longer chapter extracts when that is the smallest coherent unit."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Explicit project directory. Overrides --library. The master is pulled "
            "into <output-root>/media/source.mp4 and excerpts|ads/ + project.json "
            "are generated there."
        ),
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help=(
            "Library folder to create the project in, as <library>/<source-slug>/. "
            "Defaults to the input video's own directory."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id suffix for named variants. Defaults to per-file stage dirs.",
    )
    parser.add_argument(
        "--augmentation-assets",
        type=Path,
        default=None,
        help=(
            "Optional JSON catalog of local, rights-reviewed assets. Stage 3 may "
            "reference assets as asset://id, and the renderer will resolve them."
        ),
    )
    parser.add_argument(
        "--approve-api-cost",
        type=float,
        default=None,
        metavar="USD",
        help="Explicit total approval cap for all API calls made by this pipeline run.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=stage_cache.CACHE_MODE_CHOICES,
        default=None,
        help=(
            "Forward a local response cache identity mode to OpenAI stages. "
            "Defaults to each stage's env setting when omitted."
        ),
    )
    parser.add_argument(
        "--transcribe-provider",
        choices=TRANSCRIBE_PROVIDER_CHOICES,
        default=None,
        help="Transcription provider for the first pipeline stage.",
    )
    parser.add_argument(
        "--transcribe-model",
        default=None,
        help="Override the transcription model for the selected provider.",
    )
    parser.add_argument(
        "--frame-interval-seconds",
        type=float,
        default=DEFAULT_FRAME_INTERVAL_SECONDS,
        help="Seconds between sampled local evidence and visual-index frames.",
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
        help="Downscaled width for each preprocessing contact-sheet cell.",
    )
    parser.add_argument(
        "--contact-sheet-columns",
        type=int,
        default=DEFAULT_CONTACT_SHEET_COLUMNS,
        help="Number of columns in the preprocessing overview contact sheet.",
    )
    parser.add_argument(
        "--window-sheet-seconds",
        type=float,
        default=DEFAULT_WINDOW_SHEET_SECONDS,
        help="Seconds covered by each labeled preprocessing window sheet.",
    )
    parser.add_argument(
        "--window-sheet-frame-width",
        type=int,
        default=DEFAULT_WINDOW_SHEET_FRAME_WIDTH,
        help="Downscaled width for each cell in labeled preprocessing window sheets.",
    )
    parser.add_argument(
        "--window-sheet-columns",
        type=int,
        default=DEFAULT_WINDOW_SHEET_COLUMNS,
        help="Number of columns in labeled preprocessing window sheets.",
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
        help="Hardware decode mode for local video frame extraction and rendering.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate local artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_pipeline(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
