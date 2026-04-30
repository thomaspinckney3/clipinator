from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.preprocess_video import safe_stem
from extractor.transcribe import load_env_file


DEFAULT_ENV_FILE = Path(".env")
DEFAULT_TRANSCRIBE_MODEL = "whisper-1"
DEFAULT_FRAME_INTERVAL_SECONDS = 1.0


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

    @property
    def transcript_dir(self) -> Path:
        return self.run_root / "transcripts"

    @property
    def transcript_json(self) -> Path:
        return self.transcript_dir / f"{self.stem}.{DEFAULT_TRANSCRIBE_MODEL}.json"

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
    def visual_index_dir(self) -> Path:
        return self.run_root / "visual-index"

    @property
    def visual_index_manifest(self) -> Path:
        return self.visual_index_dir / self.stem / "manifest.json"

    @property
    def stage1_run_id(self) -> str:
        return f"{self.run_id_base}-stage1"

    @property
    def stage2_run_id(self) -> str:
        return f"{self.run_id_base}-stage2"

    @property
    def stage3_run_id(self) -> str:
        return f"{self.run_id_base}-stage3"

    @property
    def stage4_run_id(self) -> str:
        return f"{self.run_id_base}-stage4"

    @property
    def stage5_run_id(self) -> str:
        return f"{self.run_id_base}-stage5"

    @property
    def stage1_output(self) -> Path:
        return self.openai_runs_dir / self.stage1_run_id / "stage1-output.json"

    @property
    def stage2_output(self) -> Path:
        return self.openai_runs_dir / self.stage2_run_id / "stage2-output.json"

    @property
    def stage3_output(self) -> Path:
        return self.openai_runs_dir / self.stage3_run_id / "stage3-output.json"

    @property
    def stage4_output(self) -> Path:
        return self.openai_runs_dir / self.stage4_run_id / "stage4-output.json"

    @property
    def stage4_review_package(self) -> Path:
        return self.openai_runs_dir / self.stage4_run_id / "review-package.json"

    @property
    def stage5_stage3_output(self) -> Path:
        return self.openai_runs_dir / self.stage5_run_id / "stage3-output.json"

    @property
    def stage3_render_dir(self) -> Path:
        return self.run_root / "renders" / "stage3"

    @property
    def stage3_render_manifest(self) -> Path:
        return self.stage3_render_dir / "render-manifest.json"

    @property
    def final_render_dir(self) -> Path:
        return self.run_root / "renders" / "stage5"

    @property
    def final_render_manifest(self) -> Path:
        return self.final_render_dir / "render-manifest.json"


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def default_run_root(video: Path, timestamp: str) -> Path:
    return video.parent / f"full-pipeline-{safe_stem(video)}-{timestamp}"


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
        "--model",
        DEFAULT_TRANSCRIBE_MODEL,
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


def write_pipeline_manifest(paths: PipelinePaths, state: PipelineState) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "video": str(paths.video),
        "run_root": str(paths.run_root),
        "transcript": str(paths.transcript_json),
        "video_evidence_manifest": str(paths.evidence_manifest),
        "visual_index_manifest": str(paths.visual_index_manifest),
        "stage_outputs": {
            "stage1": str(paths.stage1_output),
            "stage2": str(paths.stage2_output),
            "stage3": str(paths.stage3_output),
            "stage4": str(paths.stage4_output),
            "stage5_stage3": str(paths.stage5_stage3_output),
        },
        "renders": {
            "stage3": str(paths.stage3_render_manifest),
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
    (paths.run_root / "pipeline-manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_pipeline_state(paths: PipelinePaths, approved_cap_usd: float | None) -> PipelineState:
    state = PipelineState(approved_cap_usd=approved_cap_usd)
    manifest_path = paths.run_root / "pipeline-manifest.json"
    if not manifest_path.exists():
        return state
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("costs", []) or []:
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


def run_pipeline(args: argparse.Namespace) -> PipelinePaths:
    video = args.video.expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(video)

    load_env_file(args.env_file)
    timestamp = args.run_id or utc_timestamp()
    stem = safe_stem(video)
    run_root = (args.output_root.expanduser().resolve() if args.output_root else default_run_root(video, timestamp))
    paths = PipelinePaths(
        video=video,
        run_root=run_root,
        stem=stem,
        run_id_base=f"{stem}-{timestamp}",
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
                "--write-markdown",
            ],
            paths.evidence_manifest,
            force=args.force,
        )

        stage1 = [
            *python_module("extractor.openai_stage1"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(paths.transcript_json),
            "--manifest",
            str(paths.evidence_manifest),
            "--contact-sheet",
            str(paths.contact_sheet),
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.stage1_run_id,
        ]
        run_api_step(state, "stage1", stage1, stage1, paths.stage1_output)

        stage2 = [
            *python_module("extractor.openai_stage2"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(paths.transcript_json),
            "--manifest",
            str(paths.evidence_manifest),
            "--contact-sheet",
            str(paths.contact_sheet),
            "--stage1-output",
            str(paths.stage1_output),
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.stage2_run_id,
            "--max-output-tokens",
            "24000",
        ]
        run_api_step(state, "stage2", stage2, stage2, paths.stage2_output)

        run_local_step(
            "visual-index",
            [
                *python_module("extractor.visual_index"),
                str(paths.video),
                "--output-dir",
                str(paths.visual_index_dir),
                "--stage2-output",
                str(paths.stage2_output),
                "--frame-interval-seconds",
                str(args.frame_interval_seconds),
                "--write-markdown",
                "--force",
            ],
            paths.visual_index_manifest,
            force=True,
        )

        stage3 = [
            *python_module("extractor.openai_stage3"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(paths.transcript_json),
            "--manifest",
            str(paths.evidence_manifest),
            "--contact-sheet",
            str(paths.contact_sheet),
            "--visual-index-manifest",
            str(paths.visual_index_manifest),
            "--stage1-output",
            str(paths.stage1_output),
            "--stage2-output",
            str(paths.stage2_output),
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.stage3_run_id,
        ]
        run_api_step(state, "stage3", stage3, stage3, paths.stage3_output)

        run_local_step(
            "render-stage3",
            [
                *python_module("extractor.render_stage3"),
                "--source-video",
                str(paths.video),
                "--stage3-output",
                str(paths.stage3_output),
                "--output-dir",
                str(paths.stage3_render_dir),
                "--use-plan-render-settings",
            ],
            paths.stage3_render_manifest,
            force=args.force,
        )

        stage4 = [
            *python_module("extractor.openai_stage4_review"),
            "--env-file",
            str(args.env_file),
            "--render-manifest",
            str(paths.stage3_render_manifest),
            "--stage3-output",
            str(paths.stage3_output),
            "--output-dir",
            str(paths.openai_runs_dir),
            "--assets-dir",
            str(paths.openai_runs_dir / "_stage4-review-assets" / paths.stage4_run_id),
            "--run-id",
            paths.stage4_run_id,
        ]
        run_api_step(state, "stage4", stage4, stage4, paths.stage4_output)

        stage5 = [
            *python_module("extractor.openai_stage5_reedit"),
            "--env-file",
            str(args.env_file),
            "--transcript",
            str(paths.transcript_json),
            "--manifest",
            str(paths.evidence_manifest),
            "--contact-sheet",
            str(paths.contact_sheet),
            "--stage3-output",
            str(paths.stage3_output),
            "--render-manifest",
            str(paths.stage3_render_manifest),
            "--stage4-output",
            str(paths.stage4_output),
            "--review-package",
            str(paths.stage4_review_package),
            "--output-dir",
            str(paths.openai_runs_dir),
            "--run-id",
            paths.stage5_run_id,
        ]
        run_api_step(state, "stage5", stage5, stage5, paths.stage5_stage3_output)

        run_local_step(
            "render-final",
            [
                *python_module("extractor.render_stage3"),
                "--source-video",
                str(paths.video),
                "--stage3-output",
                str(paths.stage5_stage3_output),
                "--output-dir",
                str(paths.final_render_dir),
                "--use-plan-render-settings",
            ],
            paths.final_render_manifest,
            force=args.force,
        )
    finally:
        write_pipeline_manifest(paths, state)

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
        "--output-root",
        type=Path,
        default=None,
        help="Optional run directory. Defaults next to the input video.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id suffix. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--approve-api-cost",
        type=float,
        default=None,
        metavar="USD",
        help="Explicit total approval cap for all API calls made by this pipeline run.",
    )
    parser.add_argument(
        "--frame-interval-seconds",
        type=float,
        default=DEFAULT_FRAME_INTERVAL_SECONDS,
        help="Dense visual-index frame interval.",
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
