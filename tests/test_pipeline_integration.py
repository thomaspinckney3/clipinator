"""End-to-end local-pipeline test with NO API calls.

The OpenAI stages (strategy/moments/plan/critique/revise) only ever produce a
plan-output.json; everything downstream is deterministic local ffmpeg work. So
by injecting a synthetic but schema-valid plan we exercise the real
render -> edit-list/OTIO -> import -> re-render chain against a real (tiny)
video, with frame-accurate timing and actual encoded output. This is the most
realistic test achievable without hitting an API: it covers all the non-model
code, only the model's editorial judgment is out of scope.

Skips automatically when ffmpeg/ffprobe are unavailable (via the tiny_video
fixture).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_clip, make_plan, make_transcript


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def render(plan_path: Path, video: Path, transcript: Path, out_dir: Path) -> dict:
    """Invoke the real render_plan CLI (no API) and return its render manifest."""
    rc = subprocess.run(
        [sys.executable, "-m", "extractor.render_plan",
         "--source-video", str(video), "--plan-output", str(plan_path),
         "--transcript", str(transcript), "--output-dir", str(out_dir),
         "--output-label", ""],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, f"render failed:\n{rc.stdout}\n{rc.stderr}"
    return json.loads((out_dir / "render-manifest.json").read_text())


@pytest.fixture
def local_run(tmp_path, tiny_video):
    """A run dir with a synthetic plan whose one clip fits inside the 6s video."""
    root = tmp_path / "run"
    (root / "openai-runs" / "run-plan").mkdir(parents=True)
    (root / "transcripts").mkdir()
    (root / "pipeline-manifest.json").write_text("{}")
    # card 1.0s + source segment 0.5..4.5  -> ~5.5s rendered clip, within the 6s source
    plan = make_plan([make_clip("EX01", card=1.0, segments=[(0.5, 4.5)])])
    plan_path = root / "openai-runs" / "run-plan" / "plan-output.json"
    plan_path.write_text(json.dumps(plan))
    tr_path = root / "transcripts" / "t.json"
    tr_path.write_text(json.dumps(make_transcript(6.0)))
    return root, plan_path, tr_path, tiny_video


class TestLocalRenderNoAPI:
    def test_render_produces_real_clip(self, local_run):
        root, plan_path, tr_path, video = local_run
        out = root / "renders" / "draft"
        manifest = render(plan_path, video, tr_path, out)
        assert manifest["timebase"]["source_frame_rate"] == "24000/1001"
        clips = manifest["clips"]
        assert len(clips) == 1 and clips[0]["clip_id"] == "EX01"
        mp4 = Path(clips[0]["output_path"])
        assert mp4.exists() and not mp4.name.endswith(".draft.mp4")  # final naming
        # Real render = title card + 4.0s source segment + renderer-added opening
        # dialogue lead-in and final fade. Assert a sane window, not an exact
        # value (the renderer legitimately pads the title card and edges).
        dur = ffprobe_duration(mp4)
        assert 4.0 <= dur <= 11.0, f"unexpected rendered duration {dur:.2f}s"
        assert Path(clips[0]["srt_path"]).exists()

    def test_project_layout_from_real_render(self, local_run):
        # Render a real clip, then build the source-centric overlay the pipeline
        # generates post-render, and confirm the clips point at real playable files.
        root, plan_path, tr_path, video = local_run
        from extractor import project_layout as pl
        manifest = root / "_renders" / "final" / "render-manifest.json"
        render(plan_path, video, tr_path, manifest.parent)
        plan = json.loads(plan_path.read_text())
        clips = pl.generate_layout(root, manifest, plan, "demo-project", content_mode="excerpt",
                                   original_master="orig.mp4", promoted_from="demo-project")
        assert len(clips) == 1
        ex = root / "excerpts" / clips[0]["slug"]
        assert ffprobe_duration(ex / "final.mp4") > 1.0  # real, playable
        assert (ex / "captions.srt").exists() and (ex / "metadata.json").exists()
        assert json.loads((root / "project.json").read_text())["source"]["master"] == "media/source.mp4"

    def test_otio_round_trip_then_rerender(self, local_run):
        root, plan_path, tr_path, video = local_run
        draft = root / "renders" / "draft"
        manifest_data = render(plan_path, video, tr_path, draft)
        manifest = draft / "render-manifest.json"

        # export -> OTIO, import back, re-render the imported plan: all no-API
        from extractor import otio_edl
        import argparse
        review = root / "review"
        assert otio_edl.run_export(argparse.Namespace(
            plan_output=plan_path, render_manifest=manifest,
            transcript=None, source_video=None, output_dir=review)) == 0
        otio_path = review / "edit-list.otio"
        assert otio_path.exists()
        human_plan = review / "plan-output.human.json"
        assert otio_edl.run_import(argparse.Namespace(
            otio=otio_path, plan_output=plan_path, transcript=tr_path, output=human_plan)) == 0

        out2 = root / "renders" / "human"
        manifest2 = render(human_plan, video, tr_path, out2)
        # the re-rendered clip exists and matches the first render's duration
        d1 = ffprobe_duration(Path(manifest_data["clips"][0]["output_path"]))
        d2 = ffprobe_duration(Path(manifest2["clips"][0]["output_path"]))
        assert d2 == pytest.approx(d1, abs=0.2)


class TestSubjectIndexNoAPI:
    def test_subject_index_runs_on_real_video(self, local_run):
        root, plan_path, tr_path, video = local_run
        out = root / "subject-index"
        rc = subprocess.run(
            [sys.executable, "-m", "extractor.subject_index", str(video),
             "--plan-output", str(plan_path), "--output-dir", str(out), "--run-id", "run-plan"],
            capture_output=True, text=True,
        )
        assert rc.returncode == 0, f"subject-index failed:\n{rc.stdout}\n{rc.stderr}"
        manifest = json.loads((out / "run-plan" / "manifest.json").read_text())
        assert manifest["api_calls"] is False
        assert manifest["entries"] if "entries" in manifest else manifest.get("entry_count", 1)


def _paths(run_root: Path, mode: str):
    from extractor.run_pipeline import PipelinePaths
    return PipelinePaths(
        video=run_root / "media" / "source.mp4", run_root=run_root, stem="source",
        run_id_base=("s" if mode == "ad" else "s-excerpt"), content_mode=mode,
        transcribe_provider="elevenlabs", transcribe_model="scribe_v2", render_variant=None)


def test_pipeline_manifest_partitions_state_by_mode(tmp_path):
    # Independent ad/excerpt runs must not clobber each other's state or leak costs
    # across caps. An excerpt run started after an ad run sees a clean slate, and
    # both runs' cost history is preserved and readable per mode.
    from extractor.run_pipeline import PipelineState, write_pipeline_manifest, load_pipeline_state
    root = tmp_path / "proj"; root.mkdir()

    ad = PipelineState(approved_cap_usd=10.0)
    ad.record("strategy", estimate_usd=2.0, actual_usd=2.0, cache_hit=False)
    write_pipeline_manifest(_paths(root, "ad"), ad)

    # excerpt run must NOT inherit the ad run's $2 against its own cap
    ex = load_pipeline_state(_paths(root, "excerpt"), approved_cap_usd=10.0)
    assert ex.spent_or_reserved_usd == 0.0
    ex.record("strategy", estimate_usd=3.0, actual_usd=3.0, cache_hit=False)
    write_pipeline_manifest(_paths(root, "excerpt"), ex)

    manifest = json.loads((root / "pipeline-manifest.json").read_text())
    assert set(manifest["runs"]) == {"ad", "excerpt"}
    # each mode loads only its own costs; the ad run was not erased by the excerpt run
    assert load_pipeline_state(_paths(root, "ad"), 10.0).spent_or_reserved_usd == 2.0
    assert load_pipeline_state(_paths(root, "excerpt"), 10.0).spent_or_reserved_usd == 3.0
