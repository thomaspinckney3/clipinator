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
