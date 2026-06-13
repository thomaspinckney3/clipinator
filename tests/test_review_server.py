import json
from pathlib import Path

from extractor import review_server as rs

from conftest import make_plan, make_clip, make_transcript


def build_run_root(root: Path, *, variant: str = "draft", mode: str = "excerpt") -> Path:
    """A standard run-root layout the server should discover."""
    (root / "openai-runs" / "run-plan").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(exist_ok=True)
    (root / "pipeline-manifest.json").write_text("{}")
    source = root / "source.mp4"
    source.write_bytes(b"\x00" * 1024)
    plan = make_plan([make_clip("EX01", card=1.0, segments=[(0.5, 4.5)])])
    plan_path = root / "openai-runs" / "run-plan" / "plan-output.json"
    plan_path.write_text(json.dumps(plan))
    tr = root / "transcripts" / "t.json"
    tr.write_text(json.dumps(make_transcript(6.0)))
    rdir = root / "renders" / mode / variant
    rdir.mkdir(parents=True, exist_ok=True)
    clip_mp4 = rdir / "EX01.mp4"
    clip_mp4.write_bytes(b"\x00" * 512)
    (rdir / "render-manifest.json").write_text(json.dumps({
        "source_video": str(source),
        "transcript": str(tr),
        "stage3_output": str(plan_path),
        "content_mode": mode,
        "clips": [{"clip_id": "EX01", "output_path": str(clip_mp4)}],
    }))
    return rdir / "render-manifest.json"


class TestDiscovery:
    def test_finds_project_in_library(self, tmp_path):
        lib = tmp_path / "lib"
        build_run_root(lib / "My Project")
        projects, allowed = rs.discover_projects(lib)
        assert len(projects) == 1
        assert projects[0]["clip_count"] == 1
        assert projects[0]["source_found"] is True
        # source + clip mp4 are allowlisted for /video
        assert any(p.endswith("source.mp4") for p in allowed)
        assert any(p.endswith("EX01.mp4") for p in allowed)

    def test_label_uses_run_root_name(self, tmp_path):
        # whether we scan the library or point at the run root, the label uses
        # the run-root folder name + variant (not "renders").
        lib = tmp_path / "lib"
        manifest = build_run_root(lib / "My Project", variant="final")
        from_lib = rs.discover_projects(lib)[0][0]["label"]
        from_proj = rs.discover_projects(lib / "My Project")[0][0]["label"]
        assert from_lib == "My Project — final"
        assert from_proj == "My Project — final"

    def test_finds_multiple_variants(self, tmp_path):
        root = tmp_path / "lib" / "Proj"
        build_run_root(root, variant="draft")
        build_run_root(root, variant="final")
        projects, _ = rs.discover_projects(root)
        labels = sorted(p["label"] for p in projects)
        assert labels == ["Proj — draft", "Proj — final"]

    def test_missing_dir_is_empty(self, tmp_path):
        projects, allowed = rs.discover_projects(tmp_path / "nope")
        assert projects == [] and allowed == set()

    def test_skips_heavy_dirs(self, tmp_path):
        lib = tmp_path / "lib"
        build_run_root(lib / "Proj")
        (lib / "node_modules").mkdir()
        # a stray manifest inside a skipped dir must not be discovered
        bad = lib / "node_modules" / "renders" / "draft"
        bad.mkdir(parents=True)
        (bad / "render-manifest.json").write_text("{}")
        projects, _ = rs.discover_projects(lib)
        assert all("node_modules" not in p["manifest_path"] for p in projects)
        assert [p["label"] for p in projects] == ["Proj — draft"]


class TestProjectDetail:
    def test_resolves_clips_and_captions(self, tmp_path):
        manifest = build_run_root(tmp_path / "Proj")
        detail = rs.project_detail(manifest)
        assert len(detail["clips"]) == 1
        clip = detail["clips"][0]
        assert clip["clip_id"] == "EX01"
        assert clip["segments"] == [{"source_in": 0.5, "source_out": 4.5}]
        assert clip["preview_url"].startswith("/video?path=")
        assert detail["source_url"].startswith("/video?path=")
        assert detail["source_found"] is True
        # full transcript is sent at the project level for the expandable window
        assert detail["transcript_words"] and "start" in detail["transcript_words"][0]

    def test_asset_flag(self, tmp_path):
        root = tmp_path / "Proj"
        (root / "openai-runs" / "run-plan").mkdir(parents=True)
        (root / "transcripts").mkdir()
        clip = make_clip("EX01", card=1.0, segments=[(0.5, 4.5)])
        clip["media_request_ids"] = ["req-1"]
        plan = make_plan([clip])
        plan_path = root / "openai-runs" / "run-plan" / "plan-output.json"
        plan_path.write_text(json.dumps(plan))
        tr = root / "transcripts" / "t.json"; tr.write_text(json.dumps(make_transcript(6.0)))
        rdir = root / "renders" / "excerpt" / "draft"; rdir.mkdir(parents=True)
        (rdir / "render-manifest.json").write_text(json.dumps({
            "source_video": str(root / "s.mp4"), "transcript": str(tr),
            "stage3_output": str(plan_path), "content_mode": "excerpt",
            "clips": [{"clip_id": "EX01", "output_path": str(rdir / "EX01.mp4")}],
        }))
        detail = rs.project_detail(rdir / "render-manifest.json")
        assert detail["clips"][0]["uses_assets"] is True


class TestClipWords:
    def _words(self):
        # one word per 0.5s from 0..10s
        return [{"text": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(20)]

    def test_flags_in_and_out(self):
        words = self._words()
        segs = [{"source_in": 2.0, "source_out": 4.0}]
        cw = rs.clip_words(words, segs, pad=1.0)
        # words inside [2,4] are "in"; padding words [1,2) and (4,5] are "out"
        in_times = [w["start"] for w in cw if w["in"]]
        assert in_times and all(2.0 - 1e-6 <= t <= 4.0 + 1e-6 for t in in_times)
        assert any(not w["in"] for w in cw)  # padding present

    def test_round_trip_reconstructs_segments(self):
        # rebuilding segments from contiguous in-words recovers the span
        words = self._words()
        segs = [{"source_in": 2.0, "source_out": 4.0}, {"source_in": 6.0, "source_out": 8.0}]
        cw = rs.clip_words(words, segs, pad=1.0)
        rebuilt, run = [], None
        for w in cw:
            if w["in"]:
                run = run or {"source_in": w["start"], "source_out": w["end"]}
                run["source_out"] = w["end"]
            elif run:
                rebuilt.append(run); run = None
        if run:
            rebuilt.append(run)
        assert len(rebuilt) == 2
        assert rebuilt[0]["source_in"] == 2.0
        assert rebuilt[1]["source_in"] == 6.0

    def test_empty_inputs(self):
        assert rs.clip_words([], [{"source_in": 0, "source_out": 1}]) == []
        assert rs.clip_words([{"text": "a", "start": 0, "end": 1}], []) == []


class TestRunRoot:
    def test_anchors_on_renders(self, tmp_path):
        mp = tmp_path / "MyRun" / "renders" / "excerpt" / "draft" / "render-manifest.json"
        assert rs.run_root_of(mp) == tmp_path / "MyRun"

    def test_ad_mode_layout(self, tmp_path):
        mp = tmp_path / "MyRun" / "renders" / "final" / "render-manifest.json"
        assert rs.run_root_of(mp) == tmp_path / "MyRun"


class TestVideoUrl:
    def test_none(self):
        assert rs.video_url(None) is None

    def test_quotes_path(self):
        assert rs.video_url("/a b/c.mp4").startswith("/video?path=")
        assert "%2F" in rs.video_url("/a/b.mp4")
