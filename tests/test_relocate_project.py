"""Re-pointing a project whose directory was moved or renamed.

Manifests store absolute paths, so a renamed library breaks every one of them —
including source_video, which is what the review app checks before it will serve
a clip. The move is self-describing: pipeline-manifest.json records the run_root
the project believed it had.
"""
from __future__ import annotations

import json
from pathlib import Path

from extractor import relocate_project as rp


def make_project(lib: Path, slug: str = "proj", claimed_root: str | None = None) -> Path:
    """A project whose manifests name `claimed_root` (default: a stale location)."""
    proj = lib / slug
    claimed = claimed_root or f"/Volumes/T9/old-library/{slug}"
    (proj / "media").mkdir(parents=True)
    (proj / "media" / "source.mp4").write_bytes(b"v")
    rdir = proj / "_renders" / "ad" / "final"
    rdir.mkdir(parents=True)
    (rdir / "clip_01.mp4").write_bytes(b"v")
    (rdir / "render-manifest.json").write_text(json.dumps({
        "source_video": f"{claimed}/media/source.mp4",
        "clips": [{"clip_id": "clip_01", "output_path": f"{claimed}/_renders/ad/final/clip_01.mp4"}],
    }))
    (proj / "pipeline-manifest.json").write_text(json.dumps({
        "run_root": claimed,
        "runs": {"ad": {"run_root": claimed, "video": f"{claimed}/media/source.mp4"}},
    }))
    (proj / "openai-runs" / f"{slug}-ad-plan").mkdir(parents=True)
    (proj / "openai-runs" / f"{slug}-ad-plan" / "request-manifest.json").write_text(
        json.dumps({"video": f"{claimed}/media/source.mp4"}))
    return proj


class TestMoves:
    def test_detects_the_move_from_the_manifest(self, tmp_path):
        proj = make_project(tmp_path)
        assert rp.moves(proj) == {"/Volumes/T9/old-library/proj": str(proj.resolve())}

    def test_no_move_when_already_correct(self, tmp_path):
        proj = make_project(tmp_path, claimed_root=str((tmp_path / "proj").resolve()))
        assert rp.moves(proj) == {}

    def test_normalizes_a_case_only_difference(self, tmp_path):
        # An external volume can remount as /Volumes/t9 rather than /Volumes/T9.
        # Harmless to rewrite where the filesystem ignores case, and required
        # where it does not, so it counts as a move either way.
        actual = str((tmp_path / "proj").resolve())
        proj = make_project(tmp_path, claimed_root=actual.upper())
        assert rp.moves(proj) == {actual.upper(): actual}
        assert rp.relocate(proj, apply=True)["broken_after"] == 0

    def test_collects_a_second_stale_root(self, tmp_path):
        proj = make_project(tmp_path)
        pm = json.loads((proj / "pipeline-manifest.json").read_text())
        pm["runs"]["excerpt"] = {"run_root": "/Volumes/T9/older-still/proj"}
        (proj / "pipeline-manifest.json").write_text(json.dumps(pm))
        assert set(rp.moves(proj)) == {"/Volumes/T9/old-library/proj",
                                       "/Volumes/T9/older-still/proj"}

    def test_from_override_wins(self, tmp_path):
        proj = make_project(tmp_path)
        assert rp.moves(proj, "/somewhere/else") == {"/somewhere/else": str(proj.resolve())}


class TestRelocate:
    def test_dry_run_changes_nothing(self, tmp_path):
        proj = make_project(tmp_path)
        report = rp.relocate(proj, apply=False)
        assert report["rewrites"] and report["broken_before"] > 0
        assert "old-library" in (proj / "pipeline-manifest.json").read_text()

    def test_every_reference_resolves_afterwards(self, tmp_path):
        proj = make_project(tmp_path)
        report = rp.relocate(proj, apply=True)
        assert report["broken_before"] > 0
        assert report["broken_after"] == 0
        rm = json.loads((proj / "_renders" / "ad" / "final" / "render-manifest.json").read_text())
        assert Path(rm["source_video"]).is_file()
        assert Path(rm["clips"][0]["output_path"]).is_file()
        pm = json.loads((proj / "pipeline-manifest.json").read_text())
        assert pm["run_root"] == str(proj.resolve())
        assert Path(pm["runs"]["ad"]["video"]).is_file()

    def test_leaves_the_api_record_alone(self, tmp_path):
        proj = make_project(tmp_path)
        rp.relocate(proj, apply=True)
        req = json.loads((proj / "openai-runs" / "proj-ad-plan"
                          / "request-manifest.json").read_text())
        assert "old-library" in req["video"]

    def test_is_idempotent(self, tmp_path):
        proj = make_project(tmp_path)
        rp.relocate(proj, apply=True)
        assert rp.relocate(proj, apply=True)["rewrites"] == []

    def test_writes_an_undo_record(self, tmp_path):
        proj = make_project(tmp_path)
        report = rp.relocate(proj, apply=True)
        undo = json.loads(Path(report["undo"]).read_text())
        assert undo["moves"] == {"/Volumes/T9/old-library/proj": str(proj.resolve())}


class TestCli:
    def test_relocates_a_whole_library(self, tmp_path, capsys):
        lib = tmp_path / "lib"
        make_project(lib, "a")
        make_project(lib, "b")
        assert rp.main([str(lib), "--apply"]) == 0
        assert all(rp.relocate(lib / s, apply=False)["broken_after"] == 0 for s in ("a", "b"))
