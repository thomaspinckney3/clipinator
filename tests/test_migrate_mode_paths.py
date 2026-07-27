"""Migration of existing projects onto mode-scoped paths + hardlinked subtitles.

The stakes are resume: stage skipping keys off the existence of an output file,
so if an ad run's outputs stay at the old unscoped paths, a re-run finds nothing
and re-spends every LLM stage. These tests pin the moves and the reference
rewrites that keep the moved project wired together.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from extractor import migrate_mode_paths as mm


def make_project(root: Path, slug: str = "proj") -> Path:
    """A project as the pre-change pipeline left it: ad output unscoped, excerpt
    output mode-scoped, and a .subtitled.mp4 that is a byte copy of its clip."""
    proj = root / slug
    for stage in ("draft", "final"):
        d = proj / "_renders" / stage
        d.mkdir(parents=True)
        (d / "clip_01.mp4").write_bytes(b"video-bytes")
        (d / "clip_01.subtitled.mp4").write_bytes(b"video-bytes")  # full copy
        (d / "render-manifest.json").write_text(json.dumps({
            "output_dir": f"{proj}/_renders/{stage}",
            "stage3_output": f"{proj}/openai-runs/{slug}-revise/plan-output.json",
            "clips": [{"clip_id": "clip_01",
                       "output_path": f"{proj}/_renders/{stage}/clip_01.mp4"}],
        }))
        (d / "_work" / "clip_01").mkdir(parents=True)
        (d / "_work" / "clip_01" / "segment-001.video.concat.txt").write_text(
            f"file '{proj}/_renders/{stage}/_work/clip_01/segment-001.mp4'\n")
    (proj / "_renders" / "excerpt" / "final").mkdir(parents=True)

    for stage in ("outline", "strategy", "moments", "plan", "critique", "revise"):
        (proj / "openai-runs" / f"{slug}-{stage}").mkdir(parents=True)
        (proj / "openai-runs" / f"{slug}-excerpt-{stage}").mkdir(parents=True)
    (proj / "openai-runs" / "_critique-assets" / f"{slug}-critique").mkdir(parents=True)
    for stage in ("plan", "revise"):
        d = proj / "subject-index" / f"{slug}-{stage}"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(
            {"plan": f"{proj}/openai-runs/{slug}-{stage}/plan-output.json"}))

    # The API record: must survive untouched.
    (proj / "openai-runs" / f"{slug}-plan" / "request-manifest.json").write_text(
        json.dumps({"run_id": f"{slug}-plan"}))
    (proj / "openai-runs" / f"{slug}-plan" / "task-prompt.txt").write_text(
        f"materials under {proj}/_renders/draft\n")
    # ...but critique-package.json is re-read by revise, so it must stay accurate.
    (proj / "openai-runs" / f"{slug}-critique" / "critique-package.json").write_text(
        json.dumps({"clips": [
            {"clip_id": "clip_01",
             "rendered_video_path": f"{proj}/_renders/draft/clip_01.mp4"}]}))
    cache = proj / "openai-runs" / "_response-cache" / "ab" / "abc"
    cache.mkdir(parents=True)
    (cache / "cache-manifest.json").write_text(json.dumps({"run_id": f"{slug}-plan"}))

    (proj / "pipeline-manifest.json").write_text(json.dumps({"runs": {"ad": {
        "stage_outputs": {"plan": f"{proj}/openai-runs/{slug}-plan/plan-output.json"},
        "renders": {"draft": f"{proj}/_renders/draft/render-manifest.json"},
    }}}))
    return proj


class TestPlannedRenames:
    def test_scopes_legacy_mode_output_without_touching_named_modes(self, tmp_path):
        renames = mm.planned_renames(make_project(tmp_path))
        assert renames["_renders/draft"] == "_renders/ad/draft"
        assert renames["_renders/final"] == "_renders/ad/final"
        assert renames["openai-runs/proj-plan"] == "openai-runs/proj-ad-plan"
        assert (renames["openai-runs/_critique-assets/proj-critique"]
                == "openai-runs/_critique-assets/proj-ad-critique")
        assert renames["subject-index/proj-revise"] == "subject-index/proj-ad-revise"
        # already-named modes are left alone
        assert not any(k.startswith("_renders/excerpt") for k in renames)
        assert not any("-excerpt-" in k for k in renames)

    def test_is_idempotent(self, tmp_path):
        proj = make_project(tmp_path)
        mm.migrate(proj, apply=True, dedupe=True)
        assert mm.planned_renames(proj) == {}


class TestMigrate:
    def test_dry_run_changes_nothing(self, tmp_path):
        proj = make_project(tmp_path)
        report = mm.migrate(proj, apply=False, dedupe=True)
        assert report["renames"] and report["rewrites"]
        assert (proj / "_renders" / "draft").is_dir()
        assert not (proj / "_renders" / "ad").exists()

    def test_moves_and_rewires(self, tmp_path):
        proj = make_project(tmp_path)
        mm.migrate(proj, apply=True, dedupe=True)

        assert (proj / "_renders" / "ad" / "draft" / "clip_01.mp4").is_file()
        assert (proj / "openai-runs" / "proj-ad-plan").is_dir()
        assert not (proj / "_renders" / "draft").exists()

        # every reference now resolves to a file that exists
        pm = json.loads((proj / "pipeline-manifest.json").read_text())["runs"]["ad"]
        assert "openai-runs/proj-ad-plan" in pm["stage_outputs"]["plan"]
        assert Path(pm["renders"]["draft"]).is_file()
        rm = json.loads((proj / "_renders" / "ad" / "draft" / "render-manifest.json").read_text())
        assert Path(rm["clips"][0]["output_path"]).is_file()
        assert "openai-runs/proj-ad-revise" in rm["stage3_output"]
        si = json.loads((proj / "subject-index" / "proj-ad-plan" / "manifest.json").read_text())
        assert "openai-runs/proj-ad-plan" in si["plan"]
        concat = (proj / "_renders" / "ad" / "draft" / "_work" / "clip_01"
                  / "segment-001.video.concat.txt").read_text()
        assert "_renders/ad/draft" in concat

    def test_rewires_the_critique_package_revise_re_reads(self, tmp_path):
        proj = make_project(tmp_path)
        mm.migrate(proj, apply=True, dedupe=True)
        pkg = json.loads((proj / "openai-runs" / "proj-ad-critique"
                          / "critique-package.json").read_text())
        assert Path(pkg["clips"][0]["rendered_video_path"]).is_file()

    def test_leaves_the_api_record_alone(self, tmp_path):
        # Request manifests, prompts and the response cache record what was actually
        # sent. Rewriting them would falsify that; cache keys are location-independent,
        # so nothing there needs to move.
        proj = make_project(tmp_path)
        mm.migrate(proj, apply=True, dedupe=True)
        run = proj / "openai-runs" / "proj-ad-plan"
        cached = json.loads((proj / "openai-runs" / "_response-cache" / "ab" / "abc"
                             / "cache-manifest.json").read_text())
        assert json.loads((run / "request-manifest.json").read_text())["run_id"] == "proj-plan"
        assert "_renders/draft" in (run / "task-prompt.txt").read_text()
        assert cached["run_id"] == "proj-plan"

    def test_refuses_on_collision(self, tmp_path):
        proj = make_project(tmp_path)
        (proj / "_renders" / "ad" / "draft").mkdir(parents=True)
        report = mm.migrate(proj, apply=True, dedupe=False)
        assert report["problems"]
        assert (proj / "_renders" / "draft").is_dir()  # nothing moved


class TestDedupeSubtitled:
    def test_identical_copies_become_one_inode(self, tmp_path):
        proj = make_project(tmp_path)
        report = mm.migrate(proj, apply=True, dedupe=True)
        assert report["bytes_reclaimed"] == 2 * len(b"video-bytes")
        d = proj / "_renders" / "ad" / "draft"
        assert (d / "clip_01.subtitled.mp4").read_bytes() == b"video-bytes"
        assert (d / "clip_01.subtitled.mp4").stat().st_ino == (d / "clip_01.mp4").stat().st_ino

    def test_leaves_a_genuinely_different_subtitled_file(self, tmp_path):
        proj = make_project(tmp_path)
        sub = proj / "_renders" / "draft" / "clip_01.subtitled.mp4"
        sub.write_bytes(b"burned-in-subs")  # a real, separate render
        mm.migrate(proj, apply=True, dedupe=True)
        moved = proj / "_renders" / "ad" / "draft" / "clip_01.subtitled.mp4"
        assert moved.read_bytes() == b"burned-in-subs"
        assert moved.stat().st_ino != (moved.parent / "clip_01.mp4").stat().st_ino

    def test_skips_files_already_linked(self, tmp_path):
        proj = make_project(tmp_path)
        sub = proj / "_renders" / "draft" / "clip_01.subtitled.mp4"
        sub.unlink()
        os.link(proj / "_renders" / "draft" / "clip_01.mp4", sub)
        # only the still-duplicated "final" pair remains to reclaim
        assert [s.parent.name for _, s in mm.duplicate_subtitled(proj)] == ["final"]


class TestFindProjects:
    def test_accepts_a_library_or_a_single_project(self, tmp_path):
        lib = tmp_path / "lib"
        make_project(lib, "a")
        make_project(lib, "b")
        assert [p.name for p in mm.find_projects(lib)] == ["a", "b"]
        assert mm.find_projects(lib / "a") == [lib / "a"]
