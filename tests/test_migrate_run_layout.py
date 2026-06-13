import json
from pathlib import Path

import pytest

from extractor import migrate_run_layout as mig


def build_old_run(root: Path) -> Path:
    """A pre-rename run directory (stage1..stage5 layout) with the nested
    _stage4-review-assets case and a non-renamed manifest under a renamed dir."""
    base = "vid-excerpt"
    oa = root / "openai-runs"
    for st, fn in [("stage1", "stage1-output.json"), ("stage2", "stage2-output.json"),
                   ("stage3", "stage3-output.json"), ("stage4", "stage4-output.json")]:
        d = oa / f"{base}-{st}"
        d.mkdir(parents=True)
        (d / fn).write_text("{}")
    (oa / f"{base}-stage4" / "review-package.json").write_text(
        json.dumps({"x": str(root / f"openai-runs/{base}-stage3/stage3-output.json")}))
    d5 = oa / f"{base}-stage5"
    d5.mkdir()
    (d5 / "stage5-output.json").write_text("{}")
    (d5 / "stage3-output.json").write_text("{}")
    assets = oa / "_stage4-review-assets" / f"{base}-stage4"
    assets.mkdir(parents=True)
    (assets / "sheet.jpg").write_bytes(b"x")
    rd = root / "renders" / "excerpt" / "stage3"
    rd.mkdir(parents=True)
    (rd / "render-manifest.json").write_text(
        json.dumps({"plan": str(root / f"openai-runs/{base}-stage3/stage3-output.json")}))
    (root / "renders" / "excerpt" / "stage5").mkdir()
    (root / "renders" / "excerpt" / "stage5" / "render-manifest.json").write_text("{}")
    (root / "pipeline-manifest.json").write_text(json.dumps({
        "stage_outputs": {"stage1": "a", "stage2": "b", "stage3": "c",
                          "stage4": "d", "stage5_stage3": "e"},
        "renders": {"stage3": "x", "final": "y"},
        "subject_index_manifests": {"stage3": "p", "final": "q"},
        "costs": [{"stage": "stage1"}, {"stage": "render-stage3"}]}))
    return root


class TestNameMappers:
    @pytest.mark.parametrize("old,new", [
        ("foo-stage1", "foo-strategy"), ("foo-stage2", "foo-moments"),
        ("foo-stage3", "foo-plan"), ("foo-stage4", "foo-critique"),
        ("foo-stage5", "foo-revise"),
    ])
    def test_run_dir(self, old, new):
        assert mig.rename_run_dir_name(old) == new

    def test_run_dir_no_match(self):
        assert mig.rename_run_dir_name("foo-bar") is None

    @pytest.mark.parametrize("old,new", [
        ("stage3", "draft"), ("stage5", "final"),
        ("variant-stage3", "variant-draft"), ("variant-stage5", "variant-final"),
    ])
    def test_render_dir(self, old, new):
        assert mig.rename_render_dir_name(old) == new


class TestManifestMigration:
    def test_pipeline_manifest_keys(self):
        out = mig.migrate_pipeline_manifest({
            "stage_outputs": {"stage1": "a", "stage2": "b", "stage3": "c",
                              "stage4": "d", "stage5_stage3": "e"},
            "renders": {"stage3": "x", "final": "y"},
            "subject_index_manifests": {"stage3": "p", "final": "q"},
            "costs": [{"stage": "stage1"}, {"stage": "render-stage3"}]})
        assert set(out["stage_outputs"]) == {"strategy", "moments", "plan", "critique", "revised_plan"}
        assert set(out["renders"]) == {"draft", "final"}
        assert [c["stage"] for c in out["costs"]] == ["strategy", "draft-render"]

    def test_visual_index_key(self):
        out = mig.migrate_visual_index_manifest({"stage2_output": "x", "frame_width": 480})
        assert out["moments_output"] == "x" and "stage2_output" not in out


class TestFinalLocation:
    def test_nested_asset_dir_composes(self, tmp_path):
        root = build_old_run(tmp_path / "run")
        renames = mig.plan_renames(root)
        child = str(root / "openai-runs" / "_stage4-review-assets" / "vid-excerpt-stage4")
        assert mig.final_location(child, renames) == str(
            root / "openai-runs" / "_critique-assets" / "vid-excerpt-critique")

    def test_renamed_file_in_renamed_dir(self, tmp_path):
        root = build_old_run(tmp_path / "run")
        renames = mig.plan_renames(root)
        rp = str(root / "openai-runs" / "vid-excerpt-stage4" / "review-package.json")
        assert mig.final_location(rp, renames) == str(
            root / "openai-runs" / "vid-excerpt-critique" / "critique-package.json")

    def test_nonrenamed_file_moved_by_parent(self, tmp_path):
        root = build_old_run(tmp_path / "run")
        renames = mig.plan_renames(root)
        rm = str(root / "renders" / "excerpt" / "stage3" / "render-manifest.json")
        assert mig.final_location(rm, renames) == str(
            root / "renders" / "excerpt" / "draft" / "render-manifest.json")

    def test_unrenamed_file_unchanged(self, tmp_path):
        root = build_old_run(tmp_path / "run")
        renames = mig.plan_renames(root)
        pm = str(root / "pipeline-manifest.json")
        assert mig.final_location(pm, renames) == pm


class TestMigrateRun:
    def test_dry_run_changes_nothing(self, tmp_path):
        root = build_old_run(tmp_path / "run")
        assert mig.migrate_run(root, apply=False) is True
        assert (root / "openai-runs" / "vid-excerpt-stage1").exists()  # untouched

    def test_apply_renames_and_rewrites(self, tmp_path):
        root = build_old_run(tmp_path / "run")
        assert mig.migrate_run(root, apply=True) is True
        assert (root / "openai-runs" / "vid-excerpt-strategy" / "strategy-output.json").exists()
        assert (root / "openai-runs" / "vid-excerpt-revise" / "plan-output.json").exists()
        assert (root / "openai-runs" / "vid-excerpt-critique" / "critique-package.json").exists()
        assert (root / "openai-runs" / "_critique-assets" / "vid-excerpt-critique").exists()
        assert (root / "renders" / "excerpt" / "draft" / "render-manifest.json").exists()
        pm = json.loads((root / "pipeline-manifest.json").read_text())
        assert set(pm["stage_outputs"]) == {"strategy", "moments", "plan", "critique", "revised_plan"}

    def test_undo_map_final_paths_exist(self, tmp_path):
        root = build_old_run(tmp_path / "run")
        mig.migrate_run(root, apply=True)
        undo = json.loads(next(root.glob("migration-undo-*.json")).read_text())
        assert undo["content_backups"]
        for cb in undo["content_backups"]:
            assert (root / cb["final_path"]).exists()  # F4: live file is at final_path
            assert (root / cb["backup"]).exists()
        assert "undo_order" in undo

    def test_idempotent(self, tmp_path, capsys):
        root = build_old_run(tmp_path / "run")
        mig.migrate_run(root, apply=True)
        capsys.readouterr()
        assert mig.migrate_run(root, apply=True) is True
        assert "already migrated" in capsys.readouterr().out

    def test_preflight_refuses_on_collision(self, tmp_path):
        root = build_old_run(tmp_path / "run")
        # pre-create a target dir so the rename would collide
        (root / "openai-runs" / "vid-excerpt-strategy").mkdir()
        assert mig.migrate_run(root, apply=True) is False
        assert (root / "openai-runs" / "vid-excerpt-stage1").exists()  # nothing renamed


class TestReplacePaths:
    def test_anchored_no_false_positive(self):
        pairs = [("/r/x-stage3", "/r/x-plan")]
        # a longer sibling must not be partially rewritten
        text = '"/r/x-stage3-extra/f"'
        assert mig.replace_paths_in_text(text, pairs, None) == text

    def test_quoted_and_prefix(self):
        pairs = [("/r/x-stage3", "/r/x-plan")]
        assert mig.replace_paths_in_text('"/r/x-stage3"', pairs, None) == '"/r/x-plan"'
        assert mig.replace_paths_in_text('"/r/x-stage3/out.json"', pairs, None) == '"/r/x-plan/out.json"'
