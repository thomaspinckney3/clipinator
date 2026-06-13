from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Migrate pre-rename pipeline run directories (stage1..stage5 layout) to the
# human-named layout (strategy/moments/plan/critique/revise, renders draft/
# final) so old runs resume cleanly under the renamed pipeline instead of
# re-running every LLM stage.
#
# Dry-run by default: prints the full operation plan and changes nothing.
# --apply performs it, after a preflight that refuses on any target collision.
# Content-edited files are backed up under <run>/.migration-backup/ and every
# rename is recorded in <run>/migration-undo-<timestamp>.json.
#
# Deliberately untouched: model-visible legacy JSON keys inside stage outputs,
# render manifests, and review packages ("stage3_output" and friends), raw
# LLM responses, transcripts, video evidence, and the _response-cache (its
# strict-mode keys are invalidated by the code rename regardless).

RUN_DIR_SUFFIXES = {
    "-stage1": "-strategy",
    "-stage2": "-moments",
    "-stage3": "-plan",
    "-stage4": "-critique",
    "-stage5": "-revise",
}
OUTPUT_FILES_BY_NEW_SUFFIX = {
    "-strategy": {"stage1-output.json": "strategy-output.json"},
    "-moments": {"stage2-output.json": "moments-output.json"},
    "-plan": {"stage3-output.json": "plan-output.json"},
    "-critique": {
        "stage4-output.json": "critique-output.json",
        "review-package.json": "critique-package.json",
    },
    "-revise": {
        "stage5-output.json": "revise-output.json",
        "stage3-output.json": "plan-output.json",
    },
}
RENDER_DIR_NAMES = {"stage3": "draft", "stage5": "final"}
COST_LABELS = {
    "stage1": "strategy",
    "stage2": "moments",
    "stage3": "plan",
    "stage4": "critique",
    "stage5": "revise",
    "render-stage3": "draft-render",
    "subject-index-stage3": "subject-index-draft",
}
STAGE_OUTPUT_KEYS = {
    "stage1": "strategy",
    "stage2": "moments",
    "stage3": "plan",
    "stage4": "critique",
    "stage5_stage3": "revised_plan",
}


def is_run_root(path: Path) -> bool:
    return (path / "pipeline-manifest.json").exists()


def find_run_roots(scan_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for manifest in sorted(scan_dir.rglob("pipeline-manifest.json")):
        if ".migration-backup" in manifest.parts:
            continue
        roots.append(manifest.parent)
    return roots


def rename_run_dir_name(name: str) -> str | None:
    for old, new in RUN_DIR_SUFFIXES.items():
        if name.endswith(old):
            return name[: -len(old)] + new
    return None


def rename_render_dir_name(name: str) -> str | None:
    if name in RENDER_DIR_NAMES:
        return RENDER_DIR_NAMES[name]
    for old, new in RENDER_DIR_NAMES.items():
        if name.endswith(f"-{old}"):
            return name[: -len(old)] + new
    return None


def plan_renames(run_root: Path) -> list[tuple[Path, Path]]:
    """All (old, new) renames for one run, as absolute paths in OLD locations."""
    renames: list[tuple[Path, Path]] = []

    openai_runs = run_root / "openai-runs"
    if openai_runs.is_dir():
        for child in sorted(openai_runs.iterdir()):
            if not child.is_dir() or ".incomplete-" in child.name:
                continue
            if child.name == "_stage4-review-assets":
                for asset_dir in sorted(child.iterdir()):
                    new_name = rename_run_dir_name(asset_dir.name)
                    if asset_dir.is_dir() and new_name:
                        renames.append((asset_dir, asset_dir.with_name(new_name)))
                renames.append((child, child.with_name("_critique-assets")))
                continue
            new_name = rename_run_dir_name(child.name)
            if not new_name:
                continue
            new_suffix = next(s for s in OUTPUT_FILES_BY_NEW_SUFFIX if new_name.endswith(s))
            for old_file, new_file in OUTPUT_FILES_BY_NEW_SUFFIX[new_suffix].items():
                if (child / old_file).exists():
                    renames.append((child / old_file, child / new_file))
            renames.append((child, child.with_name(new_name)))

    subject_index = run_root / "subject-index"
    if subject_index.is_dir():
        for child in sorted(subject_index.iterdir()):
            new_name = rename_run_dir_name(child.name)
            if child.is_dir() and new_name:
                renames.append((child, child.with_name(new_name)))

    renders = run_root / "renders"
    if renders.is_dir():
        for child in sorted(renders.iterdir()):
            if not child.is_dir():
                continue
            new_name = rename_render_dir_name(child.name)
            if new_name:
                renames.append((child, child.with_name(new_name)))
                continue
            for grandchild in sorted(child.iterdir()):  # renders/<mode>/<stage>
                new_name = rename_render_dir_name(grandchild.name)
                if grandchild.is_dir() and new_name:
                    renames.append((grandchild, grandchild.with_name(new_name)))
    return renames


def _apply_dir_renames(path_str: str, dir_pairs: list[tuple[str, str]]) -> str:
    """Apply applicable directory renames to a path prefix until stable.

    Each rename's `new` only renames its own leaf, so an ancestor rename must
    still be applied to the prefix; iterate to compose chains and nesting
    (.../_stage4-review-assets/Y-stage4 -> .../_critique-assets/Y-critique).
    Bounded by the number of directory renames.
    """
    for _ in range(len(dir_pairs) + 1):
        for d_old, d_new in dir_pairs:
            if path_str == d_old:
                path_str = d_new
                break
            if path_str.startswith(d_old + "/"):
                path_str = d_new + path_str[len(d_old) :]
                break
        else:
            break
    return path_str


def final_location(path_str: str, renames: list[tuple[Path, Path]]) -> str:
    """Final path of any file after all renames apply.

    Handles both a file renamed in place (its leaf changes) and a file moved
    because an ancestor directory was renamed.
    """
    dir_pairs = [(str(old), str(new)) for old, new in renames if old.is_dir()]
    leaf_renames = {str(old): Path(str(new)).name for old, new in renames}
    leaf = leaf_renames.get(path_str, Path(path_str).name)
    return _apply_dir_renames(str(Path(path_str).parent), dir_pairs) + "/" + leaf


def composed_path_pairs(run_root: Path, renames: list[tuple[Path, Path]]) -> list[tuple[str, str]]:
    """Old->new path strings, composed across nested renames, longest first.

    A path inside a renamed directory maps to its FINAL location, e.g.
    .../X-stage5/stage3-output.json -> .../X-revise/plan-output.json.
    """
    pairs = [(str(old), final_location(str(old), renames)) for old, _ in renames]
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return pairs


def replace_paths_in_text(text: str, pairs: list[tuple[str, str]], relative_root: Path | None) -> str:
    for old, new in pairs:
        text = text.replace(old + "/", new + "/")
        text = text.replace(old + '"', new + '"')
        if relative_root is not None:
            try:
                old_rel = str(Path(old).relative_to(relative_root))
                new_rel = str(Path(new).relative_to(relative_root))
            except ValueError:
                continue
            text = text.replace(old_rel + "/", new_rel + "/")
            text = text.replace(old_rel + '"', new_rel + '"')
    return text


def content_edit_targets(run_root: Path) -> list[Path]:
    targets: list[Path] = []
    manifest = run_root / "pipeline-manifest.json"
    if manifest.exists():
        targets.append(manifest)
    for pattern in (
        "renders/**/render-manifest.json",
        "subject-index/**/manifest.json",
        "visual-index/**/manifest.json",
        "openai-runs/**/request-manifest.json",
        "openai-runs/**/review-package.json",
        "review/**/edit-list.toml",
        "review/**/edit-list.html",
        "review/**/plan-output.human.json",
    ):
        targets.extend(sorted(run_root.glob(pattern)))
    return [t for t in targets if "_response-cache" not in t.parts and ".incomplete-" not in str(t)]


def migrate_pipeline_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    stage_outputs = payload.get("stage_outputs")
    if isinstance(stage_outputs, dict):
        payload["stage_outputs"] = {
            STAGE_OUTPUT_KEYS.get(key, key): value for key, value in stage_outputs.items()
        }
    for section, mapping in (("renders", {"stage3": "draft"}), ("subject_index_manifests", {"stage3": "draft"})):
        block = payload.get(section)
        if isinstance(block, dict):
            payload[section] = {mapping.get(key, key): value for key, value in block.items()}
    for record in payload.get("costs", []) or []:
        if isinstance(record, dict) and record.get("stage") in COST_LABELS:
            record["stage"] = COST_LABELS[str(record["stage"])]
    return payload


def migrate_visual_index_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if "stage2_output" in payload and "moments_output" not in payload:
        payload["moments_output"] = payload.pop("stage2_output")
    return payload


def migrate_run(run_root: Path, apply: bool) -> bool:
    renames = plan_renames(run_root)
    targets = content_edit_targets(run_root)
    if not renames:
        print(f"{run_root}: already migrated (no legacy names found).")
        return True
    pairs = composed_path_pairs(run_root, renames)

    print(f"\n{run_root}")
    for old, new in renames:
        print(f"  rename: {old.relative_to(run_root)} -> {new.relative_to(run_root)}")
    for target in targets:
        print(f"  rewrite paths: {target.relative_to(run_root)}")
    if not apply:
        return True

    # Preflight: every rename target must be free (checked against final paths).
    final_targets = [Path(new) for _, new in composed_path_pairs(run_root, renames)]
    collisions = [t for t in final_targets if t.exists()]
    if collisions:
        print(f"  REFUSED: target(s) already exist: {[str(c) for c in collisions]}")
        return False

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = run_root / ".migration-backup" / timestamp
    undo: dict[str, Any] = {
        "timestamp": timestamp,
        "undo_order": (
            "1) reverse `renames` shallowest-first (new -> old); "
            "2) restore each content backup over its `final_path`."
        ),
        "renames": [],
        "content_backups": [],
    }

    # Content edits first (paths inside files reference OLD locations, and the
    # files themselves may be about to move). Record each backup with the file's
    # final post-rename location so an undo can restore it unambiguously.
    for target in targets:
        original = target.read_text(encoding="utf-8")
        text = replace_paths_in_text(original, pairs, relative_root=run_root)
        if target.name == "pipeline-manifest.json":
            text = json.dumps(migrate_pipeline_manifest(json.loads(text)), indent=2, ensure_ascii=False) + "\n"
        elif "visual-index" in target.parts and target.name == "manifest.json":
            text = json.dumps(migrate_visual_index_manifest(json.loads(text)), indent=2, ensure_ascii=False) + "\n"
        if text == original:
            continue
        backup_path = backup_dir / target.relative_to(run_root)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_path)
        target.write_text(text, encoding="utf-8")
        final_abs = final_location(str(target), renames)
        undo["content_backups"].append(
            {
                "backup": str(backup_path.relative_to(run_root)),
                "final_path": str(Path(final_abs).relative_to(run_root)),
            }
        )

    # Renames, deepest first, so files move before their parent directories.
    for old, new in sorted(renames, key=lambda pair: len(pair[0].parts), reverse=True):
        old.rename(new)
        undo["renames"].append({"old": str(old), "new": str(new)})

    undo_path = run_root / f"migration-undo-{timestamp}.json"
    undo_path.write_text(json.dumps(undo, indent=2) + "\n", encoding="utf-8")
    print(f"  applied {len(undo['renames'])} renames, rewrote {len(undo['content_backups'])} files")
    print(f"  undo map: {undo_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate pre-rename pipeline run directories to the human-named layout."
    )
    parser.add_argument("run_roots", nargs="*", type=Path, help="Run directories to migrate.")
    parser.add_argument("--scan", type=Path, default=None, help="Find run roots (pipeline-manifest.json) under this directory.")
    parser.add_argument("--apply", action="store_true", help="Perform the migration. Default is a dry run.")
    args = parser.parse_args(argv)

    roots = [r.expanduser().resolve() for r in args.run_roots]
    if args.scan:
        roots.extend(find_run_roots(args.scan.expanduser().resolve()))
    roots = sorted(set(roots))
    if not roots:
        print("No run roots given. Pass directories or --scan.", file=sys.stderr)
        return 1
    bad = [r for r in roots if not is_run_root(r)]
    if bad:
        print(f"Not pipeline run roots (no pipeline-manifest.json): {[str(b) for b in bad]}", file=sys.stderr)
        return 1

    ok = True
    for root in roots:
        ok = migrate_run(root, apply=args.apply) and ok
    if not args.apply:
        print("\nDry run only. Re-run with --apply to perform the migration.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
