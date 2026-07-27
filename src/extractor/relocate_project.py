from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.project_layout import (
    find_projects,
    pending_rewrites,
    rewriter,
    wiring_files,
)

# Re-point a project that has been moved or renamed on disk.
#
# A project's manifests record absolute paths, so renaming the library that holds
# it (or moving it between volumes) leaves every one of them naming a location
# that no longer exists: the review app then reports the source as missing and
# refuses to serve the clips, because it reads source_video from the manifest.
#
# No --from is needed. pipeline-manifest.json records the run_root the project
# believed it occupied; comparing that to where the project actually sits gives
# the move that happened. Matching is case-insensitive, so a volume that comes
# back as /Volumes/t9 instead of /Volumes/T9 is handled too.
#
# Dry-run by default; --apply performs it and records the move in
# <project>/relocate-undo-<timestamp>.json. Only the wiring files are rewritten
# (see project_layout.WIRING_FILES) — the API record and the location-independent
# response cache are left exactly as they were.


def recorded_roots(project: Path) -> list[str]:
    """Every distinct run_root the project's manifest claims, newest first.

    Per-mode runs carry their own run_root and can have been written at different
    times, so a project that moved twice may name more than one stale location."""
    try:
        manifest = json.loads((project / "pipeline-manifest.json").read_text())
    except (OSError, ValueError):
        return []
    roots = [manifest.get("run_root")]
    roots += [run.get("run_root") for run in (manifest.get("runs") or {}).values()
              if isinstance(run, dict)]
    seen, out = set(), []
    for root in roots:
        root = str(root).rstrip("/") if root else ""
        if root and root.lower() not in seen:
            seen.add(root.lower())
            out.append(root)
    return out


def moves(project: Path, from_root: str | None = None) -> dict[str, str]:
    """old absolute root -> the project's actual location, for roots that moved.

    Compared exactly, so a case-only difference (a volume remounted as /Volumes/t9
    instead of /Volumes/T9) is still normalized: harmless where the filesystem
    ignores case, and necessary where it does not."""
    actual = str(project.resolve())
    claimed = [from_root] if from_root else recorded_roots(project)
    return {root: actual for root in claimed if root and root != actual}


def broken_references(project: Path) -> int:
    """Wiring-file references naming a path that does not exist. The number this
    tool is trying to drive to zero."""
    missing = 0
    for path in wiring_files(project):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for value in _strings(data):
            if value.startswith("/") and "/" in value[1:] and not Path(value).exists():
                missing += 1
    return missing


def _strings(node: Any):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def relocate(project: Path, *, apply: bool, from_root: str | None = None) -> dict[str, Any]:
    substitutions = moves(project, from_root)
    substitute = rewriter(substitutions, ignore_case=True)
    rewrites = pending_rewrites(project, substitute)
    report: dict[str, Any] = {
        "project": str(project),
        "moves": substitutions,
        "rewrites": [str(f.relative_to(project)) for f, _ in rewrites],
        "broken_before": broken_references(project),
    }
    if not apply or not rewrites:
        report["broken_after"] = report["broken_before"]
        return report

    for path, text in rewrites:
        path.write_text(text, encoding="utf-8")
    report["broken_after"] = broken_references(project)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    undo = project / f"relocate-undo-{stamp}.json"
    undo.write_text(json.dumps({
        "relocated_at": datetime.now(UTC).isoformat(),
        "moves": substitutions,
        "rewrites": report["rewrites"],
    }, indent=2))
    report["undo"] = str(undo)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-point a moved or renamed project's manifests at its actual location.")
    parser.add_argument("roots", nargs="+", type=Path,
                        help="Project directories, or a library containing them.")
    parser.add_argument("--apply", action="store_true", help="Perform it. Default is a dry run.")
    parser.add_argument("--from", dest="from_root", default=None,
                        help="Old absolute path to replace, when the manifest's own "
                             "run_root is missing or wrong.")
    args = parser.parse_args(argv)

    projects = [p for root in args.roots for p in find_projects(root)]
    if not projects:
        print("No projects found (looked for pipeline-manifest.json).", file=sys.stderr)
        return 1

    unresolved = 0
    for project in projects:
        report = relocate(project, apply=args.apply, from_root=args.from_root)
        print(f"\n== {project.name} ==")
        if not report["moves"]:
            print(f"  already at its recorded location ({report['broken_before']} broken ref(s))")
            unresolved += report["broken_after"]
            continue
        for old, new in report["moves"].items():
            print(f"  {old}  ->  {new}")
        if args.apply:
            print(f"  rewrote {len(report['rewrites'])} path-reference file(s); "
                  f"broken references {report['broken_before']} -> {report['broken_after']}")
            unresolved += report["broken_after"]
        else:
            print(f"  would rewrite {len(report['rewrites'])} path-reference file(s) "
                  f"({report['broken_before']} broken reference(s) today)")

    print(f"\n{'Relocated' if args.apply else 'Would relocate'} {len(projects)} project(s).")
    if args.apply and unresolved:
        print(f"{unresolved} reference(s) still name a missing path — these point outside "
              f"the project (an original master, say) and need a --from run or a look.")
    if not args.apply:
        print("Dry run — nothing changed. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
