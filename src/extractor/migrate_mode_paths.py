from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractor.project_layout import (
    WIRING_SKIP_DIRS,
    find_projects,
    pending_rewrites,
    rewriter,
)

# Bring existing projects onto two layout rules the pipeline now applies to every
# content mode:
#
#   1. Renders and run ids name their mode. "ad" used to be the unnamed default
#      (_renders/draft, <slug>-plan) while excerpt named itself (_renders/excerpt/
#      draft, <slug>-excerpt-plan). Both now read <mode>. Until an existing project
#      is moved, a re-run writes to the new paths, sees no output at the old ones,
#      and re-spends every LLM stage — so this migration is what preserves resume.
#
#   2. A clip's .subtitled.mp4 is a hardlink, not a second copy. When captions are
#      burned into the clip itself the two files are the same bytes; older runs
#      wrote a full copy, doubling the size of every delivered clip.
#
# Dry-run by default: prints the plan and changes nothing. --apply performs it
# after a preflight that refuses on any target collision, and records every move
# in <project>/mode-paths-undo-<timestamp>.json.
#
# Path references are rewritten only in the files the pipeline and review app read
# to *locate* things (pipeline-manifest.json, render manifests, index manifests,
# ffmpeg concat lists). The API record — openai-runs/<run>/ request manifests and
# prompts, and _response-cache/ — is left exactly as it was sent; response-cache
# keys are location-independent, so nothing there needs to move.

STAGES = ("outline", "strategy", "moments", "plan", "critique", "revise")
CONTENT_MODES = ("ad", "excerpt")
LEGACY_MODE = "ad"  # the mode that used to go unnamed
SKIP_DIRS = WIRING_SKIP_DIRS


def planned_renames(project: Path) -> dict[str, str]:
    """Relative old -> new directory paths for this project, as POSIX strings.

    Relative form on purpose: every embedded reference is either relative or an
    absolute path that contains the relative one, so one substitution set covers
    both without needing to know where the library was mounted."""
    slug = project.name
    renames: dict[str, str] = {}

    renders = project / "_renders"
    if renders.is_dir():
        for child in sorted(renders.iterdir()):
            # A mode dir is already correct; anything else directly under
            # _renders/ is a legacy-mode stage dir (draft, final, <variant>-draft).
            if not child.is_dir() or child.name in CONTENT_MODES:
                continue
            renames[f"_renders/{child.name}"] = f"_renders/{LEGACY_MODE}/{child.name}"

    for stage in STAGES:
        for parent, prefix in ((project / "openai-runs", "openai-runs"),
                               (project / "openai-runs" / "_critique-assets", "openai-runs/_critique-assets"),
                               (project / "subject-index", "subject-index")):
            old = parent / f"{slug}-{stage}"
            if old.is_dir():
                renames[f"{prefix}/{slug}-{stage}"] = f"{prefix}/{slug}-{LEGACY_MODE}-{stage}"
    return renames


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def duplicate_subtitled(project: Path) -> list[tuple[Path, Path]]:
    """(video, its .subtitled twin) pairs that hold identical bytes in separate
    inodes — the copies that should be hardlinks."""
    pairs: list[tuple[Path, Path]] = []
    for dirpath, dirnames, filenames in os.walk(project):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if not name.endswith(".subtitled.mp4"):
                continue
            sub = Path(dirpath) / name
            video = sub.with_name(name[: -len(".subtitled.mp4")] + ".mp4")
            if not video.is_file():
                continue
            s1, s2 = video.stat(), sub.stat()
            if s1.st_ino == s2.st_ino or s1.st_size != s2.st_size:
                continue
            if sha256_file(video) == sha256_file(sub):
                pairs.append((video, sub))
    return pairs


def preflight(project: Path, renames: dict[str, str]) -> list[str]:
    problems = []
    for old, new in renames.items():
        if not (project / old).exists():
            problems.append(f"missing source: {old}")
        if (project / new).exists():
            problems.append(f"target already exists: {new}")
    return problems


def migrate(project: Path, *, apply: bool, dedupe: bool) -> dict[str, Any]:
    renames = planned_renames(project)
    dupes = duplicate_subtitled(project) if dedupe else []
    substitute = rewriter(renames)
    report: dict[str, Any] = {
        "project": str(project),
        "renames": renames,
        "rewrites": [str(f.relative_to(project)) for f, _ in pending_rewrites(project, substitute)],
        "deduped": [str(s.relative_to(project)) for _, s in dupes],
        "bytes_reclaimed": sum(s.stat().st_size for _, s in dupes),
        "problems": preflight(project, renames),
    }
    if not apply or report["problems"]:
        return report

    # Dedupe first, while the scanned paths still resolve — hardlinks survive the
    # renames that follow, and a rename would invalidate these paths.
    for video, sub in dupes:
        # The bytes are already proven identical, so the copy is redundant; link
        # via a temp name so an interrupted run never leaves the path missing.
        tmp = sub.with_name(sub.name + ".relink")
        os.link(video, tmp)
        tmp.replace(sub)
    for old, new in sorted(renames.items(), key=lambda kv: len(kv[0]), reverse=True):
        (project / new).parent.mkdir(parents=True, exist_ok=True)
        (project / old).rename(project / new)
    # Rescanned after the moves: a wiring file that lived inside a renamed dir is
    # only reachable at its new path now.
    for path, text in pending_rewrites(project, substitute):
        path.write_text(text, encoding="utf-8")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    undo = project / f"mode-paths-undo-{stamp}.json"
    undo.write_text(json.dumps({
        "migrated_at": datetime.now(UTC).isoformat(),
        "renames": renames,
        "rewrites": report["rewrites"],
        "deduped": report["deduped"],
    }, indent=2))
    report["undo"] = str(undo)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move existing projects onto mode-scoped render/run paths and "
                    "hardlink duplicated .subtitled.mp4 copies.")
    parser.add_argument("roots", nargs="+", type=Path,
                        help="Project directories, or a library containing them.")
    parser.add_argument("--apply", action="store_true", help="Perform it. Default is a dry run.")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Skip the .subtitled.mp4 hardlink pass (paths only).")
    args = parser.parse_args(argv)

    projects = [p for root in args.roots for p in find_projects(root)]
    if not projects:
        print("No projects found (looked for pipeline-manifest.json).", file=sys.stderr)
        return 1

    failed = False
    reclaimed = 0
    for project in projects:
        report = migrate(project, apply=args.apply, dedupe=not args.no_dedupe)
        print(f"\n== {project.name} ==")
        for old, new in sorted(report["renames"].items()):
            print(f"  move   {old}  ->  {new}")
        print(f"  rewrite {len(report['rewrites'])} path-reference file(s)")
        if report["deduped"]:
            print(f"  hardlink {len(report['deduped'])} subtitled copy(ies), "
                  f"{report['bytes_reclaimed'] / 1e9:.2f} GB reclaimed")
        reclaimed += report["bytes_reclaimed"]
        for problem in report["problems"]:
            failed = True
            print(f"  REFUSED: {problem}", file=sys.stderr)

    verb = "Migrated" if args.apply else "Would migrate"
    print(f"\n{verb} {len(projects)} project(s); {reclaimed / 1e9:.2f} GB reclaimed.")
    if not args.apply:
        print("Dry run — nothing changed. Re-run with --apply.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
