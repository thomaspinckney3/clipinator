"""Source-centric project layout: the human-facing structure shared by the
pipeline (new projects) and the archive migrator (existing runs), so both produce
identical trees:

    <library>/<source-slug>/
      project.json               # source meta + provenance + clip index
      media/source.mp4           # the master, in the project (+ source.capture.json)
      excerpts/NN-title-slug/    # (or ads/) final.mp4, final.subtitled.mp4, captions.srt, metadata.json
      _archive/<older runs>/
      ... (the run's own transcripts/ plan/ _renders/ etc. stay in place) ...

The finished clips live in a folder named for the content mode — excerpts/ or ads/
— so it reads as the product it holds (the old generic "extracts/" needlessly
rhymed with the "excerpt" mode). Internal working output stays under _renders/,
underscore-prefixed like the other machine dirs (_work, _response-cache).

The run's internal structure is left untouched (find_run_root / review_server
discovery key off pipeline-manifest.json / render-manifest.json, not names), so
media/, excerpts/, project.json are a human-facing overlay, not a rewrite.
Stdlib-only. `slug`/`clip_slug` match the migrator so names line up.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HASH = re.compile(r"\.mp4-[0-9a-f]{32}")

# Files whose job is to point at other files, and which a later run re-reads.
# Rewriting these is what keeps a moved or renamed project wired together.
# Everything else is left alone — in particular openai-runs/<run>/request-manifest.json,
# task-prompt.txt and source-materials.txt, which record what was actually sent to
# the API. critique-package.json is here because revise re-reads it, not as a record;
# _response-cache/ is skipped because its keys are location-independent by design.
WIRING_FILES = ("pipeline-manifest.json", "render-manifest.json", "manifest.json",
                "critique-package.json")
WIRING_SUFFIXES = (".concat.txt",)
WIRING_SKIP_DIRS = {"_response-cache", "_archive", "_legacy", ".migration-backup"}


def is_project(path: Path) -> bool:
    """A project is a directory the pipeline has written a run manifest into."""
    return (path / "pipeline-manifest.json").is_file()


def find_projects(root: Path) -> list[Path]:
    """The projects at `root` — `root` itself if it is one, else its children."""
    root = Path(root).expanduser()
    if is_project(root):
        return [root]
    return sorted(p.parent for p in root.glob("*/pipeline-manifest.json"))


def wiring_files(project: Path) -> list[Path]:
    """Every path-carrying file in a project, in walk order."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(project):
        dirnames[:] = [d for d in dirnames if d not in WIRING_SKIP_DIRS]
        for name in filenames:
            if name in WIRING_FILES or name.endswith(WIRING_SUFFIXES):
                found.append(Path(dirpath) / name)
    return sorted(found)


def rewriter(substitutions: dict[str, str], *, ignore_case: bool = False):
    """A single-pass substitution over all pairs, longest key first, so no
    replacement is applied twice or inside another's result. None if empty."""
    if not substitutions:
        return None
    keys = sorted(substitutions, key=len, reverse=True)
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile("|".join(re.escape(k) for k in keys), flags)
    lookup = {k.lower(): v for k, v in substitutions.items()} if ignore_case else substitutions
    key_of = (lambda m: m.group(0).lower()) if ignore_case else (lambda m: m.group(0))
    return lambda text: pattern.sub(lambda m: lookup[key_of(m)], text)


def pending_rewrites(project: Path, substitute) -> list[tuple[Path, str]]:
    """(file, rewritten text) for every wiring file holding a stale reference."""
    if substitute is None:
        return []
    out = []
    for path in wiring_files(project):
        before = path.read_text(encoding="utf-8", errors="ignore")
        after = substitute(before)
        if after != before:
            out.append((path, after))
    return out


def slug(text: str, max_words: int = 6, max_len: int = 50) -> str:
    text = _HASH.sub("", str(text))
    text = re.sub(r"\.(mp4|mov|mkv|json)$", "", text, flags=re.I).replace("_master", "")
    return "-".join(re.findall(r"[A-Za-z0-9]+", text.lower())[:max_words])[:max_len].strip("-") or "untitled"


def source_slug(video: Path) -> str:
    """Human, stable project-dir name for a source video (drops capture-hash and
    excerpt/rerun/timestamp variant tails)."""
    s = _HASH.sub("", Path(video).name)
    s = re.sub(r"\.(mp4|mov|mkv)$", "", s, flags=re.I).replace("_master", "")
    s = re.sub(r"-(excerpt|rerun|insight)(-[a-z-]+)?(-\d{8}T\d{6}Z)?$", "", s)
    s = re.sub(r"-\d{8}T\d{6}Z$", "", s)
    return slug(s)


def clip_slug(title: str, order: int) -> str:
    return f"{order:02d}-{slug(title)}"


def clip_index(plan: dict[str, Any]) -> dict[str, tuple[int, str, str, dict]]:
    """clip_id -> (order, title, slug, clip) from a plan's clip_ads."""
    out: dict[str, tuple[int, str, str, dict]] = {}
    for i, c in enumerate(plan.get("clip_ads", []) or [], 1):
        title = (c.get("concept_title") or (c.get("opening_hook") or {}).get("text") or c.get("clip_id") or "").strip()
        out[str(c.get("clip_id"))] = (i, title, clip_slug(title, i), c)
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)          # same-volume: instant, space-free
    except OSError:
        shutil.copy2(src, dst)     # cross-volume (e.g. off-disk master): copy in


def establish_media(project_dir: Path, video: Path) -> Path:
    """Ensure the master lives at <project>/media/source.mp4 (hardlink if same
    volume, else copy — so an off-volume input is pulled onto the project's disk).
    Idempotent; returns the media path."""
    dst = project_dir / "media" / "source.mp4"
    if dst.exists():
        return dst
    _link_or_copy(video, dst)
    side = Path(str(video) + ".json")
    if side.exists():
        _link_or_copy(side, project_dir / "media" / "source.capture.json")
    return dst


def deliverables_dirname(content_mode: str) -> str:
    """The human-facing folder for a mode's finished clips: 'excerpt' -> 'excerpts',
    'ad' -> 'ads'. Named by product so it reads as what it holds and never collides
    with the 'excerpt' mode word the way the old generic 'extracts/' did."""
    return f"{content_mode}s"


def build_clips(project_dir: Path, render_manifest: Path, plan: dict[str, Any],
                content_mode: str) -> list[dict[str, Any]]:
    """Create the human-named <excerpts|ads>/ view from a render manifest's clips.
    Each <mode>/NN-slug/ gets final.mp4 (+ .subtitled.mp4) + captions.srt +
    metadata.json, hardlinked from the render outputs. Idempotent. Returns the
    clip index for project.json."""
    rmd = json.loads(render_manifest.read_text())
    rdir = render_manifest.parent
    idx = clip_index(plan)
    clips_dir = project_dir / deliverables_dirname(content_mode)
    index: list[dict[str, Any]] = []
    for c in rmd.get("clips", []):
        cid = str(c.get("clip_id"))
        base = Path(str(c.get("output_path"))).name
        order, title, cslug, clip = idx.get(cid, (99, cid, clip_slug(cid, 99), {}))
        ex = clips_dir / cslug
        stem = base[:-4] if base.endswith(".mp4") else base
        for suf, out in ((".mp4", "final.mp4"), (".subtitled.mp4", "final.subtitled.mp4"), (".srt", "captions.srt")):
            f = rdir / (stem + suf)
            target = ex / out
            if f.exists() and not target.exists():
                _link_or_copy(f, target)
        ex.mkdir(parents=True, exist_ok=True)
        meta = ex / "metadata.json"
        if not meta.exists():
            meta.write_text(json.dumps({
                "clip_id": cid, "order": order, "title": title, "slug": cslug,
                "source_ranges": clip.get("source_ranges", []),
            }, indent=2))
        index.append({"order": order, "slug": cslug, "title": title, "clip_id": cid})
    index.sort(key=lambda e: e["order"])
    return index


def write_project_json(project_dir: Path, project_slug: str, *, content_mode: str,
                       original_master: str, promoted_from: str, archived: list[str],
                       clips: list[dict[str, Any]], created_at: str | None = None) -> None:
    """Record this run's finished clips under project.json, partitioned by content
    mode. A run updates ONLY its own mode's slot (`modes.<mode>`) and preserves the
    others — so any mix/order of independent ad and excerpt runs leaves the project
    reflecting the latest run of each mode, with neither clobbering the other."""
    path = project_dir / "project.json"
    now = created_at or datetime.now(timezone.utc).isoformat()
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError):
            existing = {}
    modes = existing.get("modes") if isinstance(existing.get("modes"), dict) else {}
    modes[content_mode] = {
        "clips_dir": deliverables_dirname(content_mode),
        "generated_at": now,
        "provenance": {"promoted_from": promoted_from, "archived": archived},
        "clips": clips,
    }
    path.write_text(json.dumps({
        "schema_version": 2,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "source": {"slug": project_slug, "master": "media/source.mp4",
                   "original_master": original_master},
        "modes": modes,
    }, indent=2))


def generate_layout(project_dir: Path, render_manifest: Path | None, plan: dict[str, Any],
                    project_slug: str, *, content_mode: str, original_master: str,
                    promoted_from: str, archived: list[str] | None = None,
                    created_at: str | None = None) -> list[dict[str, Any]]:
    """Post-render convenience: build the <excerpts|ads>/ view (if a render manifest
    exists) and write project.json. Returns the clip index."""
    clips = (build_clips(project_dir, render_manifest, plan, content_mode)
             if render_manifest and render_manifest.exists() else [])
    write_project_json(project_dir, project_slug, content_mode=content_mode,
                       original_master=original_master, promoted_from=promoted_from,
                       archived=archived or [], clips=clips, created_at=created_at)
    return clips
