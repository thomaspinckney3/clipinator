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
    (project_dir / "project.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "content_mode": content_mode,
        "source": {"slug": project_slug, "master": "media/source.mp4",
                   "original_master": original_master},
        "provenance": {"promoted_from": promoted_from, "archived": archived},
        "clips_dir": deliverables_dirname(content_mode),
        "clips": clips,
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
