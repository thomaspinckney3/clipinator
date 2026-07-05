from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import opentimelineio as otio

from extractor.edit_list import (
    SCHEMA_VERSION,
    apply_edits,
    clip_segments,
    export_captions,
    find_run_root,
    read_json,
    relative_path,
    resolve_recorded_path,
    review_kind,
    title_card_seconds,
    write_json,
)
from extractor.render_plan import transcript_words_from_data
from extractor.timebase import probe_frame_rate

# OpenTimelineIO interchange for the human review gate. OTIO is the canonical
# editor-facing edit list (alongside the lighter TOML/HTML path): plan ->
# .otio carries each excerpt as a video track of source-range clips plus a
# generated title card, with the clipinator-specific decision/title/captions/
# notes payload in per-track metadata. Premiere/Resolve read .otio natively;
# CMX 3600 EDL and FCPXML are produced as one-way views via OTIO's adapters
# (cut geometry only — they cannot carry our metadata). Import reads an .otio
# back into the same edit-list overlay the TOML path uses, so both funnel
# through edit_list.apply_edits and stay consistent.
#
# When the run has an `outline`, a "Source outline" track is added to the .otio: a
# full-source reference clip carrying one Marker per content unit (title + one-line
# summary, colored by whether the unit made the cut), so an editor can grasp the
# whole video from the timeline without watching it. Premiere/Resolve read these
# markers natively. The single-track EDL/FCPXML views can't hold a second overview
# track, so they stay cut-geometry-only and omit it. It's a scannable map, not a cut.

METADATA_NS = "clipinator"
DEFAULT_FRAME_RATE = Fraction(24000, 1001)
NLE_FORMATS = {"edl": "cmx_3600", "fcpxml": "fcpx_xml"}
TITLE_GENERATOR_KIND = "clipinator_black_title_card"


def _rt(seconds: float, rate: Fraction) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(round(float(seconds) * float(rate)), float(rate))


def _range(start: float, duration: float, rate: Fraction) -> otio.opentime.TimeRange:
    return otio.opentime.TimeRange(_rt(start, rate), _rt(duration, rate))


def probe_duration(video: Path) -> float | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# plan -> OTIO


def _kept_source_ranges(plan: dict[str, Any]) -> list[tuple[float, float]]:
    """Every source span the plan actually keeps, so the outline can mark which
    content units made the cut."""
    ranges: list[tuple[float, float]] = []
    for clip in plan.get("clip_ads", []) or []:
        for seg in clip_segments(clip):
            ranges.append((float(seg["source_in"]), float(seg["source_out"])))
    return ranges


def _unit_is_kept(start: float, end: float, kept: list[tuple[float, float]]) -> bool:
    return any(ks < end and start < ke for ks, ke in kept)


def outline_track(
    units: list[dict[str, Any]] | None,
    *,
    source_url: str,
    source_rate: Fraction,
    source_duration: float | None,
    kept_ranges: list[tuple[float, float]],
) -> otio.schema.Track:
    """A full-source reference track whose markers summarize each content unit
    (title + one-line gist, at true source timecodes), so an editor can grasp the
    whole video from the timeline without watching it. It's a scannable map, not a
    cut: markers ride in the canonical .otio only (Premiere/Resolve read them
    natively; the single-track EDL/FCPXML views can't carry a second overview track
    and omit it). Tagged role=source_outline so import ignores it — never a clip."""
    track = otio.schema.Track(name="Source outline", kind=otio.schema.TrackKind.Video)
    track.metadata[METADATA_NS] = {"role": "source_outline"}
    units = [u for u in (units or [])
             if isinstance(u, dict) and u.get("start") is not None and u.get("end") is not None]
    if not units:
        return track
    span = source_duration or max(float(u["end"]) for u in units)
    frame = 1.0 / float(source_rate)
    source = otio.schema.Clip(
        name="Full source",
        source_range=_range(0.0, span, source_rate),
        media_reference=otio.schema.ExternalReference(
            target_url=source_url, available_range=_range(0.0, span, source_rate)),
    )
    for u in units:
        start, end = float(u["start"]), float(u["end"])
        kept = _unit_is_kept(start, end, kept_ranges)
        title = str(u.get("title") or u.get("summary") or "").strip() or f"Section {u.get('index')}"
        marker = otio.schema.Marker(
            name=title,
            marked_range=_range(start, max(end - start, frame), source_rate),
            color=otio.schema.MarkerColor.GREEN if kept else otio.schema.MarkerColor.RED,
        )
        marker.metadata[METADATA_NS] = {
            "index": u.get("index"),
            "summary": str(u.get("summary") or ""),
            "kept": kept,
        }
        source.markers.append(marker)
    track.append(source)
    return track


def load_outline_units(run_root: Path | None, explicit: Path | None) -> list[dict[str, Any]]:
    """The `outline` stage's content units (index/start/end/title/summary) for the
    storyboard markers. Prefer an explicit path; else discover the run's
    outline-output.json. Missing outline is fine — markers are just omitted."""
    path: Path | None = None
    if explicit is not None:
        path = explicit.expanduser().resolve()
    elif run_root is not None:
        matches = sorted(run_root.glob("openai-runs/*-outline/outline-output.json"))
        path = matches[-1] if matches else None
    if path is None or not path.exists():
        return []
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return []
    return [u for u in (data.get("units") or []) if isinstance(u, dict)]


def plan_to_otio(
    plan: dict[str, Any],
    *,
    source_url: str,
    source_rate: Fraction,
    transcript_words: list[dict[str, Any]],
    plan_rel: str,
    transcript_rel: str | None,
    outline_units: list[dict[str, Any]] | None = None,
    source_duration: float | None = None,
) -> otio.schema.Timeline:
    timeline = otio.schema.Timeline(name=str(plan.get("video_title", "clipinator")))
    timeline.metadata[METADATA_NS] = {
        "schema_version": SCHEMA_VERSION,
        "content_mode": str(plan.get("content_mode", "ad")),
        "source_video": source_url,
        "source_frame_rate": str(source_rate),
        "plan": plan_rel,
        "transcript": transcript_rel,
    }
    for clip in plan.get("clip_ads", []) or []:
        clip_id = str(clip.get("clip_id", ""))
        segments = clip_segments(clip)
        card = title_card_seconds(clip)
        track = otio.schema.Track(name=clip_id, kind=otio.schema.TrackKind.Video)
        track.metadata[METADATA_NS] = {
            "clip_id": clip_id,
            "decision": "keep",
            "title": str(clip.get("concept_title", "")),
            "title_card_seconds": card,
            "notes": "",
            "captions": [
                {"source_start": c["source_start"], "source_end": c["source_end"], "text": c["text"]}
                for c in export_captions(clip, card, segments, transcript_words)
            ],
            "source_candidate_id": clip.get("source_candidate_id"),
        }
        if card > 0:
            title = otio.schema.Clip(
                name="Title card",
                source_range=_range(0.0, card, source_rate),
                media_reference=otio.schema.GeneratorReference(
                    name="title-card", generator_kind=TITLE_GENERATOR_KIND,
                    parameters={"text": str(clip.get("concept_title", ""))},
                ),
            )
            track.append(title)
        for seg in segments:
            track.append(otio.schema.Clip(
                name=clip_id,
                source_range=_range(seg["source_in"], seg["source_out"] - seg["source_in"], source_rate),
                media_reference=otio.schema.ExternalReference(target_url=source_url),
            ))
        timeline.tracks.append(track)
    outline = outline_track(
        outline_units, source_url=source_url, source_rate=source_rate,
        source_duration=source_duration, kept_ranges=_kept_source_ranges(plan),
    )
    if len(outline):  # only when the run has an outline
        timeline.tracks.append(outline)
    return timeline


def flatten_for_nle(
    plan: dict[str, Any],
    *,
    source_url: str,
    source_rate: Fraction,
    source_duration: float | None,
) -> otio.schema.Timeline:
    """A single-track cuts list for EDL/FCPXML export (one-way view).

    EDL is single-track and FCPXML's adapter cannot represent our generated
    title cards, so the NLE views carry only the source segments, laid end to
    end across all excerpts with clip names marking boundaries. Metadata,
    captions, and title cards are intentionally dropped — these are disposable
    views, not the canonical edit list.
    """
    timeline = otio.schema.Timeline(name=str(plan.get("video_title", "clipinator")))
    track = otio.schema.Track(kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)
    available = (
        otio.opentime.TimeRange(_rt(0.0, source_rate), _rt(source_duration, source_rate))
        if source_duration else None
    )
    for clip in plan.get("clip_ads", []) or []:
        clip_id = str(clip.get("clip_id", ""))
        for seg in clip_segments(clip):
            track.append(otio.schema.Clip(
                name=clip_id,
                source_range=_range(seg["source_in"], seg["source_out"] - seg["source_in"], source_rate),
                media_reference=otio.schema.ExternalReference(
                    target_url=source_url, available_range=available),
            ))
    return timeline


# ---------------------------------------------------------------------------
# OTIO -> edit-list overlay


def otio_to_edits(timeline: otio.schema.Timeline) -> dict[str, Any]:
    meta = dict(timeline.metadata.get(METADATA_NS, {}) or {})
    clips: list[dict[str, Any]] = []
    for track in timeline.tracks:
        tmeta = dict(track.metadata.get(METADATA_NS, {}) or {})
        if tmeta.get("role") == "source_outline":
            continue  # the storyboard overview track is a map, not a clip
        clip_id = str(tmeta.get("clip_id") or track.name or "")
        segments: list[dict[str, Any]] = []
        for child in track:
            if not isinstance(child, otio.schema.Clip):
                continue
            if isinstance(child.media_reference, otio.schema.GeneratorReference):
                continue  # title card / generated beat, not a source segment
            sr = child.source_range
            if sr is None:
                continue
            start = sr.start_time.to_seconds()
            end = (sr.start_time + sr.duration).to_seconds()
            segments.append({"source_in": round(start, 3), "source_out": round(end, 3)})
        edit: dict[str, Any] = {
            "clip_id": clip_id,
            "decision": str(tmeta.get("decision", "keep")),
            "title": tmeta.get("title", ""),
            "title_card_seconds": tmeta.get("title_card_seconds", 0.0),
            "notes": tmeta.get("notes", ""),
            "segments": segments,
        }
        captions = tmeta.get("captions")
        if captions is not None:
            edit["captions"] = [
                {"source_start": float(c["source_start"]), "source_end": float(c["source_end"]),
                 "text": str(c.get("text", ""))}
                for c in captions
            ]
        clips.append(edit)
    return {
        "schema_version": int(meta.get("schema_version", SCHEMA_VERSION)),
        "content_mode": meta.get("content_mode"),
        "source_video": meta.get("source_video"),
        "plan": meta.get("plan"),
        "transcript": meta.get("transcript"),
        "clips": clips,
    }


# ---------------------------------------------------------------------------
# export / import


def write_nle_views(timeline: otio.schema.Timeline, output_dir: Path, stem: str) -> dict[str, str]:
    """Write one-way EDL/FCPXML views; warn (don't fail) if a format can't represent the timeline."""
    written: dict[str, str] = {}
    for ext, adapter in NLE_FORMATS.items():
        path = output_dir / f"{stem}.{ext}"
        try:
            otio.adapters.write_to_file(timeline, str(path), adapter_name=adapter)
            written[ext] = str(path)
        except Exception as exc:  # adapters raise various types; best-effort views
            print(f"note: could not write {ext} view ({type(exc).__name__}: {str(exc)[:80]})",
                  file=sys.stderr)
    return written


def run_export(args: argparse.Namespace) -> int:
    plan_path = args.plan_output.expanduser().resolve()
    plan = read_json(plan_path)
    render_manifest = None
    manifest_dir: Path | None = None
    if args.render_manifest:
        manifest_path = args.render_manifest.expanduser().resolve()
        render_manifest = read_json(manifest_path)
        manifest_dir = manifest_path.parent
    run_root = find_run_root(manifest_dir or plan_path.parent)

    source_video = args.source_video
    if source_video is None and render_manifest and render_manifest.get("source_video"):
        relink = [run_root.parent, run_root] if run_root else []
        source_video = resolve_recorded_path(
            str(render_manifest["source_video"]), manifest_dir or plan_path.parent, relink)
    if source_video is None:
        raise RuntimeError("--source-video is required when the render manifest has no resolvable path.")
    source_video = source_video.expanduser().resolve()

    transcript_path = args.transcript
    if transcript_path is None and render_manifest and render_manifest.get("transcript"):
        relink = [run_root / "transcripts"] if run_root else []
        transcript_path = resolve_recorded_path(
            str(render_manifest["transcript"]), manifest_dir or plan_path.parent, relink)
    words: list[dict[str, Any]] = []
    if transcript_path is not None:
        transcript_path = transcript_path.expanduser().resolve()
        words = transcript_words_from_data(read_json(transcript_path))

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    elif run_root is not None:
        output_dir = run_root / "review" / review_kind(plan_path, manifest_dir)
    else:
        output_dir = manifest_dir if manifest_dir else plan_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rate = probe_frame_rate(source_video) or DEFAULT_FRAME_RATE
    source_duration = probe_duration(source_video)
    outline_units = load_outline_units(run_root, getattr(args, "outline", None))
    timeline = plan_to_otio(
        plan,
        source_url=relative_path(source_video, output_dir) or str(source_video),
        source_rate=source_rate,
        transcript_words=words,
        plan_rel=relative_path(plan_path, output_dir) or str(plan_path),
        transcript_rel=relative_path(transcript_path, output_dir) if transcript_path else None,
        outline_units=outline_units,
        source_duration=source_duration,
    )
    otio_path = output_dir / "edit-list.otio"
    otio.adapters.write_to_file(timeline, str(otio_path), adapter_name="otio_json")
    nle_timeline = flatten_for_nle(
        plan, source_url=relative_path(source_video, output_dir) or str(source_video),
        source_rate=source_rate, source_duration=source_duration,
    )
    views = write_nle_views(nle_timeline, output_dir, "edit-list")

    print(f"Exported {len(plan.get('clip_ads', []) or [])} clip(s) to OTIO:")
    if outline_units:
        print(f"  + storyboard: {len(outline_units)} content-unit markers on a Source outline track")
    print(f"  OTIO (Premiere/Resolve): {otio_path}")
    for ext, path in views.items():
        print(f"  {ext.upper()}: {path}")
    print(f"Apply edits with: otio-edl import {otio_path}")
    return 0


def run_import(args: argparse.Namespace) -> int:
    otio_path = args.otio.expanduser().resolve()
    timeline = otio.adapters.read_from_file(str(otio_path), adapter_name="otio_json")
    edits = otio_to_edits(timeline)
    if int(edits.get("schema_version", 0)) != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported OTIO schema_version {edits.get('schema_version')!r}; expected {SCHEMA_VERSION}."
        )

    otio_dir = otio_path.parent
    run_root = find_run_root(otio_dir)

    if args.plan_output:
        plan_path: Path | None = args.plan_output.expanduser().resolve()
    else:
        raw_plan = str(edits.get("plan") or "")
        relink = [run_root / "openai-runs" / Path(raw_plan).parent.name] if run_root and raw_plan else []
        plan_path = resolve_recorded_path(raw_plan, otio_dir, relink)
    if plan_path is None or not plan_path.exists():
        raise RuntimeError(f"Plan not found: {edits.get('plan')!r}. Pass --plan-output to override.")
    plan = read_json(plan_path)

    if args.transcript:
        transcript_path: Path | None = args.transcript.expanduser().resolve()
    else:
        relink = [run_root / "transcripts"] if run_root else []
        transcript_path = resolve_recorded_path(str(edits.get("transcript") or ""), otio_dir, relink)
    words: list[dict[str, Any]] = []
    if transcript_path and transcript_path.exists():
        words = transcript_words_from_data(read_json(transcript_path))
    else:
        print("warning: transcript not found; caption regeneration is unavailable.")

    relink = [run_root.parent, run_root] if run_root else []
    source_video = resolve_recorded_path(str(edits.get("source_video") or ""), otio_dir, relink)
    duration = probe_duration(source_video) if source_video and source_video.exists() else None
    if duration is None:
        print("WARNING: source video unresolved; segment end-of-video bounds are NOT validated.")

    revised = apply_edits(plan, edits, words, duration, source=str(otio_path))
    output_path = args.output.expanduser().resolve() if args.output else otio_dir / "plan-output.human.json"
    write_json(output_path, revised)
    he = revised["human_edit"]
    print(f"Applied OTIO edit list: {len(revised['clip_ads'])} kept "
          f"({len(he['rework_clip_ids'])} rework), {len(he['dropped_clip_ids'])} dropped.")
    print(f"Wrote {output_path}")
    print("Render it with: render-plan --plan-output " + str(output_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a plan as an OpenTimelineIO edit list (+ EDL/FCPXML views) and import it back."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Write edit-list.otio (+ EDL/FCPXML) from a plan.")
    export.add_argument("--plan-output", type=Path, required=True)
    export.add_argument("--render-manifest", type=Path, default=None)
    export.add_argument("--transcript", type=Path, default=None)
    export.add_argument("--source-video", type=Path, default=None)
    export.add_argument("--outline", type=Path, default=None,
                        help="outline-output.json for storyboard markers (auto-discovered if omitted).")
    export.add_argument("--output-dir", type=Path, default=None)
    export.set_defaults(func=run_export)

    imp = sub.add_parser("import", help="Apply an edited edit-list.otio to its plan.")
    imp.add_argument("otio", type=Path, help="Path to the edited edit-list.otio.")
    imp.add_argument("--plan-output", type=Path, default=None)
    imp.add_argument("--transcript", type=Path, default=None)
    imp.add_argument("--output", type=Path, default=None)
    imp.set_defaults(func=run_import)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
