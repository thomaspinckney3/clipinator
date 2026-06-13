"""Shared fixtures: synthetic plan/transcript builders and a tiny test video.

Everything here is pure/in-memory except `tiny_video`, which shells out to
ffmpeg and is skipped when ffmpeg is unavailable.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

TITLE_CARD_VISUAL = "Generated black title card."
TITLE_CARD_AUDIO = "No source dialogue; silent generated title card."


def make_transcript(duration: float = 10.0, words_per_second: float = 2.0) -> dict:
    """A normalized transcript with evenly spaced word-level timestamps."""
    words = []
    t = 0.0
    step = 1.0 / words_per_second
    i = 0
    while t < duration:
        words.append({"text": f"word{i:03d}", "start": round(t, 3), "end": round(t + step * 0.8, 3)})
        t += step
        i += 1
    segments = [{"start": 0.0, "end": round(duration, 3),
                 "text": " ".join(w["text"] for w in words), "words": words}]
    return {
        "schema_version": 1,
        "provider": "test",
        "source": {"duration_seconds": round(duration, 3),
                   "effective_duration_seconds": round(duration, 3)},
        "text": segments[0]["text"],
        "segments": segments,
        "words": words,
    }


def make_clip(clip_id: str = "EX01", *, card: float = 4.5,
              segments: list[tuple[float, float]] | None = None,
              captions: list[tuple[float, float, str]] | None = None) -> dict:
    """One excerpt clip: a title-card timeline item plus source segments."""
    segments = segments if segments is not None else [(1.0, 9.0)]
    timeline = [{
        "order": 1, "screen_time_start": 0.0, "screen_time_end": card,
        "source_start": 0, "source_end": 0,
        "audio_source": TITLE_CARD_AUDIO, "visual_source": TITLE_CARD_VISUAL,
        "edit_action": "Show opening title card.", "caption_or_overlay": clip_id,
        "purpose": "title",
    }]
    cursor = card
    for order, (s_in, s_out) in enumerate(segments, start=2):
        dur = s_out - s_in
        timeline.append({
            "order": order, "screen_time_start": round(cursor, 3),
            "screen_time_end": round(cursor + dur, 3),
            "source_start": s_in, "source_end": s_out,
            "audio_source": f"Source dialogue {s_in}-{s_out}.",
            "visual_source": f"Source video {s_in}-{s_out}.",
            "edit_action": "Play source.", "caption_or_overlay": "", "purpose": "body",
        })
        cursor += dur
    caption_plan = [
        {"start_offset": round(s, 3), "end_offset": round(e, 3), "text": txt, "style_note": "x"}
        for (s, e, txt) in (captions or [])
    ]
    total = card + sum(b - a for a, b in segments)
    return {
        "clip_id": clip_id,
        "source_candidate_id": "cand-1",
        "concept_title": f"Title {clip_id}",
        "target_platform": "test",
        "target_aspect_ratio": "9:16",
        "target_duration_seconds": round(total, 3),
        "editorial_objective": "test",
        "why_this_ad_sells_the_full_video": "test",
        "source_ranges": [{"start": a, "end": b, "role": "core", "reason": "x"} for a, b in segments],
        "opening_hook": {"text": f"Title {clip_id}", "visual": TITLE_CARD_VISUAL,
                         "audio": TITLE_CARD_AUDIO, "duration_seconds": card},
        "timeline": timeline,
        "caption_plan": caption_plan,
        "audio_plan": {"primary_audio": "source", "sync_strategy": "native",
                       "music_or_roomtone": "none", "cleanup_notes": "none"},
        "visual_plan": {"first_frame": "card", "b_roll_strategy": "none",
                        "crop_strategy": "center", "needed_visual_checks": []},
        "media_request_ids": [],
        "verification_steps": [], "risk_notes": [], "success_criteria": [],
    }


def make_plan(clips: list[dict] | None = None) -> dict:
    return {
        "content_mode": "excerpt",
        "video_title": "Synthetic",
        "plan_summary": "test",
        "global_editing_rules": [],
        "clip_ads": clips if clips is not None else [make_clip()],
        "media_requests": [],
        "manual_review_queue": [{"start": 20.0, "end": 22.0, "reason": "x", "required_action": "none"}],
        "deferred_candidates": [],
        "notes_for_rendering": [],
    }


@pytest.fixture
def plan() -> dict:
    return make_plan()


@pytest.fixture
def transcript() -> dict:
    return make_transcript()


@pytest.fixture
def run_dir(tmp_path: Path, plan: dict, transcript: dict) -> Path:
    """A run root with a plan, transcript, and draft render manifest on disk."""
    root = tmp_path / "run"
    (root / "openai-runs" / "run-plan").mkdir(parents=True)
    (root / "transcripts").mkdir()
    (root / "renders" / "excerpt" / "draft").mkdir(parents=True)
    (root / "pipeline-manifest.json").write_text("{}", encoding="utf-8")
    (root / "source.mp4").write_bytes(b"\x00")  # placeholder; export resolves the path, not the bytes
    plan_path = root / "openai-runs" / "run-plan" / "plan-output.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    tr_path = root / "transcripts" / "t.json"
    tr_path.write_text(json.dumps(transcript), encoding="utf-8")
    manifest = {
        "source_video": str(root / "source.mp4"),
        "transcript": str(tr_path),
        "clips": [{"clip_id": c["clip_id"],
                   "output_path": str(root / "renders" / "excerpt" / "draft" / f"{c['clip_id']}.draft.mp4")}
                  for c in plan["clip_ads"]],
    }
    (root / "renders" / "excerpt" / "draft" / "render-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> Path:
    """A short 24000/1001 fps test video, for probe/render tests."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    path = tmp_path_factory.mktemp("video") / "tiny.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24000/1001:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=False, capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not build test video: {result.stderr.decode()[:200]}")
    return path
