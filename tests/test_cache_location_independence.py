"""The stage caches are content-addressed, so relocating a project — e.g. when the
source-centric layout pulls a master to <project>/media/source.mp4, renaming it —
must not change any cache key. These tests pin that the location-bearing fields
(absolute paths, OS filenames, stems) are stripped before a manifest/transcript is
rendered into a stage's prompt/fingerprint, while genuine content still moves the key.
"""
from __future__ import annotations

import json

from extractor import openai_moments, openai_plan, openai_strategy, stage_cache


def _manifest(video_dir: str, name: str) -> dict:
    """A minimal pipeline manifest as it would look at one filesystem location."""
    stem = name.rsplit(".", 1)[0]
    return {
        "video": {"path": f"{video_dir}/{name}", "filename": name, "stem": stem},
        "media": {"duration_seconds": 12.0, "video": {"codec": "h264", "width": 1920}},
        "visual": {"frame_interval_seconds": 2.0, "sampled_frames": [], "scene_changes": []},
        "audio": {"silence_summary": None, "silences": []},
        "transcript": {
            "path": f"{video_dir}/{stem}/transcript.json",
            "segment_count": 3,
            "word_count": 40,
            "qa": {"score": 90, "finding_counts": {}, "top_findings": []},
        },
    }


def _transcript(video_dir: str, name: str) -> dict:
    stem = name.rsplit(".", 1)[0]
    return {
        "provider": "whisper",
        "model": "large-v3",
        "language": "en",
        "source": {
            "path": f"{video_dir}/{name}",
            "filename": name,
            "duration_seconds": 12.0,
            "truncated": False,
        },
        "segments": [{"start": 0.0, "end": 4.0, "text": "the moon is bright"}],
    }


def test_compact_manifest_ignores_where_the_master_lives():
    original = _manifest("/Users/tp/Downloads", "Ergo-Braver-1.4.mp4")
    relocated = _manifest("/Volumes/T9/extract/ergo-braver-1-4/media", "source.mp4")
    # Same content, different location (and even a different on-disk filename) ->
    # identical compact form -> identical source-materials hash -> cache hit.
    assert openai_strategy.compact_manifest(original) == openai_strategy.compact_manifest(relocated)
    assert "video" not in openai_strategy.compact_manifest(original)


def test_compact_transcript_ignores_source_path_and_filename():
    original = _transcript("/Users/tp/Downloads", "Ergo-Braver-1.4.mp4")
    relocated = _transcript("/Volumes/T9/extract/ergo-braver-1-4/media", "source.mp4")
    a = openai_strategy.compact_transcript(original)
    b = openai_strategy.compact_transcript(relocated)
    assert a == b
    # Content that matters is preserved.
    assert a["source"]["duration_seconds"] == 12.0
    assert a["segments"][0]["text"] == "the moon is bright"
    assert "path" not in a["source"] and "filename" not in a["source"]


def test_compact_transcript_still_tracks_spoken_content():
    original = _transcript("/Users/tp/Downloads", "Ergo-Braver-1.4.mp4")
    reworded = _transcript("/Users/tp/Downloads", "Ergo-Braver-1.4.mp4")
    reworded["segments"][0]["text"] = "the sun is bright"
    assert openai_strategy.compact_transcript(original) != openai_strategy.compact_transcript(reworded)


def _write_visual_index(tmp_path, video_dir: str, name: str):
    stem = name.rsplit(".", 1)[0]
    manifest = {
        "schema_version": 1,
        "video": {"path": f"{video_dir}/{name}", "filename": name, "stem": stem},
        "frame_interval_seconds": 2.0,
        "frame_count": 6,
        "candidate_sheets": [],
        "usage_note": "note",
    }
    path = tmp_path / f"{stem}-manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_compact_visual_index_manifest_ignores_master_location(tmp_path):
    for module in (openai_moments, openai_plan):
        original = _write_visual_index(tmp_path, "/Users/tp/Downloads", "Ergo-Braver-1.4.mp4")
        relocated = _write_visual_index(
            tmp_path, "/Volumes/T9/extract/ergo/media", "source.mp4"
        )
        a = module.compact_visual_index_manifest(original, max_images=4)
        b = module.compact_visual_index_manifest(relocated, max_images=4)
        assert a == b, module.__name__
        assert "video" not in a, module.__name__


def test_without_location_helper():
    src = {"path": "/x/y.mp4", "filename": "y.mp4", "stem": "y", "duration_seconds": 3.0}
    assert stage_cache.without_location(src) == {"duration_seconds": 3.0}
    assert stage_cache.without_location(None) == {}
