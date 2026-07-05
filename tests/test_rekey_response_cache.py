"""The re-key tool's safety gate: a relocated run's response may only be revived
when the current prompt differs from the stored one purely in filesystem paths."""
from __future__ import annotations

from pathlib import Path

from extractor import rekey_response_cache as rk


JUNE = """\
COMPACT VIDEO EVIDENCE MANIFEST JSON:
{
  "video": {
    "path": "/old/place/Foo.mp4",
    "filename": "Foo.mp4",
    "stem": "Foo"
  },
  "media": {
    "duration_seconds": 12.0,
    "video": {
      "codec": "h264",
      "width": 1920
    }
  },
  "transcript": {
    "path": "/old/place/Foo/transcript.json",
    "segment_count": 3
  }
}
"""

# What the current (path-stripped) code renders for the same, relocated content.
CURRENT = """\
COMPACT VIDEO EVIDENCE MANIFEST JSON:
{
  "media": {
    "duration_seconds": 12.0,
    "video": {
      "codec": "h264",
      "width": 1920
    }
  },
  "transcript": {
    "segment_count": 3
  }
}
"""


def test_gate_passes_when_only_paths_differ():
    ok, offenders = rk.gate_paths_only(CURRENT, JUNE)
    assert ok, offenders


def test_strip_location_keeps_media_video_but_drops_manifest_video():
    stripped = rk.strip_location(JUNE)
    assert '"codec": "h264"' in stripped          # media.video is content, kept
    assert "/old/place/Foo.mp4" not in stripped   # location video block dropped
    assert "transcript.json" not in stripped       # stray path line dropped
    assert '"segment_count": 3' in stripped        # real content kept


def test_gate_fails_when_real_content_differs():
    changed = CURRENT.replace('"duration_seconds": 12.0', '"duration_seconds": 99.0')
    ok, offenders = rk.gate_paths_only(changed, JUNE)
    assert not ok
    assert any("duration_seconds" in o for o in offenders)


def test_gate_fails_when_a_prompt_section_is_missing():
    # e.g. a template change that added/removed a whole block — must not be revived.
    truncated = "\n".join(CURRENT.splitlines()[:6])
    ok, _ = rk.gate_paths_only(truncated, JUNE)
    assert not ok


def test_stage_and_content_mode_detection():
    assert rk.stage_of(Path("run-x-excerpt-strategy")) == "strategy"
    assert rk.stage_of(Path("run-x-moments")) == "moments"
    assert rk.stage_of(Path("run-x-critique")) == "critique"   # detected, but unsupported
    assert rk.stage_of(Path("run-x-render")) is None
    assert rk.content_mode_of(Path("run-x-excerpt-strategy")) == "excerpt"
    assert rk.content_mode_of(Path("run-x-strategy")) == "ad"
