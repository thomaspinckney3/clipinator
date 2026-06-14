import json
import tomllib
from pathlib import Path

import pytest

from extractor import edit_list
from extractor.edit_list import (
    MIN_SEGMENT_SECONDS,
    build_caption_plan,
    chunk_captions,
    clip_segments,
    find_run_root,
    rebuild_clip,
    render_offset_to_source,
    resolve_recorded_path,
    review_kind,
    run_export,
    run_import,
    source_to_render_offset,
    title_card_seconds,
    validate_segments,
)
from extractor.render_plan import transcript_words_from_data

from conftest import make_clip, make_plan, make_transcript


# --- caption offset mapping (round-trip between source and render time) -------

class TestOffsetMapping:
    def test_round_trip_within_segment(self):
        segs = [{"source_in": 1.0, "source_out": 9.0}]
        card = 4.5
        # render offset 6.0 -> source 2.5 -> back to render 6.0
        src = render_offset_to_source(6.0, card, segs)
        assert src == pytest.approx(2.5)
        assert source_to_render_offset(src, card, segs) == pytest.approx(6.0)

    def test_multi_segment_continuity(self):
        segs = [{"source_in": 1.0, "source_out": 3.0}, {"source_in": 10.0, "source_out": 14.0}]
        card = 0.0
        # second segment starts at render time 2.0 (after first 2s segment)
        assert source_to_render_offset(10.0, card, segs) == pytest.approx(2.0)
        assert render_offset_to_source(2.5, card, segs) == pytest.approx(10.5)

    def test_out_of_range_returns_none_not_clamped(self):
        # F5: an offset past the assembled timeline must not clamp into the last segment
        segs = [{"source_in": 1.0, "source_out": 3.0}]
        card = 0.0
        assert render_offset_to_source(100.0, card, segs) is None

    def test_before_any_segment_returns_none(self):
        segs = [{"source_in": 5.0, "source_out": 6.0}]
        assert source_to_render_offset(0.0, 0.0, segs) is None


class TestChunkCaptions:
    def test_groups_within_limits(self):
        words = [{"text": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.3} for i in range(10)]
        caps = chunk_captions(words)
        assert caps
        for c in caps:
            assert c["source_end"] > c["source_start"]
            assert len(c["text"].split()) <= edit_list.CAPTION_MAX_WORDS

    def test_empty(self):
        assert chunk_captions([]) == []

    def test_splits_on_large_gap(self):
        words = [{"text": "a", "start": 0.0, "end": 0.3}, {"text": "b", "start": 5.0, "end": 5.3}]
        assert len(chunk_captions(words)) == 2


class TestBuildCaptionPlan:
    def test_maps_onto_segments(self):
        segs = [{"source_in": 1.0, "source_out": 9.0}]
        anchored = [{"source_start": 2.0, "source_end": 4.0, "text": "hi"}]
        plan, dormant = build_caption_plan(anchored, card=4.5, segments=segs)
        assert dormant == 0
        assert plan[0]["text"] == "hi"
        assert plan[0]["start_offset"] == pytest.approx(5.5)  # 4.5 card + (2.0-1.0)

    def test_trimmed_caption_is_dormant_not_dropped_silently(self):
        segs = [{"source_in": 1.0, "source_out": 9.0}]
        anchored = [{"source_start": 50.0, "source_end": 52.0, "text": "gone"}]
        plan, dormant = build_caption_plan(anchored, card=0.0, segments=segs)
        assert plan == [] and dormant == 1

    def test_partial_overlap_clamped(self):
        segs = [{"source_in": 5.0, "source_out": 9.0}]
        anchored = [{"source_start": 3.0, "source_end": 7.0, "text": "spans in-point"}]
        plan, dormant = build_caption_plan(anchored, card=0.0, segments=segs)
        assert dormant == 0
        assert plan[0]["start_offset"] == pytest.approx(0.0)  # clamped to segment start


# --- segment validation ------------------------------------------------------

class TestValidateSegments:
    def test_ok(self):
        validate_segments("c", [{"source_in": 0.0, "source_out": 5.0}], duration=10.0)

    def test_empty_rejected(self):
        with pytest.raises(RuntimeError, match="at least one segment"):
            validate_segments("c", [], duration=10.0)

    def test_too_short_rejected(self):
        with pytest.raises(RuntimeError, match="minimum"):
            validate_segments("c", [{"source_in": 1.0, "source_out": 1.0}], duration=10.0)

    def test_negative_start_rejected(self):
        with pytest.raises(RuntimeError, match="before 0"):
            validate_segments("c", [{"source_in": -1.0, "source_out": 5.0}], duration=10.0)

    def test_beyond_duration_rejected(self):
        with pytest.raises(RuntimeError, match="source"):
            validate_segments("c", [{"source_in": 0.0, "source_out": 99.0}], duration=10.0)

    def test_overlap_rejected(self):
        segs = [{"source_in": 0.0, "source_out": 5.0}, {"source_in": 3.0, "source_out": 8.0}]
        with pytest.raises(RuntimeError, match="order and non-overlapping"):
            validate_segments("c", segs, duration=10.0)

    def test_no_duration_skips_upper_bound(self):
        # bounds unverifiable -> still validates the rest, no crash
        validate_segments("c", [{"source_in": 0.0, "source_out": 99.0}], duration=None)


# --- rebuild_clip ------------------------------------------------------------

class TestRebuildClip:
    def _words(self):
        return transcript_words_from_data(make_transcript(12.0))

    def test_unedited_captions_not_locked(self):
        # F1: captions matching the transcript seed must NOT set human_edited_captions
        clip = make_clip(segments=[(1.0, 9.0)])
        edit = {"clip_id": "EX01", "decision": "keep", "title": "T",
                "title_card_seconds": 4.5, "segments": [{"source_in": 1.0, "source_out": 9.0}]}
        out = rebuild_clip(clip, edit, self._words())
        assert out["human_edited_captions"] is False
        assert out["caption_plan_source"] is None

    def test_edited_captions_locked(self):
        # F1: genuinely changed caption text sets the flag so it renders
        clip = make_clip(segments=[(1.0, 9.0)])
        edit = {"clip_id": "EX01", "decision": "keep", "title": "T", "title_card_seconds": 4.5,
                "segments": [{"source_in": 1.0, "source_out": 9.0}],
                "captions": [{"source_start": 2.0, "source_end": 4.0, "text": "HUMAN EDIT"}]}
        out = rebuild_clip(clip, edit, self._words())
        assert out["human_edited_captions"] is True
        assert out["caption_plan_source"] == "render_time"
        assert any("HUMAN EDIT" in c["text"] for c in out["caption_plan"])

    def test_timeline_and_duration(self):
        clip = make_clip(segments=[(1.0, 9.0)])
        edit = {"clip_id": "EX01", "decision": "keep", "title": "New Title",
                "title_card_seconds": 3.0,
                "segments": [{"source_in": 2.0, "source_out": 6.0}]}
        out = rebuild_clip(clip, edit, self._words())
        assert out["concept_title"] == "New Title"
        # 3.0 card + 4.0 segment
        assert out["target_duration_seconds"] == pytest.approx(7.0)
        orders = [item["order"] for item in out["timeline"]]
        assert orders == sorted(orders)
        body = out["timeline"][-1]
        assert body["source_start"] == 2.0 and body["source_end"] == 6.0

    def test_framing_offset_persisted_and_clamped(self):
        clip = make_clip(segments=[(1.0, 9.0)])
        base = {"clip_id": "EX01", "title": "T", "title_card_seconds": 4.5,
                "segments": [{"source_in": 1.0, "source_out": 9.0}]}
        out = rebuild_clip(clip, {**base, "framing_offset": 0.2}, self._words())
        assert out["framing_offset"] == 0.2
        # clamped to the supported range
        out = rebuild_clip(clip, {**base, "framing_offset": 5.0}, self._words())
        assert out["framing_offset"] == 0.4

    def test_framing_offset_zero_not_stored(self):
        clip = make_clip(segments=[(1.0, 9.0)])
        clip["framing_offset"] = 0.3  # a prior reframe
        edit = {"clip_id": "EX01", "title": "T", "title_card_seconds": 4.5,
                "segments": [{"source_in": 1.0, "source_out": 9.0}], "framing_offset": 0.0}
        out = rebuild_clip(clip, edit, self._words())
        assert "framing_offset" not in out  # recentering clears the override

    def test_empty_title_rejected(self):
        clip = make_clip()
        edit = {"clip_id": "EX01", "title": "  ", "title_card_seconds": 4.5,
                "segments": [{"source_in": 1.0, "source_out": 9.0}]}
        with pytest.raises(RuntimeError, match="title"):
            rebuild_clip(clip, edit, self._words())

    def test_preserves_timeline_item_when_segments_unchanged(self):
        clip = make_clip(segments=[(1.0, 9.0)])
        clip["timeline"][-1]["edit_action"] = "SPECIAL B-ROLL CUE"
        edit = {"clip_id": "EX01", "title": "T", "title_card_seconds": 4.5,
                "segments": [{"source_in": 1.0, "source_out": 9.0}]}
        out = rebuild_clip(clip, edit, self._words())
        assert out["timeline"][-1]["edit_action"] == "SPECIAL B-ROLL CUE"


# --- path resolution / relink (F6) -------------------------------------------

class TestResolveRecordedPath:
    def test_relative_resolves_against_anchor(self, tmp_path):
        (tmp_path / "a").mkdir()
        target = tmp_path / "a" / "f.json"
        target.write_text("{}")
        got = resolve_recorded_path("f.json", tmp_path / "a", [])
        assert got == target.resolve()

    def test_relink_single_match(self, tmp_path):
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "vid.mp4").write_text("x")
        got = resolve_recorded_path("/gone/vid.mp4", tmp_path, [tmp_path / "b"])
        assert got == (tmp_path / "b" / "vid.mp4").resolve()

    def test_relink_ambiguous_raises(self, tmp_path):
        for d in ("a", "b"):
            (tmp_path / d).mkdir()
            (tmp_path / d / "vid.mp4").write_text("x")
        with pytest.raises(RuntimeError, match="Ambiguous relink"):
            resolve_recorded_path("/gone/vid.mp4", tmp_path, [tmp_path / "a", tmp_path / "b"])

    def test_no_match_returns_none(self, tmp_path):
        assert resolve_recorded_path("/gone/x.mp4", tmp_path, [tmp_path]) is None


class TestRunRootHelpers:
    def test_find_run_root(self, run_dir):
        deep = run_dir / "renders" / "excerpt" / "draft"
        assert find_run_root(deep) == run_dir

    def test_review_kind(self, tmp_path):
        assert review_kind(tmp_path / "x-revise" / "plan-output.json", None) == "final"
        assert review_kind(tmp_path / "x-plan" / "plan-output.json", None) == "draft"
        assert review_kind(tmp_path / "x" / "p.json", tmp_path / "stage5") == "final"


# --- export / import round-trip ----------------------------------------------

class TestExportImport:
    def _args(self, **kw):
        import argparse
        return argparse.Namespace(**kw)

    def test_export_emits_parseable_toml(self, run_dir, capsys):
        plan_path = run_dir / "openai-runs" / "run-plan" / "plan-output.json"
        manifest = run_dir / "renders" / "excerpt" / "draft" / "render-manifest.json"
        rc = run_export(self._args(plan_output=plan_path, render_manifest=manifest,
                                   transcript=None, source_video=None, output_dir=None))
        assert rc == 0
        toml_path = run_dir / "review" / "draft" / "edit-list.toml"
        html_path = run_dir / "review" / "draft" / "edit-list.html"
        assert toml_path.exists() and html_path.exists()
        data = tomllib.loads(toml_path.read_text())
        assert data["schema_version"] == edit_list.SCHEMA_VERSION
        assert data["clips"][0]["clip_id"] == "EX01"

    def test_export_paths_are_relative(self, run_dir):
        plan_path = run_dir / "openai-runs" / "run-plan" / "plan-output.json"
        manifest = run_dir / "renders" / "excerpt" / "draft" / "render-manifest.json"
        run_export(self._args(plan_output=plan_path, render_manifest=manifest,
                              transcript=None, source_video=None, output_dir=None))
        data = tomllib.loads((run_dir / "review" / "draft" / "edit-list.toml").read_text())
        assert not Path(data["plan"]).is_absolute()
        assert data["plan"].startswith("..")

    def test_unedited_round_trip(self, run_dir):
        plan_path = run_dir / "openai-runs" / "run-plan" / "plan-output.json"
        manifest = run_dir / "renders" / "excerpt" / "draft" / "render-manifest.json"
        run_export(self._args(plan_output=plan_path, render_manifest=manifest,
                              transcript=None, source_video=None, output_dir=None))
        toml_path = run_dir / "review" / "draft" / "edit-list.toml"
        out = run_dir / "out.json"
        rc = run_import(self._args(edit_list=toml_path, plan_output=None,
                                   transcript=None, output=out))
        assert rc == 0
        result = json.loads(out.read_text())
        assert [c["clip_id"] for c in result["clip_ads"]] == ["EX01"]
        assert "human_edit" in result

    def test_drop_removes_clip(self, run_dir):
        plan_path = run_dir / "openai-runs" / "run-plan" / "plan-output.json"
        manifest = run_dir / "renders" / "excerpt" / "draft" / "render-manifest.json"
        run_export(self._args(plan_output=plan_path, render_manifest=manifest,
                              transcript=None, source_video=None, output_dir=None))
        toml_path = run_dir / "review" / "draft" / "edit-list.toml"
        text = toml_path.read_text().replace('decision = "keep"', 'decision = "drop"', 1)
        toml_path.write_text(text)
        out = run_dir / "out.json"
        run_import(self._args(edit_list=toml_path, plan_output=None, transcript=None, output=out))
        result = json.loads(out.read_text())
        assert result["clip_ads"] == []
        assert result["human_edit"]["dropped_clip_ids"] == ["EX01"]

    def test_trim_segment_through_toml(self, run_dir):
        # Edit a segment's in/out in the TOML and confirm the trim reaches the plan.
        plan_path = run_dir / "openai-runs" / "run-plan" / "plan-output.json"
        manifest = run_dir / "renders" / "excerpt" / "draft" / "render-manifest.json"
        run_export(self._args(plan_output=plan_path, render_manifest=manifest,
                              transcript=None, source_video=None, output_dir=None))
        toml_path = run_dir / "review" / "draft" / "edit-list.toml"
        text = toml_path.read_text()
        # default clip segment is (1.0, 9.0); trim to (2.5, 7.5)
        text = text.replace("source_in = 1.000", "source_in = 2.500", 1)
        text = text.replace("source_out = 9.000", "source_out = 7.500", 1)
        toml_path.write_text(text)
        out = run_dir / "trimmed.json"
        run_import(self._args(edit_list=toml_path, plan_output=None, transcript=None, output=out))
        sr = json.loads(out.read_text())["clip_ads"][0]["source_ranges"]
        assert sr[0]["start"] == pytest.approx(2.5)
        assert sr[0]["end"] == pytest.approx(7.5)

    def test_unknown_clip_id_rejected(self, run_dir):
        plan_path = run_dir / "openai-runs" / "run-plan" / "plan-output.json"
        manifest = run_dir / "renders" / "excerpt" / "draft" / "render-manifest.json"
        run_export(self._args(plan_output=plan_path, render_manifest=manifest,
                              transcript=None, source_video=None, output_dir=None))
        toml_path = run_dir / "review" / "draft" / "edit-list.toml"
        toml_path.write_text(toml_path.read_text().replace('clip_id = "EX01"', 'clip_id = "ghost"'))
        out = run_dir / "out.json"
        with pytest.raises(RuntimeError, match="unknown clip_id"):
            run_import(self._args(edit_list=toml_path, plan_output=None, transcript=None, output=out))
        assert not out.exists()


# --- F2: export/import threshold agreement -----------------------------------

def test_export_import_threshold_agree():
    # every segment the exporter keeps must pass import validation
    assert MIN_SEGMENT_SECONDS == 0.05
    clip = make_clip(segments=[(1.0, 1.1)])  # 0.1s: above the shared 0.05 floor
    segs = clip_segments(clip)
    assert len(segs) == 1
    validate_segments("c", [{"source_in": s["source_in"], "source_out": s["source_out"]}
                            for s in segs], duration=100.0)
