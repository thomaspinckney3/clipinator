from fractions import Fraction

import opentimelineio as otio
import pytest

from extractor import edit_list, otio_edl
from extractor.otio_edl import (
    METADATA_NS,
    flatten_for_nle,
    otio_to_edits,
    plan_to_otio,
)

from conftest import make_clip, make_plan, make_transcript

RATE = Fraction(24000, 1001)
FRAME = 1.0 / float(RATE)


def build_timeline(plan, words=None):
    return plan_to_otio(
        plan, source_url="../../source.mp4", source_rate=RATE,
        transcript_words=words or [], plan_rel="../../plan.json", transcript_rel="../../t.json",
    )


class TestPlanToOtio:
    def test_one_track_per_clip(self):
        plan = make_plan([make_clip("EX01"), make_clip("EX02")])
        tl = build_timeline(plan)
        names = [t.name for t in tl.tracks]
        assert names == ["EX01", "EX02"]

    def test_timeline_metadata(self):
        tl = build_timeline(make_plan())
        meta = tl.metadata[METADATA_NS]
        assert meta["schema_version"] == edit_list.SCHEMA_VERSION
        assert meta["content_mode"] == "excerpt"
        assert meta["plan"] == "../../plan.json"

    def test_title_card_is_generator(self):
        tl = build_timeline(make_plan([make_clip("EX01", card=4.5)]))
        first = list(list(tl.tracks)[0])[0]
        assert isinstance(first.media_reference, otio.schema.GeneratorReference)

    def test_segment_source_range(self):
        tl = build_timeline(make_plan([make_clip("EX01", card=0.0, segments=[(2.0, 6.0)])]))
        clips = [c for c in list(tl.tracks)[0]
                 if not isinstance(c.media_reference, otio.schema.GeneratorReference)]
        assert len(clips) == 1
        sr = clips[0].source_range
        assert sr.start_time.to_seconds() == pytest.approx(2.0, abs=FRAME)
        assert sr.duration.to_seconds() == pytest.approx(4.0, abs=FRAME)


class TestOtioToEdits:
    def test_reads_back_fields(self):
        plan = make_plan([make_clip("EX01", card=4.5, segments=[(1.0, 9.0)])])
        tl = build_timeline(plan, make_transcript(12.0)["words"])
        edits = otio_to_edits(tl)
        assert edits["schema_version"] == edit_list.SCHEMA_VERSION
        clip = edits["clips"][0]
        assert clip["clip_id"] == "EX01"
        assert clip["decision"] == "keep"
        assert clip["title_card_seconds"] == pytest.approx(4.5)
        assert clip["segments"] == [{"source_in": pytest.approx(1.0, abs=FRAME),
                                     "source_out": pytest.approx(9.0, abs=FRAME)}]

    def test_generator_not_a_segment(self):
        tl = build_timeline(make_plan([make_clip("EX01", card=4.5, segments=[(1.0, 9.0)])]))
        edits = otio_to_edits(tl)
        assert len(edits["clips"][0]["segments"]) == 1  # title card excluded

    def test_clip_id_falls_back_to_track_name(self):
        # an NLE-exported .otio may lack our metadata namespace
        tl = otio.schema.Timeline()
        tl.metadata[METADATA_NS] = {"schema_version": edit_list.SCHEMA_VERSION}
        track = otio.schema.Track(name="EX99", kind=otio.schema.TrackKind.Video)
        track.append(otio.schema.Clip(
            name="EX99", source_range=otio.opentime.TimeRange(
                otio.opentime.RationalTime(24, 24), otio.opentime.RationalTime(48, 24)),
            media_reference=otio.schema.ExternalReference(target_url="s.mp4")))
        tl.tracks.append(track)
        edits = otio_to_edits(tl)
        assert edits["clips"][0]["clip_id"] == "EX99"
        assert edits["clips"][0]["segments"][0]["source_in"] == pytest.approx(1.0, abs=FRAME)


class TestRoundTrip:
    def test_serialize_deserialize_preserves_edits(self):
        plan = make_plan([make_clip("EX01"), make_clip("EX02")])
        tl = build_timeline(plan, make_transcript(12.0)["words"])
        text = otio.adapters.write_to_string(tl, "otio_json")
        tl2 = otio.adapters.read_from_string(text, "otio_json")
        edits = otio_to_edits(tl2)
        assert [c["clip_id"] for c in edits["clips"]] == ["EX01", "EX02"]

    def test_full_round_trip_through_apply_edits(self):
        words = make_transcript(12.0)["words"]
        plan = make_plan([make_clip("EX01", segments=[(1.0, 9.0)]),
                          make_clip("EX02", segments=[(2.0, 5.0)])])
        tl = build_timeline(plan, words)
        edits = otio_to_edits(tl)
        revised = edit_list.apply_edits(plan, edits, words, duration=100.0, source="x.otio")
        assert [c["clip_id"] for c in revised["clip_ads"]] == ["EX01", "EX02"]
        for orig, out in zip(plan["clip_ads"], revised["clip_ads"]):
            so = [(s["start"], s["end"]) for s in orig["source_ranges"]]
            sr = [(s["start"], s["end"]) for s in out["source_ranges"]]
            for (a0, b0), (a1, b1) in zip(so, sr):
                assert a1 == pytest.approx(a0, abs=FRAME)
                assert b1 == pytest.approx(b0, abs=FRAME)

    def test_drop_via_metadata(self):
        words = make_transcript(12.0)["words"]
        plan = make_plan([make_clip("EX01"), make_clip("EX02")])
        tl = build_timeline(plan, words)
        list(tl.tracks)[1].metadata[METADATA_NS]["decision"] = "drop"
        edits = otio_to_edits(tl)
        revised = edit_list.apply_edits(plan, edits, words, duration=100.0, source="x.otio")
        assert [c["clip_id"] for c in revised["clip_ads"]] == ["EX01"]
        assert revised["human_edit"]["dropped_clip_ids"] == ["EX02"]

    def test_trim_clip_enter_exit_time(self):
        # Edit the actual timeline clip's source_range (trim in- and out-points),
        # the way an editor moves a clip's enter/exit, and verify the trim flows
        # through to the rebuilt plan.
        words = make_transcript(12.0)["words"]
        plan = make_plan([make_clip("EX01", card=0.0, segments=[(1.0, 9.0)])])
        tl = build_timeline(plan, words)
        seg_clip = [c for c in list(tl.tracks)[0]
                    if not isinstance(c.media_reference, otio.schema.GeneratorReference)][0]
        # trim: enter 1.0 -> 3.0, exit 9.0 -> 7.0  (new start 3.0, duration 4.0)
        seg_clip.source_range = otio.opentime.TimeRange(
            otio.opentime.RationalTime(round(3.0 * float(RATE)), float(RATE)),
            otio.opentime.RationalTime(round(4.0 * float(RATE)), float(RATE)),
        )
        edits = otio_to_edits(tl)
        assert edits["clips"][0]["segments"] == [
            {"source_in": pytest.approx(3.0, abs=FRAME), "source_out": pytest.approx(7.0, abs=FRAME)}]
        revised = edit_list.apply_edits(plan, edits, words, duration=100.0, source="x.otio")
        sr = revised["clip_ads"][0]["source_ranges"]
        assert len(sr) == 1
        assert sr[0]["start"] == pytest.approx(3.0, abs=FRAME)
        assert sr[0]["end"] == pytest.approx(7.0, abs=FRAME)
        # the rebuilt body timeline item carries the trimmed source span
        body = [t for t in revised["clip_ads"][0]["timeline"]
                if "title" not in str(t.get("purpose", "")).lower()][-1]
        assert body["source_start"] == pytest.approx(3.0, abs=FRAME)
        assert body["source_end"] == pytest.approx(7.0, abs=FRAME)

    def test_caption_edit_marks_human_edited(self):
        words = make_transcript(12.0)["words"]
        plan = make_plan([make_clip("EX01", segments=[(1.0, 9.0)])])
        tl = build_timeline(plan, words)
        caps = list(tl.tracks)[0].metadata[METADATA_NS]["captions"]
        caps[0]["text"] = "HUMAN OTIO CAPTION"
        edits = otio_to_edits(tl)
        revised = edit_list.apply_edits(plan, edits, words, duration=100.0, source="x.otio")
        clip = revised["clip_ads"][0]
        assert clip["human_edited_captions"] is True
        assert any("HUMAN OTIO CAPTION" in c["text"] for c in clip["caption_plan"])


class TestNleViews:
    def test_flatten_single_track(self):
        plan = make_plan([make_clip("EX01", segments=[(1.0, 3.0)]),
                          make_clip("EX02", segments=[(5.0, 7.0)])])
        tl = flatten_for_nle(plan, source_url="s.mp4", source_rate=RATE, source_duration=60.0)
        assert len(list(tl.tracks)) == 1
        clips = list(list(tl.tracks)[0])
        assert [c.name for c in clips] == ["EX01", "EX02"]
        assert all(not isinstance(c.media_reference, otio.schema.GeneratorReference) for c in clips)

    def test_edl_export(self, tmp_path):
        plan = make_plan([make_clip("EX01", segments=[(1.0, 3.0)])])
        tl = flatten_for_nle(plan, source_url="s.mp4", source_rate=RATE, source_duration=60.0)
        written = otio_edl.write_nle_views(tl, tmp_path, "edit-list")
        # EDL should always succeed for a single track; FCPXML best-effort
        assert "edl" in written
        assert (tmp_path / "edit-list.edl").exists()


class TestExportImportCLI:
    def _args(self, **kw):
        import argparse
        return argparse.Namespace(**kw)

    def test_export_then_import(self, run_dir):
        plan_path = run_dir / "openai-runs" / "run-plan" / "plan-output.json"
        manifest = run_dir / "renders" / "excerpt" / "draft" / "render-manifest.json"
        rc = otio_edl.run_export(self._args(
            plan_output=plan_path, render_manifest=manifest,
            transcript=None, source_video=None, output_dir=None))
        assert rc == 0
        otio_path = run_dir / "review" / "draft" / "edit-list.otio"
        assert otio_path.exists()
        out = run_dir / "from-otio.json"
        rc = otio_edl.run_import(self._args(
            otio=otio_path, plan_output=None, transcript=None, output=out))
        assert rc == 0
        import json
        result = json.loads(out.read_text())
        assert [c["clip_id"] for c in result["clip_ads"]] == ["EX01"]
        assert result["human_edit"]["edit_list"].endswith("edit-list.otio")
