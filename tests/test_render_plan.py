from extractor.render_plan import (
    apply_framing_offset,
    caption_plan_entries,
    transcript_words_from_data,
)


class TestApplyFramingOffset:
    def test_zero_is_noop(self):
        assert apply_framing_offset(0.5, 0.0) == (0.5, None)

    def test_shifts_and_reports(self):
        center, reason = apply_framing_offset(0.5, 0.2)
        assert center == 0.7
        assert "reframe" in reason

    def test_clamped_to_frame(self):
        center, reason = apply_framing_offset(0.95, 0.2)
        assert center == 1.0  # never past the frame edge
        center, _ = apply_framing_offset(0.05, -0.2)
        assert center == 0.0


class TestCaptionPlanEntries:
    """The gate behind F1: caption_plan entries are returned only when the plan
    explicitly opts in via caption_plan_source / use_caption_plan_for_srt."""

    def _clip(self, **extra):
        clip = {"caption_plan": [{"start_offset": 1.0, "end_offset": 2.0, "text": "hi"}]}
        clip.update(extra)
        return clip

    def test_no_source_returns_empty(self):
        assert caption_plan_entries(self._clip()) == []

    def test_render_time_source_returns_entries(self):
        entries = caption_plan_entries(self._clip(caption_plan_source="render_time"))
        assert entries and entries[0]["text"] == "hi"

    def test_use_caption_plan_for_srt_flag(self):
        entries = caption_plan_entries(self._clip(use_caption_plan_for_srt=True))
        assert entries and entries[0]["text"] == "hi"

    def test_skips_empty_text_and_bad_ranges(self):
        clip = {
            "caption_plan_source": "render_time",
            "caption_plan": [
                {"start_offset": 1.0, "end_offset": 2.0, "text": "  "},   # empty
                {"start_offset": 3.0, "end_offset": 3.0, "text": "zero"},  # end<=start
                {"start_offset": 4.0, "end_offset": 5.0, "text": "ok"},
            ],
        }
        entries = caption_plan_entries(clip)
        assert [e["text"] for e in entries] == ["ok"]


class TestTranscriptWords:
    def test_top_level_words(self):
        data = {"words": [{"text": "a", "start": 0.0, "end": 0.5},
                          {"text": "b", "start": 0.5, "end": 1.0}]}
        words = transcript_words_from_data(data)
        assert [w["text"] for w in words] == ["a", "b"]

    def test_falls_back_to_segment_words(self):
        data = {"words": [], "segments": [
            {"words": [{"text": "x", "start": 0.0, "end": 0.4}]}]}
        words = transcript_words_from_data(data)
        assert [w["text"] for w in words] == ["x"]

    def test_sorted_by_time(self):
        data = {"words": [{"text": "b", "start": 1.0, "end": 1.5},
                          {"text": "a", "start": 0.0, "end": 0.5}]}
        words = transcript_words_from_data(data)
        assert [w["text"] for w in words] == ["a", "b"]
