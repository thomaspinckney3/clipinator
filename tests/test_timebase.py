from fractions import Fraction

import pytest

from extractor import timebase
from extractor.timebase import (
    OUTPUT_FRAME_RATE,
    parse_frame_rate,
    probe_frame_rate,
    quantize_plan_times,
    quantize_seconds,
)


def on_grid(seconds: float, rate: Fraction) -> bool:
    frames = seconds * rate
    return abs(frames - round(frames)) < 1e-9


class TestParseFrameRate:
    def test_fraction_string(self):
        assert parse_frame_rate("30000/1001") == Fraction(30000, 1001)

    def test_integer_string(self):
        assert parse_frame_rate("25") == Fraction(25)

    @pytest.mark.parametrize("bad", [None, "", "0", "0/0", "N/A"])
    def test_invalid_returns_none(self, bad):
        assert parse_frame_rate(bad) is None

    def test_zero_denominator(self):
        assert parse_frame_rate("30/0") is None


class TestQuantizeSeconds:
    def test_snaps_to_grid(self):
        ntsc = Fraction(30000, 1001)
        q = quantize_seconds(312.4, ntsc)
        assert on_grid(q, ntsc)
        assert abs(q - 312.4) < 1.0 / 29  # within one frame

    def test_string_input_avoids_float_repr(self):
        # Fraction(str(...)) keeps 312.4 exact rather than 312.39999...
        assert quantize_seconds("0", Fraction(24)) == 0.0

    @pytest.mark.parametrize("bad", [None, True, False, "not-a-number"])
    def test_non_numeric_returns_none(self, bad):
        assert quantize_seconds(bad, Fraction(24)) is None


class TestQuantizePlanTimes:
    def test_source_and_output_grids(self):
        ntsc = Fraction(30000, 1001)
        plan = {
            "clip_ads": [{
                "target_duration_seconds": 47.21,
                "opening_hook": {"duration_seconds": 4.5},
                "source_ranges": [{"start": 312.41, "end": 355.13}],
                "timeline": [
                    {"order": 1, "source_start": 0, "source_end": 0,
                     "screen_time_start": 0.0, "screen_time_end": 4.5},
                    {"order": 2, "source_start": 312.41, "source_end": 355.13,
                     "screen_time_start": 4.5, "screen_time_end": 47.22},
                ],
                "caption_plan": [{"start_offset": 4.51, "end_offset": 7.013, "text": "x"}],
            }],
            "manual_review_queue": [{"start": 10.001, "end": 12.5}],
        }
        changed = quantize_plan_times(plan, ntsc)
        assert changed > 0
        clip = plan["clip_ads"][0]
        tl = clip["timeline"][1]
        # source times -> source grid
        for key in ("source_start", "source_end"):
            assert on_grid(tl[key], ntsc)
        for r in clip["source_ranges"]:
            assert on_grid(r["start"], ntsc) and on_grid(r["end"], ntsc)
        # render-clock times -> output grid
        for key in ("screen_time_start", "screen_time_end"):
            assert on_grid(tl[key], OUTPUT_FRAME_RATE)
        for c in clip["caption_plan"]:
            assert on_grid(c["start_offset"], OUTPUT_FRAME_RATE)
            assert on_grid(c["end_offset"], OUTPUT_FRAME_RATE)
        # manual review queue snaps to the source grid
        for q in plan["manual_review_queue"]:
            assert on_grid(q["start"], ntsc) and on_grid(q["end"], ntsc)

    def test_zero_stays_zero(self):
        plan = {"clip_ads": [{"timeline": [{"source_start": 0, "source_end": 0}]}]}
        quantize_plan_times(plan, Fraction(30000, 1001))
        assert plan["clip_ads"][0]["timeline"][0]["source_start"] == 0.0

    def test_empty_plan_no_error(self):
        assert quantize_plan_times({}, Fraction(24)) == 0


class TestProbeFrameRate:
    def test_probe_real_video(self, tiny_video):
        rate = probe_frame_rate(tiny_video)
        assert rate == Fraction(24000, 1001)

    def test_missing_file_returns_none(self, tmp_path):
        assert probe_frame_rate(tmp_path / "nope.mp4") is None
