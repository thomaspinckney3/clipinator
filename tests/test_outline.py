"""Tests for the content-outline stage and shared unit derivation."""
from __future__ import annotations

from extractor import openai_outline
from extractor.transcript_units import pithy_summary, transcript_units_from_data


def test_units_from_segments_carry_text_and_summary():
    data = {
        "segments": [
            {"start": 0.0, "end": 4.5, "text": "Uh, we're beginning with René Descartes."},
            {"start": 4.5, "end": 4.5, "text": "zero-length dropped"},  # e <= s
            {"start": 5.0, "end": 9.0, "text": ""},  # empty dropped
            {"start": 9.0, "end": 12.0, "text": "He studied at the best schools."},
        ]
    }
    units = transcript_units_from_data(data)
    assert len(units) == 2
    assert units[0]["start"] == 0.0 and units[0]["end"] == 4.5
    assert units[0]["text"].startswith("Uh, we're beginning")
    # heuristic summary drops disfluency/stopwords, keeps content (Unicode-safe)
    assert "René" in units[0]["summary"]
    assert "Descartes" in units[0]["summary"]


def test_units_fallback_groups_words_on_pauses():
    words = [
        {"text": "one", "start": 0.0, "end": 0.4},
        {"text": "two", "start": 0.5, "end": 0.9},
        {"text": "three", "start": 3.0, "end": 3.4},  # >0.7s gap -> new unit
    ]
    units = transcript_units_from_data({"words": words})
    assert len(units) == 2
    assert units[0]["start"] == 0.0 and units[0]["end"] == 0.9
    assert units[1]["start"] == 3.0


def test_pithy_summary_skips_contractions_and_ellipsis():
    s = pithy_summary("Uh, he couldn't cure diseases with machines.")
    assert "couldn't" not in s and "…" not in s
    assert "cure" in s


def test_outline_cache_key_is_stable_and_model_sensitive():
    units = [
        {"start": 0.0, "end": 4.5, "text": "hello"},
        {"start": 5.0, "end": 9.0, "text": "world"},
    ]
    a = openai_outline.outline_cache_key(units, "gpt-5.5", "low")
    b = openai_outline.outline_cache_key(units, "gpt-5.5", "low")
    c = openai_outline.outline_cache_key(units, "gpt-6", "low")
    assert a == b
    assert a != c


def test_outline_cache_key_is_location_independent_but_content_sensitive():
    # Same content -> same key regardless of where the transcript lived (the key no
    # longer takes a path at all), so a relocated project reuses its outline cache.
    units = [{"start": 0.0, "end": 4.5, "text": "the moon is bright"}]
    key = openai_outline.outline_cache_key(units, "gpt-5.5", "low")
    # ...but changing the spoken words (same timings) must change the key: the old
    # timings-only key collided here.
    reworded = [{"start": 0.0, "end": 4.5, "text": "the sun is bright"}]
    assert openai_outline.outline_cache_key(reworded, "gpt-5.5", "low") != key


def test_outline_output_payload_shape(tmp_path):
    from pathlib import Path
    labeled = [
        {"index": 0, "start": 0.0, "end": 4.5, "title": "Intro", "summary": "The intro."},
        {"index": 1, "start": 5.0, "end": 9.0, "title": "Body", "summary": "The body."},
    ]
    payload = openai_outline.outline_output_payload(Path("/x/t.json"), "/x/v.mp4", labeled)
    assert payload["unit_count"] == 2
    assert payload["units"][0]["title"] == "Intro"
    assert payload["units"][1]["summary"] == "The body."
    assert payload["source_video"] == "/x/v.mp4"


def test_numbered_units_text_uses_batch_offset():
    units = [{"start": 0.0, "end": 4.5, "text": "hello"}]
    line = openai_outline.numbered_units_text(units, offset=7)
    assert line.startswith("7. [")
    assert "hello" in line
