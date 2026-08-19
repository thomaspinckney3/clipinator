import json

from conftest import make_transcript
from extractor.preprocess_video import transcript_summary


def test_preprocess_transcript_summary_uses_standard_qa_policy(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps(make_transcript()), encoding="utf-8")

    summary = transcript_summary(transcript)

    assert summary is not None
    assert summary["path"] == str(transcript)
    assert summary["qa"]["score"] >= 0
    assert set(summary["qa"]["finding_counts"]) == {"error", "warning", "info"}
