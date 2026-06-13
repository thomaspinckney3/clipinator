"""Recorded-response test for an OpenAI stage with NO API call.

The stages skip the network entirely on a local-cache hit, replaying a recorded
raw-response.json. This test seeds the cache with a recorded response (the same
mechanism `--seed-cache-from-run` uses) and then runs the real strategy stage,
which computes the cache key, finds the hit, parses the recorded response, and
writes its output artifacts — exercising the full stage code path (source-
materials assembly, fingerprinting, response parsing, artifact writing) without
any API call or OPENAI_API_KEY. Only the model's actual output is recorded
rather than generated.
"""
from __future__ import annotations

import json
from pathlib import Path

from extractor import openai_strategy

from conftest import make_transcript

# A recorded strategy output, as the model would have returned it.
RECORDED_STRATEGY_OUTPUT = {
    "content_mode": "excerpt",
    "video_title": "Recorded Test Lecture",
    "video_brief": "A recorded brief used to test the cache-hit path.",
    "why_watch_full_video": "Testing only.",
    "visual_inventory": ["slide", "speaker"],
    "ad_angle_hypotheses": [
        {"stage1_rank": 1, "angle": "The recorded angle", "audience": "devs",
         "promise": "no API needed", "evidence": "0-6s"}
    ],
    "notes": [],
}


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    transcript = tmp_path / "t.json"
    transcript.write_text(json.dumps(make_transcript(6.0)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"media": {"duration_seconds": 6.0}, "visual": {}, "audio": {}}))
    cache_dir = tmp_path / "cache"
    return env_file, transcript, manifest, cache_dir


def _seed_dir(tmp_path: Path) -> Path:
    """A source run dir holding just a recorded raw-response.json."""
    d = tmp_path / "recorded"
    d.mkdir()
    raw = {
        "id": "resp_recorded",
        "output": [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": json.dumps(RECORDED_STRATEGY_OUTPUT)}],
        }],
        "usage": {"input_tokens": 1200, "output_tokens": 400, "total_tokens": 1600},
    }
    (d / "raw-response.json").write_text(json.dumps(raw))
    return d


def test_strategy_cache_hit_no_api(tmp_path, monkeypatch):
    # Prove no API path is taken: remove the key entirely.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file, transcript, manifest, cache_dir = _write_inputs(tmp_path)
    seed = _seed_dir(tmp_path)
    out_dir = tmp_path / "runs"

    common = [
        "--env-file", str(env_file),
        "--transcript", str(transcript),
        "--manifest", str(manifest),
        "--content-mode", "excerpt",
        "--cache-dir", str(cache_dir),
        "--output-dir", str(out_dir),
    ]

    # 1) seed the cache from the recorded response (no API, no key)
    rc = openai_strategy.main([*common, "--run-id", "seed", "--seed-cache-from-run", str(seed)])
    assert rc == 0
    # the cache entry now exists
    assert any(cache_dir.rglob("raw-response.json"))

    # 2) real run -> cache hit -> parses recorded response, writes output (no API)
    rc = openai_strategy.main([*common, "--run-id", "hit"])
    assert rc == 0

    run_dir = out_dir / "hit"
    produced = json.loads((run_dir / "strategy-output.json").read_text())
    assert produced == RECORDED_STRATEGY_OUTPUT

    usage = json.loads((run_dir / "usage-and-cost.json").read_text())
    assert usage["api_call_made"] is False
    assert usage["cache"]["hit"] is True
    assert usage["cost"]["estimated_usd"] == 0.0

    manifest_out = json.loads((run_dir / "request-manifest.json").read_text())
    assert manifest_out["local_response_cache"]["hit"] is True


def test_strategy_requires_approval_on_cache_miss(tmp_path, monkeypatch):
    # The negative: a cache miss without a key/approval must NOT call the API.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file, transcript, manifest, cache_dir = _write_inputs(tmp_path)
    rc = openai_strategy.main([
        "--env-file", str(env_file), "--transcript", str(transcript),
        "--manifest", str(manifest), "--content-mode", "excerpt",
        "--cache-dir", str(cache_dir), "--output-dir", str(tmp_path / "runs"),
        "--run-id", "miss",
    ])
    assert rc == 1  # refused (approval required), no network attempted
