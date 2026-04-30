from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from extractor.qa_transcript import (
    DEFAULT_MAX_COMPRESSION_RATIO,
    DEFAULT_MAX_FINDINGS,
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_MAX_NO_SPEECH_PROB,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MAX_SEGMENT_WPS,
    DEFAULT_MAX_TAIL_GAP_SECONDS,
    DEFAULT_MAX_WORDS_PER_MINUTE,
    DEFAULT_MIN_AVG_LOGPROB,
    DEFAULT_MIN_SEGMENT_WPS,
    DEFAULT_MIN_WORDS_PER_MINUTE,
    build_report,
)


DEFAULT_OUTPUT_DIR = Path("ad-plans")
DEFAULT_MIN_CLIP_SECONDS = 18.0
DEFAULT_MAX_CLIP_SECONDS = 75.0
DEFAULT_TOP_N = 6


@dataclass(frozen=True)
class AdPattern:
    pattern_id: str
    angle: str
    keywords: tuple[str, ...]
    must_have_any: tuple[str, ...]
    clip_type: str
    edit_methods: tuple[str, ...]
    visual_requirements: tuple[str, ...]
    watch_pitch: str
    base_score: float = 1.0


AD_PATTERNS: tuple[AdPattern, ...] = (
    AdPattern(
        pattern_id="moon_hammer",
        angle="NASA can't just buy a hammer for the Moon.",
        keywords=(
            "hammer",
            "geology hammer",
            "hardware store",
            "moon tool",
            "tools",
            "gloved hand",
            "spacesuit",
            "cold",
            "south pole",
        ),
        must_have_any=("hammer", "hardware store", "geology hammer"),
        clip_type="cold_open_plus_demo",
        edit_methods=("text_hook_overlay", "b_roll_cover", "caption_keyword_emphasis", "j_cut"),
        visual_requirements=("tool testing", "geology hammer", "spacesuit gloves", "rock yard"),
        watch_pitch="Watch to see how ordinary geology tools have to be redesigned for lunar exploration.",
        base_score=1.25,
    ),
    AdPattern(
        pattern_id="human_sized_sandbox",
        angle="NASA built a human-sized sandbox to practice Moon science.",
        keywords=(
            "rock yard",
            "human-sized sandbox",
            "imported rocks",
            "test subjects",
            "crew perspective",
            "collect those samples",
        ),
        must_have_any=("rock yard", "human-sized sandbox", "imported rocks"),
        clip_type="visual_first_explainer",
        edit_methods=("visual_hook_first", "b_roll_cover", "caption_keyword_emphasis"),
        visual_requirements=("rock yard", "imported rocks", "astronaut or engineer field practice"),
        watch_pitch="Watch to see how NASA rehearses lunar geology before astronauts leave Earth.",
        base_score=1.2,
    ),
    AdPattern(
        pattern_id="underwater_training",
        angle="Astronauts train underwater before they go to the Moon.",
        keywords=(
            "aquanaut",
            "underwater",
            "habitat",
            "60 feet",
            "spacewalks",
            "neutral buoyancy",
            "pool",
            "one-sixth gravity",
        ),
        must_have_any=("underwater", "aquanaut", "neutral buoyancy", "pool"),
        clip_type="visual_proof",
        edit_methods=("cold_open", "b_roll_cover", "j_cut", "music_or_roomtone_smooth"),
        visual_requirements=("underwater training", "aquanauts", "neutral buoyancy lab"),
        watch_pitch="Watch to see the Earth-based simulations NASA uses to rehearse Moon missions.",
        base_score=1.15,
    ),
    AdPattern(
        pattern_id="south_pole_extremes",
        angle="The Moon's south pole is one of the strangest places humans will explore.",
        keywords=(
            "south pole",
            "coldest temperatures",
            "frozen water",
            "water ice",
            "negative 400",
            "no human",
            "young solar system",
            "extreme",
        ),
        must_have_any=("south pole", "frozen water", "water ice", "coldest temperatures"),
        clip_type="fact_first",
        edit_methods=("cold_open", "map_or_surface_broll", "caption_keyword_emphasis"),
        visual_requirements=("lunar south pole", "moon surface", "maps or animation", "ice/shadow visuals"),
        watch_pitch="Watch to understand why Artemis is targeting the Moon's south pole.",
        base_score=1.2,
    ),
    AdPattern(
        pattern_id="one_rock_history",
        angle="One Moon rock can reveal billions of years of history.",
        keywords=(
            "one rock",
            "billions of years",
            "four plus billion",
            "solar system history",
            "witness plate",
            "geologic record",
            "naked eye",
        ),
        must_have_any=("one rock", "billions", "witness plate", "geologic record"),
        clip_type="insight_explainer",
        edit_methods=("text_hook_overlay", "insert_shots", "caption_keyword_emphasis"),
        visual_requirements=("moon rock", "sample closeup", "lab analysis", "lunar surface"),
        watch_pitch="Watch to see why tiny samples can answer enormous questions about the solar system.",
        base_score=1.25,
    ),
    AdPattern(
        pattern_id="moon_samples_changed_science",
        angle="Moon rocks changed what we know about Earth, water, and life.",
        keywords=(
            "moon rocks showed",
            "first evidence",
            "moon has water",
            "history of the sun",
            "evolution of life",
            "closely related to earth",
            "same material",
        ),
        must_have_any=("water", "life", "earth", "sun"),
        clip_type="rapid_fact_stack",
        edit_methods=("fact_stack", "caption_keyword_emphasis", "insert_shots"),
        visual_requirements=("moon samples", "lab instruments", "Earth/Moon visuals"),
        watch_pitch="Watch to see how Apollo samples still shape modern science.",
        base_score=1.1,
    ),
    AdPattern(
        pattern_id="first_moon_sample",
        angle="What does it feel like to hold a piece of the Moon?",
        keywords=(
            "first time",
            "held a moon sample",
            "lunar meteorite",
            "thin section",
            "piece of hair",
            "shine light",
            "over the moon",
        ),
        must_have_any=("moon sample", "lunar meteorite", "thin section"),
        clip_type="human_story_plus_object",
        edit_methods=("cold_open", "reaction_or_object_cutaway", "caption_keyword_emphasis"),
        visual_requirements=("moon sample", "thin section", "microscope", "scientist reaction"),
        watch_pitch="Watch for the human story behind studying rocks from the Moon.",
        base_score=1.05,
    ),
    AdPattern(
        pattern_id="spacesuit_personal_spacecraft",
        angle="A spacesuit is basically a personal spacecraft.",
        keywords=(
            "spacesuit",
            "personal spacecraft",
            "300 pounds",
            "balloon",
            "squeeze",
            "poke",
            "pound",
            "arms and legs",
        ),
        must_have_any=("spacesuit", "personal spacecraft", "300 pounds", "balloon"),
        clip_type="analogy_explainer",
        edit_methods=("quote_first", "b_roll_cover", "caption_keyword_emphasis", "cutaway"),
        visual_requirements=("spacesuit", "astronaut training", "gloved hands", "tool use"),
        watch_pitch="Watch to see why doing science in a spacesuit is its own engineering problem.",
        base_score=1.15,
    ),
    AdPattern(
        pattern_id="failed_applications",
        angle="A future NASA engineer got rejected 30-plus times before one yes.",
        keywords=(
            "30 to 35",
            "applications",
            "unsuccessful",
            "accepted to a nasa internship",
            "broke the barrier",
            "living out my dream",
            "don't give up",
        ),
        must_have_any=("30 to 35", "applications", "internship", "dream"),
        clip_type="human_motivation",
        edit_methods=("cold_open", "talking_head_locked_sync", "caption_keyword_emphasis"),
        visual_requirements=("speaker", "NASA workplace", "internship or engineering B-roll"),
        watch_pitch="Watch for the people behind Artemis, not just the hardware.",
        base_score=1.0,
    ),
    AdPattern(
        pattern_id="science_origin_story",
        angle="A childhood accident accidentally created a scientist.",
        keywords=(
            "when i was a kid",
            "burnt rice",
            "bleach",
            "muriatic acid",
            "chlorine gas",
            "chemist",
            "that was cool",
        ),
        must_have_any=("when i was a kid", "burnt rice", "bleach", "muriatic acid", "chlorine gas", "chemist"),
        clip_type="story_payoff",
        edit_methods=("cold_open", "smash_cut_payoff", "caption_keyword_emphasis"),
        visual_requirements=("speaker", "lab B-roll", "safe chemistry visual metaphor"),
        watch_pitch="Watch to see the unexpected paths that lead people into space science.",
        base_score=0.95,
    ),
)


KNOWN_BRIEFS = {
    "space-school": {
        "core_idea": (
            "NASA is preparing Artemis astronauts to do science on the Moon by training "
            "them as field geologists, testing custom tools, and simulating lunar work on Earth."
        ),
        "why_watch": (
            "The video turns Moon exploration into concrete, visual problems: underwater "
            "missions, rock yards, spacesuits, and tools that have to work in lunar extremes."
        ),
        "viewer_promise": "See how NASA practices Moon science before astronauts ever leave Earth.",
        "target_audience": "Space fans, science-curious viewers, students, and educators.",
    },
    "moon-rocks": {
        "core_idea": (
            "Moon samples are scientific time capsules that reveal how the Moon, Earth, "
            "Sun, and solar system changed over billions of years."
        ),
        "why_watch": (
            "The video makes lunar science tangible through samples, microscopes, labs, "
            "and the practical challenge of collecting and preserving rocks from Artemis."
        ),
        "viewer_promise": "See why small rocks from the Moon can answer huge questions.",
        "target_audience": "Science-curious viewers, space fans, classrooms, and geology learners.",
    },
    "artemis-generation": {
        "core_idea": (
            "Artemis is a team effort: astronauts, scientists, engineers, curators, and "
            "tool builders are preparing NASA to return humans to the Moon for long-term science."
        ),
        "why_watch": (
            "The video sells the human side of the mission through origin stories, career "
            "paths, sample science, and the work needed before astronauts collect Moon rocks."
        ),
        "viewer_promise": "Meet the people building the next era of Moon exploration.",
        "target_audience": "Space fans, STEM students, educators, and viewers drawn to human stories.",
    },
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    minutes, remainder = divmod(total_ms, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path}: invalid JSON: {exc}") from exc


def default_qa_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_findings=DEFAULT_MAX_FINDINGS,
        min_words_per_minute=DEFAULT_MIN_WORDS_PER_MINUTE,
        max_words_per_minute=DEFAULT_MAX_WORDS_PER_MINUTE,
        min_segment_wps=DEFAULT_MIN_SEGMENT_WPS,
        max_segment_wps=DEFAULT_MAX_SEGMENT_WPS,
        max_gap_seconds=DEFAULT_MAX_GAP_SECONDS,
        max_tail_gap_seconds=DEFAULT_MAX_TAIL_GAP_SECONDS,
        min_avg_logprob=DEFAULT_MIN_AVG_LOGPROB,
        max_no_speech_prob=DEFAULT_MAX_NO_SPEECH_PROB,
        max_compression_ratio=DEFAULT_MAX_COMPRESSION_RATIO,
        max_segment_seconds=DEFAULT_MAX_SEGMENT_SECONDS,
    )


def infer_brief(transcript_path: Path, data: dict[str, Any]) -> dict[str, str]:
    stem = transcript_path.stem.replace(".whisper-1", "")
    for key, brief in KNOWN_BRIEFS.items():
        if key in stem:
            return dict(brief)

    text = normalize_text(data.get("text", ""))
    if "moon" in text and "artemis" in text:
        core = "This NASA video explains a piece of the Artemis effort to return humans to the Moon."
    elif "moon" in text:
        core = "This NASA video explains why lunar exploration and lunar samples matter."
    else:
        core = "This NASA video explains a science or exploration topic for a general audience."
    return {
        "core_idea": core,
        "why_watch": "The video combines expert explanation with NASA visuals and concrete examples.",
        "viewer_promise": "Learn the story behind a NASA science or exploration effort.",
        "target_audience": "General science-curious viewers.",
    }


def text_for_segments(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()


def keyword_score(text: str, pattern: AdPattern) -> float:
    normalized = normalize_text(text)
    score = 0.0
    for keyword in pattern.keywords:
        keyword_norm = normalize_text(keyword)
        if keyword_norm in normalized:
            # Multi-word phrases are more useful as editorial hooks.
            score += 2.0 if " " in keyword_norm else 1.0
    return score


def has_required_cue(text: str, pattern: AdPattern) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(keyword) in normalized for keyword in pattern.must_have_any)


def window_has_bad_qa(start: float, end: float, qa_report: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    bad_kinds = {
        "repeated_word",
        "high_compression_ratio",
        "high_no_speech_prob",
        "missing_segment_timestamps",
        "missing_word_timestamps",
    }
    overlaps: list[dict[str, Any]] = []
    for finding in qa_report.get("findings", []):
        finding_start = finding.get("start")
        finding_end = finding.get("end")
        if finding_start is None or finding_end is None:
            continue
        if end < float(finding_start) or start > float(finding_end):
            continue
        if finding.get("kind") in bad_kinds:
            overlaps.append(finding)
    return bool(overlaps), overlaps


def hook_strength(text: str) -> float:
    normalized = normalize_text(text)
    score = 0.0
    hooks = (
        "can't",
        "cannot",
        "first",
        "one",
        "why",
        "how",
        "what",
        "coldest",
        "never",
        "no human",
        "billions",
        "water",
        "moon",
        "space",
        "great challenge",
    )
    for hook in hooks:
        if hook in normalized:
            score += 0.5
    if "?" in text:
        score += 1.0
    return min(score, 4.0)


def payoff_strength(text: str) -> float:
    normalized = normalize_text(text)
    score = 0.0
    payoff_markers = (
        "can't",
        "cannot",
        "we have to",
        "we need to",
        "so we",
        "that's why",
        "because",
        "make sure",
        "great challenge",
        "special one",
        "first time",
        "no human",
        "never",
        "turns out",
        "the reason",
        "the result",
    )
    for marker in payoff_markers:
        if marker in normalized:
            score += 0.75
    if "?" in text:
        score += 0.5
    return min(score, 4.0)


def duration_score(duration: float, min_clip: float, max_clip: float) -> float:
    target = min(max((min_clip + max_clip) / 2, 35.0), 55.0)
    distance = abs(duration - target)
    return max(0.0, 5.0 - distance / 10)


def start_fragment_penalty(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    if stripped[0].islower():
        return 7.0
    return 0.0


def select_best_window(
    segments: list[dict[str, Any]],
    pattern: AdPattern,
    qa_report: dict[str, Any],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for start_index in range(len(segments)):
        start = float(segments[start_index]["start"])
        window: list[dict[str, Any]] = []
        for end_index in range(start_index, len(segments)):
            segment = segments[end_index]
            end = float(segment["end"])
            duration = end - start
            if duration > max_clip_seconds:
                break
            window.append(segment)
            if duration < min_clip_seconds:
                continue

            window_text = text_for_segments(window)
            if not has_required_cue(window_text, pattern):
                continue
            early_text = text_for_segments(
                segment for segment in window if float(segment["start"]) <= start + 12.0
            )
            if not has_required_cue(early_text, pattern):
                continue

            keyword_points = keyword_score(window_text, pattern)
            if keyword_points <= 0:
                continue

            qa_bad, qa_overlaps = window_has_bad_qa(start, end, qa_report)
            first_sentence = str(window[0].get("text", ""))
            tail_text = text_for_segments(window[-max(1, min(4, len(window))) :])
            payoff_points = max(payoff_strength(tail_text), payoff_strength(window_text) * 0.65)
            if payoff_points <= 0:
                continue
            score = (
                pattern.base_score * 10
                + keyword_points * 2.2
                + hook_strength(first_sentence)
                + payoff_points * 2.5
                + duration_score(duration, min_clip_seconds, max_clip_seconds)
                - start_fragment_penalty(first_sentence)
            )
            if qa_bad:
                score -= 12.0
            if duration < 25:
                score -= 1.5
            if duration > 65:
                score -= 2.0

            candidate = {
                "score": round(score, 3),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_seconds": round(duration, 3),
                "segments": window,
                "text": window_text,
                "payoff_score": round(payoff_points, 3),
                "qa_overlaps": qa_overlaps,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


def purpose_for_index(index: int, total: int) -> str:
    if total <= 1:
        return "core evidence"
    if index == 0:
        return "opening hook or setup"
    if index == total - 1:
        return "payoff or transition out"
    return "explanation or visual evidence"


def source_segment_plan(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    total = len(segments)
    planned = []
    for index, segment in enumerate(segments):
        planned.append(
            {
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "purpose": purpose_for_index(index, total),
                "text": str(segment.get("text", "")).strip(),
            }
        )
    return planned


def build_clip_plan(
    pattern: AdPattern,
    window: dict[str, Any],
    source_path: str | None,
    rank: int,
) -> dict[str, Any]:
    start = float(window["start"])
    end = float(window["end"])
    video_start = max(0.0, start - 0.8)
    audio_start = start
    transcript_text = str(window["text"])
    first_line = transcript_text.split(". ")[0].strip()
    if len(first_line) > 130:
        first_line = first_line[:127].rstrip() + "..."

    sync_policy = (
        "broll_offset_allowed"
        if any(method in pattern.edit_methods for method in ("b_roll_cover", "visual_hook_first", "insert_shots"))
        else "locked"
    )
    return {
        "rank": rank,
        "ad_angle": pattern.angle,
        "pattern_id": pattern.pattern_id,
        "clip_type": pattern.clip_type,
        "score": window["score"],
        "why_this_sells_the_full_video": pattern.watch_pitch,
        "opening_hook": {
            "text_overlay": pattern.angle,
            "spoken_or_contextual_hook": first_line,
        },
        "source": {
            "video_path": source_path,
            "transcript_start": start,
            "transcript_end": end,
            "duration_seconds": window["duration_seconds"],
            "display_range": f"{timestamp(start)}-{timestamp(end)}",
        },
        "edit_plan": {
            "video_start": round(video_start, 3),
            "audio_start": round(audio_start, 3),
            "clip_end": end,
            "video_fade_in_seconds": 0.5 if sync_policy == "broll_offset_allowed" else 0.2,
            "audio_fade_in_seconds": 0.0,
            "video_fade_out_seconds": 0.4,
            "audio_fade_out_seconds": 0.25,
            "sync_policy": sync_policy,
            "methods": list(pattern.edit_methods),
            "reframe": "9:16 subject-aware crop after visual review",
            "caption_style": "large captions with keyword emphasis",
        },
        "visual_requirements_to_verify": list(pattern.visual_requirements),
        "source_segments": source_segment_plan(window["segments"]),
        "qa_notes": window["qa_overlaps"],
    }


def rank_plans(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(plans, key=lambda plan: plan["score"], reverse=True)
    for plan in ranked:
        start = float(plan["source"]["transcript_start"])
        end = float(plan["source"]["transcript_end"])
        overlapping_angles = []
        for other in ranked:
            if other is plan:
                continue
            other_start = float(other["source"]["transcript_start"])
            other_end = float(other["source"]["transcript_end"])
            if not (end < other_start or start > other_end):
                overlapping_angles.append(other["ad_angle"])
        plan["overlapping_variant_angles"] = overlapping_angles
    for rank, plan in enumerate(ranked, start=1):
        plan["rank"] = rank
    return ranked


def build_plan_for_transcript(
    transcript_path: Path,
    min_clip_seconds: float,
    max_clip_seconds: float,
    top_n: int,
) -> dict[str, Any]:
    data = read_json(transcript_path)
    segments = [
        segment
        for segment in data.get("segments", [])
        if segment.get("start") is not None
        and segment.get("end") is not None
        and str(segment.get("text", "")).strip()
    ]
    qa_report = build_report(transcript_path, default_qa_args())
    brief = infer_brief(transcript_path, data)
    source_path = data.get("source", {}).get("path")

    plans: list[dict[str, Any]] = []
    for pattern in AD_PATTERNS:
        window = select_best_window(
            segments=segments,
            pattern=pattern,
            qa_report=qa_report,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
        )
        if window is None:
            continue
        plans.append(build_clip_plan(pattern, window, source_path, rank=0))

    ranked = rank_plans(plans)[:top_n]
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline": {
            "name": "local_heuristic_micro_trailer_planner",
            "api_calls": False,
            "stages": [
                "infer_full_video_ad_brief",
                "generate_ad_angle_hypotheses",
                "ground_angles_in_transcript_windows",
                "apply_transcript_qa_penalties",
                "assemble_standard_edit_plans",
            ],
            "limitations": [
                "Visual requirements are hypotheses until verified against video frames.",
                "Narration/visual alignment is not measured yet.",
                "Ad angles are generated from deterministic templates, not an LLM.",
            ],
        },
        "input": {
            "transcript_path": str(transcript_path),
            "source_video_path": source_path,
            "duration_seconds": data.get("source", {}).get("duration_seconds"),
        },
        "full_video_ad_brief": brief,
        "transcript_quality": {
            "score": qa_report["score"],
            "finding_counts": qa_report["finding_counts"],
            "top_findings": qa_report["findings"][:5],
        },
        "clip_ad_plans": ranked,
    }


def markdown_for_plan(plan: dict[str, Any]) -> str:
    brief = plan["full_video_ad_brief"]
    lines = [
        f"# {Path(plan['input']['transcript_path']).name}",
        "",
        "## Full Video Ad Brief",
        "",
        f"**Core idea:** {brief['core_idea']}",
        "",
        f"**Why watch:** {brief['why_watch']}",
        "",
        f"**Viewer promise:** {brief['viewer_promise']}",
        "",
        "## Clip Ad Plans",
        "",
    ]
    for clip in plan["clip_ad_plans"]:
        source = clip["source"]
        edit = clip["edit_plan"]
        lines.extend(
            [
                f"### {clip['rank']}. {clip['ad_angle']}",
                "",
                f"- Range: `{source['display_range']}` ({source['duration_seconds']:.1f}s)",
                f"- Score: `{clip['score']}`",
                f"- Type: `{clip['clip_type']}`",
                f"- Why it sells the full video: {clip['why_this_sells_the_full_video']}",
                f"- Edit methods: {', '.join(edit['methods'])}",
                f"- Sync policy: `{edit['sync_policy']}`",
                f"- Visuals to verify: {', '.join(clip['visual_requirements_to_verify'])}",
                f"- Hook overlay: {clip['opening_hook']['text_overlay']}",
                "",
                "Evidence:",
                "",
            ]
        )
        for segment in clip["source_segments"][:5]:
            lines.append(
                f"- `{timestamp(segment['start'])}-{timestamp(segment['end'])}` "
                f"{segment['text']}"
            )
        if len(clip["source_segments"]) > 5:
            lines.append(f"- ... {len(clip['source_segments']) - 5} more segment(s)")
        if clip["qa_notes"]:
            lines.extend(["", "QA notes:"])
            for note in clip["qa_notes"]:
                lines.append(f"- {note['severity']} `{note['kind']}`: {note['message']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(plan: dict[str, Any], output_dir: Path, write_markdown: bool) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_stem = Path(plan["input"]["transcript_path"]).name.replace(".json", "")
    json_path = output_dir / f"{transcript_stem}.ad-plan.json"
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path: Path | None = None
    if write_markdown:
        markdown_path = output_dir / f"{transcript_stem}.ad-plan.md"
        markdown_path.write_text(markdown_for_plan(plan), encoding="utf-8")
    return json_path, markdown_path


def print_summary(plan: dict[str, Any], json_path: Path | None = None, markdown_path: Path | None = None) -> None:
    transcript_name = Path(plan["input"]["transcript_path"]).name
    brief = plan["full_video_ad_brief"]
    quality = plan["transcript_quality"]
    print(transcript_name)
    print(f"  core idea: {brief['core_idea']}")
    print(f"  why watch: {brief['why_watch']}")
    print(f"  transcript QA: {quality['score']}/100")
    for clip in plan["clip_ad_plans"]:
        source = clip["source"]
        print(
            f"  {clip['rank']}. {clip['ad_angle']} "
            f"[{source['display_range']}, {source['duration_seconds']:.1f}s] "
            f"score={clip['score']}"
        )
    if json_path:
        print(f"  wrote: {json_path}")
    if markdown_path:
        print(f"  wrote: {markdown_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create local micro-trailer ad plans from normalized transcripts. "
            "No API calls are made."
        )
    )
    parser.add_argument("transcripts", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-clip-seconds", type=float, default=DEFAULT_MIN_CLIP_SECONDS)
    parser.add_argument("--max-clip-seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS)
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--stdout-json", action="store_true", help="Write all plans as JSON to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_clip_seconds <= 0 or args.max_clip_seconds <= args.min_clip_seconds:
        print("error: invalid clip duration bounds", file=sys.stderr)
        return 1
    if args.top_n <= 0:
        print("error: --top-n must be positive", file=sys.stderr)
        return 1

    try:
        plans = [
            build_plan_for_transcript(
                transcript_path=path.expanduser().resolve(),
                min_clip_seconds=args.min_clip_seconds,
                max_clip_seconds=args.max_clip_seconds,
                top_n=args.top_n,
            )
            for path in args.transcripts
        ]
        if args.stdout_json:
            print(json.dumps({"plans": plans}, indent=2, ensure_ascii=False))
            return 0
        for index, plan in enumerate(plans):
            json_path, markdown_path = write_outputs(plan, args.output_dir, args.write_markdown)
            if index:
                print()
            print_summary(plan, json_path, markdown_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
