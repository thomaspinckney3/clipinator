from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MIN_WORDS_PER_MINUTE = 70.0
DEFAULT_MAX_WORDS_PER_MINUTE = 220.0
DEFAULT_MIN_SEGMENT_WPS = 0.35
DEFAULT_MAX_SEGMENT_WPS = 6.0
DEFAULT_MAX_GAP_SECONDS = 12.0
DEFAULT_MAX_TAIL_GAP_SECONDS = 12.0
DEFAULT_MIN_AVG_LOGPROB = -0.8
DEFAULT_MAX_NO_SPEECH_PROB = 0.6
DEFAULT_MAX_COMPRESSION_RATIO = 2.4
DEFAULT_MAX_SEGMENT_SECONDS = 30.0
DEFAULT_MAX_FINDINGS = 25


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    message: str
    start: float | None = None
    end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "kind": self.kind,
            "message": self.message,
        }
        if self.start is not None:
            payload["start"] = round(self.start, 3)
        if self.end is not None:
            payload["end"] = round(self.end, 3)
        return payload


def read_transcript(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path}: invalid JSON: {exc}") from exc


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def segment_duration(segment: dict[str, Any]) -> float | None:
    start = number(segment.get("start"))
    end = number(segment.get("end"))
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def detect_repeated_tokens(words: list[dict[str, Any]]) -> list[Finding]:
    tokens = [normalize_token(str(word.get("text", ""))) for word in words]
    findings: list[Finding] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token:
            index += 1
            continue
        end_index = index + 1
        while end_index < len(tokens) and tokens[end_index] == token:
            end_index += 1
        repeat_count = end_index - index
        if repeat_count >= 5:
            start = number(words[index].get("start"))
            end = number(words[end_index - 1].get("end"))
            findings.append(
                Finding(
                    severity="warning",
                    kind="repeated_word",
                    start=start,
                    end=end,
                    message=f"Word repeated {repeat_count} times: {words[index].get('text')!r}",
                )
            )
        index = end_index
    return findings


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def severity_penalty(severity: str) -> int:
    return {"error": 18, "warning": 4, "info": 1}.get(severity, 1)


def score_from_findings(findings: list[Finding]) -> int:
    penalty = sum(severity_penalty(finding.severity) for finding in findings)
    return max(0, 100 - penalty)


def build_report(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    data = read_transcript(path)
    source = data.get("source", {})
    segments = data.get("segments", [])
    words = data.get("words", [])
    duration = number(source.get("duration_seconds")) or 0.0

    findings: list[Finding] = []
    if not segments:
        findings.append(Finding("error", "missing_segments", "Transcript has no segments."))
    if not words:
        findings.append(Finding("error", "missing_words", "Transcript has no words."))

    segment_starts = [number(segment.get("start")) for segment in segments]
    segment_ends = [number(segment.get("end")) for segment in segments]
    valid_starts = [value for value in segment_starts if value is not None]
    valid_ends = [value for value in segment_ends if value is not None]
    first_segment_start = min(valid_starts) if valid_starts else None
    last_segment_end = max(valid_ends) if valid_ends else None

    missing_timestamps = sum(
        1
        for segment in segments
        if number(segment.get("start")) is None or number(segment.get("end")) is None
    )
    if missing_timestamps:
        findings.append(
            Finding(
                "error",
                "missing_segment_timestamps",
                f"{missing_timestamps} segment(s) are missing start or end timestamps.",
            )
        )

    previous_end: float | None = None
    gaps: list[float] = []
    segment_durations: list[float] = []
    avg_logprobs: list[float] = []
    no_speech_probs: list[float] = []
    compression_ratios: list[float] = []

    for index, segment in enumerate(segments):
        start = number(segment.get("start"))
        end = number(segment.get("end"))
        duration_seconds = segment_duration(segment)
        text = str(segment.get("text", "")).strip()
        segment_words = segment.get("words", [])

        if start is not None and end is not None and end < start:
            findings.append(
                Finding(
                    "error",
                    "negative_segment_duration",
                    f"Segment {index} ends before it starts.",
                    start,
                    end,
                )
            )
        if duration_seconds is not None:
            segment_durations.append(duration_seconds)
            if duration_seconds > args.max_segment_seconds:
                findings.append(
                    Finding(
                        "warning",
                        "long_segment",
                        f"Segment {index} is {duration_seconds:.1f}s long.",
                        start,
                        end,
                    )
                )
            if text and duration_seconds > 0:
                words_per_second = len(segment_words) / duration_seconds
                if words_per_second < args.min_segment_wps:
                    findings.append(
                        Finding(
                            "info",
                            "low_segment_word_density",
                            f"Segment {index} has low word density: {words_per_second:.2f} words/s.",
                            start,
                            end,
                        )
                    )
                if words_per_second > args.max_segment_wps:
                    findings.append(
                        Finding(
                            "warning",
                            "high_segment_word_density",
                            f"Segment {index} has high word density: {words_per_second:.2f} words/s.",
                            start,
                            end,
                        )
                    )

        if previous_end is not None and start is not None:
            gap = start - previous_end
            if gap < -0.25:
                findings.append(
                    Finding(
                        "warning",
                        "overlapping_segments",
                        f"Segment {index} overlaps previous segment by {abs(gap):.1f}s.",
                        start,
                        end,
                    )
                )
            elif gap > args.max_gap_seconds:
                gaps.append(gap)
                findings.append(
                    Finding(
                        "info",
                        "long_gap",
                        f"Gap before segment {index} is {gap:.1f}s.",
                        previous_end,
                        start,
                    )
                )
        if end is not None:
            previous_end = end if previous_end is None else max(previous_end, end)

        avg_logprob = number(segment.get("avg_logprob"))
        if avg_logprob is not None:
            avg_logprobs.append(avg_logprob)
            if avg_logprob < args.min_avg_logprob:
                findings.append(
                    Finding(
                        "warning",
                        "low_avg_logprob",
                        f"Segment {index} has low avg_logprob: {avg_logprob:.3f}.",
                        start,
                        end,
                    )
                )

        no_speech_prob = number(segment.get("no_speech_prob"))
        if no_speech_prob is not None:
            no_speech_probs.append(no_speech_prob)
            if text and no_speech_prob > args.max_no_speech_prob:
                findings.append(
                    Finding(
                        "warning",
                        "high_no_speech_prob",
                        f"Segment {index} has high no_speech_prob: {no_speech_prob:.3f}.",
                        start,
                        end,
                    )
                )

        compression_ratio = number(segment.get("compression_ratio"))
        if compression_ratio is not None:
            compression_ratios.append(compression_ratio)
            if compression_ratio > args.max_compression_ratio:
                findings.append(
                    Finding(
                        "warning",
                        "high_compression_ratio",
                        f"Segment {index} has high compression_ratio: {compression_ratio:.3f}.",
                        start,
                        end,
                    )
                )

    if last_segment_end is not None and duration:
        tail_gap = duration - last_segment_end
        if tail_gap > args.max_tail_gap_seconds:
            findings.append(
                Finding(
                    "info",
                    "large_tail_gap",
                    f"Transcript ends {tail_gap:.1f}s before source duration.",
                    last_segment_end,
                    duration,
                )
            )

    timed_word_count = sum(
        1
        for word in words
        if number(word.get("start")) is not None and number(word.get("end")) is not None
    )
    if timed_word_count != len(words):
        findings.append(
            Finding(
                "warning",
                "missing_word_timestamps",
                f"{len(words) - timed_word_count} word(s) are missing timestamps.",
            )
        )

    words_per_minute = len(words) / (duration / 60) if duration else 0.0
    if words and words_per_minute < args.min_words_per_minute:
        findings.append(
            Finding(
                "info",
                "low_speech_rate",
                f"Overall speech rate is low: {words_per_minute:.1f} words/minute.",
            )
        )
    if words_per_minute > args.max_words_per_minute:
        findings.append(
            Finding(
                "warning",
                "high_speech_rate",
                f"Overall speech rate is high: {words_per_minute:.1f} words/minute.",
            )
        )

    findings.extend(detect_repeated_tokens(words))

    transcript_span = (
        (last_segment_end - first_segment_start)
        if first_segment_start is not None and last_segment_end is not None
        else 0.0
    )
    segment_duration_total = sum(segment_durations)
    coverage_ratio = segment_duration_total / duration if duration else 0.0

    sorted_findings = sorted(
        findings,
        key=lambda finding: (
            {"error": 0, "warning": 1, "info": 2}.get(finding.severity, 3),
            finding.start if finding.start is not None else -1,
            finding.kind,
        ),
    )

    report = {
        "path": str(path),
        "source": source,
        "score": score_from_findings(findings),
        "metrics": {
            "duration_seconds": round(duration, 3),
            "segment_count": len(segments),
            "word_count": len(words),
            "timed_word_count": timed_word_count,
            "words_per_minute": round(words_per_minute, 1),
            "first_segment_start": first_segment_start,
            "last_segment_end": last_segment_end,
            "transcript_span_seconds": round(transcript_span, 3),
            "segment_timeline_coverage_ratio": round(coverage_ratio, 3),
            "long_gap_count": len(gaps),
            "max_gap_seconds": round(max(gaps), 3) if gaps else 0.0,
            "avg_segment_duration_seconds": round(statistics.mean(segment_durations), 3)
            if segment_durations
            else 0.0,
            "p95_segment_duration_seconds": round(percentile(segment_durations, 0.95) or 0.0, 3),
            "min_avg_logprob": round(min(avg_logprobs), 3) if avg_logprobs else None,
            "max_no_speech_prob": round(max(no_speech_probs), 3) if no_speech_probs else None,
            "max_compression_ratio": round(max(compression_ratios), 3)
            if compression_ratios
            else None,
        },
        "finding_counts": {
            "error": sum(1 for finding in findings if finding.severity == "error"),
            "warning": sum(1 for finding in findings if finding.severity == "warning"),
            "info": sum(1 for finding in findings if finding.severity == "info"),
        },
        "findings": [finding.to_dict() for finding in sorted_findings[: args.max_findings]],
        "truncated_findings": max(0, len(sorted_findings) - args.max_findings),
    }
    return report


def print_text_report(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    counts = report["finding_counts"]
    print(report["path"])
    print(f"  score: {report['score']}/100")
    print(
        "  metrics: "
        f"{metrics['segment_count']} segments, "
        f"{metrics['word_count']} words, "
        f"{metrics['words_per_minute']} wpm, "
        f"coverage {metrics['segment_timeline_coverage_ratio']}"
    )
    print(
        "  signals: "
        f"min avg_logprob={metrics['min_avg_logprob']}, "
        f"max no_speech_prob={metrics['max_no_speech_prob']}, "
        f"max compression_ratio={metrics['max_compression_ratio']}"
    )
    print(
        "  findings: "
        f"{counts['error']} errors, {counts['warning']} warnings, {counts['info']} info"
    )
    for finding in report["findings"]:
        time_range = ""
        if "start" in finding and "end" in finding:
            time_range = f" [{finding['start']:.3f}-{finding['end']:.3f}]"
        print(
            f"    - {finding['severity']} {finding['kind']}{time_range}: "
            f"{finding['message']}"
        )
    if report["truncated_findings"]:
        print(f"    ... {report['truncated_findings']} more finding(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local QA checks against normalized transcript JSON files."
    )
    parser.add_argument("transcripts", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Write JSON report to stdout.")
    parser.add_argument("--max-findings", type=int, default=DEFAULT_MAX_FINDINGS)
    parser.add_argument("--min-words-per-minute", type=float, default=DEFAULT_MIN_WORDS_PER_MINUTE)
    parser.add_argument("--max-words-per-minute", type=float, default=DEFAULT_MAX_WORDS_PER_MINUTE)
    parser.add_argument("--min-segment-wps", type=float, default=DEFAULT_MIN_SEGMENT_WPS)
    parser.add_argument("--max-segment-wps", type=float, default=DEFAULT_MAX_SEGMENT_WPS)
    parser.add_argument("--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS)
    parser.add_argument("--max-tail-gap-seconds", type=float, default=DEFAULT_MAX_TAIL_GAP_SECONDS)
    parser.add_argument("--min-avg-logprob", type=float, default=DEFAULT_MIN_AVG_LOGPROB)
    parser.add_argument("--max-no-speech-prob", type=float, default=DEFAULT_MAX_NO_SPEECH_PROB)
    parser.add_argument(
        "--max-compression-ratio",
        type=float,
        default=DEFAULT_MAX_COMPRESSION_RATIO,
    )
    parser.add_argument("--max-segment-seconds", type=float, default=DEFAULT_MAX_SEGMENT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        reports = [build_report(path.expanduser().resolve(), args) for path in args.transcripts]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"reports": reports}, indent=2, ensure_ascii=False))
    else:
        for index, report in enumerate(reports):
            if index:
                print()
            print_text_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
