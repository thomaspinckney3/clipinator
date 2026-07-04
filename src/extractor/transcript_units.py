"""Logical content units for a source transcript.

Both the review server's storyboard timeline and the `openai_outline` pipeline
stage derive units from the transcript here, so their unit lists (and therefore
indices) always line up. Stdlib-only, so the review server stays dependency-free.
"""
from __future__ import annotations

import re
from typing import Any

from extractor.render_plan import transcript_words_from_data

# Function words + verbal disfluencies dropped when distilling a unit's gist for
# the instant heuristic label. Compared against each token's base (the part
# before an apostrophe), so contractions ("we're"->"we", "didn't"->"did") are
# caught too. The LLM outline stage produces far better labels; this is the
# free, instant fallback.
_SUMMARY_SKIP = frozenset("""
a an the and or but so if then than that this these those of to in on at for with from by as
is are was were be been being am do does did doing have has had having will would can could
should may might must i you he she it we they me him her us them my your his its our their
not no nor too very just about into over under again here there
what when where who whom whose why how which while because although though since until whether
all any some many much few more most other another each every no none thing things stuff one
get got getting go going make makes made making also even still yet now well
didn doesn don wasn weren isn aren wouldn couldn shouldn hasn haven hadn won mustn needn mightn ain
uh um er ah eh hmm mm mhm yeah yep okay ok alright right like actually basically literally
really kind sort mean know gonna wanna gotta youknow
""".split())


def pithy_summary(text: str, max_words: int = 3) -> str:
    """A short heuristic label: drop disfluencies and stopwords and keep the first
    few content-bearing words (original casing/order), with no truncation ellipsis
    so it fits a timeline block. Falls back to the first raw words when a unit is
    nothing but filler."""
    # Unicode-aware so "René" survives; keep an internal apostrophe ("We're").
    tokens = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", str(text))
    picked: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        base = re.split(r"['’]", tok.lower())[0]  # "we're" -> "we", "couldn't" -> "couldn"
        if not base or base in _SUMMARY_SKIP or base in seen:
            continue
        seen.add(base)
        picked.append(tok)
        if len(picked) >= max_words:
            break
    if not picked:
        picked = tokens[:max_words]
    return " ".join(picked)


def transcript_units_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Logical content units for the storyboard timeline and the outline stage.
    Prefer the transcript's own segments (real semantic-ish units from the ASR
    stage); if a transcript has only words, group them on pauses. Each unit carries
    its source-time span, the full text, and an instant heuristic `summary`."""
    units: list[dict[str, Any]] = []
    for seg in data.get("segments") or []:
        start, end = seg.get("start"), seg.get("end")
        text = str(seg.get("text", "")).strip()
        if start is None or end is None or not text:
            continue
        try:
            s, e = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if e > s:
            units.append({"start": round(s, 3), "end": round(e, 3),
                          "summary": pithy_summary(text), "text": text})
    if units:
        return units
    # Fallback for word-only transcripts: break on >0.7s pauses or every ~10 words.
    group: list[dict[str, Any]] = []
    def flush() -> None:
        if group:
            text = " ".join(w["text"] for w in group)
            units.append({"start": round(group[0]["start"], 3),
                          "end": round(group[-1]["end"], 3),
                          "summary": pithy_summary(text), "text": text})
    for w in transcript_words_from_data(data):
        if group and (w["start"] - group[-1]["end"] > 0.7 or len(group) >= 10):
            flush(); group = []
        group.append(w)
    flush()
    return units
