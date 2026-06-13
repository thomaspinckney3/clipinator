# Review editor — UX findings

Findings from exercising the running app in the browser (you or Claude-in-Chrome).
Claude Code works from this list; it cannot see the rendered UI itself.

**How to capture a good finding** — be specific and reproducible:
- which project / clip
- what you did (the exact clicks)
- what you expected
- what actually happened
- a screenshot if visual

Vague ("feels off") is hard to act on; precise ("toggling a word in clip 3
doesn't update the segment count until I click elsewhere") becomes a fix.

Mark each: `[ ]` open · `[x]` fixed (note the commit) · `[~]` won't-fix/by-design.

---

## Template

```
[ ] <short title>
    project/clip:
    steps:
    expected:
    actual:
    screenshot:
```

## Findings

[x] Cut preview had no audio (fixed: was `<video muted>`; preview is click-
    triggered so it can play with sound).
[x] Cut preview had no pause/play controls (fixed: added native `controls`; also
    fixed it running past the end of the cut into the rest of the source).
[x] Adding dialogue (including more words) didn't show as captions (fixed:
    captions are now derived live from the in-words for the preview, and
    regenerated from the transcript on save — verified caption count grows with
    included words, human_edited_captions stays False/verbatim).
    — all three in the same commit; please re-test in the browser to confirm
    audio + controls feel right (I can't drive a real browser here).
[x] Cut preview could play the whole source start-to-finish and had no easy
    "back to clip start" (fixed: replaced native source controls with a
    cut-confined transport — play/pause, ⏮ start, and a clickable cut progress
    bar; playback is clamped to the segments so it never leaves the clip).
[x] Couldn't include content before (or after) where a clip currently starts —
    the transcript view was bounded to the clip span (fixed: backend now sends
    the full transcript; the editor works over global word indices with an
    expandable window — "⟵ 30s earlier" / "30s later ⟶" reveal more words to
    toggle in, so any part of the lecture can be pulled into a clip).

## 2026-06-13 — live verification via Claude-in-Chrome (no bugs found)

Drove the running app in a real browser through the official Claude Code↔Chrome
integration. All recent features confirmed working:
- project discovery + open; transcript-first UI renders (green=in, struck=out).
- "⟵ 30s earlier" expands the window (showing 1:12→0:42 of 46:56) and reveals
  pre-clip transcript to include — the "reach content before the clip" fix, live.
- clicking a struck word includes it; segment count + "source cuts" recompute
  live (first cut 1:16 → 1:15); the saved plan reflected it (start 76.2 → 75.34).
- cut preview: plays WITH audio (muted:false), confined to segments (jumped
  seg1 1:15–1:23 → seg2 2:44–3:26), dual clock "source M:SS · cut M:SS / len",
  caption overlay, 9:16 crop; pause holds; ▶/⏸ toggles.
- Save writes review/<variant>/plan-output.human.json; human_edited_captions
  False (verbatim). No console errors throughout.

## 2026-06-13 — cross-video editorial pass (Braver, Gemes, Hamkins, Roughgarden)

Drove realistic pre-publish edits across 4 lecturers via Claude-in-Chrome.
NO app bugs found — every realistic edit worked through the real click handler
and persisted on save:
- drop a clip; fix a mid-sentence START by clicking the preceding word in
  (Hamkins "0.999…": "next," → "So next,", midSentence flips false, +6s);
- fix a dangling END by striking the trailing word (Hamkins "Harmonic Series":
  removed trailing "So" → ends "…going to zero."); saved plan confirmed it;
- reach pre-clip content via "⟵ 30s earlier"; retitle; preview; render.

Editorial patterns observed (content, not bugs):
- DRAFT clips frequently start mid-sentence or on filler ("Um, so…", "next,")
  and occasionally end on a dangling discourse marker ("So"). Roughgarden draft
  clip 2 starts on lowercase "so" after "Um,".
- Stutters / false starts at clip starts ("W-what's…") — a trim candidate.
- Some titles are long enough to wrap a title card ("A Turing Machine Is a
  Person With a Long Roll of Paper").
- Draft excerpts run long (1–4.3 min); the revise stage tightens them and
  cleans boundaries — Braver FINAL clips were all clean (starts Science/There/
  Imagine, ends create./exploring./that?).

RECOMMENDATION (high value, cheap): auto-flag these inline in the editor. The
app already has the transcript + word boundaries, so it can mark each clip with
"⚠ starts mid-sentence", "⚠ ends on a filler word", "⚠ leading stutter", and
"⚠ N min — long" — pointing the editor straight at the clips that need work.
Rule-based, no LLM. (A simpler precursor to the AI-critique layer.)
