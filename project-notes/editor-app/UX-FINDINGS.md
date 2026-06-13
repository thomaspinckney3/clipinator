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
