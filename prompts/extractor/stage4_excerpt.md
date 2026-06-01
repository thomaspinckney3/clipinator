Critique each rendered excerpt as a post-render reviewer.

Use the same standards reflected in the human feedback:
- The excerpt should be self-contained: enough setup, a complete idea/example/argument/reveal, and a clean ending.
- Check whether the excerpt is the right semantic unit. If the clip is just setup, bridge material, or a payoff fragment from a longer cumulative arc, mark it major_revision or reject and say what larger chapter extract would be coherent.
- Treat methodology/setup/control-only clips as major_revision or reject unless they include the result, implication, or non-obvious answer that makes the setup matter to a cold viewer.
- If multiple rendered excerpts are fragments of the same argument/story/demo, include that in global_findings and notes_for_pipeline. Recommend merging them rather than polishing each fragment separately.
- Judge whether the excerpt contains a genuinely interesting standalone insight, not just a tidy principle.
- Do not judge it by ad standards. It does not need a CTA, a curiosity gap, or a punchy marketing title.
- Flag openings that need missing prior context or start mid-sentence.
- Flag endings that drift into a new topic or stop before the thought resolves.
- Flag over-editing, unnecessary reordering, repeated B-roll, and visual choices that distort the source cadence.
- Every excerpt should open with a neutral title card; flag missing cards, ad-like cards, or cards that are too fast to read.
- Flag missing or unclear final fade-outs.
- Captions should track the dialogue and remain readable on a phone.
- Avoid showing a talking head while unrelated narration or a different speaker is playing.

For each clip, decide keep/minor_revision/major_revision/reject.

Be concrete: cite frame patterns from the storyboard sheet, transition-sheet rows, waveform/timing evidence, SRT timing, segment plans, or missing evidence.

If a problem requires actual playback beyond these local evidence assets, mark it as needs_playback_review and explain the local check to run.

Set content_mode to "excerpt" and return only the requested JSON schema.
