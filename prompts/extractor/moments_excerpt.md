Create stage 2 of a video-to-excerpt pipeline: transcript moment analysis.

Input includes the same stable video evidence prefix used in stage 1, plus the stage-1 brief and excerpt candidate groups below.

Your job is not to create an edit plan yet. Find precise spoken windows that can become self-contained extracts.

First identify the coherence unit. It may be an atomic moment, or it may be a longer chapter-style arc whose setup, experiment/story beats, reversal, and explanation only work together.

For each stage-1 candidate group, produce one to three candidate windows. Candidate windows should be as short as possible but as long as necessary: many are 25-120 seconds, but chapter extracts may run several minutes when that is the smallest coherent unit.

Prefer windows with a natural start, a complete idea/example/argument/reveal, and a clean ending. Missing context is the main rejection signal.

A methodology, setup, or experimental-control passage is primary only if it includes the resulting insight, observed consequence, or non-obvious answer that makes the control matter. If it only prepares the viewer for a later experiment, mark it alternate or review_only.

The payoff must be more than 'now we can run the test' or 'this controls for an objection'; it should leave a cold viewer with a resolved takeaway.

A good excerpt may focus on one interesting idea, novel insight, concrete reveal, or surprising subject detail. Do not over-prefer tidy textbook principles when a more specific insight would be more satisfying to watch.

Do not split a cumulative explanation into separate primary windows if each window is only setup for the later payoff. Merge interdependent beats into one broader primary candidate, and mark the fragments as alternate or review_only.

Do not optimize for curiosity gaps, CTA potential, or selling the full video. recommended_use=primary means this can stand alone with light editing.

Favor source-contiguous and source-order-preserving windows. Flag any window that would require heavy reordering or a marketing-style title to make sense.

Use story_function to identify atomic_excerpt, chapter_extract, setup_fragment, payoff_fragment, or bridge_context. Use required_context to say whether the context is included in the window or requires merging with another window.

If dense candidate visual-index sheets are included, use them as visual evidence. Candidate windows should cite visual expectations from exact sheet timestamps/cells when concrete objects or actions are visible.

Penalize windows where the narration, speaker identity, transcript QA, or visual evidence seems unreliable.

Use transcript excerpts sparingly: quote only the exact words needed to identify the moment.

Rank the best windows across all candidates by likely usefulness as standalone extracts, preferring one coherent chapter extract over several individually weak fragments.

Set content_mode to "excerpt".

STAGE 1 OUTPUT JSON:
{{stage1_output_json}}
