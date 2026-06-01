Create stage 2 of a video-to-clip-ad pipeline: transcript moment analysis.

Input includes the same stable video evidence prefix used in stage 1, plus the stage-1 brief and ad angles below.

Your job is not to create an edit plan yet. Your job is to find precise spoken moments and nearby transcript windows that could support each ad angle.

For each stage-1 angle, produce one to three candidate windows. Candidate windows should usually be 15-75 seconds, but can be shorter when they are only supporting B-roll or need to be combined later.

Prefer windows with a clean hook, enough context, and a clear payoff. A missing or weak payoff is a strong signal that the window is a bad clip candidate, not merely an editing problem.

Do not mark a window as primary unless its payoff field names a specific spoken or visual payoff. Use weak_or_rejected_windows or recommended_use=review_only for windows that are only setup, atmosphere, or information without a payoff.

Rank for cold-viewer comprehension as well as hook strength. High surprise is not enough when the window requires too much outside knowledge, too many unintroduced terms, or several dense reasoning steps before the payoff makes sense.

Across the ranked list, avoid letting one cognitive shape dominate when strong alternatives exist. A useful portfolio can mix concrete stories, demonstrations, visual processes, emotional stakes, practical takeaways, and deeper explanations; do not rank only the densest or most technical moments by default.

Use why_this_ranks_here and edit_implication to call out accessibility risks, missing context, or cognitive load when they affect ranking.

If an angle depends on a concrete visible object or action, do not mark it primary unless the evidence indicates that object/action is likely visible in the window or nearby B-roll. For example, a clip about a specific tool needs visible tool use, not just generic related footage.

If dense candidate visual-index sheets are included, use them as visual evidence. Candidate windows should cite visual expectations from exact sheet timestamps/cells when concrete objects or actions are visible.

Penalize windows where the narration, speaker identity, transcript QA, or visual evidence seems unreliable.

Flag any window that must be checked against the original audio/video before editing. Do not use the known repeated-text artifact as a substantive candidate.

Use transcript excerpts sparingly: quote only the exact words needed to identify the moment.

Rank the best moments across all angles by likely usefulness for the next edit-plan stage.

Set content_mode to "ad".

STAGE 1 OUTPUT JSON:
{{stage1_output_json}}
