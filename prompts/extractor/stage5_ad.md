Create stage 5 of a video-to-clip-ad pipeline: re-edit after post-render critique.

{{evidence_scope}}

The output must include a full revised Stage 3 plan under revised_stage3_output. It must be directly renderable by extractor.render_stage3.

Set revised_stage3_output.content_mode to "ad".

Do not produce a one-off patch for the current example. Express fixes as general edit decisions driven by critique categories and evidence.

Hard editorial rules:
- If there is no clear payoff, reject or defer the clip instead of trying to rescue it with captions or faster cuts.
- Preserve portfolio diversity. Do not keep multiple clips with the same named object, prop, demo type, source section, audience promise, or cognitive shape when a strong distinct candidate is available in the supplied Stage 3/critique/source evidence.
- Preserve cold-viewer accessibility. If Stage 4 flags that a clip is too high-context, jargon-heavy, or explanation-dense for the rendered short format, simplify, extend context, swap to a more accessible candidate, or defer it instead of adding more dense overlays.
- If Stage 4 flags redundant selected hooks, replace or defer the weaker duplicate rather than retitling it.
- Every revised clip ad should open with a short, intentional title/tagline card unless the visual action itself is the payoff.
- Title/tagline cards are ad headlines, not summaries. Make concept_title and the first rendered caption_or_overlay short, concrete, and punchy: target 3-7 words, with a soft cap of 9 words or about 55 characters unless a source quote or essential number forces more.
- Title/tagline cards must be understandable to a cold viewer before source audio starts. Rewrite cryptic shorthand, unexplained numbers, ambiguous either/or phrases, insider labels, vague hype, and wordy institutional summaries into a concrete subject plus surprising payoff.
- Put extra explanatory context in opening_hook.visual as a small subtitle or in the first spoken beat, not in the main title-card caption_or_overlay.
- The first generated title card's caption_or_overlay should normally be identical to concept_title. If Stage 4 says concept_title is clearer than the visible title card, rewrite concept_title itself into the shortest clear title-card text and use that same text as caption_or_overlay.
- Use sentence case for title/tagline cards. Do not preserve multi-word all-caps cards except acronyms or one intentionally emphasized word.
- Title/tagline cards should normally last 4-5 seconds so the viewer can read and understand the hook. Do not preserve a 1-2 second title flash.
- For any generated title, hook, or CTA card, set source_start and source_end to 0, visual_source exactly to 'Generated black title card.', and audio_source to 'No source dialogue; silent generated title card.' This is the renderer contract; do not use source_start=0 footage as a fake card.
- The payoff or surprising line should be the first spoken beat after the title/tagline card whenever it can open cleanly.
- Do not put a payoff line first if it begins with a connective such as 'but', 'and', or 'so', depends on prior context, or sounds like a mid-sentence start. Include the setup before it or reject/defer the clip.
- Title, overlay, and caption wording must preserve the source claim. Do not turn a narrator interpretation into a literal factual claim; for example, 'the next words were about science' is not the same as 'the next words were science'.
- Prefer an intentional black title card over silent source video before dialogue.
- If Stage 4 flags a broken title, repair the title card text/timing; do not remove the title card and start directly on interview or narration footage.
- Do not start or end spoken audio mid-sentence or mid-word. Leave a small tail when needed, but trim before topic drift.
- Avoid showing a talking head while unrelated narration or a different speaker is heard; use B-roll, title cards, or matching-source visuals instead.
- If B-roll repeats to stretch the clip, shorten the clip unless verified alternate visuals are available.
- If Stage 4 says the concrete object/action named by the clip is not visually legible, either find explicit visual proof from the supplied source evidence or mark the clip reject/defer with should_render=false.
- Do not turn a hard visual-proof gap into a soft note for later review. If a clip names a specific object or action that cannot be shown or supported by unambiguous source evidence, do not include it in revised_stage3_output.clip_ads.
- Smooth abrupt cuts with crossfades, J/L cuts, cut-on-action choices, or shorter ranges, but do not hide story problems with transitions.
- The ending must stay on the same story and should land after the thought resolves.
- Phrase-level captions should be explicit render-time SRT entries. When doing that, set caption_plan_source to absolute_render_time on the revised clip.
- Assume many viewers will watch muted on a phone. Preserve or rebuild captions so the core argument is understandable without audio, use short phrase-level beats, and keep essential words out of platform UI safe-zone areas.
- Use render_settings when the critique calls for crossfades, internal visual dissolves, foreground punch-in, or exposure/contrast fixes.
- Preserve and update revised_stage3_output.media_requests. If a critique calls for related visuals, graphics, or audio that are not already in the local asset catalog, add a media_requests entry instead of inventing asset:// references.
- Use media_request://id only as a planning placeholder in placement_notes or timeline source fields. It is not renderable until the requested media has been reviewed, generated or acquired, and added to the augmentation asset catalog as asset://id.
- For third-party media requests, do not invent file URLs. Provide search_query, suggested_sources, license_requirements, review_notes, and a rights_risk value. For generated media, provide a concrete source-faithful generation_prompt, negative_prompt, duration/aspect ratio, style notes, and placement_notes.
- Always include revised_stage3_output.media_requests, even when empty. Every revised clip must include media_request_ids listing related request IDs, or an empty array when none are needed.

For each clip, explain which Stage 4 issues you addressed and which feedback remains unresolved. If a requested fix needs source evidence that is not present, say so and choose the conservative edit.

PREVIOUS STAGE 3 OUTPUT JSON:
{{stage3_output_json}}

RENDER MANIFEST JSON:
{{render_manifest_json}}

STAGE 4 CRITIQUE JSON:
{{stage4_output_json}}

OPTIONAL LOCAL REVIEW PACKAGE JSON:
{{review_package_json}}
