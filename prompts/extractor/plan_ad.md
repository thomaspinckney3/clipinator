Create stage 3 of a video-to-clip-ad pipeline: concrete edit plans.

Input includes the stable video evidence prefix, stage-1 strategy, and stage-2 candidate transcript moments.

Create edit plans for the strongest {{clip_count}} short-form clip ads.

Assume a default target of {{target_seconds}} seconds and {{target_aspect_ratio}} vertical delivery unless a clip clearly needs a different duration.

This is not a render request. Produce editor-facing instructions that could drive a later FFmpeg/Premiere/Resolve assembly step.

Each clip plan must specify the opening hook, source in/out ranges, audio strategy, visual strategy, B-roll coverage, captions/overlays, and verification steps.

Set content_mode to "ad".

Treat the Stage 2 cross-angle ranking as the default selection order. Only skip a higher-ranked candidate for a lower-ranked one when the higher-ranked candidate has a concrete defect such as no clear payoff, unusable visual proof, severe transcript risk, topic duplication, or a cold-viewer accessibility problem that cannot be fixed inside the short format. Record skipped strong candidates in deferred_candidates with the specific reason.

The selected clips should work as a diverse portfolio advertising the full video. Avoid choosing multiple clips with the same named object, source section, prop, demo type, audience promise, or cognitive shape when a strong distinct candidate is available. Cognitive shape includes story, demonstration, practical takeaway, emotional beat, visual process, expert explanation, and dense technical reasoning.

Prefer the strongest distinct story over a second variation on an already selected motif. A lower-ranked candidate that repeats the same object or hook family must be clearly better than the higher-ranked distinct alternative, not merely easier to title.

Balance surprise with accessibility. Do not let the final set become uniformly high-context, jargon-heavy, or explanation-dense unless the entire source offers no stronger accessible alternatives. At least one selected ad should normally function as an approachable on-ramp for a cold viewer.

Only create a clip plan when there is a clear payoff. A missing or weak payoff is a rejection signal; put that candidate in deferred_candidates instead of trying to rescue it with pacing, captions, or B-roll.

Default story order: title/tagline card first, then the strongest payoff or surprise line as the first spoken beat, then supporting context. The payoff is the hook; do not bury it at the end when it can cleanly open the clip.

Only move a payoff line to the first spoken beat if that source line is syntactically and semantically self-contained. If it starts with a connective such as 'but', 'and', or 'so', depends on prior context, or would sound like a mid-sentence start, either include the needed setup before it or defer the candidate.

Every planned clip ad should open with a short, intentional title/tagline card unless the visual action itself is the payoff. For interview, narration, or explanation-led hooks, use a black title card first; do not replace it with source video just because dialogue can start immediately.

Title/tagline cards are ad headlines, not summaries. Make concept_title and the first rendered caption_or_overlay short, concrete, and punchy: target 3-7 words, with a soft cap of 9 words or about 55 characters unless a source quote or essential number forces more.

A cold viewer still needs to understand the promise before source audio plays. Use a concrete subject plus a surprising payoff, but do not explain the whole clip in the title. Put extra context in opening_hook.visual as a small subtitle or in the following spoken beat, not in caption_or_overlay.

Prefer specific high-curiosity wording like 'The spare part saved the launch', 'Ancient Rome had takeout', 'One cut reveals the hidden layer', or 'Rejected 30 times, then accepted'. Avoid long institutional, academic, or generic summary titles like 'This program preserves important samples for decades of future research'.

Do not use cryptic shorthand, unexplained numbers, ambiguous binaries, insider labels, or source-specific abbreviations. If a short accurate title cannot be written without sounding vague, defer the candidate.

The first generated title card's caption_or_overlay should normally be identical to concept_title. If it differs, it must be a clearer sentence-case variant, not a vaguer slogan. Do not make concept_title clearer than the rendered title card.

Use sentence case for title/tagline cards. Do not use all caps except acronyms or one intentionally emphasized word.

Title/tagline card duration should normally be 4-5 seconds. Do not use a 1-2 second title flash unless the visual action itself immediately explains the hook.

For any generated title, hook, or CTA card, set source_start and source_end to 0, visual_source exactly to 'Generated black title card.', and audio_source to 'No source dialogue; silent generated title card.' Do not describe it as a music bed only or source roomtone; the renderer treats that as ambiguous.

Title, overlay, and caption wording must preserve the source claim. Do not compress 'the next words were about science' into the false literal claim 'the next words were science'; distinguish narrator interpretation from the actual quoted words.

Assume many viewers will watch muted on a phone. Caption plans should make the argument understandable without audio, use short phrase-level beats, and keep essential words away from platform UI safe-zone areas.

If the payoff occurs later in the source than its setup, jump to the payoff first, then cut back to the setup/context. Make this chronology shift explicit in the edit_action and cover it with B-roll when needed.

Prefer a black title card or other fully intentional title beat over silent source video before dialogue. Do not let source video run for multiple seconds before the audio starts unless the visual action itself is the hook.

Prefer plans that use visually provable moments. If the best spoken hook starts with a talking head, recommend a title card, B-roll cover, visual-first cold open, or J-cut when appropriate.

If an OPTIONAL AUGMENTATION ASSET CATALOG is provided in source materials, you may use a listed video/image/graphic asset as related cutaway material by writing asset://id or asset://id#t=SECONDS in timeline.visual_source. Use augmentation only when it makes the story more emotionally legible, provides needed visual proof/metaphor, or covers an edit that would otherwise be static or confusing. Do not use generic or misleading imagery.

If an OPTIONAL AUGMENTATION ASSET CATALOG includes audio, you may mention asset://id in timeline.audio_source only as a quiet bed or standalone card audio when the license and usage notes allow it. Source dialogue remains primary and must stay synchronized.

If the best augmentation asset is not already in the catalog, add a media_requests entry instead of pretending the asset exists. Use media_request://id in placement_notes and, when useful, in timeline.visual_source or timeline.audio_source as a non-renderable placeholder. These requests can ask for third-party license-friendly media or newly generated video, image, graphic, or audio.

For third-party media requests, do not invent a specific file URL. Provide search_query, suggested_sources, license_requirements, and review_notes. Prefer public-domain, CC0, CC BY with attribution, institutional open media libraries, or clearly licensable stock sources. Mark rights_risk high if any recognizable person, brand, artwork, location release, or ambiguous license issue is likely.

For generated asset requests, provide a concrete generation_prompt, negative_prompt, desired duration/aspect ratio, style notes, and placement_notes. Prompts must be source-faithful and must not imply facts not in the source.

Always include a top-level media_requests array, even when empty. Every clip must include media_request_ids listing the related request IDs, or an empty array when none are needed.

Before creating a plan, check whether the named object/action in the hook can be visually proven. If the plan would have to say 'verify whether the object is visible' for a core premise, defer the candidate instead of rendering it.

Do not use generic related visuals as a substitute for a concrete object named by the hook. A clip titled around a specific object or action must show that object/action clearly or provide unambiguous source evidence; otherwise it belongs in deferred_candidates.

If dense candidate visual-index sheets are included, use them as the authority for exact visual proof. Cite specific timestamps from those sheets in visual_source, source_ranges, visual_plan, and verification_steps.

Be explicit about where audio and video may need to be offset, covered with B-roll, faded in, or manually checked for speaker/narration alignment.

Do not include transcript-artifact text as dialogue. Put questionable ranges in the manual review queue.

STAGE 1 OUTPUT JSON:
{{stage1_output_json}}

STAGE 2 OUTPUT JSON:
{{stage2_output_json}}
