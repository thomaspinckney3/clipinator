Create stage 5 of a video-to-excerpt pipeline: re-edit after post-render critique.

{{evidence_scope}}

The output must include a full revised Stage 3 plan under revised_stage3_output. It must be directly renderable by extractor.render_stage3.

Do not produce a one-off patch for the current example. Express fixes as general edit decisions driven by critique categories and evidence.

Hard editorial rules:
- Preserve or improve self-contained understanding. The excerpt needs a natural beginning, enough context, complete idea/example/argument/reveal, and a clean ending.
- Re-evaluate the excerpt unit before making local fixes. If Stage 4 or the source evidence shows that several prior excerpts are fragments of one cumulative argument, merge them into one longer chapter_extract and reject/defer the fragment clips.
- Do not preserve a one-input-clip to one-output-clip mapping when the correct editorial fix is fewer, longer extracts. render_queue may contain new merged clip IDs.
- When missing context is structural, extend or merge the extract. Do not paper over missing setup with a title card, summary caption, or punchier opening.
- If an excerpt is only methodology, setup, or an experimental control, keep it only when the revised endpoint includes the result, implication, or non-obvious answer that makes the setup worth watching. Otherwise merge it into the experiment it prepares, or reject/defer it.
- Preserve or strengthen the excerpt's interesting standalone idea, novel insight, concrete reveal, or surprising subject detail. Do not make the re-edit merely a tidy principle summary.
- Do not convert excerpts into ads. Do not add CTA cards, 'watch the full lecture' endings, curiosity-gap copy, or marketing-style titles.
- Every excerpt must open with a neutral generated black title card, normally 4-5 seconds, using a chapter-label version of concept_title.
- For the opening title card, set source_start and source_end to 0, visual_source exactly to 'Generated black title card.', and audio_source to 'No source dialogue; silent generated title card.'
- Prefer source-contiguous and source-order-preserving edits. Only reorder or cover with B-roll when it repairs clarity or visual continuity.
- If Stage 4 flags missing context, extend, merge, or shift the source range rather than making a punchier title.
- If Stage 4 flags topic drift, trim earlier; do not pad with generic visuals.
- Opening title cards should be neutral chapter labels and should be revised if they make the excerpt feel like an ad.
- Do not start or end spoken audio mid-sentence or mid-word. Leave a small tail when needed, but trim before topic drift.
- End at a resolved thought; the renderer will apply a clear fade to black, so do not add an outro card.
- Avoid showing a talking head while unrelated narration or a different speaker is heard; use B-roll, title cards, or matching-source visuals instead.
- Smooth abrupt cuts with crossfades, J/L cuts, cut-on-action choices, or shorter ranges, but do not hide story problems with transitions.
- Preserve transcript-word captions where possible; captions should track dialogue and remain readable on a phone.
- Use render_settings when the critique calls for crossfades, internal visual dissolves, foreground punch-in, or exposure/contrast fixes.
- Preserve and update revised_stage3_output.media_requests. If a critique calls for related visuals, graphics, or audio that are not already in the local asset catalog, add a media_requests entry instead of inventing asset:// references.
- Use media_request://id only as a planning placeholder in placement_notes or timeline source fields. It is not renderable until the requested media has been reviewed, generated or acquired, and added to the augmentation asset catalog as asset://id.
- For third-party media requests, do not invent file URLs. Provide search_query, suggested_sources, license_requirements, review_notes, and a rights_risk value. For generated media, provide a concrete source-faithful generation_prompt, negative_prompt, duration/aspect ratio, style notes, and placement_notes.
- Always include revised_stage3_output.media_requests, even when empty. Every revised clip must include media_request_ids listing related request IDs, or an empty array when none are needed.

Set revised_stage3_output.content_mode to "excerpt". For each clip, fill excerpt_type, excerpt_scope, coherence_unit, merge_or_split_rationale, depends_on_prior_excerpt, self_contained_score, context_needed, source_order_preserved, and cta_policy.

For each clip, explain which Stage 4 issues you addressed and which feedback remains unresolved. If a requested fix needs source evidence that is not present, say so and choose the conservative edit.

PREVIOUS STAGE 3 OUTPUT JSON:
{{stage3_output_json}}

RENDER MANIFEST JSON:
{{render_manifest_json}}

STAGE 4 REVIEW JSON:
{{stage4_output_json}}

REVIEW PACKAGE JSON:
{{review_package_json}}
