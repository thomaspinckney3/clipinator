Create stage 3 of a video-to-excerpt pipeline: concrete edit plans.

Input includes the stable video evidence prefix, stage-1 excerpt strategy, and stage-2 candidate transcript moments.

Create edit plans for up to {{clip_count}} standalone extracts.

Assume a default target of {{target_seconds}} seconds and {{target_aspect_ratio}} vertical delivery unless an extract clearly needs a different duration.

This is not an ad render request. Produce editor-facing instructions that preserve a self-contained part of the longer video as a complete watch.

Each excerpt plan must specify source in/out ranges, audio strategy, visual strategy, light editing, captions/overlays, and verification steps.

Do not force weak excerpts just to fill the count. Put candidates that do not stand alone in deferred_candidates.

Before choosing clips, determine the smallest coherent semantic unit for each candidate. If the payoff depends on a chain of setup, experiment/story beats, reversal, and explanation, plan one longer chapter_extract instead of several short fragments.

The clip_count is an upper bound, not a quota. Prefer one coherent chapter extract over several shorter clips that are individually unclear or uninteresting.

Reject or defer methodology/setup/control-only candidates unless the planned endpoint includes the resulting insight, observed consequence, or surprising answer that makes the setup matter. A clip that only explains how a later test will be controlled is not yet a standalone excerpt.

Do not split a cumulative argument only to satisfy a short-form duration target. Split only when each resulting part has its own natural beginning, complete thought, payoff, and clean ending.

Use excerpt_scope to classify each plan as atomic_excerpt or chapter_extract. Fill coherence_unit with the idea/arc that must remain intact, merge_or_split_rationale with why it was or was not merged, and depends_on_prior_excerpt=false unless the extract is intentionally part of a series.

Default story order: preserve source order and natural cadence. Trim dead air, false starts, or topic drift, but do not move a payoff line to the front just to create a hook.

Only reorder or use separated B-roll when it solves a clarity or visual-continuity problem; explain that explicitly in edit_action.

If an OPTIONAL AUGMENTATION ASSET CATALOG is provided in source materials, you may use a listed video/image/graphic asset as related cutaway material by writing asset://id or asset://id#t=SECONDS in timeline.visual_source. Use augmentation only when it improves emotional clarity, explains an abstract point, or covers an otherwise static/awkward visual. Do not use generic or misleading imagery.

If an OPTIONAL AUGMENTATION ASSET CATALOG includes audio, you may mention asset://id in timeline.audio_source only as a quiet bed or standalone card audio when the license and usage notes allow it. Source dialogue remains primary and must stay synchronized.

If the best augmentation asset is not already in the catalog, add a media_requests entry instead of pretending the asset exists. Use media_request://id in placement_notes and, when useful, in timeline.visual_source or timeline.audio_source as a non-renderable placeholder. These requests can ask for third-party license-friendly media or newly generated video, image, graphic, or audio.

For third-party media requests, do not invent a specific file URL. Provide search_query, suggested_sources, license_requirements, and review_notes. Prefer public-domain, CC0, CC BY with attribution, institutional open media libraries, or clearly licensable stock sources. Mark rights_risk high if any recognizable person, brand, artwork, location release, or ambiguous license issue is likely.

For generated asset requests, provide a concrete generation_prompt, negative_prompt, desired duration/aspect ratio, style notes, and placement_notes. Prompts must be source-faithful and must not imply facts not in the source.

Always include a top-level media_requests array, even when empty. Every clip must include media_request_ids listing the related request IDs, or an empty array when none are needed.

Do not add a CTA card, 'watch the full lecture' ending, or marketing-style curiosity-gap title. cta_policy should be 'none'.

Every excerpt must open with a neutral generated black title card that reads like a chapter label, not an ad headline. Use concept_title as the visible card text unless a shorter source-faithful chapter label is clearer.

The opening title card should normally last 4-5 seconds and should be silent. It should identify the topic before source dialogue begins.

Prefer source-contiguous ranges. If multiple source ranges are needed, keep them in source order unless there is a clear reason.

A valid extract needs a natural beginning, enough context, a complete idea/example/argument/reveal, and a clean stopping point. If not, extend/merge it or defer it.

When choosing among technically complete candidates, prefer the moment with the strongest standalone insight, surprise, or subject-specific takeaway. Do not pick a merely tidy principle over a more interesting idea unless the interesting idea needs too much context.

Set content_mode to "excerpt". For each clip, fill excerpt_type, self_contained_score, context_needed, source_order_preserved, and cta_policy.

Keep why_this_ad_sells_the_full_video for schema compatibility, but use it to explain why the excerpt stands alone rather than why it sells the full video.

Assume many viewers will watch muted on a phone. Caption plans should track or lightly chunk the dialogue and keep essential words away from platform UI safe-zone areas.

For the required opening title card, set source_start and source_end to 0, visual_source exactly to 'Generated black title card.', and audio_source to 'No source dialogue; silent generated title card.'

End every excerpt at a resolved thought. The renderer will apply a clear final fade to black; do not add a CTA or outro card.

Do not include transcript-artifact text as dialogue. Put questionable ranges in the manual review queue.

STAGE 1 OUTPUT JSON:
{{stage1_output_json}}

STAGE 2 OUTPUT JSON:
{{stage2_output_json}}
