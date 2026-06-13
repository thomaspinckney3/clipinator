Create stage 1 of a video-to-excerpt pipeline.

Goal: understand what the full educational video is about and identify
self-contained parts that can become extracts. An extract is not an ad for the
full video. It should be satisfying on its own, while preserving the speaker's
actual thought and natural context.

Creative priors for educational short-form excerpts:
- Prefer complete ideas, examples, demonstrations, stories, arguments, or
  explanations that have a natural beginning and ending.
- Choose the semantic unit before choosing a duration. Sometimes the unit is a
  single moment; sometimes it is a longer chapter-style arc whose setup,
  complications, and payoff only make sense together.
- A strong excerpt can be a single interesting idea, surprising detail, reveal,
  or novel insight about the subject. It does not have to be a single abstract
  principle if a more specific moment is more watchable.
- Do not split a cumulative argument into separate excerpt candidates unless
  each part is independently coherent and interesting. When the punchline
  depends on a chain of setup beats, make one broader chapter-extract candidate.
- Favor source-contiguous or source-order-preserving moments. Reordering should
  be rare and only used when it preserves clarity.
- Optimize for self-contained understanding, not curiosity gaps or persuasion.
- A good extract may be quieter than an ad if the idea is coherent and useful.
- Prefer windows where the spoken idea can stand with light visual reframing,
  captions, dead-air trimming, and minimal B-roll.
- Do not require a CTA or "watch the full video" ending.
- Flag timing regions where narration, speaker identity, or visuals may not line
  up cleanly.

Use the existing ad_angles array as excerpt candidate groups for compatibility:
angle = extract topic, viewer_hook = why this self-contained extract is worth
watching, why_it_sells_full_video = why the extract is self-contained and
valuable on its own, and recommended_clip_form = atomic_excerpt,
chapter_extract, or review_only. Set content_mode to "excerpt".

Output JSON only. Rank excerpt candidates by expected usefulness as standalone
extracts, not by chronological order. Evidence ranges should cover the full
coherence unit: broad enough for the next stage to inspect the complete arc, but
narrow enough to be actionable.
