Create stage 1 of a video-to-clip-ad pipeline.

Goal: understand what the full source video is about, why a general audience
should watch it, and which short-form ad angles are worth testing before any
precise edit plan is generated.

Creative priors for educational short-form clips:
- Prefer concrete, visual, surprising claims over broad summaries.
- Favor a fast hook with conflict, scale, counterintuitive detail, stakes, or a
  myth/reframe.
- A good clip ad needs a clear payoff or surprising line. If an angle has no
  likely payoff, treat that as evidence it is a weak clip candidate.
- The payoff can be the hook: a short clip may open with the strongest payoff
  line, then backfill context, while still selling the larger full-video story.
- Optimize for cold-viewer comprehension as well as surprise. A very dense,
  expert-only, jargon-heavy, or multi-step idea is weaker unless the clip can
  make the setup and payoff legible within the short format.
- Build a balanced angle set. Avoid making every top angle the same cognitive
  shape, such as all technical explanation, all process minutiae, all abstract
  claims, all emotional backstory, or all visual spectacle. Include accessible
  on-ramps when the source offers them.
- Prefer angles that the video can prove visually. Use transcript-only angles only
  when the spoken idea is unusually strong and the likely B-roll can carry it.
- If the hook names a concrete object or action, the candidate needs visible
  proof of that object or action. A clip about a specific tool should visibly
  show that tool or unambiguous use of it; generic related footage is not enough.
- Flag timing regions where narration, speaker identity, or visuals may not line
  up cleanly.

Output JSON only. Rank ad angles by expected usefulness for downstream clip
assembly, not by chronological order. Evidence ranges should be broad enough for
the next stage to inspect but narrow enough to be actionable. Set content_mode to
"ad".
