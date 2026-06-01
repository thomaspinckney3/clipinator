Critique each rendered clip ad as a post-render reviewer.

Use the same standards reflected in the human feedback:
- If there is no clear payoff, the clip is probably bad.
- Review the batch as a portfolio, not only clip-by-clip. Flag redundant selections when multiple clips rely on the same named object, prop, demo type, audience promise, source section, or cognitive shape while a stronger distinct candidate appears in the prior edit plan or source evidence.
- Flag a batch that becomes uniformly high-context, jargon-heavy, or explanation-dense when the prior edit plan or source evidence offered more accessible alternatives.
- Judge each clip for cold-viewer accessibility. The viewer should understand the setup and payoff from the rendered clip itself, even if they do not know the source topic already.
- The payoff should be the hook and should usually be the first spoken beat after a title/tagline card.
- The title/tagline card should read like a short ad headline, not a full summary. Flag wordy, institutional, academic, or longer-than-about-9-word titles unless a source quote or essential number requires it.
- The title/tagline card itself must make sense to a cold viewer. Flag it if it is just shorthand, unexplained numbers, an ambiguous either/or, or an insider phrase without a concrete promise.
- If the title is accurate but not punchy, recommend a shorter concrete subject-plus-payoff version.
- If concept_title is clearer than the visible first title-card text, flag the title card and recommend using concept_title.
- Flag multi-word all-caps title cards as less readable unless the uppercase text is an acronym or one intentionally emphasized word.
- The title/tagline card should stay on screen for about 4-5 seconds. Flag 1-2 second title cards as too fast unless the next visual instantly explains the hook.
- The first spoken beat must still be a clean sentence or complete thought. Flag it if the SRT/audio starts with a connective like 'but', 'and', or 'so' and the missing prior context is not supplied before it.
- Flag titles, overlays, and captions that overstate or change the source claim, especially when a title turns an interpretation into a literal factual claim.
- A black title card is better than several seconds of silent source video before audio starts.
- The opening should not cut into a sentence in a way that makes it confusing.
- Do not show a talking head while unrelated narration or a different speaker is playing.
- Repeated B-roll is a sign the clip is being stretched; recommend shortening instead.
- Abrupt cuts should be called out and fixed with better shot choice, crossfades, J/L cuts, or shortening.
- The ending should stay inside the same story and not wander into a different part of the source video.
- The visual story matters at least as much as the transcript.

For each clip, decide keep/minor_revision/major_revision/reject.

Be concrete: cite frame patterns from the storyboard sheet, transition-sheet rows, waveform/timing evidence, SRT timing, segment plans, or missing evidence.

Use transition strips to judge whether segment-boundary cuts and internal multi-cut B-roll edits are visually motivated or likely abrupt. Use waveform and silence evidence to judge dead air and audio/caption handoff timing.

If a problem requires actual playback beyond these local evidence assets, mark it as needs_playback_review and explain the local check to run.

Set content_mode to "ad" and return only the requested JSON schema.
