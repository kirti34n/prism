# What Prism measures (and what it doesn't)

Prism is honest about being a **decision journal**, not an instrument for measuring
your mind. This document says exactly where the line is, because the tool's whole
point is defeated if it makes you trust a number it hasn't earned.

## What it measures (defensible)

- **Your self-reported position and conviction**, before and after seeing the
  perspectives. Conviction is a 0-100 self-report. Pre/post self-report is the
  standard the persuasion and debiasing literature actually uses, and confidence
  change is a validated, sensitive outcome.
- **Your own categorization** of what changed (same / shifted / switched sides /
  different question). You are the authority on your own stance; Prism records your
  call, it does not infer it.
- **Which perspective you say moved you.** Self-reported, stored verbatim.

From these it derives coarse categories (`shift`, `switch`, `destabilization`,
`adoption`, …). Coarse buckets are deliberate: fine-grained numbers here would imply
a precision the inputs don't support.

## What it does NOT measure

- **It does not read your mind from your text.** Prism records a cosine
  "`wording_change`" between your before/after text, but this is an **unvalidated
  auxiliary signal, kept for later research and never shown as a result.** Text
  similarity is not a valid measure of opinion change: paraphrasing the same view
  scores as a big change, and, worse, "I support X" and "I oppose X" can score as
  nearly identical, so a full reversal could look like no change. This is a known
  failure mode ("negation blindness"), and it's why Prism's classification is driven
  by your self-report, not by distance.
- **The conviction thresholds are not calibrated to you.** A 20-point drop is
  flagged as "destabilization" because that's roughly twice the noise of a 0-100
  self-report and larger than typical persuasion effects. A reasonable default, not
  a personalized measurement.
- **Adoption/convergence numbers are directional signals, not diagnoses.** "Recent
  adoption 60%" means *you told Prism* a model answer moved you in 60% of recent
  logged sessions. It's a prompt to bring outside sources, not a verdict on your
  independence.

## One thing to expect: effective dissent feels bad

In the research, the AI dissent that *most* improved people's decisions was the one
they *rated worst*, with lower perceived performance and less pleasant to use. So Prism is
deliberately not tuned for your approval. If a perspective feels comfortable and
agreeable, that's weak evidence it did anything; the uncomfortable one is often the
one working. Don't judge a session by how good it felt.

## And a caution the tool can't escape

Leaning on any AI, including this one, to do your thinking can, over time, weaken
the independent skill it's meant to protect. Prism tries to counter this by making
you write *your* position first and by offering `prism revisit` so you check your
past calls against reality. But the tool is a prosthetic for reflection, not a
replacement for it. Use it to argue with, not to outsource to.
