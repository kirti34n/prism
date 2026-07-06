---
name: prism-check
description: Challenge a conclusion before committing to it. Generates Pre-Mortem, Alt Hypothesis, Falsification, and Blind Spot challenges, evidence-grounded and calibrated.
argument-hint: <conclusion to challenge>
---

# Prism Check — Challenge a Conclusion

Stress-test a conclusion using 4 research-backed strategies before committing to it.

## When to use

- User wants to challenge or stress-test a conclusion
- User says "check this", "challenge this", "is this right", "prism check"
- User is about to commit to a technical decision based on AI advice
- User has a claim or assumption they want tested

## How to run

**Path 1 — CLI available:** Run `prism json --check "$ARGUMENTS"`, parse JSON, present results.

**Path 2 — No CLI:** Generate all 4 challenges below natively.

### Generate 4 challenges

Apply ALL of these to the conclusion. Ground each in real, verifiable examples (if unsure an example is real, say so), and state each at the confidence the evidence supports.

**Pre-Mortem** — It is 18 months from now. This conclusion was acted on and it FAILED. Write the post-mortem: the specific failure mode, the early warning signs that were rationalized away, the moment to pivot that was missed. Concrete, not abstract.

**Alt Hypothesis** — Name 3 genuinely different explanations or approaches — structurally different mechanisms, not variations. For each: (a) the core insight, (b) one scenario where it beats the conclusion, (c) the test that distinguishes them.

**Falsification** — Design the exact test that would DISPROVE this. Specific metric, threshold, scenario. If no test can disprove it, explain why that unfalsifiability is itself a red flag.

**Blind Spot** — Identify exactly ONE hidden assumption the conclusion depends on — a structural blind spot that, once seen, makes it look naive. Show how the conclusion rests on it and the mechanism that keeps people from seeing it.

### Present

For each challenge: a bold heading + a 4-8 sentence challenge (specific, evidence-grounded, calibrated).

End with: *"Does the original conclusion still hold?"*

## Important

Keep challenges pointed and specific — the point is to stress-test, not to reassure. Effective challenges often feel uncomfortable; that discomfort is the signal they are working, not a reason to soften them. But do not manufacture certainty: a well-calibrated, real objection beats a dramatic invented one.
