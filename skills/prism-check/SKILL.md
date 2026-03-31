---
name: prism-check
description: Challenge a conclusion before committing to it. Generates Pre-Mortem, Alt Hypothesis, Falsification, and Blind Spot challenges.
argument-hint: <conclusion to challenge>
---

# Prism Check — Challenge a Conclusion

Generates 4 targeted challenges against a conclusion using research-backed strategies: Pre-Mortem, Alternative Hypothesis, Falsification, and Blind Spot. Use this after AI-assisted research, before committing to an approach.

## When to use

- User wants to challenge or stress-test a conclusion
- User says "check this", "challenge this", "is this right", "prism check"
- User is about to commit to a technical decision based on AI advice
- User has a claim or assumption they want tested

## How to run

Run this command with the conclusion to challenge:

```bash
prism json --check "$ARGUMENTS"
```

## How to present results

Parse the JSON output. It contains:
- `conclusion` — what's being challenged
- `challenges` — array of 4 challenges, each with:
  - `strategy` — strategy key
  - `name` — human-readable name
  - `text` — the challenge content

**Present like this:**

For each challenge:
1. **Strategy name** in bold
2. The core challenge in 1-2 sentences

End with: *"Does the original conclusion still hold?"*

## If prism command is not found

Tell the user:
```bash
git clone https://github.com/kirti34n/prism.git && cd prism
pipx install .
prism setup claude
```

## The 4 challenge strategies

| Strategy | What it forces | Evidence |
|----------|---------------|---------|
| **Pre-Mortem** | "This failed. The failure was predictable. Why?" | Klein 2007 — 30% more failure reasons |
| **Alt Hypothesis** | 3 structurally different explanations | Hirt & Markman 1995 — debiasing |
| **Falsification** | What specific result would disprove this? | Tetlock 2015 — superforecaster habit |
| **Blind Spot** | The one thing everyone misses | — |
