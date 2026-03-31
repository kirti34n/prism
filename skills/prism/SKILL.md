---
name: prism
description: Generate divergent perspectives on a question using research-backed cognitive strategies. Reveals how AI shapes your thinking.
argument-hint: <question or topic>
---

# Prism — Divergent Perspectives

Generate structurally different perspectives on a question using 10 research-backed cognitive strategies (Devil's Advocate, Pre-Mortem, Falsification, Blind Spot, etc.). Each perspective is ranked by how far it diverges from the AI default answer.

## When to use

- User asks for different angles on a question or decision
- User wants to challenge their own thinking
- User is evaluating a technical approach and wants to stress-test it
- User says "prism", "perspectives", "different angles", "think about this differently"

## How to run

Run this command with the user's question:

```bash
prism json "$ARGUMENTS"
```

## How to present results

Parse the JSON output. It contains:
- `question` — the question asked
- `default` — the standard AI answer
- `perspectives` — array of divergent perspectives, each with:
  - `strategy` — the cognitive strategy used
  - `name` — human-readable strategy name
  - `text` — the perspective content
  - `divergence` — how different from the default (0-1, higher = more divergent)

**Present like this:**

1. Show the **default answer** briefly (2-3 sentences)
2. For each perspective (usually 3), show:
   - **Strategy name** and divergence score
   - The key insight in 2-3 sentences
3. End with: *"Do any of these shift how you're thinking about this?"*

## If prism command is not found

Tell the user:
```bash
git clone https://github.com/kirti34n/prism.git && cd prism
pipx install .
prism setup claude
```

## Example strategies

| Strategy | What it forces |
|----------|---------------|
| Devil's Advocate | Argue AGAINST the common position |
| Pre-Mortem | "This already failed. Why?" |
| Falsification | What would disprove this? |
| Blind Spot | The one thing everyone overlooks |
| First Principles | List assumptions, negate each |
| Inversion | Answer the opposite question |
| Systems | Only 2nd/3rd order effects |
| Adjacent Field | How would a different discipline frame this? |
