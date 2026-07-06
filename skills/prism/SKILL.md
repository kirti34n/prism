---
name: prism
description: Generate divergent, evidence-grounded perspectives on a question using research-backed cognitive strategies. Reveals how the AI-default answer shapes your thinking.
argument-hint: <question or topic>
---

# Prism — Divergent Perspectives

Generate structurally different perspectives that refute the obvious answer, using research-backed cognitive strategies.

## When to use

- User asks for perspectives, different angles, challenges to their thinking, or says "prism"
- User is making a decision or evaluating approaches
- User wants to stress-test a technical approach

## How to run

**Path 1 — CLI available:** Run `prism json "$ARGUMENTS"`, parse the JSON output (it includes divergence scores). Present using the format in Step 4.

**Path 2 — No CLI:** Generate natively using the steps below.

### Step 1: Default answer

Give the most practical, specific answer to the question — 3-4 sentences, concrete technologies/approaches, not vague principles. This is the answer the perspectives will push against.

### Step 2: Generate 3 perspectives

Pick 3 strategies from the table. Each one must **identify a key claim in the default answer and refute it specifically**. Ground each perspective in real, verifiable examples — if you are not sure an example is real, say so rather than inventing one. State each point at the confidence the evidence supports; calibrated is more persuasive than table-pounding.

| Strategy | Constraint |
|----------|-----------|
| Devil's Advocate | Refute the default's strongest claim. The mechanism by which it fails, not just that it can. |
| Pre-Mortem | 18 months out, the default answer failed. Write the post-mortem: specific failure mode, ignored warning signs. |
| Falsification | Design the exact test that would disprove the default. Metric, threshold, timeframe. |
| Blind Spot | ONE hidden assumption the default depends on. Show how it depends on it; name the mechanism that hides it. |
| Alt Hypothesis | 3 structurally different approaches. Core insight, where each beats the default, the distinguishing test. |
| First Principles | 2-3 "everyone knows" assumptions behind the default. Where each breaks. Rebuild without them. |
| Inversion | Answer the opposite question in detail. What was the default hiding that the inversion reveals? |
| Systems | Only 2nd/3rd-order effects of the default. Follow causal chains 3 steps. Name feedback loops. |
| Stakeholder | Who the default harms or locks out. Their perspective, their friction, made concrete. |
| Adjacent Field | A field that solved an analogous problem. Map its technique onto this; one idea the default would never generate. |

### Step 3: Rank by divergence

Order perspectives by how different each is from the default. Most divergent first.

### Step 4: Present

**Default Answer**
[3-4 sentences]

---

**Divergent Perspectives**

**1. [Strategy Name]** — [full perspective, 4-8 sentences, refuting the default]

**2. [Strategy Name]** — [full perspective]

**3. [Strategy Name]** — [full perspective]

---

*Do any of these shift how you're thinking about this?*

## Important

Do NOT paraphrase or compress perspectives — the value is in the specifics, examples, and concrete mechanisms. Keep each perspective evidence-grounded and calibrated (real examples, honest confidence), and make sure each one actually refutes the default answer rather than drifting into a neutral survey.
