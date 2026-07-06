# Prism Project Instructions

## Architecture
- Single file: `prism.py`. Everything lives here. Do not split into modules.
- Zero external dependencies. stdlib only. This is a hard rule.
- ~1800 lines. Use Grep to find the section you need.

## Key Concepts
- Prism generates divergent perspectives on questions using 6 LLM providers (Ollama, OpenAI, Anthropic, Gemini, OpenRouter, Custom)
- Measures cognitive shift via self-report: the user's before/after position plus a 0-100 conviction rating, classified into categories (reframing / destabilization / adoption / switch / shift / unshaken / unmeasured) derived from the user's own stated change category and conviction delta. No cosine-on-word-vectors, no embedder.
- Has three modes: `explore` (full flow), `check` (challenge a conclusion), `research` (deep analysis)
- `explore` includes an interactive rebuttal round: the user can push back on one perspective and that strategy replies once
- `prism revisit` resurfaces a past session and asks whether the revised position turned out right — a decision-journal / calibration loop
- Claude Code skills live as repo files under `skills/` (SKILL.md folders), not as Python strings in prism.py

## Code Patterns
- Provider logic uses `_PROVIDERS` dict mapping to (build_fn, parse_fn) pairs
- Config cascades: project `.prism.json` → global `~/.config/prism/config.json` → auto-detect
- All LLM calls go through `_llm_call()` with provider-specific adapters
- Strategy selection is plain `random.sample`, with an explicit config-list override — see `_select_strategies(config)`. No weights, no bandit.
- State stored as JSON in `~/.config/prism/state.json` (rewritten atomically via tmp+rename), schema v3. `_load_state` migrates v2 state via `_migrate_v2` (tags old sessions `v2-legacy`, keeps them, drops weights)
- `_bow_distance` (bag-of-words cosine) still exists, but only orders perspectives for display and feeds an unvalidated `wording_change` field that is stored but never displayed
- AI-tool integration uses committed static files: a Claude Code plugin marketplace (`.claude-plugin/marketplace.json`, `.claude-plugin/plugin.json`) and SKILL.md folders under `skills/` (the open Agent Skills standard). Keep these in sync with README. The CLI no longer injects files into other tools' internal config — `prism setup` just prints install/marketplace instructions

## Testing
- Tests in `test_prism.py`. Run: `python -m unittest test_prism` (pytest is NOT installed in this environment)
- Test without API keys by mocking `_llm_call`
- If adding a new provider: mock `_llm_call` in tests, don't require real API keys
- If changing CLI behavior: test both `prism "question"` and `prism check "conclusion"` paths
- If changing config: test cascade order (project → global → default)

## When Modifying
- Keep the single-file constraint. If it feels too big, refactor within the file.
- Match existing code style: compact, minimal comments, no docstrings on private funcs.
- Don't add type hints to existing functions unless you're already modifying them.
- `prism setup` only prints instructions now; the committed `skills/` and `.claude-plugin/` files are the integration surface — keep them in sync with README.
- Provider functions have similar signatures. Read the target provider AND one working provider before modifying.
- Keep functions under 40 lines. If longer, extract a helper within the same section.
