# Changelog

All notable changes to Prism are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [3.0.0] - 2026-07-06

A research-backed honesty pass. The premise - that AI shifts your thinking and
that shift is worth tracking - held up under review. The *measurement* did not,
and this release rebuilds it around what the psychology/HCI literature actually
uses.

### Changed (breaking)
- **State schema v2 → v3.** Existing sessions are migrated automatically and kept
  (tagged `v2-legacy`); they show in `history` and `revisit` but are excluded from
  v3 metrics (different measurement scale). No data is deleted.
- **Measurement is now self-report, not text distance.** Conviction is captured on
  a 0-100 scale (was confidence 1-10). Change is classified from your own
  before/after category (same / shifted / switched / different question) plus the
  conviction delta - never from cosine distance on your text. Cosine distance
  (`wording_change`) is still recorded, but as an explicitly unvalidated auxiliary
  signal, and is no longer displayed. See `LIMITATIONS.md`.
- **Session types** are now `reframing / destabilization / adoption / switch /
  shift / unshaken / unmeasured` (self-report driven).
- **Strategy prompts rewritten**: each perspective now explicitly refutes the
  AI-default answer, cites only verifiable examples (accuracy guard), and speaks at
  calibrated confidence instead of "commit fully, no hedging".
- **AI-tool integration** no longer injects files into tools' internal config. Use
  the committed Claude Code plugin marketplace (`/plugin marketplace add
  kirti34n/prism`) and the `skills/` SKILL.md folder (the open Agent Skills
  standard). `prism setup install` now prints the correct `pipx`/`uv` command
  instead of creating a POSIX symlink - fixing install on Windows.

### Added
- **Interactive rebuttal round**: after seeing the perspectives you can push back on
  one, and that strategy responds once. (Interactive dissent is what the evidence
  shows improves decisions - a static wall of counterarguments does not.)
- **`prism revisit`**: resurfaces a past session and asks whether your revised
  position turned out right - a lightweight decision-journal / calibration loop.
- Cross-platform CI (Linux/macOS/Windows × Python 3.9/3.12).

### Removed
- The bandit strategy-weight system and all cosine-threshold classification
  (`_update_weights`, convergence-slope regression, independence score): they
  adapted to a measurement signal that isn't valid. Strategy selection is now simple
  random (with explicit config override and research-mode substitution preserved).
- The optional `sentence-transformers` embedder path (measurement no longer uses
  text similarity; ranking needs only bag-of-words).
- The per-tool config injectors (Cursor/Copilot/Gemini/etc.) - superseded by the
  single `skills/` folder.

### Fixed
- Ollama thinking models (qwen3+, deepseek-r1, qwq, magistral) now return usable
  answers. Prism sends `think: false` so the whole token budget goes to the answer
  instead of reasoning that gets cut off and stripped to nothing. Verified
  end-to-end against a local `qwen3.5:4b`.
