# Releasing

Prism publishes to PyPI automatically when you push a version tag. This file is the
checklist so you never have to remember the details.

## One-time setup (about five minutes, do this once)

`prism-think` 3.0.0 was published once by hand to bootstrap the project. To let the
workflow publish every future version without a stored token, do these two steps once.

**Step 1: add a PyPI Trusted Publisher.**

1. Sign in at [pypi.org](https://pypi.org).
2. Open the project's publishing settings:
   `https://pypi.org/manage/project/prism-think/settings/publishing/`.
3. Under "Add a new publisher", fill in exactly:
   - **Owner:** `kirti34n`
   - **Repository name:** `prism`
   - **Workflow name:** `publish.yml`
   - **Environment:** leave blank
4. Save. No token gets stored anywhere; the workflow proves its identity over OIDC.

**Step 2: arm the workflow.**

The publish job is gated on a repository variable so it can never fire before the
publisher above exists. Turn it on once:

```bash
gh variable set PYPI_READY --repo kirti34n/prism --body true
```

That is it. From now on, pushing a version tag publishes automatically.

## Cutting a release

The version lives in one source of truth: `__version__` in `prism.py`
(`pyproject.toml` reads it automatically). Keep three things in sync:

1. Bump `__version__` in `prism.py` (for example `3.0.1` or `3.1.0`).
2. Bump the matching `"version"` in `.claude-plugin/plugin.json`.
3. Add a dated section to `CHANGELOG.md` describing what changed.

Then commit, tag, and push the tag:

```bash
git add prism.py .claude-plugin/plugin.json CHANGELOG.md
git commit -m "Release v3.1.0"
git tag v3.1.0        # the tag must start with 'v' to trigger the workflow
git push origin master
git push origin v3.1.0
```

Pushing the `v3.1.0` tag triggers `publish.yml`, which builds the wheel and sdist and
uploads them to PyPI. Watch it run:

```bash
gh run watch --repo kirti34n/prism
```

## Versioning

Prism follows [Semantic Versioning](https://semver.org/):

- **Patch** (`3.0.1`): bug fixes, no behavior change for users.
- **Minor** (`3.1.0`): new commands or options, backward compatible.
- **Major** (`4.0.0`): a breaking change, such as a new state schema. When the state
  schema changes, bump `VERSION` in `prism.py` and add a migration in `_load_state`.

## Verifying a release

```bash
pip install --upgrade prism-think
prism --version          # should print the version you tagged
```

If a tag push fails to publish, the usual cause is the Trusted Publisher values not
matching the one-time setup above. Check the failed run in the Actions tab for the exact
error, fix the mismatch on PyPI, then re-run the job (no need to re-tag).
