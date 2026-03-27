# Contributing to PanODE-DPMM

Thanks for helping improve the repository.

## Before you start

- Check existing issues and pull requests before starting duplicate work.
- For substantial changes, open an issue or draft pull request first so the approach can be discussed early.
- Keep generated outputs, checkpoints, benchmark artefacts, manuscript source files, and local symlinks out of commits.

## Local setup

1. Create and activate a Python environment.
2. Install the project in editable mode:
   - core: `pip install -e ".[dev]"`
   - single-cell extras: `pip install -e ".[dev,bio]"`
   - graph encoder extras: `pip install -e ".[dev,graph]"`
3. Enable the local checks: `pre-commit install`

Optional local configuration can go in `.env` by copying `.env.example`.

## Development expectations

- Prefer small, reviewable pull requests over large mixed changes.
- Add or update tests when changing behavior.
- Run `pytest` before opening a pull request.
- Update `README.md`, `CHANGELOG.md`, or module-level docs when behavior, setup, or outputs change.
- Preserve the source-first layout: code and documented placeholders belong in git; generated data products do not.

## Pull request checklist

Before requesting review, make sure you have:

- described the problem and the solution clearly,
- linked the relevant issue, if one exists,
- run the relevant tests locally,
- avoided unrelated refactors, and
- updated docs or metadata when needed.

## Review and merge

- Use draft pull requests when work is still in progress.
- Prefer squash merges for small feature branches unless commit history matters for reproducibility.
- Address review comments with follow-up commits instead of force-pushing away useful context unless the branch needs cleanup before merge.

## Contribution license

By submitting a contribution, you agree that your work may be distributed under the repository license in `LICENSE`.
