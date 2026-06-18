# Contributing

Thanks for your interest in `anomaly-detector`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with the lean extras + dev tooling.
uv venv
uv pip install -e '.[data,viz,dev]'
```

`uv` creates `.venv/` and installs the dependency set. Prefix commands with
`uv run` to use that env without activating it.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src tests                                        # lint
uv run mypy src                                                    # types (strict)
uv run pytest -q -m "not integration" --cov=anomaly_detector --cov-report=term  # tests + coverage
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) must pass; new code should not add type errors.
- **Tests** (`pytest`) must pass with **core-logic coverage >= 90%** (the gate
  lives in `[tool.coverage.report] fail_under` in `pyproject.toml`).

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## Leakage discipline (read before touching detector code)

The top project risk is FULL-SAMPLE LEAKAGE. The `StandardScaler`, the Isolation
Forest, the PCA autoencoder, **and all thresholds** must be fitted on the TRAIN
slice ONLY, then transform/score the disjoint out-of-sample slice. A day's flag
may use only information available strictly before that day (the `.shift(1)`
chokepoint). The four Hypothesis property tests
(`tests/property/test_invariants.py`) exist to enforce this; do not weaken them.

The headline is DESCRIPTIVE: there is no ground-truth anomaly label, so no
alpha/tradability is claimed. Keep the summary honest.

## Commit hygiene

- Use clear, present-tense commit messages.
- Do not add co-author or generated-with trailers to commits or pull requests.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
