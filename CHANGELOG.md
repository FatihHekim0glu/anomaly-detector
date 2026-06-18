# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-17

First public release: a pure, typed, import-pure market-anomaly compute library
with a strictly causal walk-forward pipeline and an honest, descriptive headline.

### Added

- Pure, typed src-layout package (`anomaly_detector`, `py.typed`) with **zero
  import-time side effects** (scikit-learn / plotly / typer imported lazily).
- Core helpers reused from the HRP infra: `_constants`, `_typing`,
  `_exceptions`, `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), and `_rng` (seeded PCG64 generator + substream spawning).
- `features/engineer`: causal per-day feature vector (log-return, rolling
  realized vol, return z-score, range/ATR, volume z-score, short
  return-autocorrelation) from `.shift(1)`-safe rolling windows and
  `pct_change(fill_method=None)`.
- `detectors/iforest`: Isolation Forest wrapper (score = `-score_samples`);
  `detectors/autoencoder`: PCA reconstruction-error autoencoder (**no torch**);
  the shared frozen-slots `AnomalyResult` dataclass with `to_dict()`. The scaler,
  both models, and all flag thresholds are fitted on the TRAIN slice only.
- `scan.run_anomaly_scan` / `ScanResult`: the single-call orchestration surface,
  an anchored/expanding walk-forward refit, disjoint OOS folds concatenated into
  one zero-look-ahead score/flag series, plus the JSON-safe descriptive summary.
- `evaluation/agreement`: Jaccard, proxy precision/recall against the
  transparent `|z-return| > 3` causal proxy, and regime alignment with known
  stress windows; `evaluation/overlay`: optional, clearly-labeled-diagnostic
  fade-the-anomaly overlay with a multiplicity-aware Deflated Sharpe
  (`n_trials = #detectors x #contamination x #windows`).
- Vendored reusable infra: `backtest/walk_forward` (anchored walk-forward with
  purge/embargo), `backtest/costs`, `backtest/stats`, `evaluation/dsr`
  (Probabilistic / Deflated Sharpe).
- `data`: synthetic anomaly-injected generator (`generate_injected_series`) plus
  the Polygon EOD loader (`data_providers/polygon`), degrading to synthetic on any
  upstream failure; `plots`: lazy Plotly `{data, layout}` figure builders
  (price-with-markers, score-with-threshold); `cli`: Typer `scan`/`demo`.
- Curated top-level `__init__` re-exporting the public API.
- Partitioned test suite (unit / parity / property / regression / integration)
  with seeded `conftest` fixtures (`clean_series`, `injected_anomalies`,
  `pure_noise`): core-logic coverage about 97 % (gate >= 90 %, network provider
  omitted).
- `docs/DESIGN.md` and ADRs 0001 to 0005 (train-only fit, PCA-AE-not-torch, the
  `.shift(1)` walk-forward chokepoint, the no-ground-truth descriptive framing,
  and the transparent proxy label); `CITATION.cff`; a `no-ai-attribution` CI
  guard rejecting AI co-author trailers.

### Validation

- **parity**: Isolation Forest `score_samples` and PCA `inverse_transform`
  reconstruction MSE reproduced against raw scikit-learn to `atol=1e-10`; the
  autoencoder flag threshold to `1e-12`.
- **property**: future-perturbation invariance, prefix-determinism, z-feature
  scale-invariance, and flag-count monotonicity in `contamination` (Hypothesis).
- **regression**: golden injected-anomaly recovery (no lookahead) and the
  honest-headline guard (Jaccard in `[0.20, 0.65]`, proxy precision <= `0.20`).

### Headline (measured, seeded synthetic series)

- Jaccard about **0.50** (modest), proxy precision about **0.04** (low), proxy
  recall about **0.32**. Flags are **diagnostic, not tradable**: there is no
  ground-truth label, so no alpha is claimed.

[Unreleased]: https://github.com/FatihHekim0glu/anomaly-detector/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/anomaly-detector/releases/tag/v0.1.0
