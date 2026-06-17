# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-17

### Added

- Initial package skeleton (src-layout, import name `anomaly_detector`).
- Core helpers reused from the HRP infra: `_constants`, `_typing`,
  `_exceptions`, `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), and `_rng` (seeded PCG64 generator + substream spawning).
- Vendored reusable infra: `backtest/walk_forward` (anchored walk-forward with
  purge/embargo), `backtest/costs`, `backtest/stats`, `evaluation/dsr`
  (Probabilistic / Deflated Sharpe), and `data_providers/polygon` (Polygon EOD).
- Typed stubs with full contracts for the NEW modules: `features/engineer`
  (causal `.shift(1)`-safe per-day features), `detectors/iforest` (Isolation
  Forest) + `detectors/autoencoder` (PCA reconstruction-error autoencoder, no
  torch) + the shared frozen `AnomalyResult` dataclass, `evaluation/agreement`
  (Jaccard / proxy precision-recall / regime alignment) + `evaluation/overlay`
  (toy fade-the-anomaly overlay), `data` (synthetic anomaly-injected generator +
  Polygon loader), `plots`, and `cli`.
- Curated top-level `__init__` re-exporting the public API.
- Partitioned test suite (unit / parity / property / regression / integration)
  with seeded `conftest` fixtures (`clean_series`, `injected_anomalies`,
  `pure_noise`).

[Unreleased]: https://github.com/FatihHekim0glu/anomaly-detector/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/anomaly-detector/releases/tag/v0.1.0
