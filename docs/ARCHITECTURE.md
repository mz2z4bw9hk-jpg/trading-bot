# TITAN Architecture

## Principles

- **Dependency injection everywhere.** Every component receives its config
  object and collaborators in its constructor; nothing reads globals. Any
  piece can be unit-tested or swapped in isolation.
- **One code path.** The scanner, the backtester and the walk-forward all
  consume the same `SignalGenerator`, `RiskEngine` and `CostModel`
  instances. There is no "sim version" to drift away from "prod".
- **Statistics upstream, mechanics downstream.** The backtest engine sees
  only `TradePlan`s; it cannot peek at features or probabilities. Anything
  statistical happened strictly earlier, under the causality contract.
- **Artifacts as the interface.** The dashboard renders only files written
  by validated runs. If it is on screen, it is auditable on disk.

## Data flow

```
DataConfig ─► build_provider ─► SyntheticProvider | YahooProvider | CSVProvider
                                     │  fetch(symbol, bars)
                              normalize_ohlcv (canonical schema, UTC)
                                     │
                              assess_quality ─► reliability ∈ [0,1]
                                     │  (below floor -> excluded, loudly)
                              MarketDataset {frames, benchmark, reliability}
                                     │
      FeatureMatrixBuilder: per-symbol registry + cross-sectional features
                                     │  FeaturePanel (date, symbol) x ~70
      build_label_panel: triple-barrier labels + uniqueness weights
                                     │
      PurgedWalkForward.split(dates, t1) ─► folds  ──►  assert_no_leakage
                                     │
   per fold: univariate IC ─► redundancy prune ─► top-K ─► CalibratedEnsemble
             RegimeDetector.fit(train bench) ─► frozen transform over test
             AnalogueIndex.fit(train)        ─► LocalExplainer(train)
                                     │
             SignalGenerator.generate(...)   ─► Signal | None   (EV gate)
                                     │
      TradePlans ─► BacktestEngine.run(frames, plans, RiskEngine.approve)
                                     │
      WalkForwardReport {folds, pooled stats, backtest, bootstrap, DSR,
                         regime breakdown, signals, importance}
                                     │
      artifacts.py ─► report.json / equity.csv / signals.json / ...
                                     │
      titan.server (FastAPI) ─► dashboard.html      (read-only)
      ModelRegistry ─► bundle vNNN (candidate) ─► promotion gate ─► production
```

## Module contracts

| Module | Contract |
|---|---|
| `data.providers.DataProvider` | `fetch(symbol, bars) -> raw frame`; normalization/QC are NOT its job |
| `features.registry.FeatureSpec` | `fn(ohlcv) -> Series`, value at `t` uses only bars `<= t`; verified by `verify_causality` |
| `labels.triple_barrier` | label + `t1` event end per decision bar; pessimistic same-bar tie |
| `models.cv.PurgedWalkForward` | forward-chained folds; train events end before embargoed test start |
| `models.ensemble.CalibratedEnsemble` | `fit(X, y, dates, w)`; `predict_proba` calibrated; `uncertainty` = member disagreement |
| `regime.RegimeDetector` | `fit(train bench)` then frozen, causal `transform`; never a model feature |
| `signals.SignalGenerator` | returns `None` unless every gate passes; a `Signal` is complete or absent |
| `risk.RiskEngine` | implements `RiskApprover.approve(plan, snapshot) -> size`; pure function of config + state |
| `backtest.BacktestEngine` | next-bar open execution, cost-adjusted fills, pessimistic intrabar stops |
| `monitor.compare.promotion_gate` | ALL conditions or no promotion; reasons are returned, not logged away |

## Extension points

- **New data source**: implement `DataProvider`, register in
  `build_provider`, ship reliability scoring via `assess_quality` (or a
  source-specific scorer). Alternative-data features enter as ordinary
  `FeatureSpec`s and inherit causality verification for free.
- **New feature family**: return `list[FeatureSpec]` from a builder and
  `register_all` it in `features.pipeline.build_default_registry`. The
  causality test sweeps the whole registry automatically.
- **New ensemble member**: add a builder in `models.ensemble._build_member`
  + a param sampler in `_sample_params`. Members must expose
  `predict_proba`; everything else (weighting, calibration, uncertainty)
  is inherited. This is where deep models would plug in.
- **Bayesian hyperparameter search**: `_sample_params(member, rng)` is the
  sampler seam — replace random draws with a TPE/GP proposer without
  touching the fit/calibration protocol.
- **Short side**: mirror barriers in `triple_barrier_labels`, flip fills in
  the engine (already parameterized by `Side`), and validate borrow costs
  before flipping `allow_short`.

## Operational loop (self-improvement)

1. `titan validate` retrains per fold, writes artifacts, saves the final
   bundle as a **candidate** in the `ModelRegistry`.
2. The candidate is compared to production on overlapping OOS returns via a
   paired block bootstrap (`monitor.compare`). Promotion requires
   significance, higher Sharpe and non-degraded drawdown — otherwise the
   candidate stays shelved with the reasons recorded.
3. In operation, `feature_drift_report` (PSI vs the training reference) and
   `PredictionTracker` (rolling Brier, calibration table, CUSUM alarm)
   decide when a retrain is *forced* rather than scheduled.
4. Every bundle version keeps its manifest (metrics, features, window,
   config fingerprint) — the registry is the audit trail.

## Performance notes

- Feature build: vectorized pandas/numpy; ~1.2 s for 10 symbols × 2,500
  bars × ~70 features on the dev container. Cross-sectional features are
  wide-frame operations, O(symbols) memory.
- Rolling regressions use sliding-window matmuls (no Python loops).
- Walk-forward wall time is dominated by member fits; `tuning_iterations`
  is the budget knob and `model.members` the second one.
- The scanner reuses the trained bundle — a scan is pure inference and
  runs in seconds on the full universe.
