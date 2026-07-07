# TITAN User Guide — from zero to reading your first signal

This is the hands-on tutorial. The theory lives in `RESEARCH.md`, the
safety rules in `VALIDATION.md`, and the internals in `ARCHITECTURE.md`.

---

## 1. Install (one time)

```bash
git clone <repo> && cd trading-bot
pip install -e ".[server,dev]"
titan --version        # sanity check the CLI landed
```

Python 3.11+. No GPU, no database, no API keys needed for the offline demo.

## 2. Your first run (~4 minutes)

```bash
titan validate --config configs/control-strong-signal.yaml
titan dashboard --port 8321
# open http://127.0.0.1:8321
```

`validate` is the whole research factory in one command. It will:

1. generate (or fetch) the market data and quality-score every instrument;
2. build ~70 causal features per instrument;
3. label every bar with the triple-barrier outcome;
4. run the purged walk-forward: per fold — select features, train and
   calibrate the ensemble, detect regimes, generate gated signals;
5. simulate the whole out-of-sample period through the risk engine;
6. run bootstrap / deflated-Sharpe / risk-of-ruin robustness statistics;
7. write everything to `artifacts/` and register the model bundle
   (first bundle auto-promotes to production; later ones must beat it);
8. finish with a scan of the latest bar.

The JSON block it prints at the end is your executive summary.

### The two demo configs are a controlled experiment

| Config | Planted signal | Correct outcome |
|---|---|---|
| `configs/control-strong-signal.yaml` | strong | detection: high AUC, many trades, positive Sharpe CI |
| `configs/default.yaml` | realistically weak | **rejection**: some classifier skill, few trades, CI straddles zero |

Run both. Seeing the platform *refuse* the weak market teaches you more
about it than the pretty equity curve does. (The third control is in the
test suite: on shuffled labels the model scores AUC ≈ 0.5.)

## 3. Reading the dashboard

Top to bottom:

- **KPI row** — all out-of-sample, net of costs. `Hit rate @ gate` vs
  `Base rate` is the most important pair: the whole premise is that the
  gated tail beats the base rate.
- **Equity + drawdown + regime strip** — hover for values; the strip shows
  what the detector believed each day (blue bullish / gray neutral / red
  bearish / dark-red crash).
- **Robustness panel** — read the Sharpe CI before the Sharpe. Then DSR at
  the trial count matching how many configs you tried. Then risk of ruin.
- **Signal feed** — the actionable output (see §4).
- **Scanner ranking** — every instrument, including *why* the rejected ones
  were rejected. A scanner that only shows winners teaches you nothing.
- **Fold diagnostics** — each fold trained only on its past. One heroic
  fold among dead ones = regime luck, not edge.
- **Feature importance** — permutation, out-of-sample. ≤ 0 means unused.
- **Regime breakdown** — consistency across regimes beats any headline
  number.

## 4. Anatomy of a signal

Every signal card is a complete decision package:

```
DDD  [A]  strong_bull                conf 76 · P 66.4%
Entry zone   11.42–11.52      Limit / market  11.42 / 11.49
Stop / ATR   11.24 / 10.95    Targets         11.66 · 11.82 · 11.99
Size / risk  10.3% / 0.22%    R:R / EV        1.33 / +0.86%
Est. hold    2 bars           MAE / MFE est.  -2.1% / +3.1%
Similarity   38% (50 analogues)   Institutional 94/100
+ rsi_14 at 94th percentile (+14.6% to probability)
! 25th-percentile analogue outcome is beyond the stop
```

How to read it:

- **P** is a *calibrated* probability that price hits +2σ before −1.5σ
  within the horizon, entering at the next open. It already cleared an
  adaptive threshold derived from costs — not a hand-tuned constant.
- **EV** is expected value *after* commission, spread and impact. If it
  weren't positive, the signal would not exist.
- **Size** is the minimum of ¼-Kelly, vol-targeting and fixed-fractional
  stop risk; **risk** is what you lose if the stop is hit.
- **MAE/MFE estimates** come from the 50 nearest historical states — how
  bad the ride typically got before resolution.
- **Similarity** is an out-of-distribution alarm: low similarity means the
  market state has little precedent; trust the probability less.
- The **+ / !** lines are evidence for and against. A card with no
  conflicting evidence listed means none was material — not that none was
  looked for.

## 5. Daily operation

```bash
titan scan                    # rank the universe with the production model
titan info                    # registry status, production version, universe
titan dashboard --port 8321   # always reads the latest artifacts
```

`scan` is pure inference (seconds). Retraining happens only through
`validate`, and a retrained model reaches production **only** by beating
the incumbent through the statistical promotion gate — a challenger with a
non-significant edge or a degraded drawdown stays shelved, with reasons
recorded in `models_store/index.json`.

## 6. Connecting real data

1. Copy `configs/live-example.yaml`, edit the universe/benchmark.
   Provider `yahoo` needs network egress; provider `csv` reads
   `{symbol}.csv` files (`data.csv_dir`) with date-indexed OHLCV columns.
2. `titan validate --config configs/my-live.yaml`
3. Check `artifacts/quality.json` — drop anything below ~0.9 reliability.
4. Follow `VALIDATION.md` §1 (reading order and rejection gates) and §3
   (cost / barrier / jackknife / seed sensitivity sweeps).
5. Paper-track `titan scan` on a schedule for a meaningful period before
   any capital decision. Synthetic results certify the machinery only.

## 7. Tuning the knobs that matter

All in your YAML config (validated by pydantic — typos fail loudly):

| Knob | Effect |
|---|---|
| `labels.tp_sigma` / `sl_sigma` / `horizon_bars` | barrier geometry: what "win" means. Changes the gate automatically |
| `signals.min_probability`, `ev_margin_bps` | how picky the gate is (fewer, better trades) |
| `risk.target_annual_vol`, `risk_per_trade_pct`, `kelly_fraction` | aggressiveness; the minimum-of-three sizing keeps any one mistake bounded |
| `risk.regime_multipliers` | risk appetite per regime (crash is 0 — think hard before changing) |
| `model.tuning_iterations`, `internal_folds`, `members` | compute vs. thoroughness |
| `cv.n_folds`, `test_bars`, `min_train_bars` | how much out-of-sample history judges the model |
| `backtest.costs.*` | be honest; sweep 2×–4× per VALIDATION §3 |

CLI overrides for quick experiments:
`titan validate --folds 6 --tuning 16 --seed 13 --out artifacts_exp1`.

## 8. Monitoring a deployed model

- **Feature drift**: `titan.monitor.feature_drift_report(X_train_ref,
  X_live)` — PSI ≥ 0.25 on a model feature means the world changed;
  retrain.
- **Calibration drift**: feed realized outcomes to a `PredictionTracker`
  (`log_prediction` at scan time, `resolve` when the event closes). The
  CUSUM alarm fires on slow rot long before the PnL makes it obvious.
- On alarm: run `validate` to produce a challenger; let the promotion gate
  decide. Never hand-promote.

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `no production model: run titan validate first` | scan before any validate |
| Zero signals on a scan | usually the gate doing its job — check the scanner table for per-symbol reasons (crash regime blocks everything) |
| `training window too small` | not enough history for `cv.min_train_bars` / internal folds — more bars or fewer folds |
| Instrument missing from results | failed QC; see `artifacts/quality.json` and the log line explaining why |
| Yahoo fetch fails | no network egress from your environment; use `csv` |
| Slow validate | lower `tuning_iterations` / `internal_folds`, or trim `members` |

## 10. Development loop

```bash
pytest -q -m "not slow"     # fast unit tests (~30 s)
pytest -q                   # full suite incl. end-to-end (108 tests)
ruff check src tests scripts
python -m mypy src/titan
python scripts/screenshot_dashboard.py   # visual check of the dashboard
```

Adding a feature/provider/model member: see the extension points in
`ARCHITECTURE.md`. Anything you add inherits the causality test, the
purge invariants and the no-skill-on-noise control automatically — the
platform is built so the safe path is the easy path.
