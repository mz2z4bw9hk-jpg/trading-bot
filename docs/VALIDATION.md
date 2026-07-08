# TITAN Validation Protocol

The purpose of this document is to make it *hard to fool yourself*. Follow
it in order; do not skip gates because early numbers look good — early good
numbers are the most common symptom of leakage.

## 0. Machinery gates (run on every change, automated)

```bash
pytest -q            # 121 tests
```

- **Causality:** every feature recomputed on truncated history must be
  bit-identical in the overlap (`tests/test_causality.py`). One changed
  value = look-ahead = red build.
- **Purge/embargo invariants:** no training event's life may overlap any
  test window (`tests/test_cv.py`, plus `assert_no_leakage` re-checked at
  runtime inside every walk-forward).
- **Accounting:** engine PnL is checked against hand-computed fills,
  including slippage, commission, gap-through-stop, entry-bar stop parity
  with the labels, gap-past-level plan invalidation, and the pessimistic
  same-bar rule (`tests/test_engine.py`).
- **No fabricated skill:** on shuffled labels the ensemble must score
  AUC ≈ 0.5 (`tests/test_ensemble.py`). A pipeline that finds signal in
  noise is broken in the dangerous direction.

## 1. Research validation (per universe / config)

Run `titan validate --config <cfg>`, then read `artifacts/report.json` in
this order:

1. **Pooled OOS AUC and Brier** — classification skill across all folds.
   Reject if pooled AUC ≤ 0.52 on a serious sample, or if Brier is worse
   than the base-rate Brier (the model is anti-calibrated).
2. **Gated hit rate vs base rate** (`high_conf_hit_rate` vs `base_rate`) —
   the whole strategy premise is that the gated tail beats the base rate.
   Reject if the uplift is ≲ 3 pp.
3. **Fold dispersion** — one heroic fold and four dead ones is regime
   luck, not edge. Prefer monotone-ish, positive-in-most-folds profiles.
4. **Backtest summary** — Sharpe, Sortino, Calmar, max drawdown, profit
   factor, expectancy, exposure, turnover. Sanity-check trade count: n<50
   trades cannot support any statistical claim.
5. **Bootstrap CIs** — reject if the 5th-percentile Sharpe ≤ 0 or
   `p_sharpe_positive` < 0.90. A point-estimate Sharpe without its CI is
   noise cosplay.
6. **Deflated Sharpe** — read DSR at the trial count that honestly matches
   your research process (if you tried ~10 configs before this one, read
   `n_trials=10`). Reject below 0.95 for anything intended for production.
7. **Regime breakdown** — reject strategies whose entire PnL lives in one
   regime unless the regime gate provably keeps them out of the others.
8. **Risk of ruin** — reject if P(−30% within 3y) is not comfortably small
   (< 5% at the intended sizing).

## 2. Overfitting controls (already in the machinery)

- Feature selection, hyperparameters, member weights, calibration and
  thresholds are all fitted **inside each fold's training window only** —
  and member weights/calibration specifically on purged out-of-fold
  predictions within that window, never on anything a member trained on.
- The signal threshold is *derived* from costs + barrier geometry, not
  optimized on outcomes.
- Redundancy pruning caps effective dimensionality; permutation importance
  on held-out data exposes features the model doesn't actually use.
- DSR deflates for multiple testing; the bootstrap exposes path luck.
- The champion/challenger gate (`monitor.compare.promotion_gate`) requires
  a paired-bootstrap p-value below `monitor.promotion_p_value`, higher
  Sharpe, and non-degraded drawdown before any model reaches production.

## 3. Sensitivity analysis (manual, before production)

Sweep and re-run `titan validate`, expecting graceful degradation — cliffs
mean fragility:

- **Costs**: 2× and 4× `commission_bps` + `spread_bps`. A strategy that
  dies at 2× costs is a cost-model bet, not an alpha.
- **Barriers**: `tp_sigma`/`sl_sigma` ± 25%; `horizon_bars` ± 5.
- **Universe**: drop each instrument (jackknife); results should not hinge
  on one name.
- **Seed**: on real data a different `run.seed` moves nothing but model
  tie-breaks — if conclusions flip, there are no conclusions. On the
  synthetic provider a new seed generates a NEW market, so the check is
  stronger: the *conclusion* (planted signal detected / weak market
  rejected) must reproduce even though the numbers move.
- **Regime multipliers**: setting all to 1.0 should *hurt* (that's the
  gate earning its keep) but not zero out returns.

## 4. Real-data checklist (before trusting anything live)

1. Point `data.provider` at real data (`yahoo`/`csv`); confirm reliability
   scores ≥ 0.9 for every instrument you keep, and read `quality.json`.
2. At least 8–10 years of daily history per instrument (regime detector
   and CV minimums are hard requirements, not suggestions).
3. Re-run §1 and §3 in full. Synthetic results transfer *zero* evidence to
   real markets — they only certify the machinery.
4. Paper-track the scanner (`titan scan` on a schedule) for a meaningful
   period; feed outcomes to `PredictionTracker.resolve` and watch the
   calibration table and CUSUM alarm.
5. Only then consider capital, sized by the risk engine, with the drawdown
   throttle live and the crash-regime zero unchallenged.

## 5. Ongoing monitoring (production)

- **Feature drift**: PSI vs training reference; warn ≥ 0.10, alert ≥ 0.25
  (`monitor.drift.feature_drift_report`).
- **Calibration drift**: rolling Brier + calibration table; the CUSUM alarm
  is the tripwire for slow rot.
- **Retrain policy**: scheduled retrains produce *candidates*; only the
  promotion gate moves one to production. Manual promotion of an unproven
  model defeats the entire platform — the registry keeps the audit trail
  of who promoted what, when, on which evidence.

## Reading the demo (synthetic) results honestly

The shipped demo runs on a synthetic market with planted structure. The
correct reading is: *"the pipeline recovers structure that is provably
there, claims none where there is none, and every downstream number is
internally consistent."* It is a control experiment, not a track record.
