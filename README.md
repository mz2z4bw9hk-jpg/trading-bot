# PROJECT TITAN

**An institutional-grade quantitative research platform** for discovering,
validating and monitoring statistically defensible trading signals across
equities and crypto — built around one non-negotiable principle: *no number
reaches a human unless it survived out-of-sample, cost-aware, leakage-proof
validation.*

TITAN is a research and signal-intelligence platform, **not** an order-routing
bot. It produces graded, explainable, risk-sized decision packages and the
evidence trail behind them.

---

## Honest status — read this first

| Claim | Status |
|---|---|
| Pipeline correctness (causality, purging, accounting, calibration) | **Verified** — 108 automated tests, incl. leak-detection and hand-computed accounting checks |
| Statistical machinery (walk-forward, bootstrap CIs, PSR/DSR, drift) | **Implemented and tested** |
| Edge on real markets | **Not claimed.** The default config runs on a synthetic regime-switching market with *known planted structure* so the whole system is verifiable offline. Connect real data and run the full protocol in `docs/VALIDATION.md` before believing anything. |
| Execution / brokerage | Out of scope by design |

This sandbox has no market-data egress, so the repo ships with a synthetic
provider as the default. The provider layer (`yahoo`, `csv`) is production
code — switch `data.provider` in the config where network/data access exists.

**Nothing here is investment advice.**

---

## What it does

```
raw OHLCV ──► quality gates ──► causal features (~70, 8 families)
                                     │
                     triple-barrier labels (vol-scaled, event times)
                                     │
              purged walk-forward (embargoed, leakage-checked at runtime)
                                     │
     calibrated ensemble (HGB + RF + logistic; purged K-fold OOF stacking,
        isotonic calibration on pooled OOF, full-train member refit)
                                     │
   regime detector (GMM states + rules) ──► gates strategy + scales risk
                                     │
      adaptive EV gate (break-even prob from costs + margin, per regime)
                                     │
   Signal: grade, prices, size, analogues, evidence for AND against
                                     │
   risk engine (¼-Kelly ∧ vol-target ∧ stop-risk, heat/sector/corr/DD caps)
                                     │
   portfolio simulation ──► bootstrap CIs, deflated Sharpe, risk of ruin
                                     │
              artifacts ──► dashboard / registry / drift monitors
```

Every signal carries: calibrated probability, ensemble disagreement,
confidence score (0–100), trade grade (A+/A/B+/B), entry zone + limit/market
prices, sigma- and ATR-stops, three take-profit levels, position size and
risk-%, expected holding period and volatility, MAE/MFE estimates from its
k-nearest historical analogues, outcome quantiles, regime context,
institutional-accumulation score, top feature contributions, supporting and
conflicting evidence, and a plain-English reasoning summary. Signals with
negative expected value after costs **do not exist**.

## Quickstart

```bash
pip install -e ".[server,dev]"

# POSITIVE CONTROL (~4 min): strong planted signal — the pipeline must find
# it, emit signals, and show positive OOS statistics.
titan validate --config configs/control-strong-signal.yaml

# REALISTIC DEMO (~4 min): weak planted signal. Correct outcome: classifier
# skill above base rate but few trades and a REJECTION verdict under
# docs/VALIDATION.md — watching the platform refuse a marginal edge is the
# point of this config.
titan validate --config configs/default.yaml

# serve the dashboard over the artifacts
titan dashboard --port 8321        # http://127.0.0.1:8321

# rank the universe with the current production model
titan scan

# platform / registry status
titan info

# tests & lint  (negative control lives here: shuffled labels => AUC ~0.5)
pytest -q          # 108 tests; -m "not slow" for the fast subset
ruff check src tests
```

Docker:

```bash
docker compose up            # research run, then dashboard on :8321
```

### Real data

```yaml
# configs/live.yaml (example)
data: { provider: yahoo, bars: 2500 }
universe:
  benchmark: SPY
  instruments:
    - { symbol: AAPL, asset_class: equity, sector: technology }
    - { symbol: BTC-USD, asset_class: crypto, sector: crypto }
    # ...
```

```bash
titan validate --config configs/live.yaml
```

Then follow `docs/VALIDATION.md` — especially the promotion gate and the
deflated-Sharpe reading — before acting on anything.

## Repository map

```
configs/            research configuration (YAML, pydantic-validated)
src/titan/
  core/             types, config, logging
  data/             providers (synthetic/yahoo/csv), QC + reliability scoring, store
  features/         causal feature registry, families, redundancy pruning, importance
  labels/           triple-barrier labelling, uniqueness weights
  models/           purged walk-forward CV, calibrated ensemble, model registry
  regime/           GMM + rules regime detector (9 regimes + vol overlay)
  backtest/         cost model, event-driven engine, metrics, Monte Carlo, walk-forward
  risk/             sizing rules and portfolio-level risk engine
  signals/          analogue index, signal generator, schema
  explain/          median-counterfactual local explanations
  scanner/          universe ranking with rejection reasons
  monitor/          PSI drift, prediction tracking, champion/challenger gate
  server/           FastAPI + self-contained dashboard (dark/light, responsive)
  cli.py            validate / scan / dashboard / info
  artifacts.py      research outputs -> auditable files
tests/              108 tests: causality, leakage, accounting, calibration, e2e
docs/               RESEARCH.md, ARCHITECTURE.md, VALIDATION.md
```

## Design positions (the short version)

- **Causality is enforced, not assumed** — every feature is recomputed on
  truncated history in CI; a single changed value fails the build.
- **Labels are events, not returns** — triple-barrier with vol-scaled
  barriers matches how the engine actually exits, and event end-times feed
  the purge. The mirror is exact down to the entry bar: stops are live the
  moment a fill exists, gaps through a level invalidate the plan, and when
  capacity binds the highest-confidence candidates take the slots.
- **Calibration over accuracy** — position sizing consumes probabilities;
  an uncalibrated 0.7 is a lie that costs money.
- **The gate is derived, not tuned** — the signal threshold is the
  break-even probability implied by barrier geometry and costs, plus margin,
  tightened in hostile regimes.
- **Robustness beats frequency** — most days produce zero signals; the
  scanner shows *why* each instrument was rejected.
- **Self-distrust is a feature** — drift monitors, a CUSUM alarm on
  calibration, and a promotion gate that keeps challengers out of production
  without statistical proof.

Full methodology and the evidence behind each decision: `docs/RESEARCH.md`.
Validation protocol and rejection gates: `docs/VALIDATION.md`.
Module-level architecture and extension points: `docs/ARCHITECTURE.md`.
