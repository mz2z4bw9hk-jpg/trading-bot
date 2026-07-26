# TITAN User Guide — from zero to reading your first signal

This is the hands-on tutorial. The theory lives in `RESEARCH.md`, the
safety rules in `VALIDATION.md`, and the internals in `ARCHITECTURE.md`.

---

## 1. Install (one time)

```bash
git clone <repo> && cd trading-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[server,data,dev]"
titan --version        # sanity check the CLI landed
```

Python 3.11+ — check with `python3 --version`; the Python that ships with
macOS is 3.9 and will refuse the install. The virtualenv is not ceremony: it
is what puts the `titan` command on your PATH and makes `pip` writable.
Re-activate it (`source .venv/bin/activate`) in every new terminal.

Extras: `server` for the dashboard, `dev` for the test suite, and **`data`
for `provider: yahoo`** (it brings `yfinance`) — omit `data` and a Yahoo run
fails at fetch time with a missing-dependency error. No GPU, no database, no
API keys needed for the offline demo.

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
| `configs/control-strong-signal.yaml` | strong | detection: high AUC, many trades, positive Sharpe CI, tight Venn-ABERS bands |
| `configs/default.yaml` | realistically weak | **refusal**: AUC barely above chance and ZERO trades — no candidate survives the gate chain (EV threshold, Venn-ABERS lower bound, uncertainty ceiling) |

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
- The **[lo–hi] band** after P is the Venn-ABERS interval: distribution-free
  error bars on the calibration itself. The gate is cleared by the *lower*
  bound (`signals.conservative_gate`), so a wide band near the threshold
  kills the signal rather than flattering it.
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
titan scan --config <cfg>          # rank the universe; auto-logs predictions
titan track resolve --config <cfg> # grade elapsed predictions on real bars
titan info                         # registry, production, tracking status
titan dashboard --port 8321        # always reads the latest artifacts
titan export                       # one static HTML file of the dashboard
```

`<cfg>` is the SAME config you validated with. Each model remembers a
fingerprint of the world it was trained on (provider, universe, labels,
features — and, on synthetic data, the seed); `scan` and `track resolve`
refuse a mismatched config instead of producing plausible-looking numbers
about a market the model never saw. Bare `titan scan` means
`configs/default.yaml`, which is usually *not* what you validated with.

`scan` is pure inference (seconds). Retraining happens only through
`validate`, and a retrained model reaches production **only** by beating
the incumbent through the statistical promotion gate — a challenger with a
non-significant edge or a degraded drawdown stays shelved, with reasons
recorded in `models_store/index.json`.

### Sharing results without a server

`titan export` writes `artifacts/titan_dashboard.html`: the full dashboard
with every number baked in. It opens with a double-click — no Python, no
server, no network — and can be e-mailed or dropped on any static host
(GitHub Pages, Netlify, an S3 bucket). It is a frozen snapshot of one run
(the footer shows the export time); re-export after each `validate`/`scan`
to refresh it. The embedded payloads are byte-identical to what the live
API serves — a test enforces it.

## 6. Connecting real data

1. Start from `configs/majors.yaml` (large-cap crypto + index ETFs +
   mega-cap stocks, ready to run) or copy `configs/live-example.yaml` and
   edit the universe/benchmark. Provider `yahoo` needs network egress;
   provider `csv` reads per-symbol OHLCV files from `data.csv_dir`.
2. `titan validate --config configs/my-live.yaml`
3. Check `artifacts/quality.json` — drop anything below ~0.9 reliability.
4. Follow `VALIDATION.md` §1 (reading order and rejection gates) and §3
   (cost / barrier / jackknife / seed sensitivity sweeps).
5. Paper-track `titan scan` on a schedule for a meaningful period before
   any capital decision. Synthetic results certify the machinery only.

### From TradingView

TradingView has no public account API, but every chart exports its data:

1. Open the chart, set the timeframe to **1D**, then menu → **Export
   chart data…** and download the CSV.
2. Drop the downloaded files into one folder (say `data_tv/`) — **no
   renaming needed**: TITAN resolves TradingView names like
   `BINANCE_BTCUSDT, 1D.csv` to the symbol `BTCUSDT` (and `BTC-USD`
   matches `BITSTAMP_BTCUSD, 1D.csv`). ISO or epoch time columns both
   parse; rows are sorted before use.
3. In your config:

   ```yaml
   data: { provider: csv, csv_dir: data_tv, bars: 2500 }
   universe:
     benchmark: SPY          # export SPY too — the benchmark needs a file
     instruments:
       - { symbol: BTCUSDT, asset_class: crypto, sector: crypto }
       # ... one entry per exported chart
   ```

4. `titan validate --config configs/my-tv.yaml`

Prefer ETFs (SPY/QQQ/DIA) over raw indices — the pipeline requires a
volume column, which raw index series often lack. Export as much history
as your TradingView plan allows; fewer than ~1500 daily bars will be
flagged by QC and below `cv.min_train_bars` the run refuses to start.

## 6b. Choosing a timeframe

`data.timeframe` sets what one bar means: `1wk`, `1d`, `4h`, `1h`, `30m`,
`15m`, `5m`, `1m`. Four ready-made configs cover the usual styles:

| Style | Config | Bar | Typical hold |
|---|---|---|---|
| Position / long-term | `configs/style-longterm.yaml` | `1wk` | months |
| Swing | `configs/style-swing.yaml` | `1d` | days–weeks |
| Day trading | `configs/style-daytrading.yaml` | `1h` | hours–days |
| Scalping | `configs/style-scalping.yaml` | `5m` | minutes |

Everything annualized — Sharpe, CAGR, annual vol, the vol-targeting divisor,
the regime detector's trend thresholds, the VaR window — derives from
bars-per-year, which is resolved from the timeframe. It is also part of the
research fingerprint, so a model validated on `1h` refuses to scan a `1d`
config rather than reporting numbers about a market it never saw.

**Session vs 24/7.** A universe whose tradeable instruments are all crypto
annualizes on a 365-day calendar; anything with an equity or ETF leg uses the
6.5-hour session calendar. At `1h` those differ by 5.3x, which moves every
Sharpe by sqrt(5.3) = 2.3x. The benchmark is excluded from that test (crypto
books are routinely benchmarked against SPY). If the loaded data disagrees
with the resolved convention, the run logs a `calendar mismatch` warning —
set `data.bars_per_year` explicitly to settle it.

**Shorter bars are not more research, they are less.** Two things get worse
as the bar shrinks:

- *Costs scale with how often you cross the spread.* Same calendar exposure
  at `1h` instead of `1d` means ~6.5x the round trips. A 4 bp round trip that
  is noise against a two-week move is the entire edge against a one-hour one.
- *Vendors keep less intraday history.* Yahoo's caps are hard: 730 days of
  `1h`, 60 days of `5m`/`15m`/`30m`, 7 days of `1m`. Asking for more does not
  error — it silently returns less. TITAN clamps the request and warns, but
  no clamp creates history that isn't there. 4,700 hourly bars spanning two
  years is plenty of *rows* and only one or two *regimes*; read the regime
  breakdown before believing a walk-forward that never saw a bear market.

**`1m` and `5m` are refused** unless the config sets
`data.acknowledge_unvalidated_timeframe: true`. The machinery runs fine; the
evidence cannot meet the platform's own bar. Fills are modelled at the next
bar's open, which is a fair model of a market-on-open order over a day and
fiction over a minute. The cost model is calibrated for daily turnover. And
sub-minute price formation is driven by order-book state — queue position,
depth, order-flow imbalance — that OHLCV bars simply do not contain. Honest
scalping research needs order-book data and fills calibrated to your broker,
not a smaller bar.

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

Paper-tracking is built into the daily loop — no code needed:

1. `titan scan` logs every emitted signal to
   `models_store/paper_track.json` automatically (idempotent per
   symbol+date, so re-scanning is safe).
2. `titan track resolve` — run it any day — fetches current data and grades
   every prediction whose horizon has elapsed, using the *exact*
   triple-barrier labeller the models were trained on. Exit code 3 means
   the CUSUM alarm fired.
3. `titan track status` (or the dashboard's *Paper tracking* panel) shows
   logged/resolved counts, hit rate, rolling Brier vs the production
   model's own OOF baseline, and the calibration table.

On a CUSUM alarm: run `validate` to produce a challenger; let the promotion
gate decide. Never hand-promote.

- **Feature drift** stays available as a library call:
  `titan.monitor.feature_drift_report(X_train_ref, X_live)` — PSI ≥ 0.25 on
  a model feature means the world changed; retrain.

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `no production model: run titan validate first` | scan before any validate |
| Zero signals on a scan | usually the gate doing its job — check the scanner table for per-symbol reasons (crash regime blocks everything) |
| `training window too small` | not enough history for `cv.min_train_bars` / internal folds — more bars or fewer folds |
| Instrument missing from results | failed QC; see `artifacts/quality.json` and the log line explaining why |
| Yahoo fetch fails | no network egress from your environment; use `csv` |
| `CONFIG MISMATCH` on scan/resolve | you validated with one `--config` and scanned with another (bare `titan scan` = `configs/default.yaml`) — pass the config the model was validated with. Changing `data.timeframe` changes the fingerprint too |
| `not validated research on this platform` | `1m`/`5m` refused by design — see §6b before setting `data.acknowledge_unvalidated_timeframe` |
| `Yahoo keeps at most N days` | vendor history cap for that interval; the run continues on what exists, but check `quality.json` and the regime breakdown |
| `calendar mismatch` warning | the resolved bars-per-year disagrees with the loaded data (usually a 24/7 book on a session calendar) — set `data.bars_per_year` |
| `titan track resolve` exits 3 | that IS the CUSUM alarm — run `validate` to produce a challenger and let the promotion gate decide |
| Signal shows a wide P band | thin calibration evidence near that score; the conservative gate already priced that in |
| Slow validate | lower `tuning_iterations` / `internal_folds`, or trim `members` |

## 10. Development loop

```bash
pytest -q -m "not slow"     # fast unit tests (~30 s)
pytest -q                   # full suite incl. end-to-end (133 tests)
ruff check src tests scripts
python -m mypy src/titan
python scripts/screenshot_dashboard.py   # visual check of the dashboard
```

Adding a feature/provider/model member: see the extension points in
`ARCHITECTURE.md`. Anything you add inherits the causality test, the
purge invariants and the no-skill-on-noise control automatically — the
platform is built so the safe path is the easy path.
