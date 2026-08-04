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

- **Orders to place** — the actionable output, first on the page. One row per
  signal that cleared the whole gate chain, laid out as an order ticket: side,
  limit and market entry, entry zone, stop, three targets, size as a fraction
  of equity, risk-%, R:R, calibrated probability and grade. Prices are shown to
  significant figures rather than two decimals, so a sub-penny token's entry,
  stop and targets stay distinguishable. An empty table is a decision — every
  candidate was refused, with the reason in the scanner table below.
- **Paper account** — the forward simulated account, starting at
  `monitor.paper_starting_equity` (default $1,000,000). Equity, realized P&L,
  drawdown, win rate and profit factor; the equity curve; open positions; and
  the closed-trade ledger newest-first with entry, exit, why it closed, bars
  held, notional, return, P&L and the equity it left behind. This is the live
  loop's own track record, distinct from the backtest above it — the backtest
  is history, this is what the scanner has actually emitted since you started
  running it.
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
  were rejected. A scanner that only shows winners teaches you nothing. Each
  instrument is ranked on its OWN most recent bar, so a mixed equity/crypto
  book does not lose its equities on days when only crypto printed; anything
  lagging beyond `scanner.max_staleness_bars` is listed as stale with its date
  rather than dropped.
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
titan preflight --config <cfg>     # which symbols survive QC — run this FIRST
titan scan --config <cfg>          # rank the universe; auto-logs predictions
titan scan --config <cfg> --model v007   # ...with a specific registry version
titan account --config <cfg>       # paper account: equity, open book, ledger
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
2. `titan preflight --config configs/my-live.yaml` — **before** committing
   compute. It fetches and QCs every symbol, prints bars and reliability per
   ticker with the reason for each exclusion, and emits a cleaned `instruments:`
   block to paste back. On a 200-symbol universe this is the difference between
   finding a dead ticker in two minutes and finding it three hours in.
3. `titan validate --config configs/my-live.yaml`
4. Check `artifacts/quality.json` — drop anything below ~0.9 reliability.
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
| Day trading | `configs/style-daytrading.yaml` | `1h` | one session |
| Semi-scalping (equities) | `configs/style-semiscalp.yaml` | `1h` | ~3 hours |
| Semi-scalping (crypto) | `configs/style-semiscalp-crypto.yaml` | `3h` | ~6 hours |
| Scalping | `configs/style-scalping.yaml` | `5m` | minutes — exploratory only |

**Semi-scalping is the floor of honest research here.** The objections that
make `5m` exploratory weaken as the bar grows and are merely strains by the
hourly bar: a round trip costs ~11% of a 3-hour target move (versus ~27% at
five minutes, where it swallows the edge outright), and a next-open fill over
an hour is a fair model of an order you can actually place. Crypto is the
better venue for it — Yahoo's 730-day hourly cap yields ~3,500 bars on an
equity trading 6.5 hours a day and ~17,500 on an asset that never closes.

**Shortening the horizon means re-cutting the barriers.** Barriers sit at
±k·σ where σ is *per-bar* volatility — they are not scaled by the horizon.
Only √horizon sigmas of cumulative move are available, so carrying the daily
`tp_sigma: 2.0` onto a 3-bar horizon asks price to travel 2σ when 1.7σ is on
offer: nearly every label times out and the model trains on an almost
constant target. Each shipped config holds the validated daily reachability
ratio (`tp_sigma / √horizon ≈ 0.63`), and a test enforces it.

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

**`data.resample_from` aggregates finer bars into your timeframe.** Two
things need it. Yahoo serves no multi-hour bar, so `2h`/`3h`/`4h` are
reachable only by aggregating `1h`. And Yahoo's *hourly crypto* series
carries no volume on roughly half its bars — which QC correctly refuses,
since OBV, dollar-volume and up/down-volume features are built from that
column. Summing source bars leaves a target bar empty only where every
source bar in it was empty, so aggregating trades resolution for a real
volume column. It recovers traded volume; it does not invent any (total
volume is conserved exactly, and a test enforces it).

```yaml
data: { provider: yahoo, timeframe: 2h, resample_from: 1h }
```

Bars are stamped at the **start** of the interval they cover, matching the
convention the engine assumes — stamping right would date a bar before data
inside it and leak the future into every feature. On a market that closes,
aggregation never fuses bars across the overnight gap into one bar that never
traded as one.

**QC is session-aware below daily.** On a market that closes, an hourly
series carries an overnight boundary every seventh bar. Gap detection would
read those as dropped data and the bad-print detector would read the
overnight move as an error, failing every symbol on the exchange for keeping
normal hours. Both checks therefore evaluate intra-session structure
separately from session boundaries when the timeframe is intraday and the
universe is not 24/7 — without going blind: a hole *inside* a session and a
bad print in either population are still caught.

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

## 6c. The second order source: rule-based swing setups

The calibrated gate emits nothing when the model has no edge to defend — which
is the correct answer, and on a universe scoring AUC ~0.51 it is the answer
every day. `signals.technical` is a separate engine that fires on price
structure instead:

```yaml
signals:
  technical:
    enabled: true
    max_orders_per_scan: 5      # per asset class, strongest-first
    min_risk_reward: 1.5        # measured to the second target
    setups: [donchian_breakout, pullback_in_uptrend, ma_cross,
             oversold_bounce, macd_momentum]
```

| Setup | Fires when |
|---|---|
| `donchian_breakout` | close at a new 20-bar high on >1.2x average volume |
| `pullback_in_uptrend` | close > rising 50MA > 200MA, back within 1 ATR of the 20MA, turning up |
| `ma_cross` | 20MA crosses above the 50MA with price over the 200MA |
| `oversold_bounce` | RSI(2) < 10 while price holds above the 200MA |
| `macd_momentum` | MACD crosses its signal above zero, in an uptrend |

Stops come from structure (the swing low, or an ATR multiple — whichever is
further), and targets are 1R/2R/3R of that stop distance, so reward is always
measured against the risk actually taken. Sizing is fixed-fractional off the
stop (`risk.risk_per_trade_pct`), scaled by the regime multiplier — a crash
still zeroes the size. There is no Kelly term, because Kelly needs a
probability and a rule does not produce one.

**These are not validated alpha, and the platform does not pretend otherwise.**
The walk-forward, the Venn-ABERS bands and the deflated Sharpe apply to the
model, not to a rule fired on a chart. A breakout setup here has a definition,
not out-of-sample evidence. Every technical order is labelled with the rule
that produced it, in the orders table, the scanner status and the account
ledger, and the paper account reports P&L **by source** — so after enough
trades the ledger tells you whether the rules or the model earned. That is the
honest way to find out, and it costs simulated money rather than real money.

Order cards from this engine show no `P(hit)`: there isn't one, and printing a
number nothing computed would be worse than a dash.

## 6d. Two books: how many orders, and from where

Orders are ranked and capped **within each asset class**, not across all of
them:

```yaml
scanner:
  top_n: 10                     # fallback for classes not named below
  orders_per_asset_class:
    equity: 5
    crypto: 5
```

This is not cosmetic. Crypto's daily volatility runs several times an equity's,
so on one ranked list the coins take nearly every slot and the equity book is
never traded. Ranking inside each class is what actually produces "the five
best stocks and the five best crypto".

Two rules govern what reaches the list:

- **One order per symbol.** The model gate and the rule engine can fire on the
  same name on the same bar. That is one idea, and shipping both tickets would
  double the intended size. The model order wins the collision — it is the one
  with out-of-sample evidence behind it.
- **An empty book does not donate its slots.** Five crypto setups and no equity
  ones means five orders, not ten. Backfilling would quietly double crypto
  exposure on days the equity screen is silent.

## 6e. Leverage (crypto perpetuals)

Off by default. Turn it on per asset class:

```yaml
risk:
  leverage:
    max_leverage:
      crypto: 3.0             # equities absent -> 1.0, cash
    maintenance_margin_rate: 0.005
    funding_bps_daily: 3.0    # ~0.01% per 8h, charged on notional
    stop_buffer: 1.5          # liquidation must stay 1.5 stop-widths away
    max_account_leverage: 2.0 # ceiling on summed notional / equity
```

**Leverage multiplies risk, not just size.** A position sized to lose 0.4% of
equity at its stop loses 1.2% at 3x. There is no version of this where the
notional triples and the loss does not. The `Risk` column on the order card,
the paper ledger and the account KPIs all show the levered number, because that
is the number that is true.

`max_leverage` is a **ceiling, not a setting**. Each order solves for the
largest multiple whose liquidation price stays `stop_buffer` stop-widths beyond
its stop, and takes the smaller of that and the ceiling:

> L ≤ 1 / (stop_buffer · stop_distance + maintenance_margin_rate)

A 2% stop leaves that slack and gets the full 3x. A 25% stop resolves to ~2.6x
on its own. A stop wide enough that no multiple is safe trades unlevered rather
than being rejected. The consequence worth internalizing: **a wide-stop trade
de-levers itself**, so leverage concentrates in exactly the tight-stop setups
where it is survivable.

The order card gains three columns — `Lev`, `Liquidation`, and `Margin`
alongside `Size`. Size is notional as a percent of equity; margin is the cash it
actually ties up (`notional / leverage`). Liquidation is where the exchange
closes the trade whether or not the stop has filled; by construction it is
always further out than the stop, and it is printed so that is visibly true
rather than merely asserted.

Funding is charged on notional for the expected hold and folded into
`cost_estimate` **before** the EV gate, so a levered trade has to pay its own
rent out of the move it predicts.

In the paper account, two independent ceilings apply on entry, because leverage
separates two things a cash account conflates. `backtest.max_gross_exposure`
bounds the **cash** posted as margin — an account cannot fund what it does not
have. `max_account_leverage` bounds the summed **notional** — an account that
has funded ten 3x positions carries 30x of market exposure behind one balance,
and no per-position limit can see that. Positions refused by either are
reported, not dropped.

Closed levered trades show `return_on_margin` next to the return on notional;
on a levered trade the former is the number that matters. A position that
reaches its liquidation level is posted as a total loss of its margin and
labelled `liquidated` — the ledger will never show a loss larger than the cash
a position had, which is the one thing raw notional arithmetic gets wrong.

**The walk-forward backtest runs unlevered even when this is configured**, and
says so in a warning at `titan validate`. Simulating margin through it properly
needs intrabar liquidation and per-bar funding accrual; a half-modelled version
would report returns the account could not have produced. So backtest Sharpe,
CAGR and drawdown describe the cash strategy — the multiple shows up in live
`titan scan` orders and in the paper account, which is where you should look
for its effect.

What is **not** modelled: cross margin (one position's loss eating another's
collateral — it would let a single trade liquidate the whole book), short
perpetuals (every setup here is long), tiered maintenance margin, and funding
that varies with the basis. Equities stay at 1x unless you add them explicitly,
because margin on a stock needs a broker agreement a config file has no
business assuming exists.

## 7. Tuning the knobs that matter

All in your YAML config (validated by pydantic — typos fail loudly):

| Knob | Effect |
|---|---|
| `labels.tp_sigma` / `sl_sigma` / `horizon_bars` | barrier geometry: what "win" means. Changes the gate automatically |
| `signals.min_probability`, `ev_margin_bps` | how picky the gate is (fewer, better trades) |
| `risk.target_annual_vol`, `risk_per_trade_pct`, `kelly_fraction` | aggressiveness; the minimum-of-three sizing keeps any one mistake bounded |
| `risk.regime_multipliers` | risk appetite per regime (crash is 0 — think hard before changing) |
| `risk.leverage.max_leverage` | margin per asset class; a ceiling, and it multiplies loss as well as size (§6e) |
| `scanner.orders_per_asset_class` | how many orders each book gets, ranked within itself (§6d) |
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

### The paper account

`scan` and `track resolve` both refresh `artifacts/account.json`, so the
forward account keeps itself current on the daily loop with no third command:
a scan opens positions, a resolve closes them.

    titan account --config <cfg>

It starts at `monitor.paper_starting_equity` ($1,000,000 by default) and takes
each signal at the size the risk engine gave it, against the equity standing
at that moment, netting the round-trip cost the signal itself priced in.
Equity compounds as trades close.

It is deliberately NOT a second risk engine. Sizes are taken as emitted; the
only portfolio rules re-applied are `backtest.max_gross_exposure` and
`risk.leverage.max_account_leverage`, because an account that cannot fund a
position does not take it and silently levering past its cap would make the
curve fiction. Signals refused for that reason are counted and shown, not
dropped.

An empty ledger is the expected state until the gate emits something that has
since resolved — it is not a broken panel.

#### Marked to market: two balances, not one

Open positions are valued at the latest bar, so the account moves every day —
not only on the days something closes. It reports two figures, the way a broker
separates cash from net liquidation value:

| Figure | What it is |
|---|---|
| **Cash (realized)** | banked by closed trades. This is what sizes new positions |
| **Unrealized P&L** | floating P&L of everything still open, net of the round trip it still owes |
| **Account value** | cash + unrealized. What the account is worth right now |

**Only cash compounds into sizing.** Sizing off account value would turn a
paper gain into real exposure — the next position gets bigger because an
earlier one happens to be winning on screen, which is leverage nobody asked
for.

A position opened today shows a small loss immediately. That is not a bug: it
is charged the round trip it will owe to get out, which is what you are down if
you close it now.

The marked view also corrects a number the realized-only view understates.
`max_drawdown` counts closed trades; `max_marked_drawdown` counts the open book
too. On a run of mine the two read −26.6% and −34.1% — the open losers were
real, they just had not been booked yet.

Open-position rows show mark price, move since entry, unrealized dollars,
return **on margin** (the number that matters on a levered position), and
distance to both stop and liquidation. They are sorted worst floating loss
first, because that is what you want at the top of an open book.

**Liquidation is checked against the price path, not the outcome.** A levered
position whose low reached its liquidation level was closed by the exchange on
the way, whatever barrier the tracking log later grades it against — so the
replay realizes it there. A dip through liquidation that recovered still counts;
an account that only looked at where price ended up would report it as a live
winner.

Marking needs prices: `titan scan` and `titan track resolve` already have a
dataset loaded and pass it; `titan account` loads one for the same reason, and
`--no-marks` skips it for a realized-only view.

#### Live refresh: three clocks, not one

    titan dashboard --refresh 1 --quote-interval 15

`--refresh` recomputes the account and has the page poll to match (default 1s;
`0` serves static artifacts). `--quote-interval` is how often prices are
actually fetched. They are separate because three different things change at
three different rates, and collapsing them either wastes requests or reports
stale data as fresh:

| Clock | Rate | Why |
|---|---|---|
| **Account** | `--refresh`, default 1s | pure arithmetic, no network — recompute as fast as you want to look |
| **Prices** | `--quote-interval`, default 15s | one batched vendor request; the only clock bounded by someone else |
| **Bars** | 15 min, or when the date rolls | a `1d` config gains one bar a day; polling faster learns nothing |

**Only the open book is priced, not the universe.** Marking needs a price per
open *position* — typically a dozen names. Requesting all 201 configured
symbols every tick to value nine of them is a 20x waste that gets an IP
rate-limited into returning nothing at all. `titan scan` ranks the universe
once a bar; the refresh loop prices what the account holds.

**A live vendor has a hard floor of 5s regardless of what you ask for.** This is
a correctness guard, not politeness: being rate-limited does not give you slower
data, it gives you *none*. Fetches are batched and sequential (yfinance's
threaded mode races its own sqlite timezone cache and fails every symbol from a
background thread), on daily bars rather than 1-minute — today's daily bar is
in progress, so its close IS the last trade, and the 1m endpoint is the first
one Yahoo refuses at scale.

**A partial answer counts as a failure.** Yahoo routinely returns half a batch.
Treating that as success resets the backoff, so a feed that is chronically 60%
blind keeps being polled at full rate — useless, and the surest way to stay
rate-limited. Below 50% coverage the prices are still kept (they are real) but
the interval stretches. The status line shows `N/M priced`, so a partly-marked
balance is visibly partly marked.

Not moving? `titan quotes` runs exactly the refresh loop's fetch once and prints
what came back:

    titan quotes --config configs/top100.yaml
    titan quotes --symbols AAPL,BTC-USD     # test the feed with no open book

When quotes fail entirely the account still marks from the latest **bar** — it
just stops moving between closes.

**Refreshing never creates an order.** New orders need a new *bar* — the gate,
the setups and the labels are all defined on closes — so an intra-bar signal
would be research the pipeline never did. Orders come from `titan scan`. What
moves in between is the valuation of what is already open, which is what a
broker screen shows between fills.

On a **daily** config the prices behind the mark change once a day at the close,
so a 1-second refresh redraws the same numbers all day. That is worth knowing
before reading anything into a still balance: it is not frozen, there is just
nothing new. Intraday configs (`data.timeframe: 1h`, say) is where the fast
refresh earns its keep.

The page pauses polling while its tab is hidden and refreshes immediately on
return. A refresh tick re-reads only the account, the scan and the live status;
the research panels reload only when `titan validate` reruns, so a 178-symbol
correlation heatmap is not rebuilt every second. A failed fetch keeps the last
good render rather than blanking a panel.

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
| `no instrument passed data QC` on an intraday crypto run | Yahoo's intraday crypto volume is ~50% empty; aggregate with `data.resample_from: 1h` at a `2h`/`3h`/`4h` timeframe, or use exchange CSV exports |
| `Yahoo has no native 2h bar` | multi-hour bars come from aggregation — set `data.resample_from: 1h` |
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
