# TITAN Research Notes — methodology review, hypotheses, and decisions

This document records *why* the platform is built the way it is: the survey
of candidate methodologies, the hypotheses we kept, the ones we rejected and
the reasons, and the resulting hybrid design. It is a living lab notebook,
not marketing.

---

## 1. Survey of methodologies (what we considered)

### Kept, in adapted form

| Family | Evidence base | How TITAN uses it |
|---|---|---|
| **Time-series momentum** | Moskowitz, Ooi & Pedersen (2012); Jegadeesh & Titman (1993) — persistence of 1–12-month returns across asset classes | Multi-horizon return, vol-scaled momentum, skip-month momentum features; one of the two hypothesis engines |
| **Short-horizon mean reversion** | Lehmann (1990), Lo & MacKinlay (1990) — weekly reversal in equities; stronger inside low-vol ranges | z-scores, band positions, gap features; regime-gated (range regimes only get meaningful weight via the ML layer) |
| **Volatility regimes / clustering** | Engle (1982) ARCH lineage; Ang & Bekaert (2002) regime switching | Unsupervised GMM state model + vol-percentile overlay drives strategy gating and risk multipliers |
| **Relative strength / cross-sectional momentum** | Jegadeesh & Titman (1993) | Cross-sectional rank features (plain and vol-adjusted), breadth, dispersion |
| **Volume/participation microstructure proxies** | Amihud (2002) illiquidity; OBV-style flow accumulation | Liquidity features, up/down volume ratio, volume-weighted close-location (institutional accumulation proxy), Amihud measure |
| **Meta-labelling & event-based labels** | López de Prado, *Advances in Financial ML* (2018) | Triple-barrier labels with event times; purged/embargoed walk-forward; uniqueness weighting; deflated Sharpe |
| **Ensemble learning + calibration** | Breiman (1996; 2001); Niculescu-Mizil & Caruana (2005) on calibration of boosted trees | Heterogeneous ensemble (HGB/RF/logistic), out-of-fold isotonic calibration, disagreement as uncertainty |
| **Conformal / Venn prediction** | Vovk & Petej (2014), Venn-ABERS predictors | Distribution-free probability intervals per signal; the gate is cleared at the interval's lower bound |
| **Fractional Kelly & vol targeting** | Kelly (1956); Thorp (2006); Moreira & Muir (2017) vol-managed portfolios | Position size = min(¼-Kelly, vol-target, fixed-fractional stop risk) |
| **Block bootstrap inference** | Politis & Romano (1994); Bailey & López de Prado (2014) | Stationary block bootstrap CIs; PSR/DSR multiple-testing control |

### Considered and deliberately rejected (for this platform, with reasons)

- **Raw indicator stacking (RSI+MACD+BBands voting)** — rejected: popular ≠
  predictive; unvalidated indicator soup is the canonical overfitting
  machine. Individual constructs survive only as *features* subject to IC
  screening, redundancy pruning and permutation importance, where most of
  them must earn their place per fold.
- **Deep sequence models (LSTM / temporal CNN / Transformers / TFT)** —
  rejected *for the default stack*: at daily frequency on a research
  universe the sample count (10³–10⁵ events) is far below where these
  dominate gradient boosting on tabular features (see Grinsztajn et al.
  2022); they add GPU dependencies, opaque failure modes and a wide
  hyperparameter surface that inflates the trial count DSR must deflate.
  The `CalibratedEnsemble` member registry is the extension point if a
  future universe (tick data) justifies them.
- **Reinforcement learning for execution/allocation** — rejected: reward
  hacking on backtest simulators is endemic; sample-inefficient; the sizing
  problem already has closed-form, auditable solutions (Kelly/vol-target)
  that a risk committee can reason about.
- **Graph neural networks over asset graphs** — rejected for now: our
  cross-sectional features (ranks, breadth, dispersion, benchmark beta)
  capture the low-hanging relational structure; a GNN needs a much larger
  cross-section to beat that baseline.
- **HFT-style microstructure (order-book imbalance, liquidity sweeps, order
  blocks, fair-value gaps)** — out of scope on daily bars: these are
  intraday phenomena requiring L2/tick data. Where the underlying *idea*
  has a daily-bar analogue (accumulation, participation), it exists as a
  feature (CLV-volume, up/down volume). The "order block / FVG" retail
  formulations were additionally rejected for lacking any peer-reviewed
  evidence base.
- **Options-flow / gamma exposure / dark-pool prints, on-chain whale flows** —
  *architecturally supported, not shipped*: the provider interface and the
  feature registry accept any aligned series with a reliability score, but
  we refuse to ship features whose data we cannot validate in this
  environment. Stubs pretending otherwise would be exactly the fake rigor
  this platform exists to prevent.
- **Wyckoff / Elliott narrative analysis** — rejected as untestable in
  their narrative form; the testable kernel (accumulation/distribution via
  price-volume divergence) is implemented as `clv_accum_21`, `obv_slope_21`
  and the regime refinement rules.

## 2. The TITAN hybrid methodology (original synthesis)

**Regime-Conditioned Ensemble with Derived Gates (RCE-DG).** The synthesis
is: *hypothesis-driven feature engines* + *event labels that mirror
execution* + *a calibrated ensemble whose output is consumed through an
economically derived gate* + *regime-conditioned risk*. No single component
is novel science; the discipline of the assembly is the product.

### Hypotheses under continuous test

- **H1 (persistence):** vol-scaled multi-horizon past returns carry
  information about barrier-hit probability. *Status: confirmed on synthetic
  ground truth (planted AR drift recovered: pooled OOS AUC 0.53, gated
  hit-rate uplift +12pp over base); must be re-confirmed per real universe.*
- **H2 (reversion in ranges):** short-horizon overextension mean-reverts in
  low-vol regimes. *Status: mechanism planted and recovered on synthetic;
  weight learned per fold by the ensemble rather than asserted.*
- **H3 (vol regimes gate everything):** predictability and optimal risk
  differ by regime strongly enough that gating beats always-on. *Status:
  detector separates bull/bear posteriors (Δ>0.15) with bear recall ~0.7 on
  ground truth; crash regime zeroes size by construction.*
- **H4 (relative strength):** cross-sectional leaders persist within bull
  regimes. *Status: features shipped; importance monitored per fold.*

Each hypothesis lives or dies by per-fold feature importance and the
monitoring stack — a hypothesis whose features stop earning importance is
flagged by drift reports, not defended by us.

### Why these specific mechanics

- **Triple-barrier labels** (not fixed-horizon returns): the label answers
  the exact question the trade asks — *does price hit +2σ before −1.5σ
  within H bars, entering at next open?* — including the pessimistic
  same-bar tie-break the backtester also uses. Label/execution mismatch is
  a silent killer of deployed strategies.
- **Purged walk-forward with embargo** (not K-fold): overlapping event
  labels leak across naive fold boundaries; K-fold also lets models train on
  the future. Both are structurally impossible here, and
  `assert_no_leakage` re-proves it at runtime on every run.
- **Heterogeneous ensemble + purged-OOF isotonic calibration**: boosted
  trees are accurate but miscalibrated (Niculescu-Mizil & Caruana 2005);
  sizing needs calibrated probabilities. Member weights and the calibrator
  are learned on pooled out-of-fold predictions from K purged internal
  folds — calibration therefore sees several market regimes instead of only
  the tail of the training window — and the deployed members are then refit
  on the full window, wasting no data. Members with different bias
  structures also give a free uncertainty signal — their disagreement —
  which gates signals independently of the probability level.
- **Venn-ABERS intervals on the calibration itself** (Vovk & Petej 2014):
  isotonic calibration returns a point probability with no notion of how
  much evidence supports it. The inductive Venn-ABERS pair [p0, p1] — two
  isotonic fits on the pooled OOF sample with the test point labelled 0 and
  1 respectively — brackets the probability with a distribution-free
  validity guarantee. The band is wide exactly where OOF evidence is thin.
  The gate consumes the *lower* bound (`signals.conservative_gate`): a
  signal that is only positive-EV under the most favorable reading of
  sparse calibration data is refused.
- **Derived threshold**: the minimum acceptable probability is computed
  from barrier geometry and the cost model (`p* = (b + c + margin)/(a+b)`),
  not fitted. A fitted threshold is one more overfittable parameter; a
  derived one moves automatically when costs or volatility change.
- **Minimum-of-three sizing**: each sizing rule fails differently
  (mis-estimated edge, vol-blind stops, stop-blind vol); the minimum
  inherits no single failure mode. Kelly is quartered because estimated
  edges are upward-biased by selection.
- **Regime detector outside the ML feature path**: defense in depth — a
  regime misclassification can cut risk, but it cannot silently poison the
  probability model, because the model never sees it.

## 3. Synthetic ground truth as scientific control

The default market generator plants *known* structure (slow AR(1) drift ⇒
momentum; negative idio autocorrelation in ranges ⇒ reversion; Markov regime
chain ⇒ regimes) inside realistic nuisance (factor structure, vol
clustering, crash correlation spikes, volume-|return| coupling). This gives
us what real data never can: **the right answer**. The test suite asserts
that the pipeline finds planted structure *and claims nothing on shuffled
labels* (AUC pinned to ~0.5 on noise). A pipeline that passes both is
trustworthy machinery; whether any *real* market pays it is a question only
`docs/VALIDATION.md` on real data can answer.

## 4. Known limitations (kept visible on purpose)

- Daily bars only in the default stack; intraday extensions need new
  execution modeling, not just new data.
- Long-only signal surface by default; the label/engine mirror shorts
  cleanly, but borrow costs and squeeze risk need their own validation
  before enabling `allow_short`.
- Crash *onset* detection lags by a few bars (daily realized-vol inertia);
  the drawdown throttle and vol overlay are the compensating controls.
- Impact model is volatility-scaled, not ADV-participation-based; adequate
  for small size, conservative to unknown degree for large size.
- DSR's trial count is an input, not something the platform can know about
  your research process; we report DSR at N ∈ {1, 5, 10, 25} and the
  honest answer is "use the largest N resembling what you actually tried."
