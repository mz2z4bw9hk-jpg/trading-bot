# Multi-Agent Trading Pipeline

Signal, risk and execution as three agents with a typed boundary between them,
running suggest-only, shadowed before live, and developed in parallel git
worktrees.

This document covers the architecture. The code is under `src/titan/agents/`,
`src/titan/microstructure/` and `src/titan/execution/`.

---

## 0. What this is not

This subsystem is **not** the TITAN scanner. The scanner is a swing-horizon
research platform on hourly and daily bars; this is a short-horizon
microstructure pipeline. They share the config machinery and the repository
and nothing else, deliberately — a market-making risk model and a portfolio
risk model answer different questions and merging them would produce a thing
that answers neither.

**Data requirement, stated up front.** Everything below needs L2 book updates
and a trade tape with sub-second timestamps. Bar data — including the Yahoo
hourly feed the scanner uses — cannot produce queue imbalance, order flow
imbalance, or adverse selection markouts, because the information is not in a
bar. The modules are written against explicit feed protocols and are tested
against replayed and synthetic books. They are correct today and useful the
day a real feed is attached, and not before.

---

## 1. Module breakdown

```
src/titan/
├── microstructure/            # shared primitives; no agent logic
│   ├── book.py                # BookSnapshot, microprice, imbalance, integrity
│   └── toxicity.py            # OFI, VPIN, Kyle's lambda, ToxicityState
│
├── agents/
│   ├── contracts.py           # THE BOUNDARY: Intent, RiskVerdict, ClearedOrder
│   ├── signal_agent.py        # (protocol + reference impl) stat-arb, microstructure
│   ├── risk_agent.py          # hard limits, toxicity, anomaly triage, SIZING
│   └── governance.py          # ProposalGate: suggest-only, enforced
│
├── execution/
│   ├── evaluation.py          # spread capture vs toxicity; markout decomposition
│   └── slicing.py             # (next) child order schedule, venue routing
│
└── shadow/
    └── runner.py              # (next) replay a session, quote without capital
```

Ownership is strict. The signal agent may not import `risk_agent`; the
execution agent may not import `signal_agent`. Both import `contracts`. That
is the entire permitted coupling, and it is what lets any one of the three be
replaced without the other two noticing.

### The data flow

```
market data ──► Signal Agent ──► Intent ──► Risk Agent ──► ClearedOrder ──► Execution Agent
                                              │                                  │
                                              ▼                                  ▼
                                         audit log                        fills + markouts
                                              │                                  │
                                              └──────────► Proposal ◄────────────┘
                                                              │
                                                              ▼
                                                    human review → merge → deploy
```

---

## 2. The boundary (`agents/contracts.py`)

Two design decisions carry the whole separation.

**An `Intent` has no size and no price.** This is the load-bearing one. A
signal agent that sized its own orders would reduce the risk agent to an
advisor that can only shrink what it is handed — it would never be the thing
that *decided*. Sizing depends on current inventory, session drawdown, and
flow toxicity, none of which appear on the signal agent's inputs. So the
signal agent expresses an edge and a confidence, and risk decides what that is
worth in shares.

`tests/test_agent_boundary.py::test_an_intent_carries_no_size` fails the build
if a `qty`, `size`, `notional` or `limit_price` field is ever added.

**A `ClearedOrder` cannot be constructed without the risk mint.** The
execution agent accepts nothing else. Python cannot truly seal a constructor,
and pretending otherwise would be theatre; what the mint buys is a *single
greppable construction site*. `test_the_mint_is_referenced_in_exactly_one_agent`
walks the source tree and fails if any file other than `contracts.py` and
`risk_agent.py` mentions it.

That test is the actual control. The exception is just what happens when
someone ignores it.

---

## 3. Signal Agent

**Owns:** short-horizon price move probability, statistical arbitrage
residuals, order book microstructure features.

**Emits:** `Intent(symbol, side, edge_bps, horizon_s, confidence, features)`.

**Never sees:** inventory, P&L, limits, other agents' state.

`edge_bps` is the **gross** expected edge before costs and before adverse
selection. The risk agent subtracts both. A signal agent reporting an edge net
of costs it cannot observe would be reporting a number nobody can check.

Feature families, all in `microstructure/`:

| Feature | Horizon | What it says |
|---|---|---|
| Queue imbalance | ms–s | Direction of the next tick |
| Microprice − mid | ms–s | Where fair value sits inside the spread |
| Order flow imbalance | s | Whether the touch is being consumed |
| Cointegration residual | min–h | Stat-arb dislocation vs a hedge basket |
| VPIN | min–h | Whether making markets is viable at all right now |

No MACD, no RSI. Those are functions of a price series that discard the book,
and the book is where the information at these horizons lives.

---

## 4. Risk & Monitoring Agent

*(Full implementation: `src/titan/agents/risk_agent.py`; tests:
`tests/test_risk_agent.py`)*

Four properties, each a design decision rather than an implementation detail:

**Fail closed.** Anything unevaluable is a rejection — a missing book, an
unmeasurable VPIN, an exception inside a check. The opposite convention
(unknown means fine) fails exactly when measurement breaks, which is
correlated with the market doing something unusual. `ToxicityState` fields
default to `nan` and `nan` rejects.

**Risk sizes, not signal.** Base size comes from the agent's own budget model
scaled by the intent's confidence, then every check caps it.

**Hard limits reject; soft limits scale.** A check either fails outright or
returns a multiplicative `scale` in `[0, 1]`. Composition is a single `min`,
so the binding constraint is always identifiable *by name* — which is the
thing you need at 4pm, not a total.

**Every decision is recorded whole**, including the checks that passed. The
interesting question after a bad day is never "what rejected this" but "what
nearly did".

### Check order

Cheap and absolute first, so a dead intent never reaches a costly estimator.

| # | Check | Rejects on | Scales on |
|---|---|---|---|
| 1 | `kill_switch` | halted symbol; session drawdown ≥ kill | drawdown between warn and kill |
| 2 | `book_integrity` | crossed, locked, empty, unordered, stale, spread blowout | — |
| 3 | `toxicity` | VPIN unmeasurable or over limit; OFI against the intent | VPIN in the top half of tolerance |
| 4 | `edge_vs_costs` | impact unmeasurable; edge < half-spread + impact + floor | — |
| 5 | `position_limits` | per-symbol / gross / net exhausted | room remaining |
| 6 | `participation` | no size at the far touch | fraction of the far touch |

Then: if the resulting notional is below `min_order_notional`, reject as dust
— a fee and a queue cost against an edge that no longer covers them.

Two behaviours worth calling out because they are easy to get wrong:

- **A position at its cap can still be reduced.** `position_limits` checks
  whether the order increases exposure before applying the per-symbol ceiling.
  Without that, limits create traps: a book at its limit cannot be closed.
- **The toxicity taper starts at half the VPIN limit, not at zero.** A haircut
  beginning at VPIN 0 charges every order for toxicity that is merely normal,
  and makes `APPROVED` unreachable — the disposition stops distinguishing a
  routine fill from one a limit actually bound.

---

## 5. Execution Agent

**Owns:** child order scheduling, venue routing, passive/aggressive choice,
and measuring its own fill quality.

**Accepts:** `ClearedOrder`, and nothing else.

### The evaluation loop

*(Full implementation: `src/titan/execution/evaluation.py`; tests:
`tests/test_execution_evaluation.py`)*

Everything rests on one identity:

```
gross spread capture  =  realized spread  +  adverse selection
```

You earn the spread at the fill. The mid then moves, and the part of that move
running against your new position is handed back. What remains is realized
spread — the only one of the three that is P&L.

That is why "we captured 1.8bps" is not a result. A desk capturing 1.8bps and
paying 2.4bps of adverse selection is losing money while every spread-capture
dashboard it owns shows green. `EvaluationReport.verdict()` therefore reads
the sign of the **net**, never the capture.

Per fill, with `sign = +1` for a buy:

```
gross_capture_bps    = 1e4 * sign * (mid_at_fill - fill_price) / mid_at_fill
adverse_selection_bps = -1e4 * sign * (mid_at_horizon - mid_at_fill) / mid_at_fill
realized_spread_bps  = gross_capture - adverse_selection
net_bps              = realized_spread - fee_bps          # rebate is a negative fee
```

`test_capture_equals_realized_plus_adverse_selection` asserts the identity
across both sides and five drift scenarios. If it ever fails, every number the
evaluator reports is describing something other than the trade.

### Reading the markout curve

Horizons are the diagnostic, which is why the report keeps them separate
rather than collapsing to one number:

| Adverse selection at 1s | at 60s+ | Diagnosis | Fix |
|---|---|---|---|
| High | Low | Picked off by faster participants, price reverts | Queue position, latency, quote wider on thin books |
| Low | High | Quoting against genuine information | Signal problem — the alpha is real and it is not yours |
| High | High | Both | Stop quoting this name |

### Two aggregation rules that are not incidental

- **Notional weighting.** An equal-weighted mean of per-fill bps is the classic
  way to report a profitable desk that is losing money: losses arrive in size,
  gains in odd lots. `test_aggregates_are_notional_weighted` constructs exactly
  that case — nine clean 1-lots and one 1000-lot that gets run over — and
  asserts the weighted net is negative while the median stays positive.
- **Unmeasured is not zero.** A horizon reaching past the end of the quote data
  is `measured=False`, excluded from aggregates, and counted in
  `n_unmeasured`. Averaging it as zero biases toxicity toward zero, which is
  the direction that flatters.

---

## 6. Suggest-only

*(`src/titan/agents/governance.py`; tests: `tests/test_governance.py`)*

No agent process may write to a path a running system reads. As a comment that
lasts until someone is in a hurry; as `ProposalGate` it survives contact.

```python
gate = ProposalGate(repo_root)
gate.propose(
    "widen quotes above VPIN 0.5",
    rationale="Adverse selection exceeds capture above VPIN 0.5 in shadow.",
    configs={"configs/live.yaml": proposed_yaml},   # staged, NOT written there
    evidence={"decision_horizon_s": 30, "toxicity_ratio": 1.3},
    artifacts={"markout_curve": "artifacts/markouts.png"},
)
```

- Writes outside the proposal directory raise `ProtectedPathError`.
- `configs/`, `models_store/`, `src/` and `.github/` are protected by prefix.
- `..` traversal is resolved *before* the check, not after.
- The `configs` mapping is metadata recording where a human would apply the
  file. Nothing is written to the live path.
- `TITAN_AGENT_MODE` defaults to suggest-only: an agent that cannot tell what
  mode it is in must assume it is the restricted one.

**Turning a proposal into a pull request is deliberately not automated here.**
It belongs in CI under a human-held credential. An agent that can open and
merge its own pull request has a governance diagram, not governance.

The generated `RATIONALE.md` carries a reviewer checklist:

- [ ] Shadow results cover a period containing at least one stress day
- [ ] Risk limits are within the signed-off envelope
- [ ] Markouts positive at the decision horizon, net of fees
- [ ] The change was not fitted to the period it is evaluated on

---

## 7. Shadow mode

Three stages, and nothing skips one:

| Stage | Capital | Orders | Question it answers |
|---|---|---|---|
| **Replay** | none | none | Do the estimators reproduce on recorded books? |
| **Shadow** | none | proposed, never sent | Would these quotes have been hit, and at what markout? |
| **Live** | yes | sent, size-capped | Does the fill rate survive being a participant? |

Shadow's honest limitation, stated because it determines how much the numbers
are worth: **a shadow quote does not change the book.** Fill estimation from
queue position and observed trades is an assumption, not a measurement, and it
is optimistic — a real resting order alters what other participants do. So
shadow results bound the *upside*. A strategy unprofitable in shadow is
definitively unprofitable; one profitable in shadow is a candidate.

Promotion out of shadow requires positive net bps at the decision horizon
after fees, across a period containing at least one stress day, with the
markout curve flat or improving from 1s to 5min.

---

## 8. Worktree isolation

The three research loops have incompatible working states — the signal loop
rewrites features, the risk loop rewrites limits, the execution loop replays
tapes. Sharing one checkout means one agent's uncommitted edit changes another
agent's result, and neither notices.

```bash
git worktree add ../titan-signal   -b agent/signal
git worktree add ../titan-risk     -b agent/risk
git worktree add ../titan-exec     -b agent/execution
```

```
trading-bot/            main            human, review and merge only
../titan-signal/        agent/signal    feature research; owns features/, signals/
../titan-risk/          agent/risk      limit calibration; owns agents/risk_agent.py
../titan-exec/          agent/execution routing + markouts; owns execution/
```

Rules that make this work:

1. **One agent, one worktree, one branch.** Never two agents in a checkout.
2. **Ownership is by directory.** An agent editing outside its directories is
   a review flag — the module boundaries in §1 exist so this is checkable.
3. **Artifacts are per-worktree** (`artifacts/` is gitignored). A shared
   artifacts directory silently mixes three agents' runs.
4. **The data cache is shared and read-only** — symlink `data_cache/` into
   each worktree. It is large, immutable, and re-downloading it three times
   wastes the rate limit that actually binds.
5. **Merge to main is human.** Each agent lands via a proposal and a pull
   request; `main` stays the only branch a live process reads.

---

## 9. What is built, and what is not

**Built and tested (83 tests):**

- `microstructure/book.py`, `microstructure/toxicity.py` — book state, OFI,
  VPIN, Kyle's lambda
- `agents/contracts.py` — the boundary, with the mint enforced by test
- `agents/risk_agent.py` — all six checks, sizing, audit log
- `agents/governance.py` — `ProposalGate`
- `execution/evaluation.py` — markout decomposition, aggregation, bucketing

**Not built:**

- `signal_agent.py` — only the protocol is settled; the reference stat-arb
  implementation is the next piece
- `execution/slicing.py` — child order scheduling and venue routing
- `shadow/runner.py` — the session replayer
- **A live feed.** Nothing above runs on Yahoo bars, and no amount of code
  changes that.
