"""Market scanner: rank the whole universe on the latest bar.

The scanner reuses the most recently validated fold artifacts (ensemble,
selected features, explainer, analogue index) — it never trains anything
itself, so a scan can run at data frequency with negligible latency. Every
instrument gets a row: actionable signals ranked by confidence at the top,
and for everything else the explicit reason it was rejected (below threshold,
hostile regime, uncertainty, negative EV, sizing). Surfacing the rejects and
their reasons is deliberate: a scanner that only shows winners teaches its
operator nothing about the gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from titan.core.config import TitanConfig
from titan.core.log import get_logger
from titan.core.types import Regime
from titan.data.store import MarketDataset
from titan.explain.evidence import LocalExplainer
from titan.features.registry import FeaturePanel
from titan.models.ensemble import CalibratedEnsemble
from titan.regime.detector import RegimeDetector, RegimeSnapshot
from titan.risk.leverage import LeverageTerms
from titan.signals.analogues import AnalogueIndex
from titan.signals.generator import SignalGenerator
from titan.signals.schema import Signal

logger = get_logger(__name__)


def _fixed_band(b: tuple[float, float]) -> Callable[[], tuple[float, float]]:
    def provide() -> tuple[float, float]:
        return b

    return provide


@dataclass(slots=True)
class ScanRow:
    symbol: str
    probability: float
    uncertainty: float
    status: str          # "signal" or the rejection reason
    asset_class: str = "equity"
    probability_low: float | None = None   # Venn-ABERS band, when available
    probability_high: float | None = None
    confidence: float = 0.0
    grade: str = ""
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "probability": round(self.probability, 4),
            "probability_low": None if self.probability_low is None else round(self.probability_low, 4),
            "probability_high": None if self.probability_high is None else round(self.probability_high, 4),
            "uncertainty": round(self.uncertainty, 4),
            "status": self.status,
            "confidence": round(self.confidence, 1),
            "grade": self.grade,
        }


@dataclass(slots=True)
class ScanResult:
    date: pd.Timestamp
    regime: RegimeSnapshot
    rows: list[ScanRow] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)

    def signals_by_asset_class(self) -> dict[str, list[Signal]]:
        out: dict[str, list[Signal]] = {}
        for s in self.signals:
            out.setdefault(s.asset_class, []).append(s)
        return out

    @property
    def portfolio_heat(self) -> float:
        """Summed risk-at-stop across the emitted orders, in percent of equity.

        Reported, not enforced. Leverage makes this worth looking at: five
        crypto orders at 3x carry three times the heat the same five would
        unlevered, and the per-position risk figure gives no hint of the total.
        The scanner's job is to say what it found and what it would cost; how
        many of those orders to actually place is the operator's call, and the
        account applies its own funding ceilings besides.
        """
        return sum(s.risk_percentage for s in self.signals)

    def to_dict(self) -> dict:
        return {
            "date": str(self.date.date()),
            "regime": self.regime.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "signals": [s.to_dict() for s in self.signals],
            "orders_by_asset_class": {
                k: len(v) for k, v in sorted(self.signals_by_asset_class().items())
            },
            "portfolio_heat_pct": round(self.portfolio_heat, 3),
            "gross_notional_pct": round(
                100.0 * sum(s.position_size_fraction for s in self.signals), 2
            ),
            "margin_required_pct": round(
                100.0 * sum(s.margin_fraction or s.position_size_fraction
                            for s in self.signals), 2
            ),
        }


class MarketScanner:
    def __init__(
        self,
        cfg: TitanConfig,
        ensemble: CalibratedEnsemble,
        selected_features: list[str],
        generator: SignalGenerator,
        detector: RegimeDetector,
        explainer: LocalExplainer | None = None,
        analogues: AnalogueIndex | None = None,
    ) -> None:
        self._cfg = cfg
        self._ensemble = ensemble
        self._selected = selected_features
        self._generator = generator
        self._detector = detector
        self._explainer = explainer
        self._analogues = analogues
        self._terms_cache: dict[str, LeverageTerms] = {}

    # ------------------------------------------------------------------ #

    def _asset_class(self, dataset: MarketDataset, symbol: str) -> str:
        """The symbol's class, defaulting to equity for anything unregistered."""
        try:
            return str(dataset.universe.instrument(symbol).asset_class)
        except KeyError:
            return "equity"

    def _leverage_terms(self, asset_class: str) -> LeverageTerms:
        if asset_class not in self._terms_cache:
            self._terms_cache[asset_class] = LeverageTerms.resolve(
                self._cfg.risk.leverage, asset_class, self._cfg.data.timeframe
            )
        return self._terms_cache[asset_class]

    # ------------------------------------------------------------------ #

    def scan(self, dataset: MarketDataset, panel: FeaturePanel) -> ScanResult:
        """Rank every instrument on ITS OWN most recent bar.

        Scanning one global date silently drops every instrument that does not
        trade on it. In a mixed equity/crypto universe the newest panel date is
        routinely a crypto-only day — a weekend, or simply a fresher print —
        and the entire equity book vanishes from the scan without a word. An
        equity's Friday close is not stale on a Saturday; it is that
        instrument's current state.

        Instruments lagging further than ``scanner.max_staleness_bars`` panel
        dates ARE reported as stale rather than ranked, so a delisted or halted
        symbol cannot produce a signal from months-old prices.
        """
        dates = panel.dates()
        last_date = dates[-1]
        snapshot = self._detector.snapshot(dataset.benchmark_frame.loc[:last_date])

        # Each symbol's own freshest row, and how far behind the panel it is.
        symbols = panel.X.index.get_level_values(1)
        own_date = panel.X.groupby(symbols, observed=True).apply(
            lambda g: g.index.get_level_values(0).max()
        )
        date_rank = {d: i for i, d in enumerate(dates)}
        lag = {sym: len(dates) - 1 - date_rank[d] for sym, d in own_date.items()}

        fresh = [s for s in own_date.index if lag[s] <= self._cfg.scanner.max_staleness_bars]
        stale = [s for s in own_date.index if lag[s] > self._cfg.scanner.max_staleness_bars]
        if stale:
            logger.info(
                "scan: %d instrument(s) stale beyond %d bars and not ranked: %s",
                len(stale), self._cfg.scanner.max_staleness_bars,
                ", ".join(map(str, stale[:8])) + ("..." if len(stale) > 8 else ""),
            )

        X_last = pd.DataFrame(
            [panel.X.loc[(own_date[s], s)] for s in fresh], index=pd.Index(fresh)
        )
        X_sel = X_last.reindex(columns=self._selected)
        probs = self._ensemble.predict_proba(X_sel)[:, 1]
        uncs = self._ensemble.uncertainty(X_sel)
        # A scan is a handful of rows, so every instrument gets its band —
        # older registry bundles saved before interval support degrade to None.
        bands = self._ensemble.probability_interval(X_sel) if self._ensemble.has_intervals else None

        rows: list[ScanRow] = []
        signals: list[Signal] = []
        for i, sym in enumerate(X_sel.index):
            p, unc = float(probs[i]), float(uncs[i])
            band: tuple[float, float] | None = (
                (float(bands[i][0]), float(bands[i][1])) if bands is not None else None
            )
            asset_class = self._asset_class(dataset, str(sym))
            row = ScanRow(
                symbol=str(sym), probability=p, uncertainty=unc, status="",
                asset_class=asset_class,
                probability_low=band[0] if band else None,
                probability_high=band[1] if band else None,
            )
            if snapshot.regime is Regime.CRASH:
                row.status = "blocked: crash regime"
            elif p < self._cfg.signals.min_probability:
                row.status = f"below probability floor ({self._cfg.signals.min_probability:.2f})"
            elif unc > self._cfg.signals.max_uncertainty:
                row.status = "model disagreement too high"
            else:
                signal = self._generator.generate(
                    symbol=str(sym),
                    date=own_date[sym],
                    probability=p,
                    uncertainty=unc,
                    frame=dataset.frames[str(sym)].loc[: own_date[sym]].tail(60),
                    feature_row=X_sel.loc[sym],
                    regime=snapshot.regime,
                    vol_state=snapshot.vol_state,
                    regime_confidence=snapshot.confidence,
                    analogue=self._analogues.query(X_sel.loc[sym]) if self._analogues else None,
                    reliability=float(dataset.reliability.get(str(sym), 1.0)),
                    explainer=self._explainer,
                    model_version="scanner",
                    interval_provider=_fixed_band(band) if band is not None else None,
                    asset_class=asset_class,
                    leverage_terms=self._leverage_terms(asset_class),
                )
                if signal is None:
                    row.status = "failed adaptive EV gate"
                else:
                    row.status = "signal"
                    row.confidence = signal.confidence_score
                    row.grade = signal.trade_grade.value
                    signals.append(signal)
            rows.append(row)

        for sym in stale:
            rows.append(ScanRow(
                symbol=str(sym), probability=float("nan"), uncertainty=float("nan"),
                asset_class=self._asset_class(dataset, str(sym)),
                status=f"stale: last bar {own_date[sym].date()} ({lag[sym]} bars behind)",
            ))

        # ---- second order source: rule-based swing setups ----------------
        # These do not consult the model. They fire on price structure and are
        # labelled with the rule that produced them, so the ledger can later
        # say whether the rules or the model made the money.
        tech_cfg = self._cfg.signals.technical
        n_tech = 0
        if tech_cfg.enabled and not (
            tech_cfg.skip_in_crash and snapshot.regime is Regime.CRASH
        ):
            tech_signals = self._technical_signals(dataset, fresh, own_date, snapshot)
            n_tech = len(tech_signals)
            signals.extend(tech_signals)
            emitted = {s.symbol for s in tech_signals}
            for row in rows:
                if row.symbol in emitted and row.status != "signal":
                    setup = next(
                        s.source.split(":", 1)[1]
                        for s in tech_signals if s.symbol == row.symbol
                    )
                    row.status = f"signal ({setup})"
                    row.grade = next(
                        s.trade_grade.value for s in tech_signals if s.symbol == row.symbol
                    )
                    row.confidence = next(
                        s.confidence_score for s in tech_signals if s.symbol == row.symbol
                    )

        rows.sort(key=lambda r: (not r.status.startswith("signal"),
                                 -r.confidence, -r.probability))
        for rank, row in enumerate(rows, start=1):
            row.rank = rank

        selected = self._select(signals)
        result = ScanResult(date=last_date, regime=snapshot, rows=rows, signals=selected)
        by_class = ", ".join(
            f"{k} {len(v)}" for k, v in sorted(result.signals_by_asset_class().items())
        )
        logger.info(
            "scan %s: %d instruments (%d ranked, %d stale), %d candidates "
            "(%d model, %d technical) -> %d orders [%s] (regime=%s)",
            last_date.date(), len(rows), len(fresh), len(stale), len(signals),
            len(signals) - n_tech, n_tech, len(selected), by_class or "none",
            snapshot.regime.value,
        )
        return result

    # ------------------------------------------------------------------ #

    def _select(self, candidates: list[Signal]) -> list[Signal]:
        """Rank candidates within each asset class and fill that class's quota.

        Two rules, both about not deceiving the reader of the order list:

        One order per symbol. The model gate and the rule engine can fire on
        the same name on the same bar, and shipping both would put two tickets
        on one instrument — double the intended size, from what looks like two
        independent ideas but is one. The model order wins the collision: it is
        the one with out-of-sample evidence behind it.

        Quotas are per asset class, not global. Crypto and equities differ by
        an order of magnitude in volatility, so a single ranked list is not a
        fair fight — the louder class takes every slot, and the account ends up
        concentrated in whichever one happened to be moving.
        """
        best: dict[str, Signal] = {}
        for s in sorted(candidates, key=lambda s: -s.confidence_score):
            prior = best.get(s.symbol)
            if prior is None or (prior.source != "model" and s.source == "model"):
                best[s.symbol] = s

        by_class: dict[str, list[Signal]] = {}
        for s in best.values():
            by_class.setdefault(s.asset_class, []).append(s)

        out: list[Signal] = []
        for asset_class, group in by_class.items():
            group.sort(key=lambda s: -s.confidence_score)
            out.extend(group[: self._cfg.scanner.quota_for(asset_class)])
        out.sort(key=lambda s: (s.asset_class, -s.confidence_score))
        return out

    # ------------------------------------------------------------------ #

    def _technical_signals(self, dataset, symbols, own_date, snapshot) -> list[Signal]:
        """Fire the rule set over each instrument's own freshest bar.

        Ranked by setup strength and capped PER ASSET CLASS, because a trending
        day fires dozens across a large universe and an account that took them
        all would be fully committed to a single day's worth of patterns. The
        cap is per class so a broad equity rally cannot crowd crypto out of the
        list, or the reverse.
        """
        from titan.signals.technical import detect, to_signal

        cfg = self._cfg.signals.technical
        candidates: list[tuple[float, Signal]] = []
        for sym in symbols:
            frame = dataset.frames.get(str(sym))
            if frame is None:
                continue
            window = frame.loc[: own_date[sym]]
            asset_class = self._asset_class(dataset, str(sym))
            for setup in detect(window, cfg.setups, min_risk_reward=cfg.min_risk_reward):
                signal = to_signal(
                    setup,
                    symbol=str(sym),
                    date=own_date[sym],
                    frame=window,
                    risk_cfg=self._cfg.risk,
                    cost_model=self._generator._costs,
                    regime=snapshot.regime,
                    vol_state=snapshot.vol_state,
                    regime_confidence=snapshot.confidence,
                    reliability=float(dataset.reliability.get(str(sym), 1.0)),
                    asset_class=asset_class,
                    leverage_terms=self._leverage_terms(asset_class),
                )
                if signal is not None:
                    candidates.append((setup.strength, signal))

        candidates.sort(key=lambda c: -c[0])
        # One order per symbol: two rules firing on the same name is one idea.
        seen: set[str] = set()
        taken: dict[str, int] = {}
        out: list[Signal] = []
        for _, signal in candidates:
            if signal.symbol in seen:
                continue
            if taken.get(signal.asset_class, 0) >= cfg.max_orders_per_scan:
                continue
            seen.add(signal.symbol)
            taken[signal.asset_class] = taken.get(signal.asset_class, 0) + 1
            out.append(signal)
        return out
