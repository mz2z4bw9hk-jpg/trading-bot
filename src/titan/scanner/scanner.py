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
    probability_low: float | None = None   # Venn-ABERS band, when available
    probability_high: float | None = None
    confidence: float = 0.0
    grade: str = ""
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "symbol": self.symbol,
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

    def to_dict(self) -> dict:
        return {
            "date": str(self.date.date()),
            "regime": self.regime.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "signals": [s.to_dict() for s in self.signals],
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
            row = ScanRow(
                symbol=str(sym), probability=p, uncertainty=unc, status="",
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
                status=f"stale: last bar {own_date[sym].date()} ({lag[sym]} bars behind)",
            ))

        rows.sort(key=lambda r: (r.status != "signal", -r.confidence, -r.probability))
        for rank, row in enumerate(rows, start=1):
            row.rank = rank
        signals.sort(key=lambda s: -s.confidence_score)
        top_n = self._cfg.scanner.top_n
        result = ScanResult(
            date=last_date, regime=snapshot, rows=rows, signals=signals[:top_n]
        )
        logger.info(
            "scan %s: %d instruments (%d ranked, %d stale), %d actionable signals "
            "(regime=%s)",
            last_date.date(), len(rows), len(fresh), len(stale), len(signals),
            snapshot.regime.value,
        )
        return result
