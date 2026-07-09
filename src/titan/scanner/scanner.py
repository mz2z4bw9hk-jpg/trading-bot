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
        """Rank every instrument on the panel's most recent date."""
        last_date = panel.dates()[-1]
        snapshot = self._detector.snapshot(dataset.benchmark_frame.loc[:last_date])

        X_last = panel.X.loc[last_date]
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
                    date=last_date,
                    probability=p,
                    uncertainty=unc,
                    frame=dataset.frames[str(sym)].loc[:last_date].tail(60),
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

        rows.sort(key=lambda r: (r.status != "signal", -r.confidence, -r.probability))
        for rank, row in enumerate(rows, start=1):
            row.rank = rank
        signals.sort(key=lambda s: -s.confidence_score)
        top_n = self._cfg.scanner.top_n
        result = ScanResult(
            date=last_date, regime=snapshot, rows=rows, signals=signals[:top_n]
        )
        logger.info(
            "scan %s: %d instruments, %d actionable signals (regime=%s)",
            last_date.date(), len(rows), len(signals), snapshot.regime.value,
        )
        return result
