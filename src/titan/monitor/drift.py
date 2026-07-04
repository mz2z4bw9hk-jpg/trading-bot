"""Drift detection and live prediction tracking.

Models decay. The platform assumes decay and measures it three ways:

- **Feature drift (PSI)**: population stability index of each model feature
  between its training reference distribution and the live window. PSI > 0.10
  is a warning, > 0.25 an alert (industry-standard bands): the world the
  model sees no longer resembles the world it learned.
- **Calibration drift**: rolling Brier score and hit-rate-vs-confidence of
  live predictions against subsequent outcomes.
- **CUSUM alarm** on Brier degradation: a one-sided cumulative-sum detector
  that fires when the recent error consistently exceeds the training
  baseline, catching slow rot that a fixed threshold misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from titan.core.log import get_logger

logger = get_logger(__name__)


def population_stability_index(
    reference: np.ndarray | pd.Series,
    live: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """PSI between a reference and a live sample of one feature.

    Bin edges come from reference deciles; both distributions are floored to
    avoid log(0). Identical distributions score ~0.
    """
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    liv = pd.Series(live).dropna().to_numpy(dtype=float)
    if len(ref) < 30 or len(liv) < 30:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: any change is structural
        return 0.0 if np.allclose(np.median(ref), np.median(liv)) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(ref, bins=edges)[0] / len(ref)
    liv_frac = np.histogram(liv, bins=edges)[0] / len(liv)
    ref_frac = np.clip(ref_frac, 1e-4, None)
    liv_frac = np.clip(liv_frac, 1e-4, None)
    return float(np.sum((liv_frac - ref_frac) * np.log(liv_frac / ref_frac)))


def feature_drift_report(
    X_reference: pd.DataFrame,
    X_live: pd.DataFrame,
    warn: float = 0.10,
    alert: float = 0.25,
) -> dict:
    """Per-feature PSI with warning/alert flags, worst first."""
    rows = {}
    for col in X_reference.columns:
        if col not in X_live.columns:
            continue
        psi = population_stability_index(X_reference[col], X_live[col])
        rows[col] = psi
    series = pd.Series(rows).sort_values(ascending=False)
    alerts = [c for c, v in series.items() if np.isfinite(v) and v >= alert]
    warnings_ = [c for c, v in series.items() if np.isfinite(v) and warn <= v < alert]
    if alerts:
        logger.warning("feature drift ALERT (psi>=%.2f): %s", alert, alerts[:10])
    return {
        "psi": {k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in series.items()},
        "alerts": alerts,
        "warnings": warnings_,
        "max_psi": float(series.max()) if len(series) else float("nan"),
    }


@dataclass(slots=True)
class PredictionRecord:
    date: str
    symbol: str
    probability: float
    outcome: int | None = None  # filled once the event resolves


@dataclass(slots=True)
class PredictionTracker:
    """Append-only log of live predictions with rolling quality metrics."""

    baseline_brier: float
    cusum_k: float = 0.005   # slack per observation before drift accumulates
    cusum_h: float = 0.15    # alarm threshold on the cumulative excess
    records: list[PredictionRecord] = field(default_factory=list)

    def log_prediction(self, date: str, symbol: str, probability: float) -> None:
        self.records.append(PredictionRecord(date=date, symbol=symbol, probability=probability))

    def resolve(self, date: str, symbol: str, outcome: int) -> None:
        for rec in reversed(self.records):
            if rec.date == date and rec.symbol == symbol and rec.outcome is None:
                rec.outcome = int(outcome)
                return
        raise KeyError(f"no open prediction for {symbol}@{date}")

    # ------------------------------------------------------------------ #

    def _resolved(self) -> list[PredictionRecord]:
        return [r for r in self.records if r.outcome is not None]

    def rolling_brier(self, window: int = 100) -> float:
        resolved = self._resolved()[-window:]
        if not resolved:
            return float("nan")
        errors = [
            (r.probability - r.outcome) ** 2 for r in resolved if r.outcome is not None
        ]
        return float(np.mean(errors)) if errors else float("nan")

    def calibration_table(self, n_bins: int = 5) -> list[dict]:
        resolved = self._resolved()
        if len(resolved) < n_bins * 4:
            return []
        p = np.array([r.probability for r in resolved])
        y = np.array([r.outcome for r in resolved], dtype=float)
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        out = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
            if mask.sum() < 3:
                continue
            out.append(
                {
                    "bin": f"[{lo:.2f},{hi:.2f}]",
                    "n": int(mask.sum()),
                    "mean_p": round(float(p[mask].mean()), 4),
                    "hit_rate": round(float(y[mask].mean()), 4),
                }
            )
        return out

    def cusum_alarm(self) -> tuple[bool, float]:
        """One-sided CUSUM on per-prediction Brier excess over baseline."""
        s = 0.0
        for r in self._resolved():
            if r.outcome is None:
                continue
            err = (r.probability - r.outcome) ** 2
            s = max(0.0, s + (err - self.baseline_brier - self.cusum_k))
            if s >= self.cusum_h:
                return True, float(s)
        return False, float(s)

    def summary(self) -> dict:
        alarm, stat = self.cusum_alarm()
        return {
            "n_predictions": len(self.records),
            "n_resolved": len(self._resolved()),
            "rolling_brier_100": self.rolling_brier(100),
            "baseline_brier": self.baseline_brier,
            "cusum_alarm": alarm,
            "cusum_stat": round(stat, 4),
            "calibration": self.calibration_table(),
        }
