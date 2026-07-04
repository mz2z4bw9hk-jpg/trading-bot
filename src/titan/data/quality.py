"""Data quality control and reliability scoring.

Every data source is scored before use. The composite reliability score in
``[0, 1]`` propagates downstream: instruments below ``data.min_reliability``
are excluded from research, and signal confidence is discounted by the score
of the data that produced it. Garbage in, nothing out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DataQualityReport:
    symbol: str
    n_bars: int
    checks: dict[str, float] = field(default_factory=dict)  # check -> subscore in [0,1]
    issues: list[str] = field(default_factory=list)
    reliability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_bars": self.n_bars,
            "reliability": round(self.reliability, 4),
            "checks": {k: round(v, 4) for k, v in self.checks.items()},
            "issues": list(self.issues),
        }


def _linear_penalty(value: float, ok_at: float, zero_at: float) -> float:
    """Map ``value`` to a [0,1] subscore: 1.0 up to ``ok_at``, 0.0 at ``zero_at``."""
    if zero_at <= ok_at:
        raise ValueError("zero_at must exceed ok_at")
    if value <= ok_at:
        return 1.0
    if value >= zero_at:
        return 0.0
    return float(1.0 - (value - ok_at) / (zero_at - ok_at))


def assess_quality(symbol: str, df: pd.DataFrame, min_bars: int = 400) -> DataQualityReport:
    """Run the QC battery on a canonical OHLCV frame and score reliability.

    Subscores (each in [0,1]) are combined geometrically so that a single
    catastrophic dimension collapses the composite score — a series with
    perfect volume data but 30% stale closes is still unusable.
    """
    report = DataQualityReport(symbol=symbol, n_bars=len(df))
    checks = report.checks

    # -- history depth ------------------------------------------------------
    checks["history"] = min(1.0, len(df) / max(min_bars, 1))
    if len(df) < min_bars:
        report.issues.append(f"only {len(df)} bars (< {min_bars})")

    # -- OHLC internal consistency ------------------------------------------
    bad_hl = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12) | (
        df["low"] > df[["open", "close"]].min(axis=1) + 1e-12
    )
    frac_bad = float(bad_hl.mean())
    checks["ohlc_consistency"] = _linear_penalty(frac_bad, ok_at=0.0, zero_at=0.02)
    if frac_bad > 0:
        report.issues.append(f"{frac_bad:.2%} bars violate OHLC bounds")

    # -- non-positive prices -------------------------------------------------
    nonpos = float((df[["open", "high", "low", "close"]] <= 0).any(axis=1).mean())
    checks["positive_prices"] = 1.0 if nonpos == 0 else 0.0
    if nonpos > 0:
        report.issues.append(f"{nonpos:.2%} bars with non-positive prices")

    # -- staleness (repeated closes suggest fill-forward or halted data) -----
    stale = float((df["close"].diff() == 0).mean()) if len(df) > 1 else 0.0
    checks["staleness"] = _linear_penalty(stale, ok_at=0.05, zero_at=0.40)
    if stale > 0.05:
        report.issues.append(f"{stale:.2%} stale closes")

    # -- extreme returns (data errors, not crashes: robust 15-MAD threshold) --
    log_ret = np.log(df["close"]).diff().dropna()
    if len(log_ret) > 30:
        mad = float((log_ret - log_ret.median()).abs().median())
        robust_sigma = max(1.4826 * mad, 1e-6)
        frac_extreme = float((log_ret.abs() > 15 * robust_sigma).mean())
    else:
        frac_extreme = 0.0
    checks["return_outliers"] = _linear_penalty(frac_extreme, ok_at=0.0005, zero_at=0.02)
    if frac_extreme > 0.0005:
        report.issues.append(f"{frac_extreme:.3%} returns beyond 15 robust sigmas")

    # -- gaps in the calendar -------------------------------------------------
    if len(df) > 2:
        spacing = df.index.to_series().diff().dropna()
        median_step = spacing.median()
        big_gaps = float((spacing > 5 * median_step).mean())
    else:
        big_gaps = 0.0
    checks["calendar_gaps"] = _linear_penalty(big_gaps, ok_at=0.002, zero_at=0.05)
    if big_gaps > 0.002:
        report.issues.append(f"{big_gaps:.2%} large calendar gaps")

    # -- volume sanity ---------------------------------------------------------
    zero_vol = float((df["volume"] <= 0).mean())
    checks["volume"] = _linear_penalty(zero_vol, ok_at=0.02, zero_at=0.50)
    if zero_vol > 0.02:
        report.issues.append(f"{zero_vol:.2%} zero-volume bars")

    scores = np.array(list(checks.values()), dtype=float)
    report.reliability = float(np.exp(np.log(np.clip(scores, 1e-6, 1.0)).mean()))
    return report
