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


def _session_continuation(index: pd.DatetimeIndex) -> np.ndarray:
    """Per-bar mask: True where a bar continues the previous bar's session.

    Session boundaries are detected as a change of calendar date, which needs
    no exchange calendar and holds for overnight gaps, weekends and holidays
    alike.
    """
    dates = index.normalize()
    return np.asarray(dates[1:] == dates[:-1])


def _outlier_fraction(log_ret: pd.Series) -> float:
    """Fraction of returns beyond 15 robust sigmas of their own population."""
    if len(log_ret) <= 30:
        return 0.0
    mad = float((log_ret - log_ret.median()).abs().median())
    robust_sigma = max(1.4826 * mad, 1e-6)
    return float((log_ret.abs() > 15 * robust_sigma).mean())


def assess_quality(
    symbol: str,
    df: pd.DataFrame,
    min_bars: int = 400,
    *,
    intraday_sessions: bool = False,
) -> DataQualityReport:
    """Run the QC battery on a canonical OHLCV frame and score reliability.

    Subscores (each in [0,1]) are combined geometrically so that a single
    catastrophic dimension collapses the composite score — a series with
    perfect volume data but 30% stale closes is still unusable.

    ``intraday_sessions`` marks bars finer than a day on a market that closes
    (US equities, not crypto). Two checks are otherwise calendar-naive and
    fail every such series on its normal structure: the overnight boundary
    looks like a dropped-bar gap, and the overnight return looks like a data
    error next to intra-session hourly moves. Both are then evaluated
    session-aware — the defects they hunt for stay detectable, but the
    market's own clock stops counting as one.
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
    # Non-positive closes are already scored above; they must not reach the log.
    # A single zero price yields -inf, which propagates into the median absolute
    # deviation and makes the robust sigma itself infinite — after which the
    # outlier fraction is not a measurement of anything. Sub-penny tokens hit
    # this routinely: Yahoo rounds a $0.00001 quote to exactly 0.
    positive_close = df["close"].where(df["close"] > 0)
    log_ret = np.log(positive_close).diff().replace([np.inf, -np.inf], np.nan).dropna()
    if intraday_sessions and len(log_ret) > 1:
        # Overnight and intra-session returns are different populations: a
        # gap-up open is not an error, it just dwarfs an hourly move. Score
        # each against its own robust sigma so real bad prints in either are
        # still caught.
        cont = _session_continuation(df.index)[-len(log_ret):]
        intra, overnight = log_ret[cont], log_ret[~cont]
        n = len(log_ret)
        frac_extreme = (
            _outlier_fraction(intra) * len(intra) + _outlier_fraction(overnight) * len(overnight)
        ) / max(n, 1)
    else:
        frac_extreme = _outlier_fraction(log_ret)
    checks["return_outliers"] = _linear_penalty(frac_extreme, ok_at=0.0005, zero_at=0.02)
    if frac_extreme > 0.0005:
        report.issues.append(f"{frac_extreme:.3%} returns beyond 15 robust sigmas")

    # -- gaps in the calendar -------------------------------------------------
    # A dropped bar shows up as a hole INSIDE a session; the hole between
    # sessions is the market being closed. On a session market at intraday
    # resolution the latter is every seventh bar of an hourly series, which
    # would fail every symbol on the exchange's opening hours.
    if len(df) > 2:
        spacing = df.index.to_series().diff().dropna()
        if intraday_sessions:
            spacing = spacing[_session_continuation(df.index)]
        if len(spacing) > 1:
            big_gaps = float((spacing > 5 * spacing.median()).mean())
        else:
            big_gaps = 0.0
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
