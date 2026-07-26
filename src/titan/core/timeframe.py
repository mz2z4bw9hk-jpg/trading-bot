"""Bar interval: what one row of the panel means in wall-clock time.

Every annualized number the platform reports — Sharpe, CAGR, annual vol, the
vol-targeting divisor, the regime detector's trend thresholds — converts a
per-bar quantity into a per-year one by multiplying by bars-per-year. On daily
bars that constant is 252 and it is safe to hard-code. On any other interval a
hard-coded 252 does not produce a slightly-off number, it produces a wrong one:
hourly bars annualize at ~6.5x the rate, so a Sharpe printed as 2.5 is really
2.5/sqrt(6.5) = 0.98, and vol-targeted position sizes come out ~2.5x too large.

Two conventions coexist and differ by more than 3x at the same interval:
markets with a 6.5-hour session (US equities) and markets that never close
(crypto). The resolver below picks by universe composition and warns when the
loaded data disagrees with the choice.

What this module does NOT do is make sub-15-minute research honest — see
``UNVALIDATED_TIMEFRAMES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from titan.core.log import get_logger

logger = get_logger(__name__)

Timeframe = Literal["1wk", "1d", "4h", "3h", "2h", "1h", "30m", "15m", "5m", "1m"]

TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0

# Bars per year on a 6.5-hour US equity session.
_SESSION_BARS_PER_YEAR: dict[str, float] = {
    "1wk": 52.0,
    "1d": TRADING_DAYS_PER_YEAR,
    "4h": TRADING_DAYS_PER_YEAR * 2,
    "3h": TRADING_DAYS_PER_YEAR * 2.5,
    "2h": TRADING_DAYS_PER_YEAR * 3.25,
    "1h": TRADING_DAYS_PER_YEAR * 6.5,
    "30m": TRADING_DAYS_PER_YEAR * 13,
    "15m": TRADING_DAYS_PER_YEAR * 26,
    "5m": TRADING_DAYS_PER_YEAR * 78,
    "1m": TRADING_DAYS_PER_YEAR * 390,
}

# Bars per year on a market that never closes.
_CONTINUOUS_BARS_PER_YEAR: dict[str, float] = {
    "1wk": 52.0,
    "1d": CALENDAR_DAYS_PER_YEAR,
    "4h": CALENDAR_DAYS_PER_YEAR * 6,
    "3h": CALENDAR_DAYS_PER_YEAR * 8,
    "2h": CALENDAR_DAYS_PER_YEAR * 12,
    "1h": CALENDAR_DAYS_PER_YEAR * 24,
    "30m": CALENDAR_DAYS_PER_YEAR * 48,
    "15m": CALENDAR_DAYS_PER_YEAR * 96,
    "5m": CALENDAR_DAYS_PER_YEAR * 288,
    "1m": CALENDAR_DAYS_PER_YEAR * 1440,
}

# yfinance interval string per timeframe. Yahoo serves no multi-hour bar:
# 2h/3h/4h are reachable only by aggregating 1h via data.resample_from.
YAHOO_INTERVALS: dict[str, str] = {
    "1wk": "1wk",
    "1d": "1d",
    "1h": "1h",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "1m": "1m",
}

# Yahoo's hard cap on how far back each interval reaches, in calendar days.
# These are vendor limits, not preferences: asking for more silently returns
# less, which is how an intraday run ends up validating on three weeks of data
# while believing it used ten years.
YAHOO_MAX_HISTORY_DAYS: dict[str, int] = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": 0,  # 0 = no limit
    "1wk": 0,
}

# Intervals at which this platform cannot honestly certify a result, and why.
# The machinery runs; the evidence does not meet the platform's own bar:
#
#  - The engine fills at the NEXT BAR'S OPEN. Over a day that is a reasonable
#    model of a market-on-open order. Over a minute it is fiction — the price
#    you are modelled as receiving is not one a retail order reliably gets.
#  - The cost model is a fixed spread plus a square-root impact term, which is
#    calibrated for daily turnover. At one-minute frequency the spread IS the
#    signal: a 1 bp per-bar edge is entirely consumed by a 3 bp round trip.
#  - Sub-minute price formation is driven by order-book state that OHLCV bars
#    do not contain, so the features cannot see the thing that moves the price.
#
# Refused unless the config sets ``data.acknowledge_unvalidated_timeframe``.
UNVALIDATED_TIMEFRAMES: frozenset[str] = frozenset({"1m", "5m"})

# Intervals that run, but where the above pressures are already visible.
CAUTION_TIMEFRAMES: frozenset[str] = frozenset({"15m", "30m"})


@dataclass(frozen=True)
class BarClock:
    """Resolved bar interval plus the annualization constant it implies."""

    timeframe: str
    bars_per_year: float
    continuous: bool

    @property
    def periods_per_year(self) -> float:
        return self.bars_per_year


def resolve_bars_per_year(
    timeframe: str,
    *,
    continuous: bool,
    override: float | None = None,
) -> float:
    """Bars per year for ``timeframe``; ``override`` wins when supplied."""
    if override is not None:
        return float(override)
    table = _CONTINUOUS_BARS_PER_YEAR if continuous else _SESSION_BARS_PER_YEAR
    if timeframe not in table:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    return table[timeframe]


def session_bars_per_day(timeframe: str) -> float:
    """Bars in one 6.5-hour session — what a vendor day-count request needs."""
    if timeframe not in _SESSION_BARS_PER_YEAR:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    return _SESSION_BARS_PER_YEAR[timeframe] / TRADING_DAYS_PER_YEAR


def empirical_bars_per_year(index: pd.DatetimeIndex) -> float | None:
    """Bars per year implied by the data itself, or ``None`` if too short.

    Counting bars per unit of calendar span (rather than inverting the median
    gap) is what makes this correct across weekends and holidays: 252 daily
    equity bars spanning one year give 252, not 365.
    """
    if len(index) < 30:
        return None
    span_days = (index[-1] - index[0]).total_seconds() / 86400.0
    if span_days < 20:
        return None
    return len(index) / (span_days / CALENDAR_DAYS_PER_YEAR)


def warn_on_calendar_mismatch(clock: BarClock, index: pd.DatetimeIndex) -> None:
    """Log when the loaded calendar disagrees with the assumed convention.

    The usual cause is a 24/7 universe annualized on a session convention (or
    the reverse), which scales every Sharpe by a constant and every
    vol-targeted position size with it.
    """
    observed = empirical_bars_per_year(index)
    if observed is None:
        return
    ratio = observed / clock.bars_per_year
    if 0.75 <= ratio <= 1.33:
        return
    logger.warning(
        "calendar mismatch: %s bars assumed %.0f/year but the data implies "
        "%.0f/year (%.2fx). Every annualized figure is off by sqrt(%.2f)=%.2fx. "
        "Set data.bars_per_year to override.",
        clock.timeframe, clock.bars_per_year, observed, ratio, ratio, ratio**0.5,
    )


# Bars finer than one day. On a market that closes, these carry session
# boundaries that calendar-naive checks misread as missing data.
INTRADAY_TIMEFRAMES: frozenset[str] = frozenset(
    {"4h", "3h", "2h", "1h", "30m", "15m", "5m", "1m"}
)


# Ordering from coarsest to finest, for validating that a resample source
# is genuinely finer than its target.
_ORDER: tuple[str, ...] = (
    "1wk", "1d", "4h", "3h", "2h", "1h", "30m", "15m", "5m", "1m",
)


def is_finer(source: str, target: str) -> bool:
    """True when ``source`` bars are shorter than ``target`` bars."""
    if source not in _ORDER or target not in _ORDER:
        raise ValueError(f"unknown timeframe in ({source!r}, {target!r})")
    return _ORDER.index(source) > _ORDER.index(target)
