"""Data provider abstraction.

Providers only fetch raw frames; normalization and QC happen in the store.
New sources (order book, options chains, on-chain metrics, ...) plug in by
implementing :class:`DataProvider` and registering in :func:`build_provider`.

Note on scope: this platform runs research on daily/periodic OHLCV bars. The
Yahoo adapter works wherever network egress to Yahoo exists; the CSV adapter
covers any vendor exports; the synthetic provider needs nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from titan.core.config import DataConfig, UniverseConfig
from titan.core.log import get_logger
from titan.core.timeframe import (
    YAHOO_INTERVALS,
    YAHOO_MAX_HISTORY_DAYS,
    session_bars_per_day,
)
from titan.data.schema import SchemaError
from titan.data.synthetic import SyntheticMarket, SyntheticResult

logger = get_logger(__name__)


@runtime_checkable
class DataProvider(Protocol):
    """Fetches raw (pre-normalization) OHLCV frames for symbols."""

    name: str

    def fetch(self, symbol: str, bars: int) -> pd.DataFrame:
        """Return a raw OHLCV frame with at most ``bars`` most-recent rows."""
        ...


class SyntheticProvider:
    """Serves the seeded synthetic market. Deterministic per (universe, seed)."""

    name = "synthetic"

    def __init__(
        self,
        symbols: list[str],
        benchmark: str,
        bars: int,
        seed: int,
        drift_sigma: float | None = None,
    ) -> None:
        self._benchmark = benchmark
        if drift_sigma is None:
            market = SyntheticMarket(symbols=list(symbols), bars=bars, seed=seed)
        else:
            market = SyntheticMarket(
                symbols=list(symbols), bars=bars, seed=seed, drift_sigma=drift_sigma
            )
        self._result: SyntheticResult = market.generate()

    @property
    def result(self) -> SyntheticResult:
        return self._result

    def fetch(self, symbol: str, bars: int) -> pd.DataFrame:
        if symbol == self._benchmark:
            return self._result.index_frame.tail(bars)
        if symbol not in self._result.frames:
            raise KeyError(f"synthetic universe does not contain {symbol}")
        return self._result.frames[symbol].tail(bars)


_TIME_COLUMNS = ("time", "date", "datetime", "timestamp")


def _squash(s: str) -> str:
    """Comparison key: ``BTC-USD`` == ``BTCUSD`` == ``btcusd``."""
    return "".join(ch for ch in s.upper() if ch.isalnum())


def _tv_stem(stem: str) -> str:
    """Reduce a TradingView export filename stem to its ticker.

    ``"BINANCE_BTCUSDT, 1D"`` -> ``"BTCUSDT"``; plain stems pass through.
    """
    stem = stem.split(",")[0].strip()
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return stem


def _set_time_index(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Move the time column to a UTC DatetimeIndex, sorted ascending.

    Handles the formats seen in vendor exports: a named time/date column (or
    the first column), holding ISO strings, epoch seconds (TradingView) or
    epoch milliseconds.
    """
    lower = {str(c).strip().lower(): c for c in df.columns}
    time_col = next((lower[k] for k in _TIME_COLUMNS if k in lower), df.columns[0])
    ts = df[time_col]
    try:
        if pd.api.types.is_numeric_dtype(ts):
            # Epoch magnitude discriminates the unit: dates this century are
            # ~1e9 s, ~1e12 ms, ~1e15 us, ~1e18 ns.
            v = float(ts.iloc[-1])
            unit = "ns" if v > 1e17 else "us" if v > 1e14 else "ms" if v > 1e11 else "s"
            idx = pd.to_datetime(ts, unit=unit, utc=True)
        else:
            idx = pd.to_datetime(ts, utc=True)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"cannot parse time column {time_col!r} in {path.name}: {exc}") from exc
    out = df.drop(columns=[time_col])
    out.index = pd.DatetimeIndex(idx)
    return out.sort_index()


class CSVProvider:
    """Reads per-symbol CSV files exported from any vendor.

    Two filename conventions resolve, in order:

    1. exact ``{symbol}.csv``
    2. a unique TradingView "Export chart data" file whose name reduces to
       the symbol: ``BINANCE_BTCUSDT, 1D.csv`` matches ``BTCUSDT`` (and
       ``BITSTAMP_BTCUSD, 1D.csv`` matches ``BTC-USD``) — no renaming needed.

    The time column may be ISO strings or epoch seconds/milliseconds; rows
    are sorted before the most-recent ``bars`` are taken.
    """

    name = "csv"

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        if not self._dir.is_dir():
            raise FileNotFoundError(f"CSV data directory not found: {self._dir}")

    def _resolve(self, symbol: str) -> Path:
        path = self._dir / f"{symbol}.csv"
        if path.exists():
            return path
        want = _squash(symbol)
        matches = [p for p in sorted(self._dir.glob("*.csv")) if _squash(_tv_stem(p.stem)) == want]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(p.name for p in matches)
            raise FileNotFoundError(
                f"ambiguous CSVs for {symbol} ({names}) — keep one or rename it {symbol}.csv"
            )
        raise FileNotFoundError(f"no CSV for {symbol}: {path}")

    def fetch(self, symbol: str, bars: int) -> pd.DataFrame:
        path = self._resolve(symbol)
        df = _set_time_index(pd.read_csv(path), path)
        return df.tail(bars)


class YahooProvider:
    """Thin adapter over yfinance for real-data deployments.

    Import is lazy so the core platform has no hard dependency on yfinance,
    and sandboxes without market-data egress can still run everything else.
    """

    name = "yahoo"

    def __init__(self, timeframe: str = "1d") -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("YahooProvider requires: pip install 'titan[data]'") from exc
        if timeframe not in YAHOO_INTERVALS:
            raise ValueError(
                f"Yahoo has no native {timeframe} bar; it serves "
                f"{', '.join(sorted(YAHOO_INTERVALS))}. Reach {timeframe} by "
                f"aggregating: set data.resample_from to a finer interval "
                f"(e.g. 1h) alongside data.timeframe: {timeframe}."
            )
        self._timeframe = timeframe
        self._interval = YAHOO_INTERVALS[timeframe]

    def _period_days(self, bars: int) -> int:
        """Calendar days to request, clamped to Yahoo's cap for this interval.

        The cap matters more than it looks: Yahoo does not error when you ask
        for more intraday history than it keeps, it just returns less. Without
        the clamp an hourly run asks for ten years, silently receives two
        months, and reports walk-forward statistics computed over a single
        market regime as though they spanned a decade.
        """
        wanted = int(bars / session_bars_per_day(self._timeframe) * 1.6) + 30
        cap = YAHOO_MAX_HISTORY_DAYS[self._timeframe]
        if cap and wanted > cap:
            logger.warning(
                "Yahoo keeps at most %d days of %s bars; requesting %d instead of %d. "
                "Expect fewer bars than data.bars and check the QC report.",
                cap, self._timeframe, cap, wanted,
            )
            return cap
        return wanted

    def fetch(self, symbol: str, bars: int) -> pd.DataFrame:  # pragma: no cover
        import yfinance as yf

        # Fetch generously, then trim: calendars differ across assets.
        df = yf.Ticker(symbol).history(
            period=f"{self._period_days(bars)}d", interval=self._interval, auto_adjust=True
        )
        if df is None or len(df) == 0:
            raise RuntimeError(f"Yahoo returned no data for {symbol}")
        return df.tail(bars)


def build_provider(data_cfg: DataConfig, universe_cfg: UniverseConfig, seed: int) -> DataProvider:
    """Provider factory keyed on ``data.provider``."""
    if data_cfg.provider == "synthetic":
        symbols = [i.symbol for i in universe_cfg.instruments]
        return SyntheticProvider(
            symbols=symbols,
            benchmark=universe_cfg.benchmark,
            bars=data_cfg.bars,
            seed=seed,
            drift_sigma=data_cfg.synthetic_drift_sigma,
        )
    if data_cfg.provider == "csv":
        if data_cfg.csv_dir is None:
            raise ValueError("data.csv_dir must be set for the csv provider")
        return CSVProvider(data_cfg.csv_dir)
    if data_cfg.provider == "yahoo":
        # With resampling on, the vendor is asked for the FINER interval;
        # the store aggregates it up to data.timeframe.
        return YahooProvider(data_cfg.resample_from or data_cfg.timeframe)
    raise ValueError(f"unknown provider: {data_cfg.provider}")
