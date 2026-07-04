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
        kwargs = {} if drift_sigma is None else {"drift_sigma": drift_sigma}
        market = SyntheticMarket(symbols=list(symbols), bars=bars, seed=seed, **kwargs)
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


class CSVProvider:
    """Reads ``{symbol}.csv`` files exported from any vendor."""

    name = "csv"

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        if not self._dir.is_dir():
            raise FileNotFoundError(f"CSV data directory not found: {self._dir}")

    def fetch(self, symbol: str, bars: int) -> pd.DataFrame:
        path = self._dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"no CSV for {symbol}: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.tail(bars)


class YahooProvider:
    """Thin adapter over yfinance for real-data deployments.

    Import is lazy so the core platform has no hard dependency on yfinance,
    and sandboxes without market-data egress can still run everything else.
    """

    name = "yahoo"

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("YahooProvider requires: pip install 'titan[data]'") from exc

    def fetch(self, symbol: str, bars: int) -> pd.DataFrame:  # pragma: no cover
        import yfinance as yf

        # Fetch generously, then trim: calendars differ across assets.
        period_days = int(bars * 1.6) + 30
        df = yf.Ticker(symbol).history(period=f"{period_days}d", auto_adjust=True)
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
        return YahooProvider()
    raise ValueError(f"unknown provider: {data_cfg.provider}")
