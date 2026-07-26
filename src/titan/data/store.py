"""Market data store: fetch → normalize → QC → cache → dataset.

The store is the single entry point downstream code uses. It guarantees that
every frame in a :class:`MarketDataset` passed schema normalization and QC,
and that instruments below the reliability floor were excluded (loudly).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from titan.core.config import DataConfig, UniverseConfig
from titan.core.log import get_logger
from titan.core.timeframe import INTRADAY_TIMEFRAMES
from titan.core.types import AssetClass, Instrument, Universe
from titan.data.providers import DataProvider, SyntheticProvider, build_provider
from titan.data.quality import DataQualityReport, assess_quality
from titan.data.schema import SchemaError, normalize_ohlcv

logger = get_logger(__name__)


@dataclass(slots=True)
class MarketDataset:
    """A validated, QC-scored snapshot of the research universe."""

    frames: dict[str, pd.DataFrame]
    benchmark_frame: pd.DataFrame
    universe: Universe
    reliability: dict[str, float] = field(default_factory=dict)
    reports: dict[str, DataQualityReport] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)  # symbol -> reason
    true_regimes: pd.Series | None = None  # ground truth, synthetic data only

    @property
    def symbols(self) -> list[str]:
        return list(self.frames)

    def common_index(self) -> pd.DatetimeIndex:
        idx: pd.DatetimeIndex | None = None
        for frame in self.frames.values():
            idx = frame.index if idx is None else idx.intersection(frame.index)
        if idx is None:
            raise ValueError("dataset has no frames")
        return idx


class MarketDataStore:
    """Loads a universe through a provider with caching and QC."""

    def __init__(
        self,
        data_cfg: DataConfig,
        universe_cfg: UniverseConfig,
        seed: int = 7,
        provider: DataProvider | None = None,
    ) -> None:
        self._cfg = data_cfg
        self._universe_cfg = universe_cfg
        self._provider = provider or build_provider(data_cfg, universe_cfg, seed)
        # Bars finer than a day on a market that closes: QC must not read the
        # exchange's opening hours as dropped data (see assess_quality).
        tradeable = [
            i for i in universe_cfg.instruments if i.symbol != universe_cfg.benchmark
        ]
        continuous = bool(tradeable) and all(i.asset_class == "crypto" for i in tradeable)
        self._intraday_sessions = data_cfg.timeframe in INTRADAY_TIMEFRAMES and not continuous
        # Cache keyed by everything that determines the data. A stale cache
        # silently serving frames from a different configuration corrupts
        # research; live providers additionally get a per-day key so "most
        # recent N bars" is refetched at most daily.
        key_parts = [
            self._provider.name,
            f"bars={data_cfg.bars}",
            f"seed={seed}",
            f"drift={data_cfg.synthetic_drift_sigma}",
        ]
        if self._provider.name != "synthetic":
            key_parts.append(time.strftime("%Y-%m-%d", time.gmtime()))
        fingerprint = hashlib.sha1("|".join(key_parts).encode()).hexdigest()[:10]
        self._cache_dir = Path(data_cfg.cache_dir) / f"{self._provider.name}_{fingerprint}"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace(":", "_")
        return self._cache_dir / f"{safe}.parquet"

    def _load_one(self, symbol: str, use_cache: bool) -> pd.DataFrame:
        path = self._cache_path(symbol)
        if use_cache and path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:  # corrupt cache: refetch
                logger.warning("corrupt cache for %s, refetching", symbol)
        raw = self._provider.fetch(symbol, self._cfg.bars)
        frame = normalize_ohlcv(raw, max_forward_fill=self._cfg.max_forward_fill)
        if use_cache:
            try:
                frame.to_parquet(path)
            except Exception as exc:  # parquet engine missing: cache is best-effort
                logger.debug("cache write failed for %s: %s", symbol, exc)
        return frame

    # ------------------------------------------------------------------ #

    def load(self, use_cache: bool = True) -> MarketDataset:
        universe = Universe(
            instruments=[
                Instrument(
                    symbol=item.symbol,
                    asset_class=AssetClass(item.asset_class),
                    sector=item.sector,
                )
                for item in self._universe_cfg.instruments
            ],
            benchmark=self._universe_cfg.benchmark,
            name=self._universe_cfg.name,
        )

        frames: dict[str, pd.DataFrame] = {}
        reliability: dict[str, float] = {}
        reports: dict[str, DataQualityReport] = {}
        excluded: dict[str, str] = {}

        for symbol in universe.symbols:
            try:
                frame = self._load_one(symbol, use_cache)
            except (SchemaError, FileNotFoundError, KeyError, RuntimeError) as exc:
                excluded[symbol] = f"load failed: {exc}"
                logger.warning("excluding %s: %s", symbol, exc)
                continue
            report = assess_quality(
                symbol,
                frame,
                min_bars=self._cfg.min_history_bars,
                intraday_sessions=self._intraday_sessions,
            )
            reports[symbol] = report
            reliability[symbol] = report.reliability
            if report.reliability < self._cfg.min_reliability:
                excluded[symbol] = (
                    f"reliability {report.reliability:.2f} < floor {self._cfg.min_reliability:.2f}: "
                    + "; ".join(report.issues[:3])
                )
                logger.warning("excluding %s: %s", symbol, excluded[symbol])
                continue
            frames[symbol] = frame

        if not frames:
            raise RuntimeError("no instrument passed data QC; cannot build dataset")

        benchmark_frame = normalize_ohlcv(
            self._provider.fetch(universe.benchmark, self._cfg.bars),
            max_forward_fill=self._cfg.max_forward_fill,
        )

        true_regimes = None
        if isinstance(self._provider, SyntheticProvider):
            true_regimes = self._provider.result.true_regimes

        logger.info(
            "dataset ready: %d/%d instruments passed QC (benchmark %s, %d bars)",
            len(frames),
            len(universe.symbols),
            universe.benchmark,
            len(benchmark_frame),
        )
        return MarketDataset(
            frames=frames,
            benchmark_frame=benchmark_frame,
            universe=universe,
            reliability=reliability,
            reports=reports,
            excluded=excluded,
            true_regimes=true_regimes,
        )
