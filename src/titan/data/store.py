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
from titan.core.timeframe import INTRADAY_TIMEFRAMES, resolve_bars_per_year
from titan.core.types import AssetClass, Instrument, Universe
from titan.data.providers import DataProvider, SyntheticProvider, build_provider
from titan.data.quality import DataQualityReport, assess_quality
from titan.data.resample import RESAMPLE_RULES, resample_ohlcv, zero_volume_fraction
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
        self._continuous = continuous
        # With resampling on, the provider is asked for enough SOURCE bars to
        # yield data_cfg.bars target bars after aggregation.
        self._source_bars = data_cfg.bars
        if data_cfg.resample_from is not None:
            ratio = (
                resolve_bars_per_year(data_cfg.resample_from, continuous=continuous)
                / resolve_bars_per_year(data_cfg.timeframe, continuous=continuous)
            )
            self._source_bars = int(data_cfg.bars * ratio)
        # Cache keyed by everything that determines the data. A stale cache
        # silently serving frames from a different configuration corrupts
        # research; live providers additionally get a per-day key so "most
        # recent N bars" is refetched at most daily.
        key_parts = [
            self._provider.name,
            f"bars={data_cfg.bars}",
            f"seed={seed}",
            f"drift={data_cfg.synthetic_drift_sigma}",
            # The cache stores post-resample frames, so both the interval that
            # was fetched and the one it was aggregated to are part of a frame's
            # identity. Omitting them serves 1d bars to a 1h run: same symbol,
            # same bar count, wrong market — and nothing downstream can tell.
            f"tf={data_cfg.timeframe}",
            f"src={data_cfg.resample_from}",
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
        raw = self._provider.fetch(symbol, self._source_bars)
        frame = normalize_ohlcv(raw, max_forward_fill=self._cfg.max_forward_fill)
        frame = self._resample(symbol, frame)
        if use_cache:
            try:
                frame.to_parquet(path)
            except Exception as exc:  # parquet engine missing: cache is best-effort
                logger.debug("cache write failed for %s: %s", symbol, exc)
        return frame

    def _resample(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate source bars up to data.timeframe, if configured."""
        if self._cfg.resample_from is None:
            return frame
        before_zero = zero_volume_fraction(frame)
        out = resample_ohlcv(
            frame,
            RESAMPLE_RULES[self._cfg.timeframe],
            within_sessions=not self._continuous,
        )
        after_zero = zero_volume_fraction(out)
        logger.info(
            "%s: resampled %s -> %s (%d -> %d bars); zero-volume %.1f%% -> %.1f%%",
            symbol, self._cfg.resample_from, self._cfg.timeframe,
            len(frame), len(out), 100 * before_zero, 100 * after_zero,
        )
        return out

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

        # The benchmark goes through the SAME path as every instrument. It used
        # to be fetched and normalized inline, which quietly skipped resampling
        # and asked for target-bar counts against a source-bar interval: the
        # regime detector would then fit on 1h bars while the instruments it
        # gates traded on 2h ones. It is still not QC-gated — it is the
        # reference series, not a tradeable — but it must be the same clock.
        benchmark_frame = self._load_one(universe.benchmark, use_cache)

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
