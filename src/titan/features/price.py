"""Price-derived feature families: momentum, trend, mean reversion,
volatility, market structure, calendar.

Each builder returns a list of :class:`FeatureSpec`. Windows come from config,
so the registry scales from a compact research set to hundreds of features
without code changes. Hypothesis grounding for each family lives in
docs/RESEARCH.md; nothing here is included merely because it is popular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.core.config import FeatureConfig
from titan.features import rolling as R
from titan.features.registry import FeatureSpec


def _spec(name: str, family: str, lookback: int, fn, desc: str) -> FeatureSpec:
    return FeatureSpec(name=name, family=family, fn=fn, lookback=lookback, description=desc)


# --------------------------------------------------------------------- #
# momentum
# --------------------------------------------------------------------- #

def momentum_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for w in cfg.momentum_windows:
        specs.append(
            _spec(
                f"ret_{w}", "momentum", w + 1,
                lambda df, w=w: np.log(df["close"] / df["close"].shift(w)),
                f"{w}-bar log return",
            )
        )
        specs.append(
            _spec(
                f"tsmom_{w}", "momentum", w + 1,
                lambda df, w=w: (
                    np.log(df["close"] / df["close"].shift(w))
                    / (np.log(df["close"]).diff().rolling(w).std() * np.sqrt(w)).replace(0.0, np.nan)
                ),
                f"vol-scaled {w}-bar momentum (t-stat-like)",
            )
        )
    for w in (21, 63, 126):
        if w in cfg.momentum_windows:
            specs.append(
                _spec(
                    f"upfrac_{w}", "momentum", w + 1,
                    lambda df, w=w: (df["close"].diff() > 0).rolling(w).mean() - 0.5,
                    f"fraction of up days over {w} bars, centered",
                )
            )
    long_w = max(cfg.momentum_windows)
    if long_w >= 126 and 21 in cfg.momentum_windows:
        specs.append(
            _spec(
                f"mom_{long_w}_21", "momentum", long_w + 1,
                lambda df, w=long_w: np.log(df["close"].shift(21) / df["close"].shift(w)),
                f"{long_w}-bar momentum skipping most recent 21 bars (reversal-free)",
            )
        )
    return specs


# --------------------------------------------------------------------- #
# trend
# --------------------------------------------------------------------- #

def trend_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for w in cfg.trend_windows:
        specs.append(
            _spec(
                f"ema_dist_{w}", "trend", w * 3,
                lambda df, w=w: df["close"]
                / df["close"].ewm(span=w, adjust=False, min_periods=w).mean()
                - 1.0,
                f"distance from EMA({w})",
            )
        )
    for w in (21, 63):
        if w in cfg.trend_windows or w in cfg.structure_windows:
            specs.append(
                _spec(
                    f"slope_t_{w}", "trend", w,
                    lambda df, w=w: R.rolling_slope_stats(np.log(df["close"]), w)[1],
                    f"t-stat of OLS log-price slope over {w} bars",
                )
            )
    for w in (10, 21):
        specs.append(
            _spec(
                f"eff_ratio_{w}", "trend", w + 1,
                lambda df, w=w: R.efficiency_ratio(df["close"], w),
                f"Kaufman efficiency ratio over {w} bars",
            )
        )
    return specs


# --------------------------------------------------------------------- #
# mean reversion
# --------------------------------------------------------------------- #

def meanrev_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for w in cfg.meanrev_windows:
        specs.append(
            _spec(
                f"zscore_{w}", "meanrev", w,
                lambda df, w=w: (df["close"] - df["close"].rolling(w).mean())
                / df["close"].rolling(w).std().replace(0.0, np.nan),
                f"z-score of close vs {w}-bar mean",
            )
        )
    specs.append(
        _spec(
            "rsi_14", "meanrev", 60,
            lambda df: (R.wilder_rsi(df["close"], 14) - 50.0) / 50.0,
            "Wilder RSI(14), centered and scaled to [-1, 1]",
        )
    )
    specs.append(
        _spec(
            "bb_pos_20", "meanrev", 20,
            lambda df: (
                (df["close"] - df["close"].rolling(20).mean())
                / (2.0 * df["close"].rolling(20).std().replace(0.0, np.nan))
            ).clip(-3, 3),
            "position within Bollinger(20, 2) band",
        )
    )
    specs.append(
        _spec(
            "gap_mean_5", "meanrev", 6,
            lambda df: np.log(df["open"] / df["close"].shift(1)).rolling(5).mean(),
            "mean overnight gap over 5 bars",
        )
    )
    return specs


# --------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------- #

def volatility_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for w in cfg.vol_windows:
        specs.append(
            _spec(
                f"rv_{w}", "volatility", w + 1,
                lambda df, w=w: R.realized_vol(df["close"], w),
                f"annualized close-to-close vol over {w} bars",
            )
        )
    specs.append(
        _spec(
            "park_21", "volatility", 21,
            lambda df: R.parkinson_vol(df, 21),
            "Parkinson range vol (21)",
        )
    )
    specs.append(
        _spec(
            "gk_21", "volatility", 21,
            lambda df: R.garman_klass_vol(df, 21),
            "Garman-Klass OHLC vol (21)",
        )
    )
    specs.append(
        _spec(
            "atr_norm_14", "volatility", 60,
            lambda df: R.atr(df, 14) / df["close"],
            "ATR(14) as a fraction of price",
        )
    )
    if 5 in cfg.vol_windows and 63 in cfg.vol_windows:
        specs.append(
            _spec(
                "vol_ts_5_63", "volatility", 64,
                lambda df: R.realized_vol(df["close"], 5)
                / R.realized_vol(df["close"], 63).replace(0.0, np.nan),
                "vol term structure: rv(5)/rv(63)",
            )
        )
    specs.append(
        _spec(
            "vol_rank_252", "volatility", 252 + 21,
            lambda df: R.rolling_percentile_rank(R.realized_vol(df["close"], 21), 252),
            "percentile of rv(21) within trailing year",
        )
    )
    specs.append(
        _spec(
            "vol_of_vol_21", "volatility", 43,
            lambda df: R.realized_vol(df["close"], 21).pct_change().rolling(21).std(),
            "vol of vol (21)",
        )
    )
    specs.append(
        _spec(
            "squeeze_20", "volatility", 120 + 20,
            lambda df: (
                (df["close"].rolling(20).std() / df["close"].rolling(20).mean())
                / (df["close"].rolling(20).std() / df["close"].rolling(20).mean())
                .rolling(120)
                .median()
                .replace(0.0, np.nan)
            ),
            "Bollinger width vs its 120-bar median (volatility compression)",
        )
    )
    return specs


# --------------------------------------------------------------------- #
# market structure
# --------------------------------------------------------------------- #

def structure_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for w in cfg.structure_windows:
        specs.append(
            _spec(
                f"dist_high_{w}", "structure", w,
                lambda df, w=w: df["close"] / df["close"].rolling(w).max() - 1.0,
                f"distance below {w}-bar high",
            )
        )
        specs.append(
            _spec(
                f"range_pos_{w}", "structure", w,
                lambda df, w=w: (
                    (df["close"] - df["close"].rolling(w).min())
                    / (df["close"].rolling(w).max() - df["close"].rolling(w).min()).replace(0.0, np.nan)
                ),
                f"position of close within {w}-bar range",
            )
        )
    specs.append(
        _spec(
            "breakout_63", "structure", 80,
            lambda df: (df["close"] - df["close"].rolling(63).max().shift(1))
            / R.atr(df, 14).replace(0.0, np.nan),
            "ATR-scaled excursion above prior 63-bar high",
        )
    )
    specs.append(
        _spec(
            "streak", "structure", 30,
            lambda df: R.signed_streak(df["close"], cap=10) / 10.0,
            "signed run length of daily moves, scaled",
        )
    )
    specs.append(
        _spec(
            "clv_accum_21", "structure", 22,
            lambda df: (
                (
                    (2.0 * df["close"] - df["high"] - df["low"])
                    / (df["high"] - df["low"]).replace(0.0, np.nan)
                )
                * (df["volume"] / df["volume"].rolling(21).mean().replace(0.0, np.nan))
            )
            .rolling(21)
            .mean(),
            "volume-weighted close-location value (institutional accumulation proxy)",
        )
    )
    specs.append(
        _spec(
            "gap_freq_21", "structure", 80,
            lambda df: (
                np.log(df["open"] / df["close"].shift(1)).abs()
                > 0.5 * (R.atr(df, 14) / df["close"])
            )
            .rolling(21)
            .mean(),
            "fraction of bars opening beyond half an ATR from prior close",
        )
    )
    return specs


# --------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------- #

def calendar_specs(_cfg: FeatureConfig) -> list[FeatureSpec]:
    return [
        _spec(
            "dow", "calendar", 1,
            lambda df: pd.Series(df.index.dayofweek.astype(float), index=df.index) / 4.0 - 0.5,
            "day of week, centered",
        ),
        _spec(
            "month_phase", "calendar", 1,
            lambda df: pd.Series(
                np.sin(2.0 * np.pi * (df.index.day.astype(float) / 31.0)), index=df.index
            ),
            "sinusoidal month phase (turn-of-month effect carrier)",
        ),
    ]


def build_price_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    return (
        momentum_specs(cfg)
        + trend_specs(cfg)
        + meanrev_specs(cfg)
        + volatility_specs(cfg)
        + structure_specs(cfg)
        + calendar_specs(cfg)
    )
