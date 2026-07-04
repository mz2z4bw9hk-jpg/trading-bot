"""Volume, liquidity and participation feature family."""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.core.config import FeatureConfig
from titan.features import rolling as R
from titan.features.registry import FeatureFn, FeatureSpec


def _vol_z(window: int) -> FeatureFn:
    def fn(df: pd.DataFrame) -> pd.Series:
        return (
            (df["volume"] - df["volume"].rolling(window).mean())
            / df["volume"].rolling(window).std().replace(0.0, np.nan)
        ).clip(-5, 5)

    return fn


def _amihud_21(df: pd.DataFrame) -> pd.Series:
    return np.log1p(
        (
            np.log(df["close"]).diff().abs()
            / (df["close"] * df["volume"]).replace(0.0, np.nan)
        )
        .rolling(21)
        .mean()
        * 1e9
    )


def _obv_slope_21(df: pd.DataFrame) -> pd.Series:
    normalized = R.on_balance_volume(df) / df["volume"].rolling(63, min_periods=21).mean()
    return R.rolling_slope_stats(normalized, 21)[1]


def _updown_vol_21(df: pd.DataFrame) -> pd.Series:
    up = df["volume"].where(df["close"].diff() > 0, 0.0).rolling(21).sum()
    down = df["volume"].where(df["close"].diff() < 0, 0.0).rolling(21).sum()
    return np.log((up + 1.0) / (down + 1.0))


def _dollar_vol_z_63(df: pd.DataFrame) -> pd.Series:
    log_dv = np.log(df["close"] * df["volume"] + 1.0)
    return (
        (log_dv - log_dv.rolling(63).mean()) / log_dv.rolling(63).std().replace(0.0, np.nan)
    ).clip(-5, 5)


def build_volume_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for w in cfg.volume_windows:
        specs.append(
            FeatureSpec(
                name=f"vol_z_{w}",
                family="liquidity",
                lookback=w + 1,
                fn=_vol_z(w),
                description=f"volume z-score over {w} bars",
            )
        )
    specs.append(
        FeatureSpec(
            name="amihud_21",
            family="liquidity",
            lookback=22,
            fn=_amihud_21,
            description="Amihud illiquidity (21), log-scaled",
        )
    )
    specs.append(
        FeatureSpec(
            name="obv_slope_21",
            family="liquidity",
            lookback=22,
            fn=_obv_slope_21,
            description="t-stat of OBV trend over 21 bars (volume-normalized)",
        )
    )
    specs.append(
        FeatureSpec(
            name="updown_vol_21",
            family="liquidity",
            lookback=22,
            fn=_updown_vol_21,
            description="log ratio of up-day to down-day volume over 21 bars",
        )
    )
    specs.append(
        FeatureSpec(
            name="dollar_vol_z_63",
            family="liquidity",
            lookback=64,
            fn=_dollar_vol_z_63,
            description="dollar-volume z-score over 63 bars (participation)",
        )
    )
    return specs
