"""Volume, liquidity and participation feature family."""

from __future__ import annotations

import numpy as np

from titan.core.config import FeatureConfig
from titan.features import rolling as R
from titan.features.registry import FeatureSpec


def build_volume_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for w in cfg.volume_windows:
        specs.append(
            FeatureSpec(
                name=f"vol_z_{w}",
                family="liquidity",
                lookback=w + 1,
                fn=lambda df, w=w: (
                    (df["volume"] - df["volume"].rolling(w).mean())
                    / df["volume"].rolling(w).std().replace(0.0, np.nan)
                ).clip(-5, 5),
                description=f"volume z-score over {w} bars",
            )
        )
    specs.append(
        FeatureSpec(
            name="amihud_21",
            family="liquidity",
            lookback=22,
            fn=lambda df: np.log1p(
                (
                    np.log(df["close"]).diff().abs()
                    / (df["close"] * df["volume"]).replace(0.0, np.nan)
                )
                .rolling(21)
                .mean()
                * 1e9
            ),
            description="Amihud illiquidity (21), log-scaled",
        )
    )
    specs.append(
        FeatureSpec(
            name="obv_slope_21",
            family="liquidity",
            lookback=22,
            fn=lambda df: R.rolling_slope_stats(
                R.on_balance_volume(df) / df["volume"].rolling(63, min_periods=21).mean(), 21
            )[1],
            description="t-stat of OBV trend over 21 bars (volume-normalized)",
        )
    )
    specs.append(
        FeatureSpec(
            name="updown_vol_21",
            family="liquidity",
            lookback=22,
            fn=lambda df: np.log(
                (df["volume"].where(df["close"].diff() > 0, 0.0).rolling(21).sum() + 1.0)
                / (df["volume"].where(df["close"].diff() < 0, 0.0).rolling(21).sum() + 1.0)
            ),
            description="log ratio of up-day to down-day volume over 21 bars",
        )
    )
    specs.append(
        FeatureSpec(
            name="dollar_vol_z_63",
            family="liquidity",
            lookback=64,
            fn=lambda df: (
                (np.log(df["close"] * df["volume"] + 1.0)
                 - np.log(df["close"] * df["volume"] + 1.0).rolling(63).mean())
                / np.log(df["close"] * df["volume"] + 1.0).rolling(63).std().replace(0.0, np.nan)
            ).clip(-5, 5),
            description="dollar-volume z-score over 63 bars (participation)",
        )
    )
    return specs
