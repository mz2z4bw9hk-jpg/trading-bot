"""Cross-sectional and intermarket features.

Computed on aligned *trailing* data across the whole universe: relative
strength ranks, benchmark beta/correlation, breadth and dispersion. Breadth
and dispersion are market-level series broadcast to every symbol — they give
the models the market context a single instrument's frame cannot carry.

Causality: every value at date ``t`` uses only bars ``<= t`` across all
symbols. Verified by the causality test in the suite (truncating the panel's
future must not change values at ``t``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.core.config import FeatureConfig

CROSS_FAMILY = "cross_sectional"


def _wide(frames: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    return pd.concat({sym: f[column] for sym, f in frames.items()}, axis=1)


def build_cross_features(
    frames: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    cfg: FeatureConfig,
) -> dict[str, pd.DataFrame]:
    """Return {feature_name: wide frame (date x symbol)}."""
    close = _wide(frames, "close")
    bench_ret = np.log(benchmark["close"]).diff().reindex(close.index)
    log_close = np.log(close)
    ret_1 = log_close.diff()

    out: dict[str, pd.DataFrame] = {}

    for w in cfg.cross_windows:
        ret_w = log_close.diff(w)
        # Relative-strength rank in [-0.5, 0.5]; NaN-safe (rank ignores NaN).
        out[f"rs_rank_{w}"] = ret_w.rank(axis=1, pct=True) - 0.5
        vol_w = ret_1.rolling(w).std()
        out[f"rs_voladj_{w}"] = (ret_w / (vol_w * np.sqrt(w)).replace(0.0, np.nan)).rank(
            axis=1, pct=True
        ) - 0.5
        out[f"rel_ret_{w}"] = ret_w.sub(ret_w.median(axis=1), axis=0)

    # Benchmark beta / correlation (63 bars).
    w_b = 63
    cov = ret_1.rolling(w_b).cov(bench_ret)
    var_b = bench_ret.rolling(w_b).var()
    out["beta_63"] = cov.div(var_b.replace(0.0, np.nan), axis=0).clip(-3, 5)
    out["corr_bench_63"] = ret_1.rolling(w_b).corr(bench_ret).clip(-1, 1)

    # Breadth: fraction of universe above its own 50-bar EMA (broadcast).
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    breadth = (close > ema50).mean(axis=1)
    out["breadth_50"] = pd.DataFrame(
        np.tile(breadth.to_numpy()[:, None], (1, close.shape[1])) - 0.5,
        index=close.index,
        columns=close.columns,
    )

    # Cross-sectional dispersion of 21-bar returns (broadcast).
    disp = log_close.diff(21).std(axis=1)
    out["disp_21"] = pd.DataFrame(
        np.tile(disp.to_numpy()[:, None], (1, close.shape[1])),
        index=close.index,
        columns=close.columns,
    )

    return out


def cross_feature_names(cfg: FeatureConfig) -> list[str]:
    names = []
    for w in cfg.cross_windows:
        names += [f"rs_rank_{w}", f"rs_voladj_{w}", f"rel_ret_{w}"]
    names += ["beta_63", "corr_bench_63", "breadth_50", "disp_21"]
    return names
