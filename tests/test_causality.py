"""THE critical invariant: no feature may see the future.

Every registered per-symbol feature and every cross-sectional feature is
recomputed on truncated history; any change to a past value when the future
is removed is a look-ahead leak and fails the build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.features.cross import build_cross_features
from titan.features.pipeline import build_default_registry


def test_registry_features_are_causal(cfg, ohlcv):
    registry = build_default_registry(cfg.features)
    violations = registry.verify_causality(ohlcv.iloc[:500], checkpoints=[380, 430, 480])
    assert violations == [], f"look-ahead leaks in: {sorted({v.feature for v in violations})}"


def test_cross_features_are_causal(cfg, dataset):
    frames = {s: f.iloc[:500] for s, f in dataset.frames.items()}
    bench = dataset.benchmark_frame.iloc[:500]
    full = build_cross_features(frames, bench, cfg.features)

    cut = 450
    frames_cut = {s: f.iloc[: cut + 1] for s, f in frames.items()}
    truncated = build_cross_features(frames_cut, bench.iloc[: cut + 1], cfg.features)

    ts = frames["AAA"].index[cut]
    for name, wide in full.items():
        a = wide.loc[ts].to_numpy(dtype=float)
        b = truncated[name].loc[ts].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-9, equal_nan=True), f"cross feature {name} leaks"


def test_labels_use_only_future_within_horizon(cfg, ohlcv):
    """Labels must be identical when data BEYOND the horizon changes."""
    from titan.labels.triple_barrier import triple_barrier_labels

    full = triple_barrier_labels(ohlcv.iloc[:400], cfg.labels).frame
    longer = triple_barrier_labels(ohlcv.iloc[:500], cfg.labels).frame
    joint = full.index.intersection(longer.index)
    pd.testing.assert_series_equal(full.loc[joint, "label"], longer.loc[joint, "label"])
    pd.testing.assert_series_equal(full.loc[joint, "ret"], longer.loc[joint, "ret"])
