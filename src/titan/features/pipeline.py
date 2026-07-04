"""Feature matrix construction: per-symbol families + cross-sectional joins.

Produces the pooled ``(date, symbol)`` :class:`FeaturePanel` used everywhere
downstream. Rows keep NaNs from warm-up windows; ``finalize`` drops rows whose
feature coverage is below threshold, and models impute the remainder.
"""

from __future__ import annotations

import pandas as pd

from titan.core.config import FeatureConfig
from titan.core.log import get_logger
from titan.data.store import MarketDataset
from titan.features.cross import CROSS_FAMILY, build_cross_features
from titan.features.price import build_price_specs
from titan.features.registry import FeaturePanel, FeatureRegistry
from titan.features.volume import build_volume_specs

logger = get_logger(__name__)


def build_default_registry(cfg: FeatureConfig) -> FeatureRegistry:
    registry = FeatureRegistry()
    registry.register_all(build_price_specs(cfg))
    registry.register_all(build_volume_specs(cfg))
    return registry


class FeatureMatrixBuilder:
    """Builds the pooled feature panel for a dataset."""

    def __init__(self, cfg: FeatureConfig, registry: FeatureRegistry | None = None) -> None:
        self._cfg = cfg
        self._registry = registry or build_default_registry(cfg)

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    def build(self, dataset: MarketDataset, min_coverage: float = 0.85) -> FeaturePanel:
        # Per-symbol features.
        per_symbol: dict[str, pd.DataFrame] = {
            sym: self._registry.compute(frame) for sym, frame in dataset.frames.items()
        }

        # Cross-sectional features (wide) -> stacked to (date, symbol).
        cross_wide = build_cross_features(dataset.frames, dataset.benchmark_frame, self._cfg)

        blocks: list[pd.DataFrame] = []
        for sym, feats in per_symbol.items():
            block = feats.copy()
            for name, wide in cross_wide.items():
                block[name] = wide[sym].reindex(block.index)
            block.index = pd.MultiIndex.from_product([block.index, [sym]], names=["date", "symbol"])
            blocks.append(block)

        X = pd.concat(blocks).sort_index()

        families = self._registry.families()
        families.update(dict.fromkeys(cross_wide, CROSS_FAMILY))

        reliability = None
        if dataset.reliability:
            rel_map = pd.Series(dataset.reliability)
            reliability = pd.Series(
                rel_map.reindex(X.index.get_level_values(1)).to_numpy(),
                index=X.index,
                name="reliability",
            )

        panel = FeaturePanel(X=X, families=families, reliability=reliability)
        return finalize(panel, min_coverage=min_coverage)


def finalize(panel: FeaturePanel, min_coverage: float = 0.85) -> FeaturePanel:
    """Drop warm-up rows with insufficient feature coverage."""
    coverage = panel.X.notna().mean(axis=1)
    keep = coverage >= min_coverage
    dropped = int((~keep).sum())
    if dropped:
        logger.info(
            "dropped %d/%d rows below %.0f%% feature coverage (warm-up windows)",
            dropped, len(panel.X), 100 * min_coverage,
        )
    X = panel.X.loc[keep]
    rel = panel.reliability.loc[keep] if panel.reliability is not None else None
    return FeaturePanel(X=X, families=panel.families, reliability=rel)
