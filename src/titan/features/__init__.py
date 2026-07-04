"""Feature engineering: causal registry, families, selection."""

from titan.features.pipeline import FeatureMatrixBuilder, build_default_registry
from titan.features.registry import FeaturePanel, FeatureRegistry, FeatureSpec
from titan.features.selection import permutation_rank, redundancy_prune, univariate_ic

__all__ = [
    "FeatureMatrixBuilder",
    "FeaturePanel",
    "FeatureRegistry",
    "FeatureSpec",
    "build_default_registry",
    "permutation_rank",
    "redundancy_prune",
    "univariate_ic",
]
