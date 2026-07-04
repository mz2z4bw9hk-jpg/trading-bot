"""Feature registry with an enforced causality contract.

A :class:`FeatureSpec` wraps a function ``OHLCV frame -> Series`` that must be
*strictly causal*: the value at time ``t`` may use only bars ``<= t``. This is
the single most important invariant in the platform — one leaked feature
invalidates every backtest built on it.

The contract is enforced two ways:
- :meth:`FeatureRegistry.verify_causality` recomputes each feature on
  truncated prefixes of the data and fails if any value changes when the
  future is removed. The test suite runs this over the full registry.
- Feature functions have no access to labels or to other symbols (cross-
  sectional features are built separately in :mod:`titan.features.cross`
  from aligned *past* data and verified the same way).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from titan.core.log import get_logger

logger = get_logger(__name__)

FeatureFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A named, documented, causal feature."""

    name: str
    family: str
    fn: FeatureFn
    lookback: int
    description: str = ""

    def compute(self, df: pd.DataFrame) -> pd.Series:
        out = self.fn(df)
        if not out.index.equals(df.index):
            raise ValueError(f"feature {self.name} returned a misaligned index")
        return out.astype("float64").rename(self.name)


@dataclass(slots=True)
class CausalityViolation:
    feature: str
    timestamp: pd.Timestamp
    full_value: float
    truncated_value: float


class FeatureRegistry:
    """Ordered collection of feature specs for one instrument's OHLCV frame."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate feature name: {spec.name}")
        self._specs[spec.name] = spec

    def register_all(self, specs: Iterable[FeatureSpec]) -> None:
        for spec in specs:
            self.register(spec)

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    @property
    def specs(self) -> list[FeatureSpec]:
        return list(self._specs.values())

    def families(self) -> dict[str, str]:
        return {s.name: s.family for s in self._specs.values()}

    def __len__(self) -> int:
        return len(self._specs)

    # ------------------------------------------------------------------ #

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute every registered feature on a canonical OHLCV frame."""
        cols = [spec.compute(df) for spec in self._specs.values()]
        return pd.concat(cols, axis=1)

    # ------------------------------------------------------------------ #

    def verify_causality(
        self,
        df: pd.DataFrame,
        checkpoints: Iterable[int] | None = None,
        atol: float = 1e-9,
    ) -> list[CausalityViolation]:
        """Prove features unchanged when future data is truncated away.

        For each checkpoint ``t`` we recompute all features on ``df.iloc[:t+1]``
        and compare the value at ``t`` with the value computed on the full
        history. Any mismatch is a look-ahead leak.
        """
        n = len(df)
        if checkpoints is None:
            checkpoints = [int(n * f) for f in (0.55, 0.7, 0.85, 0.97)]
        full = self.compute(df)
        violations: list[CausalityViolation] = []
        for t in checkpoints:
            if t < 1 or t >= n:
                continue
            truncated = self.compute(df.iloc[: t + 1])
            for name in self.names:
                a = full[name].iloc[t]
                b = truncated[name].iloc[-1]
                both_nan = bool(np.isnan(a) and np.isnan(b))
                if not both_nan and not np.isclose(a, b, atol=atol, rtol=1e-7, equal_nan=True):
                    violations.append(
                        CausalityViolation(
                            feature=name,
                            timestamp=df.index[t],
                            full_value=float(a),
                            truncated_value=float(b),
                        )
                    )
        if violations:
            logger.error("causality violations detected: %s", {v.feature for v in violations})
        return violations


@dataclass(slots=True)
class FeaturePanel:
    """Pooled (date, symbol) panel of features ready for modelling."""

    X: pd.DataFrame  # MultiIndex (date, symbol) x feature columns
    families: dict[str, str] = field(default_factory=dict)
    reliability: pd.Series | None = None  # per-row data reliability weight

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def dates(self) -> pd.DatetimeIndex:
        return self.X.index.get_level_values(0).unique().sort_values()

    def for_symbol(self, symbol: str) -> pd.DataFrame:
        return self.X.xs(symbol, level=1)

    def select(self, columns: list[str]) -> FeaturePanel:
        return FeaturePanel(
            X=self.X[columns],
            families={k: v for k, v in self.families.items() if k in columns},
            reliability=self.reliability,
        )
