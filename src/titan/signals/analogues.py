"""Historical analogue retrieval.

For each candidate setup we retrieve its k nearest neighbours in
(standardized, selected) feature space from the TRAINING window and read off
what actually happened after those states: outcome quantiles, hit rate, time
to resolution, and typical adverse/favourable excursions. This grounds every
signal in precedent instead of leaving the probability floating in the
abstract, and the similarity score doubles as an out-of-distribution alarm:
a setup with no close historical analogue deserves less capital regardless
of what the model says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class AnalogueReport:
    similarity: float             # 0-1, exp(-distance / calibration scale)
    n: int
    hit_rate: float               # fraction of analogues that hit take-profit
    ret_quantiles: dict[str, float]
    median_bars_held: float
    mae_p75: float                # typical-bad adverse excursion (negative)
    mfe_p50: float


class AnalogueIndex:
    def __init__(self, k: int = 50, seed: int = 7) -> None:
        self._k = k
        self._seed = seed
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        self._nn: NearestNeighbors | None = None
        self._outcomes: pd.DataFrame | None = None
        self._scale: float = 1.0

    def fit(self, X: pd.DataFrame, outcomes: pd.DataFrame) -> AnalogueIndex:
        """
        Parameters
        ----------
        X:
            Training features (selected columns only).
        outcomes:
            Aligned frame with columns label, ret, bars_held, mae, mfe.
        """
        if len(X) != len(outcomes):
            raise ValueError("X and outcomes must align")
        Z = self._scaler.fit_transform(self._imputer.fit_transform(X.to_numpy(dtype=float)))
        k = min(self._k, len(X) - 1)
        self._nn = NearestNeighbors(n_neighbors=k).fit(Z)
        self._outcomes = outcomes.reset_index(drop=True)

        # Distance normalization: median k-th neighbour distance on a sample
        # of the training set itself defines "typical closeness".
        rng = np.random.default_rng(self._seed)
        sample = Z[rng.choice(len(Z), size=min(400, len(Z)), replace=False)]
        dist, _ = self._nn.kneighbors(sample)
        self._scale = float(np.median(dist[:, -1])) or 1.0
        return self

    def query(self, x_row: pd.Series | np.ndarray) -> AnalogueReport:
        if self._nn is None or self._outcomes is None:
            raise RuntimeError("AnalogueIndex must be fit first")
        x = np.asarray(x_row, dtype=float).reshape(1, -1)
        z = self._scaler.transform(self._imputer.transform(x))
        dist, idx = self._nn.kneighbors(z)
        rows = self._outcomes.iloc[idx[0]]
        rets = rows["ret"].to_numpy()
        similarity = float(np.exp(-float(np.median(dist)) / self._scale))
        return AnalogueReport(
            similarity=min(similarity, 1.0),
            n=len(rows),
            hit_rate=float(rows["label"].mean()),
            ret_quantiles={
                f"p{q}": float(np.quantile(rets, q / 100.0)) for q in (10, 25, 50, 75, 90)
            },
            median_bars_held=float(rows["bars_held"].median()),
            mae_p75=float(np.quantile(rows["mae"].to_numpy(), 0.25)),  # 75th worst = 25th quantile
            mfe_p50=float(rows["mfe"].median()),
        )
