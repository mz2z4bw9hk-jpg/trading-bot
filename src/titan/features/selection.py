"""Feature selection: redundancy elimination and importance ranking.

Redundancy: hierarchical clustering on the absolute Spearman correlation
matrix; within each cluster only the feature with the strongest univariate
information coefficient survives. Highly collinear features add variance to
importance estimates and invite overfitting without adding information.

Importance: permutation importance on *validation* data (never training data,
where impurity-based importances flatter high-cardinality noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import log_loss

from titan.core.log import get_logger

logger = get_logger(__name__)

_MAX_ROWS = 20_000  # correlation/IC estimates are stable well below this


def _subsample(X: pd.DataFrame, y: pd.Series | None, seed: int) -> tuple[pd.DataFrame, pd.Series | None]:
    if len(X) <= _MAX_ROWS:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), _MAX_ROWS, replace=False))
    return X.iloc[idx], (y.iloc[idx] if y is not None else None)


def univariate_ic(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.Series:
    """Spearman information coefficient of each feature vs the label."""
    Xs, ys = _subsample(X, y, seed)
    assert ys is not None
    y_rank = ys.rank()
    ics = {}
    for col in Xs.columns:
        x = Xs[col]
        mask = x.notna()
        if mask.sum() < 50:
            ics[col] = 0.0
            continue
        ics[col] = float(x[mask].rank().corr(y_rank[mask]))
    return pd.Series(ics, name="ic").fillna(0.0)


@dataclass(slots=True)
class PruneResult:
    kept: list[str]
    dropped: dict[str, str] = field(default_factory=dict)  # dropped -> surviving proxy

    def summary(self) -> str:
        return f"kept {len(self.kept)}, dropped {len(self.dropped)} redundant"


def redundancy_prune(
    X: pd.DataFrame,
    threshold: float = 0.90,
    priority: pd.Series | None = None,
    seed: int = 0,
) -> PruneResult:
    """Drop features whose |Spearman rho| to a surviving feature exceeds threshold.

    ``priority`` (higher = keep) breaks ties inside a cluster; default is the
    feature's variance rank, but callers should pass univariate IC.
    """
    cols = list(X.columns)
    if len(cols) < 2:
        return PruneResult(kept=cols)

    Xs, _ = _subsample(X, None, seed)
    ranked = Xs.rank()
    ranked = ranked.fillna(ranked.median())
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(ranked.to_numpy(), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    dist = 1.0 - np.abs(corr)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    tree = linkage(condensed, method="average")
    clusters = fcluster(tree, t=1.0 - threshold, criterion="distance")

    if priority is None:
        priority = Xs.var().rank()
    priority = priority.reindex(cols).fillna(0.0)

    kept: list[str] = []
    dropped: dict[str, str] = {}
    for cluster_id in np.unique(clusters):
        members = [cols[i] for i in np.where(clusters == cluster_id)[0]]
        best = max(members, key=lambda m: (float(priority[m]), m))
        kept.append(best)
        for m in members:
            if m != best:
                dropped[m] = best

    kept.sort(key=cols.index)
    result = PruneResult(kept=kept, dropped=dropped)
    logger.info("redundancy pruning at |rho|>%.2f: %s", threshold, result.summary())
    return result


def permutation_rank(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 3,
    seed: int = 0,
    max_rows: int = 2_500,
) -> pd.Series:
    """Permutation importance (log-loss degradation) on held-out data.

    Implemented directly rather than via sklearn's helper so any object with
    ``predict_proba`` qualifies (our ensemble is not a sklearn estimator).
    Importance = mean increase in log-loss when the feature is shuffled;
    <= 0 means the model does not use the feature out-of-sample.
    """
    if len(X_val) > max_rows:
        rng_sub = np.random.default_rng(seed)
        idx = np.sort(rng_sub.choice(len(X_val), max_rows, replace=False))
        X_val, y_val = X_val.iloc[idx], y_val.iloc[idx]
    y = y_val.to_numpy()

    def _score(frame: pd.DataFrame) -> float:
        p = np.clip(model.predict_proba(frame)[:, 1], 1e-6, 1 - 1e-6)
        return float(log_loss(y, p))

    baseline = _score(X_val)
    rng = np.random.default_rng(seed)
    importances: dict[str, float] = {}
    for col in X_val.columns:
        original = X_val[col].to_numpy()
        losses = []
        for _ in range(n_repeats):
            shuffled = X_val.copy()
            shuffled[col] = rng.permutation(original)
            losses.append(_score(shuffled))
        importances[col] = float(np.mean(losses) - baseline)
    return pd.Series(importances, name="importance").sort_values(ascending=False)
