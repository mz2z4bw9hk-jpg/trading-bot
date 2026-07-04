"""Purged walk-forward cross-validation with embargo.

Standard K-fold on overlapping financial labels leaks: a training event whose
life ``[t, t1]`` extends into the test window shares future information with
test samples. Purging removes such events; the embargo removes training events
starting within a buffer *before* the test window whose serial correlation
with early test samples would otherwise leak (AFML ch.7).

Splits are strictly forward-chaining (train always precedes test), because
markets are non-stationary and a model must be judged only on data from
*after* everything it saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from titan.core.config import CVConfig


def to_naive_utc(values) -> np.ndarray:
    """Datetime-like -> tz-naive UTC ``datetime64[ns]`` array for fast compares."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.to_numpy(dtype="datetime64[ns]")


@dataclass(slots=True)
class Fold:
    fold: int
    train_idx: np.ndarray  # positional row indices into the panel
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class PurgedWalkForward:
    """Forward-chaining splits over a (date, symbol) panel with event purging."""

    def __init__(self, cfg: CVConfig) -> None:
        self._cfg = cfg

    def split(self, dates: pd.DatetimeIndex, t1: pd.Series) -> list[Fold]:
        """Build folds.

        Parameters
        ----------
        dates:
            Per-row decision dates of the panel (level 0 of the MultiIndex).
        t1:
            Per-row event end dates (from the triple-barrier labeller).
        """
        cfg = self._cfg
        date_values = to_naive_utc(dates)
        t1_values = to_naive_utc(t1)
        unique_dates = np.unique(date_values)
        n_dates = len(unique_dates)

        usable = n_dates - cfg.min_train_bars
        if usable < cfg.test_bars:
            raise ValueError(
                f"not enough history: {n_dates} dates, need >= "
                f"{cfg.min_train_bars + cfg.test_bars}"
            )
        n_folds = min(cfg.n_folds, usable // cfg.test_bars)
        if n_folds < 1:
            raise ValueError("cannot build a single fold with the given sizes")

        folds: list[Fold] = []
        # Test blocks tile the tail of the calendar; train precedes each block.
        first_test_pos = n_dates - n_folds * cfg.test_bars
        for k in range(n_folds):
            test_start_pos = first_test_pos + k * cfg.test_bars
            test_end_pos = test_start_pos + cfg.test_bars - 1
            test_start = unique_dates[test_start_pos]
            test_end = unique_dates[test_end_pos]

            embargo_pos = max(0, test_start_pos - cfg.embargo_bars)
            embargo_date = unique_dates[embargo_pos]

            if cfg.scheme == "rolling":
                train_start_pos = max(0, test_start_pos - cfg.min_train_bars - cfg.embargo_bars)
                train_start = unique_dates[train_start_pos]
            else:  # expanding
                train_start = unique_dates[0]

            # Purge: training events must START before the embargo AND END
            # before the embargo boundary (no life overlap with the test).
            train_mask = (
                (date_values >= train_start)
                & (date_values < embargo_date)
                & (t1_values < embargo_date)
            )
            test_mask = (date_values >= test_start) & (date_values <= test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            folds.append(
                Fold(
                    fold=k,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    train_end=pd.Timestamp(date_values[train_idx].max()).tz_localize("UTC"),
                    test_start=pd.Timestamp(test_start).tz_localize("UTC"),
                    test_end=pd.Timestamp(test_end).tz_localize("UTC"),
                )
            )
        if not folds:
            raise ValueError("no valid folds produced")
        return folds


def assert_no_leakage(folds: list[Fold], dates: pd.DatetimeIndex, t1: pd.Series) -> None:
    """Hard invariant check used by tests and by the pipeline in debug mode."""
    date_values = to_naive_utc(dates)
    t1_values = to_naive_utc(t1)
    for fold in folds:
        ts = pd.Timestamp(fold.test_start)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        test_start = np.datetime64(ts.to_datetime64())
        train_t1 = t1_values[fold.train_idx]
        train_dates = date_values[fold.train_idx]
        if (train_t1 >= test_start).any():
            raise AssertionError(f"fold {fold.fold}: training event overlaps test window")
        if (train_dates >= test_start).any():
            raise AssertionError(f"fold {fold.fold}: training decision inside test window")
        test_dates = date_values[fold.test_idx]
        if (test_dates < test_start).any():
            raise AssertionError(f"fold {fold.fold}: test row before test window")
