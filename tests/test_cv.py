"""Purged walk-forward: structural leakage is impossible by construction."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from titan.core.config import CVConfig
from titan.labels.triple_barrier import build_label_panel
from titan.models.cv import PurgedWalkForward, assert_no_leakage


@pytest.fixture(scope="module")
def panel_dates_t1(dataset, cfg):
    labels, _ = build_label_panel(dataset.frames, cfg.labels)
    dates = pd.DatetimeIndex(labels.index.get_level_values(0))
    return dates, labels["t1"]


def test_folds_are_forward_chained(panel_dates_t1, cfg):
    dates, t1 = panel_dates_t1
    folds = PurgedWalkForward(cfg.cv).split(dates, t1)
    assert len(folds) >= 2
    for f in folds:
        assert f.train_end < f.test_start
    for a, b in itertools.pairwise(folds):
        assert a.test_end < b.test_start


def test_no_event_overlap_invariant(panel_dates_t1, cfg):
    dates, t1 = panel_dates_t1
    folds = PurgedWalkForward(cfg.cv).split(dates, t1)
    assert_no_leakage(folds, dates, t1)  # raises on violation


def test_embargo_is_respected(panel_dates_t1):
    dates, t1 = panel_dates_t1
    cv = CVConfig(n_folds=2, embargo_bars=10, min_train_bars=300, test_bars=100)
    folds = PurgedWalkForward(cv).split(dates, t1)
    unique = np.unique(np.asarray(dates.tz_convert("UTC").tz_localize(None), dtype="datetime64[ns]"))
    for f in folds:
        test_start_pos = int(np.searchsorted(unique, np.datetime64(
            pd.Timestamp(f.test_start).tz_convert("UTC").tz_localize(None))))
        max_train = np.asarray(
            dates[f.train_idx].tz_convert("UTC").tz_localize(None), dtype="datetime64[ns]"
        ).max()
        max_train_pos = int(np.searchsorted(unique, max_train))
        assert test_start_pos - max_train_pos >= 10


def test_purge_removes_overlapping_events():
    """A synthetic long-lived event just before the test window must be purged."""
    calendar = pd.bdate_range("2020-01-01", periods=500, tz="UTC")
    dates = pd.DatetimeIndex(calendar)
    # every event lasts 20 bars
    t1 = pd.Series(calendar[np.minimum(np.arange(500) + 20, 499)], index=calendar)
    cv = CVConfig(n_folds=2, embargo_bars=0, min_train_bars=200, test_bars=100)
    folds = PurgedWalkForward(cv).split(dates, t1)
    for f in folds:
        # the last training decision must end before the test window starts
        train_t1_max = pd.DatetimeIndex(t1.iloc[f.train_idx]).max()
        assert train_t1_max < f.test_start


def test_insufficient_history_raises(panel_dates_t1):
    dates, t1 = panel_dates_t1
    cv = CVConfig(n_folds=3, min_train_bars=100_000, test_bars=100)
    with pytest.raises(ValueError):
        PurgedWalkForward(cv).split(dates, t1)
