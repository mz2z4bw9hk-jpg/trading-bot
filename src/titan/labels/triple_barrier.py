"""Triple-barrier labelling with volatility-scaled barriers.

Methodology after López de Prado (Advances in Financial Machine Learning,
2018), adapted to TITAN's execution convention:

- A decision made on bar ``t`` (features use data through ``t``'s close) is
  executed at the *next bar's open* — the entry price is ``open[t+1]``.
- Take-profit and stop-loss barriers are placed at ``±k·sigma_t`` around the
  entry, where ``sigma_t`` is the EWMA volatility known at ``t`` (causal).
- The label is 1 if the take-profit is touched before the stop within the
  horizon, else 0. If both barriers are touched inside the same bar the stop
  is assumed to have been hit first (pessimistic; matches the backtester).
- If neither barrier is touched, the label is the sign of the terminal return.

The event end time ``t1`` is recorded for every sample — purged
cross-validation is impossible without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from titan.core.config import LabelConfig


@dataclass(slots=True)
class LabelSet:
    """Labels plus event metadata, indexed by decision time."""

    frame: pd.DataFrame  # columns: label, ret, entry, sigma, t1, bars_held, touch

    @property
    def labels(self) -> pd.Series:
        return self.frame["label"]

    @property
    def t1(self) -> pd.Series:
        return self.frame["t1"]


def ewma_volatility(close: pd.Series, span: int, floor: float = 1e-4) -> pd.Series:
    """Causal EWMA std of daily log returns."""
    ret = np.log(close).diff()
    vol = ret.ewm(span=span, adjust=False, min_periods=span).std()
    return vol.clip(lower=floor)


def triple_barrier_labels(df: pd.DataFrame, cfg: LabelConfig) -> LabelSet:
    """Compute triple-barrier labels for one instrument's canonical OHLCV frame."""
    n = len(df)
    horizon = cfg.horizon_bars
    if n < horizon + cfg.vol_span + 2:
        raise ValueError("not enough bars for labelling")

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    sigma = ewma_volatility(df["close"], cfg.vol_span, cfg.min_vol_floor).to_numpy()

    # Decision at t, entry at open[t+1]. Last usable decision needs t+horizon <= n-1.
    max_t = n - 1 - horizon
    idx = np.arange(0, max_t + 1)

    entry = open_[idx + 1]
    tp = entry * np.exp(cfg.tp_sigma * sigma[idx])
    sl = entry * np.exp(-cfg.sl_sigma * sigma[idx])

    # Path matrices: bars t+1 .. t+horizon for each decision t.
    offsets = np.arange(1, horizon + 1)
    path_rows = idx[:, None] + offsets[None, :]
    path_high = high[path_rows]
    path_low = low[path_rows]

    tp_hit = path_high >= tp[:, None]
    sl_hit = path_low <= sl[:, None]

    big = horizon + 1
    first_tp = np.where(tp_hit.any(axis=1), tp_hit.argmax(axis=1) + 1, big)
    first_sl = np.where(sl_hit.any(axis=1), sl_hit.argmax(axis=1) + 1, big)

    # Pessimistic tie-break: same-bar touch counts as stop first.
    sl_first = (first_sl <= first_tp) & (first_sl <= horizon)
    tp_first = (first_tp < first_sl) & (first_tp <= horizon)

    terminal_close = close[idx + horizon]
    label = np.where(tp_first, 1, np.where(sl_first, 0, (terminal_close > entry).astype(int)))

    ret = np.where(
        tp_first,
        cfg.tp_sigma * sigma[idx],
        np.where(sl_first, -cfg.sl_sigma * sigma[idx], np.log(terminal_close / entry)),
    )
    bars_held = np.where(tp_first, first_tp, np.where(sl_first, first_sl, horizon))
    touch = np.where(tp_first, "tp", np.where(sl_first, "sl", "time"))

    # True excursions over the event's actual life (for analogue estimates).
    row_pick = np.arange(len(idx))
    held_col = bars_held.astype(int) - 1
    mae = np.minimum.accumulate(path_low, axis=1)[row_pick, held_col] / entry - 1.0
    mfe = np.maximum.accumulate(path_high, axis=1)[row_pick, held_col] / entry - 1.0

    dates = df.index[idx]
    t1 = df.index[idx + bars_held]

    frame = pd.DataFrame(
        {
            "label": label.astype(int),
            "ret": ret,
            "entry": entry,
            "sigma": sigma[idx],
            "t1": pd.DatetimeIndex(t1),
            "bars_held": bars_held.astype(int),
            "touch": touch,
            "mae": mae,
            "mfe": mfe,
        },
        index=dates,
    )
    # Drop rows where volatility was not yet defined (warm-up).
    frame = frame[np.isfinite(frame["sigma"]) & (frame["sigma"] > 0)]
    return LabelSet(frame=frame)


def average_uniqueness(
    starts: pd.DatetimeIndex, ends: pd.DatetimeIndex, calendar: pd.DatetimeIndex
) -> pd.Series:
    """Average uniqueness weight of overlapping events (AFML ch.4).

    Overlapping labels are not independent samples; weighting by average
    uniqueness stops the model from seeing one market episode many times and
    mistaking it for many independent observations.
    """
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    s = pos.reindex(starts).to_numpy()
    e = pos.reindex(ends).to_numpy()
    if np.isnan(s).any() or np.isnan(e).any():
        raise ValueError("event start/end not on calendar")
    s = s.astype(int)
    e = e.astype(int)

    concurrency = np.zeros(len(calendar) + 1)
    np.add.at(concurrency, s, 1)
    np.add.at(concurrency, e + 1, -1)
    concurrency = np.cumsum(concurrency)[:-1]

    uniqueness = np.array(
        [np.mean(1.0 / np.clip(concurrency[si : ei + 1], 1, None)) for si, ei in zip(s, e)]
    )
    return pd.Series(uniqueness, index=starts, name="uniqueness")


def build_label_panel(
    frames: dict[str, pd.DataFrame], cfg: LabelConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Labels for every symbol, stacked to a (date, symbol) panel.

    Returns (label frame, sample weights). Weights combine average uniqueness
    (overlap correction) with return magnitude (large-move events carry more
    information about the barrier geometry than dead-zone events).
    """
    blocks: list[pd.DataFrame] = []
    weights: list[pd.Series] = []
    for sym, df in frames.items():
        ls = triple_barrier_labels(df, cfg)
        block = ls.frame.copy()
        uniq = average_uniqueness(
            pd.DatetimeIndex(block.index), pd.DatetimeIndex(block["t1"]), df.index
        )
        w = uniq * (block["ret"].abs() / block["sigma"]).clip(0.25, 4.0)
        block.index = pd.MultiIndex.from_product([block.index, [sym]], names=["date", "symbol"])
        w.index = block.index
        blocks.append(block)
        weights.append(w)
    panel = pd.concat(blocks).sort_index()
    weight = pd.concat(weights).sort_index().rename("weight")
    return panel, weight
