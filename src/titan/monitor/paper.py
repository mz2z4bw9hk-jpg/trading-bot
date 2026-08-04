"""Paper-tracking: log live scan predictions, grade them against what the
market actually did, and surface calibration decay.

The loop the VALIDATION protocol demands (§4: paper-track before capital)
reduced to two commands:

- ``titan scan`` logs every emitted signal here automatically (idempotent
  per (symbol, date) — re-scanning the same bar does not double-count).
- ``titan track resolve`` re-runs the exact triple-barrier labeller over the
  latest data and grades every open prediction whose horizon has elapsed.
  Reusing :func:`titan.labels.triple_barrier.triple_barrier_labels` is the
  point: the tracked outcome is *by construction* the same event the model
  was trained to predict — same entry convention, same barriers, same
  pessimistic tie rule. A hand-rolled re-implementation here would drift.

Resolved predictions feed a :class:`PredictionTracker` (rolling Brier,
calibration table, CUSUM alarm) whose summary lands in
``artifacts/tracking.json`` for the dashboard.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from titan.core.config import LabelConfig
from titan.core.log import get_logger
from titan.labels.triple_barrier import triple_barrier_labels
from titan.monitor.drift import PredictionTracker
from titan.signals.schema import Signal

logger = get_logger(__name__)

DEFAULT_BASELINE_BRIER = 0.25  # coin-flip Brier: alarm-only-on-worse fallback


class PaperTrackingStore:
    """Append-only JSON store of live predictions and their resolutions."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._records: list[dict] = []
        if self._path.exists():
            self._records = json.loads(self._path.read_text()).get("records", [])

    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"records": self._records}, indent=1))

    @property
    def records(self) -> list[dict]:
        """The raw log, for consumers that derive state from it (the account)."""
        return list(self._records)

    def _seen(self) -> set[tuple[str, str]]:
        return {(r["symbol"], r["date"]) for r in self._records}

    def log_signals(self, signals: Iterable[Signal]) -> int:
        """Log new signals; returns how many were new. Idempotent per (symbol, date)."""
        seen = self._seen()
        added = 0
        for s in signals:
            key = (s.symbol, str(s.date.date()))
            if key in seen:
                continue
            self._records.append(
                {
                    "symbol": s.symbol,
                    "date": str(s.date.date()),
                    "probability": round(s.probability, 4),
                    "probability_low": None if s.probability_low is None else round(s.probability_low, 4),
                    "probability_high": None if s.probability_high is None else round(s.probability_high, 4),
                    "grade": s.trade_grade.value,
                    "model_version": s.model_version,
                    # Account fields: what the signal committed to, so the
                    # forward ledger can price it in dollars later. Recorded at
                    # log time because the signal object is gone by resolve.
                    "side": s.side.value,
                    "source": s.source,
                    "asset_class": s.asset_class,
                    "size_fraction": round(s.position_size_fraction, 6),
                    "risk_percentage": round(s.risk_percentage, 4),
                    "cost_estimate": round(s.cost_estimate, 6),
                    # Margin terms as of the entry. The ledger needs them to
                    # post the right cash and to know where this position dies;
                    # config can change between now and the resolve that grades
                    # it, and re-deriving them then would rewrite history.
                    "leverage": round(s.leverage, 4),
                    "margin_fraction": round(s.margin_fraction, 6),
                    "liquidation_price": (
                        None if s.liquidation_price is None else round(s.liquidation_price, 8)
                    ),
                    "signal_entry": round(s.market_entry, 8),
                    "stop_loss": round(s.stop_loss, 8),
                    "take_profit_levels": [round(t, 8) for t in s.take_profit_levels],
                    "logged_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                    "outcome": None,
                    "touch": None,
                    "ret": None,
                    "bars_held": None,
                    "resolved_at": None,
                    "exit_date": None,
                    "entry_price": None,
                    "exit_price": None,
                }
            )
            seen.add(key)
            added += 1
        if added:
            self._save()
        return added

    # ------------------------------------------------------------------ #

    def resolve(self, frames: dict[str, pd.DataFrame], label_cfg: LabelConfig) -> int:
        """Grade every open prediction the data can now decide.

        A prediction resolves once its symbol's frame extends ``horizon_bars``
        past the decision date: the triple-barrier labeller is re-run on the
        current frame and the row at the decision date IS the outcome.
        Predictions whose horizon has not elapsed stay open; symbols missing
        from ``frames`` are left open and reported, not dropped.
        """
        open_by_symbol: dict[str, list[dict]] = {}
        for r in self._records:
            if r["outcome"] is None:
                open_by_symbol.setdefault(r["symbol"], []).append(r)
        if not open_by_symbol:
            return 0

        resolved = 0
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        for symbol, records in open_by_symbol.items():
            frame = frames.get(symbol)
            if frame is None:
                logger.warning("paper-track: %s not in dataset; %d stay open", symbol, len(records))
                continue
            try:
                labels = triple_barrier_labels(frame, label_cfg).frame
            except ValueError as exc:
                logger.warning("paper-track: cannot label %s: %s", symbol, exc)
                continue
            dates = pd.DatetimeIndex(labels.index).tz_convert("UTC").date
            by_date = {str(d): i for i, d in enumerate(dates)}
            for rec in records:
                pos = by_date.get(rec["date"])
                if pos is None:  # horizon not elapsed yet (or warm-up edge)
                    continue
                row = labels.iloc[pos]
                rec["outcome"] = int(row["label"])
                rec["touch"] = str(row["touch"])
                rec["ret"] = round(float(row["ret"]), 5)
                rec["bars_held"] = int(row["bars_held"])
                rec["resolved_at"] = now
                # Fills and the exit bar, for the forward account ledger. The
                # exit date is only knowable here, where the bar calendar is in
                # hand: bars_held alone cannot be mapped back to a date.
                entry_price = float(row["entry"])
                rec["entry_price"] = round(entry_price, 8)
                rec["exit_price"] = round(entry_price * float(np.exp(row["ret"])), 8)
                exit_pos = min(pos + int(row["bars_held"]), len(dates) - 1)
                rec["exit_date"] = str(dates[exit_pos])
                resolved += 1
        if resolved:
            self._save()
        logger.info("paper-track: resolved %d predictions", resolved)
        return resolved

    # ------------------------------------------------------------------ #

    def tracker(self, baseline_brier: float = DEFAULT_BASELINE_BRIER) -> PredictionTracker:
        """Materialize a PredictionTracker over this store (oldest first)."""
        tracker = PredictionTracker(baseline_brier=baseline_brier)
        for r in sorted(self._records, key=lambda r: r["date"]):
            tracker.log_prediction(r["date"], r["symbol"], float(r["probability"]))
            if r["outcome"] is not None:
                tracker.resolve(r["date"], r["symbol"], int(r["outcome"]))
        return tracker

    def summary(self, baseline_brier: float = DEFAULT_BASELINE_BRIER) -> dict:
        out = self.tracker(baseline_brier).summary()
        open_records = [r for r in self._records if r["outcome"] is None]
        resolved = [r for r in self._records if r["outcome"] is not None]
        out["n_open"] = len(open_records)
        out["hit_rate"] = (
            round(sum(r["outcome"] for r in resolved) / len(resolved), 4) if resolved else None
        )
        out["last_logged"] = max((r["date"] for r in self._records), default=None)
        out["last_resolved"] = max((r["date"] for r in resolved), default=None)
        out["store"] = str(self._path)
        return out
