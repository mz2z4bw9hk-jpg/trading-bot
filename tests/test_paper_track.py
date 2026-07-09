"""Paper-tracking: log -> resolve -> calibration, on hand-built price paths."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from titan.cli import main
from titan.core.config import LabelConfig
from titan.core.types import Side
from titan.labels.triple_barrier import triple_barrier_labels
from titan.monitor.paper import PaperTrackingStore
from titan.signals.schema import Signal

CFG = LabelConfig(horizon_bars=10, tp_sigma=2.0, sl_sigma=1.5, vol_span=5)


def _flat_frame(n: int = 120, price: float = 100.0) -> pd.DataFrame:
    """Low-vol random walk long enough to label, with a controllable tail."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = price * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )


def _signal(symbol: str, date: pd.Timestamp, p: float = 0.66) -> Signal:
    return Signal(symbol=symbol, date=date, side=Side.LONG, probability=p,
                  probability_low=p - 0.04, probability_high=p + 0.04,
                  model_version="test")


def test_log_is_idempotent_and_persistent(tmp_path):
    frame = _flat_frame()
    store_path = tmp_path / "paper_track.json"
    store = PaperTrackingStore(store_path)
    date = frame.index[50]
    assert store.log_signals([_signal("AAA", date), _signal("BBB", date)]) == 2
    assert store.log_signals([_signal("AAA", date)]) == 0  # same (symbol, date)
    reloaded = PaperTrackingStore(store_path)  # round-trips through disk
    assert reloaded.summary()["n_predictions"] == 2
    assert reloaded.summary()["n_open"] == 2


def test_resolution_matches_the_labeller_exactly(tmp_path):
    """The graded outcome must be the triple-barrier label, bit for bit."""
    frame = _flat_frame(160)
    labels = triple_barrier_labels(frame, CFG).frame
    store = PaperTrackingStore(tmp_path / "pt.json")

    # log predictions on three decision dates with known labels
    picks = [labels.index[30], labels.index[60], labels.index[90]]
    store.log_signals([_signal("AAA", d) for d in picks])
    resolved = store.resolve({"AAA": frame}, CFG)
    assert resolved == 3

    by_date = {r["date"]: r for r in store._records}
    for d in picks:
        rec = by_date[str(d.date())]
        row = labels.loc[d]
        assert rec["outcome"] == int(row["label"])
        assert rec["touch"] == str(row["touch"])
        assert rec["bars_held"] == int(row["bars_held"])


def test_unelapsed_and_unknown_symbols_stay_open(tmp_path):
    frame = _flat_frame(120)
    store = PaperTrackingStore(tmp_path / "pt.json")
    last_bar = frame.index[-1]  # horizon cannot have elapsed
    store.log_signals([_signal("AAA", last_bar), _signal("GHOST", frame.index[40])])
    assert store.resolve({"AAA": frame}, CFG) == 0
    s = store.summary()
    assert s["n_open"] == 2 and s["n_resolved"] == 0


def test_summary_calibration_and_cusum(tmp_path):
    """Well-calibrated stream: no alarm. Systematically wrong stream: alarm."""
    rng = np.random.default_rng(9)
    good = PaperTrackingStore(tmp_path / "good.json")
    bad = PaperTrackingStore(tmp_path / "bad.json")
    dates = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
    for i, d in enumerate(dates):
        p = float(rng.uniform(0.55, 0.9))
        good.log_signals([_signal(f"S{i}", d, p)])
        bad.log_signals([_signal(f"S{i}", d, p)])
        good._records[-1]["outcome"] = int(rng.uniform() < p)   # world agrees
        bad._records[-1]["outcome"] = int(rng.uniform() < 0.2)  # world disagrees

    baseline = 0.20
    assert good.summary(baseline)["cusum_alarm"] is False
    bad_summary = bad.summary(baseline)
    assert bad_summary["cusum_alarm"] is True
    assert bad_summary["hit_rate"] < 0.35
    assert len(good.summary(baseline)["calibration"]) > 0


def test_cli_track_status_and_resolve_exit_codes(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
run: {{ artifacts_dir: {tmp_path / "artifacts"}, log_level: WARNING }}
data: {{ provider: synthetic, cache_dir: {tmp_path / "cache"}, bars: 700 }}
model: {{ store_dir: {tmp_path / "models"} }}
universe:
  benchmark: IDX
  instruments:
    - {{ symbol: AAA, asset_class: equity, sector: tech }}
    - {{ symbol: BBB, asset_class: equity, sector: tech }}
"""
    )
    assert main(["track", "status", "--config", str(cfg)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_predictions"] == 0 and out["newly_resolved"] == 0
    assert (tmp_path / "artifacts" / "tracking.json").exists()

    # seed one prediction on a date the synthetic market can already decide,
    # then resolve end-to-end through the provider
    store = PaperTrackingStore(tmp_path / "models" / "paper_track.json")
    from titan.core.config import load_config
    from titan.data.store import MarketDataStore

    dataset = MarketDataStore(load_config(cfg, {}).data, load_config(cfg, {}).universe, seed=7).load()
    frame = dataset.frames["AAA"]
    store.log_signals([_signal("AAA", frame.index[-60])])
    assert main(["track", "resolve", "--config", str(cfg)]) in (0, 3)  # 3 = CUSUM alarm path
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["newly_resolved"] == 1
    assert resolved["n_resolved"] == 1


def test_scan_payload_survives_missing_intervals():
    """Signals logged from pre-interval bundles carry None bands, not crashes."""
    s = Signal(symbol="X", date=pd.Timestamp("2024-01-05", tz="UTC"), side=Side.LONG,
               probability=0.6)
    d = s.to_dict()
    assert d["probability_low"] is None and d["probability_high"] is None
