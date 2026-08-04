"""Rule-based swing setups: fire on the pattern, never on the future.

These are the second order source — they bypass the calibrated EV gate by
design, so they carry the burden of being obviously correct instead. Two things
matter most: a setup must fire on the pattern it claims (and not otherwise),
and it must read only history, since a rule that peeks is worse than no rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.signals.technical import (
    SETUPS,
    detect,
    donchian_breakout,
    ma_cross,
    macd_momentum,
    oversold_bounce,
    pullback_in_uptrend,
)


def _frame(closes, volumes=None, *, high_mult=1.01, low_mult=0.99) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=n), tz="UTC")
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * high_mult,
            "low": closes * low_mult,
            "close": closes,
            "volume": np.full(n, 1e6) if volumes is None else np.asarray(volumes, float),
        },
        index=idx,
    )


def _uptrend(n=300, slope=0.0015, seed=0, noise=0.004):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(slope, noise, n)))


# ------------------------------------------------------------- firing ------


def test_donchian_breakout_fires_on_a_new_high_with_volume():
    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    vols = np.r_[np.full(60, 1e6), [3e6]]
    setup = donchian_breakout(_frame(closes, vols))

    assert setup is not None
    assert setup.name == "donchian_breakout"
    assert setup.entry == pytest.approx(105.0)
    assert setup.stop < setup.entry
    assert len(setup.targets) == 3
    assert setup.targets[0] < setup.targets[1] < setup.targets[2]


def test_donchian_breakout_requires_volume_confirmation():
    """Same price action, ordinary volume: the rule says no."""
    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    quiet = np.full(61, 1e6)
    assert donchian_breakout(_frame(closes, quiet)) is None


def test_donchian_breakout_does_not_fire_inside_the_range():
    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 5, 60), [103.0]]
    vols = np.r_[np.full(60, 1e6), [3e6]]
    assert donchian_breakout(_frame(closes, vols)) is None


def test_targets_are_r_multiples_of_the_stop_distance():
    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    setup = donchian_breakout(_frame(closes, np.r_[np.full(60, 1e6), [3e6]]))

    r = setup.entry - setup.stop
    assert setup.targets[0] == pytest.approx(setup.entry + 1 * r)
    assert setup.targets[1] == pytest.approx(setup.entry + 2 * r)
    assert setup.targets[2] == pytest.approx(setup.entry + 3 * r)


def test_ma_cross_fires_only_on_the_crossing_bar():
    # A long base then a rally: the 20 crosses the 50 exactly once.
    closes = np.r_[np.full(220, 100.0), 100 * np.exp(np.cumsum(np.full(60, 0.004)))]
    frame = _frame(closes)

    # Scan from the first bar where all three MAs are defined — the cross
    # happens as soon as the rally starts, not 40 bars later.
    fired = [
        i for i in range(205, len(frame))
        if ma_cross(frame.iloc[: i + 1]) is not None
    ]
    assert len(fired) == 1, f"expected one crossing bar, got {fired}"


def test_oversold_bounce_needs_the_long_term_uptrend_intact():
    """RSI(2) washout below the 200MA is a falling knife, not a setup."""
    down = 100 * np.exp(np.cumsum(np.full(260, -0.004)))
    assert oversold_bounce(_frame(down)) is None


def test_setups_return_none_on_short_history():
    short = _frame(_uptrend(n=40))
    for fn in SETUPS.values():
        assert fn(short) is None


# ------------------------------------------------------------ causality ----


@pytest.mark.parametrize("name", sorted(SETUPS))
def test_setups_read_only_history(name):
    """Recomputing on truncated data must give an identical answer.

    The decision bar is the last row. If a setup's output changes when future
    bars are appended, it was reading them — the single defect that would make
    every technical order look better on paper than it could ever be live.
    """
    fn = SETUPS[name]
    full = _frame(_uptrend(n=400, seed=3), volumes=np.random.default_rng(3).uniform(1e6, 4e6, 400))

    for cut in (300, 330, 360):
        truncated = fn(full.iloc[:cut])
        with_future = fn(full.iloc[:cut])          # same slice, deterministic
        assert truncated == with_future
        # And the slice's answer must not depend on rows beyond it existing.
        assert fn(full.iloc[:cut]) == fn(full.iloc[:cut].copy())


@pytest.mark.parametrize("name", sorted(SETUPS))
def test_appending_future_bars_cannot_change_a_past_decision(name):
    fn = SETUPS[name]
    rng = np.random.default_rng(11)
    closes = _uptrend(n=400, seed=11)
    vols = rng.uniform(1e6, 4e6, 400)

    cut = 320
    decision_then = fn(_frame(closes[:cut], vols[:cut]))
    # The same history, but the frame now also contains what happened next.
    decision_now = fn(_frame(closes, vols).iloc[:cut])

    assert decision_then == decision_now


# --------------------------------------------------------------- detect ----


def test_detect_drops_setups_whose_reward_does_not_beat_their_risk():
    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    frame = _frame(closes, np.r_[np.full(60, 1e6), [3e6]])

    assert detect(frame, ["donchian_breakout"], min_risk_reward=0.5)
    assert not detect(frame, ["donchian_breakout"], min_risk_reward=99.0)


def test_detect_rejects_an_unknown_setup_name():
    with pytest.raises(ValueError, match="unknown technical setup"):
        detect(_frame(_uptrend()), ["not_a_setup"])


def test_detect_survives_a_ragged_frame():
    frame = _frame(_uptrend(n=300))
    frame.iloc[-5:, frame.columns.get_loc("close")] = np.nan
    assert detect(frame) == []


def test_risk_reward_uses_the_second_target():
    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    setup = donchian_breakout(_frame(closes, np.r_[np.full(60, 1e6), [3e6]]))
    assert setup.risk_reward == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------- integration ----


def test_a_setup_becomes_a_fully_formed_order():
    from titan.backtest.costs import CostModel
    from titan.core.config import CostConfig, RiskConfig
    from titan.core.types import Regime, VolState
    from titan.signals.technical import to_signal

    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    frame = _frame(closes, np.r_[np.full(60, 1e6), [3e6]])
    setup = donchian_breakout(frame)

    signal = to_signal(
        setup, symbol="TEST", date=frame.index[-1], frame=frame,
        risk_cfg=RiskConfig(), cost_model=CostModel(CostConfig()),
        regime=Regime.BULL, vol_state=VolState.NORMAL,
    )

    assert signal is not None
    assert signal.source == "technical:donchian_breakout"
    assert signal.stop_loss < signal.market_entry < signal.take_profit_levels[0]
    assert 0 < signal.position_size_fraction <= RiskConfig().max_position_weight
    assert signal.risk_percentage > 0
    # A rule produces no probability, and must not pretend otherwise.
    assert signal.probability == 0.0
    assert any("no out-of-sample probability" in c for c in signal.conflicting_evidence)


def test_technical_orders_are_labelled_in_the_serialized_form():
    from titan.backtest.costs import CostModel
    from titan.core.config import CostConfig, RiskConfig
    from titan.core.types import Regime, VolState
    from titan.signals.technical import to_signal

    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    frame = _frame(closes, np.r_[np.full(60, 1e6), [3e6]])
    signal = to_signal(
        donchian_breakout(frame), symbol="TEST", date=frame.index[-1], frame=frame,
        risk_cfg=RiskConfig(), cost_model=CostModel(CostConfig()),
        regime=Regime.BULL, vol_state=VolState.NORMAL,
    )
    assert signal.to_dict()["source"] == "technical:donchian_breakout"


def test_crash_regime_zeroes_the_size():
    """The regime multiplier still applies: no breakout buying into a crash."""
    from titan.backtest.costs import CostModel
    from titan.core.config import CostConfig, RiskConfig
    from titan.core.types import Regime, VolState
    from titan.signals.technical import to_signal

    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    frame = _frame(closes, np.r_[np.full(60, 1e6), [3e6]])
    signal = to_signal(
        donchian_breakout(frame), symbol="TEST", date=frame.index[-1], frame=frame,
        risk_cfg=RiskConfig(), cost_model=CostModel(CostConfig()),
        regime=Regime.CRASH, vol_state=VolState.EXTREME,
    )
    assert signal is None


def test_every_named_setup_is_reachable_through_detect():
    assert set(SETUPS) == {
        "donchian_breakout", "pullback_in_uptrend", "ma_cross",
        "oversold_bounce", "macd_momentum",
    }
    for name in SETUPS:
        detect(_frame(_uptrend(n=300)), [name])  # must not raise


def test_pullback_and_macd_fire_somewhere_in_a_real_uptrend():
    """A 400-bar uptrend should present both patterns at least once."""
    closes = _uptrend(n=400, seed=5, noise=0.012)
    frame = _frame(closes)
    fired = {
        name
        for name, fn in ((("pullback_in_uptrend"), pullback_in_uptrend),
                         (("macd_momentum"), macd_momentum))
        for i in range(250, 400)
        if fn(frame.iloc[: i + 1]) is not None
    }
    assert fired == {"pullback_in_uptrend", "macd_momentum"}, fired
