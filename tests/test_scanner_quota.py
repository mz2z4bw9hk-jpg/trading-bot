"""Per-asset-class order quotas, and one order per symbol.

The requirement is "the 5 best stocks and the 5 best crypto" — which a single
global top-10 does not deliver. Crypto's daily volatility runs several times an
equity's, so on a ranked-together list the coins take nearly every slot and the
equity book never trades. Ranking inside each class is what makes the split
actually happen.

The second rule tested here is subtler and was a live defect: the model gate
and the rule engine can both fire on the same symbol on the same bar, and the
order panel would show two tickets for it. That is one idea, sized twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.core.config import TitanConfig
from titan.core.types import Regime, Side, TradeGrade, VolState
from titan.scanner.scanner import MarketScanner
from titan.signals.schema import Signal


def _signal(symbol: str, asset_class: str, confidence: float, source="model") -> Signal:
    return Signal(
        symbol=symbol,
        date=pd.Timestamp("2024-06-03", tz="UTC"),
        side=Side.LONG,
        source=source,
        asset_class=asset_class,
        confidence_score=confidence,
        trade_grade=TradeGrade.A,
        market_entry=100.0,
        stop_loss=95.0,
        position_size_fraction=0.05,
    )


def _scan_result(signals):
    from titan.regime.detector import RegimeSnapshot
    from titan.scanner.scanner import ScanResult

    snapshot = RegimeSnapshot(
        regime=Regime.BULL, vol_state=VolState.NORMAL, confidence=0.9,
        posteriors={"bull": 0.9, "range": 0.1},
    )
    return ScanResult(
        date=pd.Timestamp("2024-06-03", tz="UTC"), regime=snapshot, signals=signals
    )


def _scanner(cfg: TitanConfig) -> MarketScanner:
    return MarketScanner(
        cfg, ensemble=None, selected_features=[], generator=None, detector=None
    )


def _cfg(**scanner_kwargs) -> TitanConfig:
    cfg = TitanConfig()
    for k, v in scanner_kwargs.items():
        setattr(cfg.scanner, k, v)
    return cfg


# ------------------------------------------------------------- quotas ------


def test_five_stocks_and_five_crypto_come_out_of_a_lopsided_candidate_pool():
    """Crypto outranks every equity, and the equity book still gets its five."""
    candidates = (
        [_signal(f"COIN{i}", "crypto", 90.0 - i) for i in range(20)]
        + [_signal(f"STOCK{i}", "equity", 60.0 - i) for i in range(20)]
    )
    cfg = _cfg(orders_per_asset_class={"equity": 5, "crypto": 5})

    picked = _scanner(cfg)._select(candidates)

    by_class = {}
    for s in picked:
        by_class.setdefault(s.asset_class, []).append(s)
    assert len(by_class["equity"]) == 5
    assert len(by_class["crypto"]) == 5
    # And each class kept its own best, not the global best.
    assert [s.symbol for s in by_class["crypto"]] == [f"COIN{i}" for i in range(5)]
    assert [s.symbol for s in by_class["equity"]] == [f"STOCK{i}" for i in range(5)]


def test_a_global_top_n_would_not_have_produced_that_split():
    """Guards the reason the quota exists rather than just its arithmetic."""
    candidates = (
        [_signal(f"COIN{i}", "crypto", 90.0 - i) for i in range(20)]
        + [_signal(f"STOCK{i}", "equity", 60.0 - i) for i in range(20)]
    )
    global_top_10 = sorted(candidates, key=lambda s: -s.confidence_score)[:10]

    assert all(s.asset_class == "crypto" for s in global_top_10)


def test_a_class_with_no_candidates_does_not_donate_its_slots():
    """Five crypto setups and no equity ones means five orders, not ten.

    Backfilling the empty half from the other book would quietly double the
    intended crypto exposure on days the equity screen is silent.
    """
    cfg = _cfg(orders_per_asset_class={"equity": 5, "crypto": 5})
    picked = _scanner(cfg)._select(
        [_signal(f"COIN{i}", "crypto", 90.0 - i) for i in range(20)]
    )

    assert len(picked) == 5
    assert all(s.asset_class == "crypto" for s in picked)


def test_an_unlisted_asset_class_falls_back_to_top_n():
    cfg = _cfg(top_n=3, orders_per_asset_class={"crypto": 5})
    picked = _scanner(cfg)._select(
        [_signal(f"ETF{i}", "etf", 80.0 - i) for i in range(10)]
    )
    assert len(picked) == 3


def test_no_quota_configured_keeps_the_old_global_behaviour():
    cfg = _cfg(top_n=4, orders_per_asset_class=None)
    picked = _scanner(cfg)._select(
        [_signal(f"STOCK{i}", "equity", 80.0 - i) for i in range(10)]
    )
    assert len(picked) == 4


def test_a_zero_quota_silences_a_book_entirely():
    cfg = _cfg(orders_per_asset_class={"equity": 0, "crypto": 5})
    picked = _scanner(cfg)._select(
        [_signal("AAPL", "equity", 99.0), _signal("BTC-USD", "crypto", 10.0)]
    )
    assert [s.symbol for s in picked] == ["BTC-USD"]


# ------------------------------------------------------------ one per name --


def test_a_symbol_gets_one_order_even_when_both_engines_fire():
    cfg = _cfg(orders_per_asset_class={"equity": 5})
    picked = _scanner(cfg)._select([
        _signal("AAPL", "equity", 70.0, source="model"),
        _signal("AAPL", "equity", 82.0, source="technical:donchian_breakout"),
    ])
    assert len(picked) == 1


def test_the_model_order_wins_a_collision_even_when_it_scores_lower():
    """Confidence is not comparable across the two engines.

    A rule's score is a description of the pattern; the model's is a margin
    over a validated gate. When both name the same symbol, the one with
    out-of-sample evidence behind it is the order to place.
    """
    cfg = _cfg(orders_per_asset_class={"equity": 5})
    picked = _scanner(cfg)._select([
        _signal("AAPL", "equity", 60.0, source="model"),
        _signal("AAPL", "equity", 85.0, source="technical:ma_cross"),
    ])
    assert [s.source for s in picked] == ["model"]


def test_two_rules_on_one_symbol_collapse_to_the_stronger():
    cfg = _cfg(orders_per_asset_class={"crypto": 5})
    picked = _scanner(cfg)._select([
        _signal("BTC-USD", "crypto", 62.0, source="technical:macd_momentum"),
        _signal("BTC-USD", "crypto", 78.0, source="technical:donchian_breakout"),
    ])
    assert len(picked) == 1
    assert picked[0].source == "technical:donchian_breakout"


def test_selection_is_empty_when_nothing_qualified():
    assert _scanner(_cfg())._select([]) == []


# ------------------------------------------------------- levered orders -----


def _levered_signal(asset_class: str):
    """Run a real technical setup through to_signal on the given book."""
    from titan.backtest.costs import CostModel
    from titan.core.config import CostConfig, LeverageConfig, RiskConfig
    from titan.risk.leverage import LeverageTerms
    from titan.signals.technical import donchian_breakout, to_signal

    closes = np.r_[np.full(60, 100.0) + np.linspace(0, 1, 60), [105.0]]
    n = len(closes)
    idx = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=n), tz="UTC")
    frame = pd.DataFrame(
        {"open": closes * 0.999, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": np.r_[np.full(60, 1e6), [3e6]]},
        index=idx,
    )
    lev_cfg = LeverageConfig(max_leverage={"crypto": 3.0})
    return frame, to_signal(
        donchian_breakout(frame), symbol="TEST",
        date=frame.index[-1], frame=frame,
        risk_cfg=RiskConfig(), cost_model=CostModel(CostConfig()),
        regime=Regime.BULL, vol_state=VolState.NORMAL,
        asset_class=asset_class,
        leverage_terms=LeverageTerms.resolve(lev_cfg, asset_class, "1d"),
    )


def test_a_crypto_setup_comes_out_levered_with_a_liquidation_level():
    _, signal = _levered_signal("crypto")

    assert signal.asset_class == "crypto"
    assert signal.leverage > 1.0
    assert signal.liquidation_price is not None
    # Below the stop, so the stop is a stop and not decoration.
    assert signal.liquidation_price < signal.stop_loss < signal.market_entry
    assert signal.margin_fraction == pytest.approx(
        signal.position_size_fraction / signal.leverage
    )


def test_the_same_setup_on_the_equity_book_stays_cash():
    _, signal = _levered_signal("equity")

    assert signal.leverage == 1.0
    assert signal.liquidation_price is None
    assert signal.margin_fraction == pytest.approx(signal.position_size_fraction)
    assert signal.funding_cost == 0.0


def test_the_order_card_reports_the_levered_risk_not_the_base_risk():
    _, crypto = _levered_signal("crypto")
    _, equity = _levered_signal("equity")

    assert crypto.risk_percentage == pytest.approx(
        equity.risk_percentage * crypto.leverage, rel=1e-6
    )


def test_funding_is_priced_into_the_levered_order_cost():
    _, crypto = _levered_signal("crypto")
    _, equity = _levered_signal("equity")

    assert crypto.funding_cost > 0
    assert crypto.cost_estimate > equity.cost_estimate


def test_the_levered_order_says_so_in_its_conflicting_evidence():
    """A user reading the card must not have to infer the downside."""
    _, crypto = _levered_signal("crypto")
    assert any("leveraged" in c and "liquidation" in c for c in crypto.conflicting_evidence)
    assert crypto.to_dict()["leverage"] == pytest.approx(crypto.leverage, rel=1e-3)


# ------------------------------------------------------------- totals ------


def test_the_scan_reports_what_placing_every_order_would_commit():
    """Per-row risk gives no hint of the total, and leverage triples it."""
    signals = []
    for i in range(5):
        s = _signal(f"COIN{i}", "crypto", 80.0 - i)
        s.position_size_fraction, s.margin_fraction = 0.30, 0.10
        s.leverage, s.risk_percentage = 3.0, 1.2
        signals.append(s)

    result = _scan_result(signals)

    assert result.portfolio_heat == pytest.approx(6.0)
    d = result.to_dict()
    assert d["gross_notional_pct"] == pytest.approx(150.0)
    assert d["margin_required_pct"] == pytest.approx(50.0)
    assert d["orders_by_asset_class"] == {"crypto": 5}


def test_an_empty_scan_reports_zero_rather_than_nothing():
    result = _scan_result([])
    assert result.portfolio_heat == 0.0
    assert result.to_dict()["gross_notional_pct"] == 0.0
