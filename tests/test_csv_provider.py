"""CSV provider: TradingView export filenames, epoch times, unsorted rows."""

import numpy as np
import pandas as pd
import pytest

from titan.data.providers import CSVProvider, _squash, _tv_stem
from titan.data.schema import SchemaError, normalize_ohlcv


def _tv_frame(n: int = 300, start: str = "2022-01-01") -> pd.DataFrame:
    """A frame shaped like a TradingView 'Export chart data' file."""
    rng = np.random.default_rng(3)
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({
        # epoch seconds, TV style (resolution-proof: pandas may back the
        # index with us instead of ns)
        "time": (dates - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1),
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "Volume": rng.integers(1_000, 9_999, n).astype(float),
        "Volume MA": rng.integers(1_000, 9_999, n).astype(float),
    })


def test_tv_filename_resolution(tmp_path):
    _tv_frame().to_csv(tmp_path / "BINANCE_BTCUSDT, 1D.csv", index=False)
    _tv_frame().to_csv(tmp_path / "BITSTAMP_BTCUSD, 60.csv", index=False)
    provider = CSVProvider(tmp_path)
    assert len(provider.fetch("BTCUSDT", 250)) == 250
    # dashed config symbols match dash-less TradingView tickers
    assert len(provider.fetch("BTC-USD", 250)) == 250
    with pytest.raises(FileNotFoundError, match="no CSV"):
        provider.fetch("ETH-USD", 250)


def test_exact_filename_wins_and_ambiguity_is_loud(tmp_path):
    _tv_frame().to_csv(tmp_path / "AAPL.csv", index=False)
    _tv_frame().to_csv(tmp_path / "NASDAQ_AAPL, 1D.csv", index=False)
    provider = CSVProvider(tmp_path)
    assert len(provider.fetch("AAPL", 10)) == 10  # exact name resolves despite the twin
    _tv_frame().to_csv(tmp_path / "BINANCE_ETHUSD, 1D.csv", index=False)
    _tv_frame().to_csv(tmp_path / "COINBASE_ETHUSD, 1D.csv", index=False)
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        provider.fetch("ETH-USD", 10)


def test_epoch_seconds_parse_and_normalize(tmp_path):
    _tv_frame().to_csv(tmp_path / "BINANCE_SOLUSDT, 1D.csv", index=False)
    raw = CSVProvider(tmp_path).fetch("SOLUSDT", 300)
    frame = normalize_ohlcv(raw)
    assert str(frame.index[0].date()) == "2022-01-01"  # not 1970: epoch decoded as seconds
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.is_monotonic_increasing


def test_unsorted_and_iso_times(tmp_path):
    df = _tv_frame(100)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    df = df.sample(frac=1.0, random_state=1)  # shuffle rows
    df.to_csv(tmp_path / "SPY.csv", index=False)
    fetched = CSVProvider(tmp_path).fetch("SPY", 50)
    assert fetched.index.is_monotonic_increasing
    assert len(fetched) == 50
    # tail() after sorting returns the most recent rows, not file order
    assert fetched.index[-1] == pd.Timestamp("2022-04-10", tz="UTC")


def test_garbage_time_column_fails_loudly(tmp_path):
    df = _tv_frame(20)
    df["time"] = "not a date"
    df.to_csv(tmp_path / "QQQ.csv", index=False)
    with pytest.raises(SchemaError, match="time column"):
        CSVProvider(tmp_path).fetch("QQQ", 10)


def test_stem_helpers():
    assert _tv_stem("BINANCE_BTCUSDT, 1D") == "BTCUSDT"
    assert _tv_stem("NASDAQ_AAPL, 1D") == "AAPL"
    assert _tv_stem("AAPL") == "AAPL"
    assert _squash("BTC-USD") == _squash("btcusd")
