"""Core domain types.

All timestamps in TITAN are timezone-aware UTC. All OHLCV frames follow the
canonical schema defined by :data:`OHLCV_COLUMNS` with a ``DatetimeIndex``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    CRYPTO = "crypto"
    FUTURE = "future"


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    @property
    def sign(self) -> int:
        return {"long": 1, "short": -1, "flat": 0}[self.value]


class Regime(StrEnum):
    """Primary market regime taxonomy.

    Detected states come from an unsupervised state model; the finer labels
    (strong/weak, accumulation/distribution, correction) are assigned by
    deterministic post-rules on trend, drawdown and volume/price divergence.
    """

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    WEAK_BULL = "weak_bull"
    RANGE = "range"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    CORRECTION = "correction"
    BEAR = "bear"
    CRASH = "crash"

    @property
    def is_bullish(self) -> bool:
        return self in {Regime.STRONG_BULL, Regime.BULL, Regime.WEAK_BULL, Regime.ACCUMULATION}

    @property
    def is_bearish(self) -> bool:
        return self in {Regime.BEAR, Regime.CRASH, Regime.DISTRIBUTION, Regime.CORRECTION}


class VolState(StrEnum):
    """Volatility overlay, orthogonal to the directional regime."""

    LOW = "low_volatility"
    NORMAL = "normal_volatility"
    HIGH = "high_volatility"
    EXTREME = "extreme_volatility"


class TradeGrade(StrEnum):
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradeable instrument in the research universe."""

    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    sector: str = "unknown"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Instrument symbol must be non-empty")


@dataclass(slots=True)
class Universe:
    """A named collection of instruments plus the market benchmark proxy."""

    instruments: list[Instrument] = field(default_factory=list)
    benchmark: str = "INDEX"
    name: str = "default"

    @property
    def symbols(self) -> list[str]:
        return [i.symbol for i in self.instruments]

    def instrument(self, symbol: str) -> Instrument:
        for inst in self.instruments:
            if inst.symbol == symbol:
                return inst
        raise KeyError(f"Unknown symbol: {symbol}")

    def sector_of(self, symbol: str) -> str:
        try:
            return self.instrument(symbol).sector
        except KeyError:
            return "unknown"
