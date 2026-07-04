"""Regime-switching synthetic market generator.

Purpose: offline verification of the *entire* research pipeline. The sandbox
this platform ships from has no market-data egress, and a pipeline you cannot
test end-to-end is a pipeline you cannot trust. The generator produces a
multi-asset market with known ground truth:

- a hidden Markov regime chain (bull / range / bear / crash / recovery) driving
  the common market factor's drift and volatility;
- a one-factor-plus-sectors covariance structure (betas, sector loadings);
- a slow AR(1) per-symbol drift component that creates *genuine, learnable*
  time-series momentum;
- mild negative short-horizon autocorrelation in the range regime (learnable
  mean reversion);
- volume that co-moves with absolute returns (leverage effect);
- internally consistent OHLC bars built around each close-to-close move.

Because ground truth (true regimes, presence of momentum) is known, tests can
assert that the regime detector, the feature engine and the ML ensemble find
structure that is *actually there* — and find nothing on a shuffled control.

This data is for pipeline verification only. It says nothing about edge on
real markets; see docs/VALIDATION.md for the real-data protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Per-state daily drift/vol of the common market factor (log returns).
STATE_PARAMS: dict[str, tuple[float, float]] = {
    "bull": (0.00055, 0.009),
    "range": (0.00000, 0.007),
    "bear": (-0.00075, 0.017),
    "crash": (-0.02000, 0.045),
    "recovery": (0.00220, 0.020),
}
STATES: list[str] = list(STATE_PARAMS)

# Row-stochastic transition matrix, ordered as STATES.
TRANSITIONS = np.array(
    [
        # bull    range   bear    crash   recovery
        [0.9880, 0.0080, 0.0030, 0.0010, 0.0000],  # bull
        [0.0100, 0.9800, 0.0080, 0.0020, 0.0000],  # range
        [0.0000, 0.0060, 0.9850, 0.0030, 0.0060],  # bear
        [0.0000, 0.0000, 0.0300, 0.8200, 0.1500],  # crash
        [0.0200, 0.0050, 0.0000, 0.0000, 0.9750],  # recovery
    ]
)

_MEANREV_PHI_RANGE = -0.15  # idiosyncratic AR(1) in the range regime
_DRIFT_PHI = 0.985          # persistence of the slow momentum drift
_DRIFT_SIGMA = 0.00025      # innovation scale of the slow drift


@dataclass(slots=True)
class SyntheticResult:
    """Generated market plus the ground truth needed by verification tests."""

    frames: dict[str, pd.DataFrame]
    index_frame: pd.DataFrame
    true_regimes: pd.Series          # state name per bar (hidden from models)
    betas: dict[str, float] = field(default_factory=dict)
    sectors: dict[str, str] = field(default_factory=dict)


class SyntheticMarket:
    """Deterministic (seeded) generator for a synthetic multi-asset market."""

    def __init__(
        self,
        symbols: list[str],
        bars: int = 2000,
        seed: int = 7,
        start: str = "2015-01-02",
        sectors: dict[str, str] | None = None,
        drift_sigma: float = _DRIFT_SIGMA,
        drift_phi: float = _DRIFT_PHI,
    ) -> None:
        if bars < 50:
            raise ValueError("need at least 50 bars")
        if not symbols:
            raise ValueError("need at least one symbol")
        self.symbols = list(symbols)
        self.bars = bars
        self.seed = seed
        self.start = start
        self.drift_sigma = drift_sigma
        self.drift_phi = drift_phi
        n_sectors = max(1, round(len(symbols) / 3))
        self._sectors = sectors or {
            s: f"sector_{i % n_sectors}" for i, s in enumerate(self.symbols)
        }

    # ------------------------------------------------------------------ #

    def _simulate_regimes(self, rng: np.random.Generator) -> np.ndarray:
        states = np.zeros(self.bars, dtype=int)
        states[0] = 0  # start in bull
        for t in range(1, self.bars):
            states[t] = rng.choice(len(STATES), p=TRANSITIONS[states[t - 1]])
        return states

    def _build_ohlcv(
        self,
        rng: np.random.Generator,
        index: pd.DatetimeIndex,
        log_returns: np.ndarray,
        state_sigma: np.ndarray,
        start_price: float,
        base_volume: float,
    ) -> pd.DataFrame:
        n = len(log_returns)
        closes = start_price * np.exp(np.cumsum(log_returns))
        prev_close = np.concatenate([[start_price], closes[:-1]])

        gap = rng.normal(0.0, 0.25 * state_sigma)
        opens = prev_close * np.exp(gap)
        hi_ext = np.abs(rng.normal(0.0, 0.6 * state_sigma))
        lo_ext = np.abs(rng.normal(0.0, 0.6 * state_sigma))
        highs = np.maximum(opens, closes) * np.exp(hi_ext)
        lows = np.minimum(opens, closes) * np.exp(-lo_ext)

        vol_shock = 0.8 * np.abs(log_returns) / np.maximum(state_sigma, 1e-8)
        volumes = base_volume * np.exp(vol_shock + rng.normal(0.0, 0.3, n))

        return pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.round(volumes),
            },
            index=index,
        )

    # ------------------------------------------------------------------ #

    def generate(self) -> SyntheticResult:
        rng = np.random.default_rng(self.seed)
        index = pd.bdate_range(self.start, periods=self.bars, tz="UTC")

        states = self._simulate_regimes(rng)
        mu = np.array([STATE_PARAMS[STATES[s]][0] for s in states])
        sigma = np.array([STATE_PARAMS[STATES[s]][1] for s in states])
        market_ret = mu + sigma * rng.standard_normal(self.bars)

        # Sector factors: persistent AR(1) with modest volatility.
        sector_names = sorted(set(self._sectors.values()))
        sector_ret: dict[str, np.ndarray] = {}
        for name in sector_names:
            innov = rng.normal(0.0, 0.003, self.bars)
            path = np.zeros(self.bars)
            for t in range(1, self.bars):
                path[t] = 0.70 * path[t - 1] + innov[t]
            sector_ret[name] = path

        in_crash = np.array([STATES[s] == "crash" for s in states])
        in_range = np.array([STATES[s] == "range" for s in states])

        frames: dict[str, pd.DataFrame] = {}
        betas: dict[str, float] = {}
        for sym in self.symbols:
            beta = float(np.clip(rng.normal(1.0, 0.25), 0.4, 1.8))
            betas[sym] = beta
            # Correlations tighten in crashes: effective beta amplified.
            eff_beta = np.where(in_crash, beta * 1.3, beta)

            # Slow AR(1) drift -> genuine momentum structure.
            drift = np.zeros(self.bars)
            eta = rng.normal(0.0, self.drift_sigma, self.bars)
            for t in range(1, self.bars):
                drift[t] = self.drift_phi * drift[t - 1] + eta[t]

            # Idiosyncratic noise; negatively autocorrelated inside range regime.
            idio_sigma = 0.010
            eps = rng.normal(0.0, idio_sigma, self.bars)
            idio = np.zeros(self.bars)
            for t in range(1, self.bars):
                phi = _MEANREV_PHI_RANGE if in_range[t] else 0.0
                idio[t] = phi * idio[t - 1] + eps[t]

            sym_ret = eff_beta * market_ret + sector_ret[self._sectors[sym]] + drift + idio
            state_sig = np.sqrt((eff_beta * sigma) ** 2 + idio_sigma**2)

            start_price = float(rng.uniform(20.0, 200.0))
            base_volume = float(np.exp(rng.normal(14.5, 0.5)))
            frames[sym] = self._build_ohlcv(rng, index, sym_ret, state_sig, start_price, base_volume)

        index_frame = self._build_ohlcv(
            rng, index, market_ret, sigma, start_price=1000.0, base_volume=5e8
        )
        true_regimes = pd.Series([STATES[s] for s in states], index=index, name="regime")
        return SyntheticResult(
            frames=frames,
            index_frame=index_frame,
            true_regimes=true_regimes,
            betas=betas,
            sectors=dict(self._sectors),
        )
