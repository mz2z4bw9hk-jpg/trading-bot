"""Artifact serialization: research outputs → files the dashboard serves.

Everything the dashboard shows is a static artifact of a validated run —
the server never computes statistics on the fly, so what you see is exactly
what was validated, byte for byte.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from titan.backtest.walkforward import WalkForwardReport
from titan.core.config import TitanConfig
from titan.core.jsonsafe import json_safe
from titan.core.log import get_logger
from titan.data.store import MarketDataset
from titan.scanner.scanner import ScanResult

logger = get_logger(__name__)


def _write_json(path: Path, payload: Any) -> None:
    # allow_nan=False is unreachable after json_safe — it is here so a future
    # payload type that slips past the sanitizer fails loudly instead of
    # writing an artifact the dashboard API cannot serve.
    path.write_text(json.dumps(json_safe(payload), indent=1, default=str, allow_nan=False))


def write_walkforward_artifacts(
    out_dir: str | Path,
    cfg: TitanConfig,
    dataset: MarketDataset,
    report: WalkForwardReport,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_json(out / "report.json", report.to_dict())
    _write_json(out / "signals.json", [s.to_dict() for s in report.signals])
    _write_json(out / "trades.json", [t.to_dict() for t in report.backtest.trades])
    _write_json(
        out / "quality.json",
        {sym: rep.to_dict() for sym, rep in dataset.reports.items()},
    )
    if report.importance is not None:
        _write_json(
            out / "importance.json",
            {k: round(float(v), 6) for k, v in report.importance.items()},
        )

    equity = pd.DataFrame(
        {
            "equity": report.backtest.equity,
            "exposure": report.backtest.exposure,
        }
    )
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1.0
    equity.to_csv(out / "equity.csv", index_label="date")

    report.regimes.to_csv(out / "regimes.csv", index_label="date")
    report.decisions.to_csv(out / "decisions.csv", index=False)

    # Correlation matrix of universe returns over the OOS window (dashboard).
    closes = pd.concat({s: f["close"] for s, f in dataset.frames.items()}, axis=1)
    oos_returns = closes.pct_change().loc[report.backtest.equity.index[0] :]
    corr = oos_returns.corr().round(3)
    _write_json(
        out / "correlation.json",
        {"symbols": list(corr.columns), "matrix": corr.to_numpy().tolist()},
    )

    sectors: dict[str, list[str]] = {}
    for inst in dataset.universe.instruments:
        if inst.symbol in dataset.frames:
            sectors.setdefault(inst.sector, []).append(inst.symbol)
    _write_json(out / "universe.json", {
        "name": dataset.universe.name,
        "benchmark": dataset.universe.benchmark,
        "sectors": sectors,
        "reliability": {k: round(v, 3) for k, v in dataset.reliability.items()},
        "excluded": dataset.excluded,
    })

    _write_json(out / "manifest.json", {
        "generated_unix": time.time(),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "provider": cfg.data.provider,
        "seed": cfg.run.seed,
        "n_folds": len(report.folds),
        "config": cfg.model_dump(mode="json"),
        "disclaimer": (
            "Research output. Results on the synthetic provider verify the "
            "pipeline only and say nothing about live markets. Not investment advice."
        ),
    })
    logger.info("artifacts written to %s", out.resolve())
    return out


def write_scan_artifacts(out_dir: str | Path, scan: ScanResult) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "scan.json", scan.to_dict())
    return out
