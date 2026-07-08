"""Static dashboard export: payload collection, embedding, escaping, CLI."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from titan.cli import main
from titan.server.export import export_static_dashboard, render_static_dashboard
from titan.server.payloads import MAX_POINTS, collect_payloads, downsample

EMBED_RE = re.compile(r"window\.TITAN_EMBEDDED = (.*?);</script>", re.S)


def _write_minimal_artifacts(d: Path) -> None:
    # `note` deliberately contains a script terminator: embedding must survive it.
    (d / "report.json").write_text(json.dumps({"pooled_auc": 0.61, "note": "</script>alert(1)"}))
    (d / "scan.json").write_text(json.dumps({"regime": {"regime": "bull", "confidence": 0.8}}))
    (d / "manifest.json").write_text(
        json.dumps({"provider": "synthetic", "seed": 7, "n_folds": 4, "generated": "t"})
    )
    n = 2000
    pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=n, freq="D"),
        "equity": [1e6 + i for i in range(n)],
        "drawdown": [0.0] * n,
        "exposure": [0.5] * n,
    }).to_csv(d / "equity.csv", index=False)


def test_collect_payloads_missing_map_to_none(tmp_path):
    _write_minimal_artifacts(tmp_path)
    payloads = collect_payloads(tmp_path)
    assert payloads["report"] == {"pooled_auc": 0.61, "note": "</script>alert(1)"}
    assert payloads["equity"] is not None
    assert payloads["regimes"] is None  # regimes.csv never written
    assert payloads["importance"] is None


def test_equity_payload_is_downsampled(tmp_path):
    _write_minimal_artifacts(tmp_path)
    payloads = collect_payloads(tmp_path)
    equity = payloads["equity"]
    assert len(equity["date"]) <= MAX_POINTS
    # the most recent point always survives downsampling
    last_raw = pd.read_csv(tmp_path / "equity.csv", parse_dates=["date"])["date"].iloc[-1]
    assert equity["date"][-1] == str(last_raw.date())
    df = pd.DataFrame({"x": range(10)})
    assert downsample(df, max_points=100) is df  # short frames pass through


def test_export_round_trips_payloads_and_escapes_script_close(tmp_path):
    _write_minimal_artifacts(tmp_path)
    out, payloads = export_static_dashboard(tmp_path, tmp_path / "dash.html")
    text = out.read_text()

    match = EMBED_RE.search(text)
    assert match, "embedded payload block missing"
    embedded = json.loads(match.group(1))

    # exact round trip: what the page sees is what the API would serve
    for key, value in payloads.items():
        if value is not None:
            assert embedded[key] == value, key
        else:
            assert key not in embedded  # missing artifacts are not embedded at all
    assert "exported_at" in embedded["__export"]
    # a raw "</" inside the blob would have terminated the <script> element,
    # truncating the regex capture and failing json.loads above
    assert "<\\/script>" in match.group(1)


def test_export_refuses_empty_artifacts_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="titan validate"):
        render_static_dashboard(tmp_path)


def test_cli_export(tmp_path, capsys):
    _write_minimal_artifacts(tmp_path)
    out = tmp_path / "site" / "index.html"
    assert main(["export", "--artifacts", str(tmp_path), "--out", str(out)]) == 0
    assert out.exists()
    summary = json.loads(capsys.readouterr().out)
    assert set(summary["embedded"]) == {"report", "scan", "manifest", "equity"}
    assert "regimes" in summary["missing"]


def test_cli_export_empty_dir_fails_cleanly(tmp_path, capsys):
    assert main(["export", "--artifacts", str(tmp_path)]) == 2
    assert "titan validate" in capsys.readouterr().err
