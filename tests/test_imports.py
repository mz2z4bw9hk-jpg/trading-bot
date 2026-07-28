"""Every subpackage must import in a fresh interpreter, in any order.

``titan.backtest.__init__`` eagerly imported ``walkforward``, which imports
titan.risk and titan.signals, both of which import titan.backtest.costs —
running that same __init__ again. Whether the cycle bit depended entirely on
which subpackage a process touched first: `import titan.scanner` and
`import titan.risk` both failed outright, while the test suite and CLI happened
to import in an order that masked it.

Each case below runs in its own subprocess, because once any module is in
sys.modules the cycle cannot reproduce.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

SUBPACKAGES = [
    "titan",
    "titan.backtest",
    "titan.core",
    "titan.data",
    "titan.explain",
    "titan.features",
    "titan.labels",
    "titan.models",
    "titan.monitor",
    "titan.regime",
    "titan.risk",
    "titan.scanner",
    "titan.server",
    "titan.signals",
]


def _import_in_fresh_process(statement: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True, timeout=180
    )


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_subpackage_imports_first_in_a_fresh_interpreter(module):
    result = _import_in_fresh_process(f"import {module}")
    assert result.returncode == 0, (
        f"`import {module}` fails when it is the first titan import:\n{result.stderr}"
    )


def test_walkforward_names_remain_importable_from_the_package():
    """The lazy __getattr__ must not silently drop the public API."""
    result = _import_in_fresh_process(
        "from titan.backtest import WalkForwardRunner, WalkForwardReport; "
        "print(WalkForwardRunner.__name__, WalkForwardReport.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert "WalkForwardRunner WalkForwardReport" in result.stdout


def test_unknown_package_attribute_still_raises_attribute_error():
    result = _import_in_fresh_process(
        "import titan.backtest\n"
        "try:\n"
        "    titan.backtest.NoSuchThing\n"
        "except AttributeError as exc:\n"
        "    print('AttributeError:', exc)\n"
    )
    assert result.returncode == 0, result.stderr
    assert "AttributeError" in result.stdout
