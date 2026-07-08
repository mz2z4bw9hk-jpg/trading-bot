"""Dashboard server and static export.

``create_app`` is resolved lazily so that ``titan export`` (pandas + stdlib
only) works in environments without the ``server`` extra installed.
"""

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from titan.server.app import create_app

        return create_app
    raise AttributeError(name)
