"""Versioned model store with production/challenger lifecycle.

Every trained bundle is persisted with its manifest (metrics, features,
config fingerprint, data window). Promotion to production is *mechanical*
here; the statistical gate lives in :mod:`titan.monitor.compare` — a
challenger must beat production out-of-sample with statistical significance
before anyone calls :meth:`ModelRegistry.promote`.

**"Production" is scoped to a research world, not to the registry.** A model
is only meaningful against the config fingerprint it was validated on, so one
store holds one champion *per fingerprint* rather than one champion overall.
Without that scoping a registry that has ever held a strong model from another
universe deadlocks every other config: the champion/challenger bootstrap
compares return series from two different markets, the gate rejects on a
comparison that never meant anything, and the config guard then refuses to
scan with the incumbent it just protected. Passing a fingerprint to
:meth:`production_version` or :meth:`load` asks the only answerable question —
"what is the best model *for this world*".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

from titan.core.log import get_logger

logger = get_logger(__name__)

STATUS_CANDIDATE = "candidate"
STATUS_PRODUCTION = "production"
STATUS_RETIRED = "retired"


def fingerprint_matches(recorded: str, active: str) -> bool:
    """Can a model recorded under ``recorded`` be used against ``active``?

    Kept deliberately permissive in one direction: a manifest written before
    fingerprints existed records nothing, and an absent fingerprint is not
    evidence of a mismatch, so it matches anything. The scan guard in the CLI
    lets those through for the same reason, and both call this so the two
    cannot drift apart.
    """
    return not recorded or recorded == active


@dataclass(slots=True)
class ModelManifest:
    version: str
    created_unix: float
    status: str
    description: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    config_fingerprint: str = ""
    train_start: str = ""
    train_end: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    def __init__(self, store_dir: str | Path) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"

    # ------------------------------------------------------------------ #

    def _read_index(self) -> dict[str, dict[str, Any]]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text())

    def _write_index(self, index: dict[str, dict[str, Any]]) -> None:
        self._index_path.write_text(json.dumps(index, indent=2, default=str))

    def _next_version(self, index: dict[str, dict[str, Any]]) -> str:
        nums = [int(v[1:]) for v in index if v.startswith("v") and v[1:].isdigit()]
        return f"v{(max(nums) + 1) if nums else 1:03d}"

    # ------------------------------------------------------------------ #

    def save(
        self,
        model: Any,
        metrics: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
        description: str = "",
        config_fingerprint: str = "",
        train_start: str = "",
        train_end: str = "",
    ) -> str:
        index = self._read_index()
        version = self._next_version(index)
        manifest = ModelManifest(
            version=version,
            created_unix=time.time(),
            status=STATUS_CANDIDATE,
            description=description,
            metrics=metrics or {},
            feature_names=feature_names or [],
            config_fingerprint=config_fingerprint,
            train_start=train_start,
            train_end=train_end,
        )
        vdir = self._dir / version
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, vdir / "model.joblib")
        (vdir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, default=str))
        index[version] = manifest.to_dict()
        self._write_index(index)
        logger.info("saved model %s (%s)", version, description or "no description")
        return version

    def load(
        self, version: str | None = None, fingerprint: str | None = None
    ) -> tuple[Any, ModelManifest]:
        """Load a bundle by version, or the production model when none given.

        ``fingerprint`` narrows the default to the champion of that research
        world; it is ignored when an explicit ``version`` is named, because a
        caller asking for a specific model has already decided.
        """
        index = self._read_index()
        if version is None:
            version = self.production_version(fingerprint)
            if version is None:
                raise LookupError(
                    "no production model in registry"
                    if fingerprint is None
                    else f"no production model for research config {fingerprint}"
                )
        if version not in index:
            raise LookupError(f"unknown model version: {version}")
        model = joblib.load(self._dir / version / "model.joblib")
        return model, ModelManifest(**index[version])

    # ------------------------------------------------------------------ #

    def promote(self, version: str) -> None:
        """Make ``version`` production; demote the champion it replaces.

        Callers must first pass the statistical comparison gate
        (:func:`titan.monitor.compare.compare_strategies`). Promoting an
        unproven model by hand defeats the whole platform.

        Only the champion of the *same* research world is retired, on exact
        fingerprint equality. A model validated elsewhere was never in this
        contest and losing a race it did not run should not cost it its status
        — that is what let one config's incumbent block every other config.
        """
        index = self._read_index()
        if version not in index:
            raise LookupError(f"unknown model version: {version}")
        fp = index[version].get("config_fingerprint", "")
        for v, m in index.items():
            if v == version or m.get("status") != STATUS_PRODUCTION:
                continue
            if m.get("config_fingerprint", "") != fp:
                continue
            m["status"] = STATUS_RETIRED
            logger.info("retired previous production model %s", v)
        index[version]["status"] = STATUS_PRODUCTION
        self._write_index(index)
        manifest_path = self._dir / version / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = STATUS_PRODUCTION
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        logger.info("promoted %s to production", version)

    def history(self) -> list[ModelManifest]:
        index = self._read_index()
        return [ModelManifest(**m) for _, m in sorted(index.items())]

    def production_version(self, fingerprint: str | None = None) -> str | None:
        """Latest production model, optionally within one research world.

        With ``fingerprint``, returns the champion for that config and None if
        the world has never had one — which is the signal to promote outright
        rather than to run a comparison against a stranger.
        """
        index = self._read_index()
        production = [
            v for v, m in index.items()
            if m.get("status") == STATUS_PRODUCTION
            and (
                fingerprint is None
                or fingerprint_matches(m.get("config_fingerprint", ""), fingerprint)
            )
        ]
        return sorted(production)[-1] if production else None

    def versions_for_fingerprint(self, fingerprint: str) -> list[str]:
        """Every non-retired version usable against ``fingerprint``, oldest first.

        Exists so a refused scan can name the models that *would* work instead
        of only reporting the one that would not.
        """
        index = self._read_index()
        return sorted(
            v for v, m in index.items()
            if m.get("status") != STATUS_RETIRED
            and fingerprint_matches(m.get("config_fingerprint", ""), fingerprint)
        )
