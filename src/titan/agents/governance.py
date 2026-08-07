"""Suggest-only governance: agents propose, humans deploy.

The rule is that no agent process may write to a path that a running system
reads. Stated as a comment, that lasts until someone is in a hurry. Stated as
:class:`ProposalGate`, which raises on any write outside the proposal
directory and refuses protected paths by pattern, it survives contact.

A proposal is a directory, not a commit. It holds the changed config, the
evidence for the change, and a human-readable rationale, and it is the unit a
reviewer approves. Turning it into a branch and a pull request is a mechanical
step deliberately left to CI with a human-held credential — an agent that
could open and merge its own pull request has a governance diagram, not
governance.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from titan.core.log import get_logger

logger = get_logger(__name__)

__all__ = ["Proposal", "ProposalGate", "ProtectedPathError"]

# Anything a live process reads. Agents may read these; they may never write
# them. Matched against the repo-relative path of the write target.
PROTECTED_PREFIXES: tuple[str, ...] = (
    "configs/",
    "models_store/",
    "src/",
    ".github/",
)


class ProtectedPathError(PermissionError):
    """Raised when an agent attempts to write to live configuration."""


@dataclass(frozen=True, slots=True)
class Proposal:
    """One reviewable unit of proposed change."""

    name: str
    root: Path
    rationale: str
    created_at: str
    files: tuple[str, ...] = ()
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "rationale": self.rationale,
            "files": list(self.files),
            "evidence": self.evidence,
        }


class ProposalGate:
    """The only writable surface an agent gets.

    Parameters
    ----------
    repo_root:
        Used to resolve writes to repo-relative paths for the protected-prefix
        test. Writes are resolved through ``Path.resolve()`` first, so a
        ``..`` traversal out of the proposal directory is caught rather than
        normalised into a successful escape.
    proposals_dir:
        Where proposals are written. Must be inside ``repo_root`` and must not
        itself be protected.
    """

    def __init__(self, repo_root: Path | str, proposals_dir: Path | str = "proposals") -> None:
        self._root = Path(repo_root).resolve()
        proposals = Path(proposals_dir)
        self._dir = (proposals if proposals.is_absolute() else self._root / proposals).resolve()
        rel = self._relative(self._dir)
        if rel is not None and any(rel.startswith(p) for p in PROTECTED_PREFIXES):
            raise ProtectedPathError(
                f"proposals_dir {rel!r} is itself a protected path"
            )
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------ #

    def _relative(self, path: Path) -> str | None:
        try:
            return path.resolve().relative_to(self._root).as_posix()
        except ValueError:
            return None

    def assert_writable(self, path: Path | str) -> Path:
        """Raise unless ``path`` resolves inside the proposal directory."""
        target = Path(path)
        target = target if target.is_absolute() else self._dir / target
        resolved = target.resolve()

        rel = self._relative(resolved)
        if rel is not None and any(rel.startswith(p) for p in PROTECTED_PREFIXES):
            raise ProtectedPathError(
                f"refusing to write {rel!r}: agents propose changes to live "
                "configuration, they do not make them"
            )
        if not resolved.is_relative_to(self._dir):
            raise ProtectedPathError(
                f"refusing to write outside the proposal directory: {resolved}"
            )
        return resolved

    def write_text(self, relative_path: Path | str, content: str) -> Path:
        target = self.assert_writable(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def write_json(self, relative_path: Path | str, payload: dict) -> Path:
        return self.write_text(relative_path, json.dumps(payload, indent=2, default=str))

    def copy_in(self, source: Path | str, relative_path: Path | str) -> Path:
        """Copy an artifact (a plot, a backtest log) into the proposal."""
        target = self.assert_writable(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source), target)
        return target

    # ------------------------------------------------------------------ #

    def propose(
        self,
        name: str,
        *,
        rationale: str,
        configs: dict[str, str] | None = None,
        evidence: dict | None = None,
        artifacts: dict[str, Path | str] | None = None,
    ) -> Proposal:
        """Assemble a reviewable proposal directory.

        ``configs`` maps a live path the change would eventually touch (for the
        reviewer's benefit, recorded as text) to its proposed contents. Nothing
        is written to that live path — the mapping is metadata describing where
        a human would apply the file after approving it.
        """
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base = Path(f"{stamp}-{slug}")
        written: list[str] = []

        for live_path, content in (configs or {}).items():
            staged = base / "configs" / Path(live_path).name
            self.write_text(staged, content)
            written.append(f"{staged.as_posix()} -> {live_path}")

        for label, source in (artifacts or {}).items():
            staged = base / "evidence" / Path(str(source)).name
            self.copy_in(source, staged)
            written.append(f"{staged.as_posix()} ({label})")

        proposal = Proposal(
            name=name,
            root=self._dir / base,
            rationale=rationale,
            created_at=stamp,
            files=tuple(written),
            evidence=evidence or {},
        )
        self.write_json(base / "proposal.json", proposal.to_dict())
        self.write_text(base / "RATIONALE.md", _render_rationale(proposal))
        logger.info(
            "proposal %s written to %s (%d file(s)); no live path was modified",
            name, proposal.root, len(written),
        )
        return proposal


def _render_rationale(proposal: Proposal) -> str:
    lines = [
        f"# Proposal: {proposal.name}",
        "",
        f"Generated {proposal.created_at} by an agent in suggest-only mode.",
        "**No live configuration was modified.** Applying this is a human action.",
        "",
        "## Rationale",
        "",
        proposal.rationale.strip(),
        "",
        "## Files",
        "",
    ]
    lines += [f"- `{f}`" for f in proposal.files] or ["_none_"]
    if proposal.evidence:
        lines += ["", "## Evidence", "", "```json",
                  json.dumps(proposal.evidence, indent=2, default=str), "```"]
    lines += [
        "",
        "## Reviewer checklist",
        "",
        "- [ ] Shadow-mode results cover a period containing at least one stress day",
        "- [ ] Risk limits in the proposed config are within the signed-off envelope",
        "- [ ] Execution markouts are positive at the decision horizon, net of fees",
        "- [ ] The change was not fitted to the period it is evaluated on",
        "",
    ]
    return "\n".join(lines)


def in_suggest_only_mode() -> bool:
    """Whether the process is running under the suggest-only guarantee.

    Defaults to ``True``: an agent that cannot tell what mode it is in must
    assume it is the restricted one.
    """
    return os.environ.get("TITAN_AGENT_MODE", "suggest-only").lower() != "deploy"
