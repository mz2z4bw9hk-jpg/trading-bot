"""Suggest-only, enforced. An agent that can write live config is not gated."""

from __future__ import annotations

import pytest

from titan.agents.governance import ProposalGate, ProtectedPathError


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "live.yaml").write_text("risk:\n  max_leverage: 2.0\n")
    (tmp_path / "src" / "titan").mkdir(parents=True)
    return tmp_path


def test_writing_live_config_is_refused(repo):
    gate = ProposalGate(repo)
    with pytest.raises(ProtectedPathError, match="they do not make them"):
        gate.write_text(repo / "configs" / "live.yaml", "risk:\n  max_leverage: 50.0\n")


def test_writing_source_is_refused(repo):
    gate = ProposalGate(repo)
    with pytest.raises(ProtectedPathError):
        gate.write_text(repo / "src" / "titan" / "evil.py", "print('deployed')")


def test_traversal_out_of_the_proposal_directory_is_refused(repo):
    """`..` must be resolved before the check, not after."""
    gate = ProposalGate(repo)
    with pytest.raises(ProtectedPathError):
        gate.write_text("../configs/live.yaml", "nope")


def test_absolute_paths_outside_the_repo_are_refused(repo, tmp_path):
    gate = ProposalGate(repo)
    with pytest.raises(ProtectedPathError, match="outside the proposal directory"):
        gate.write_text(tmp_path / "elsewhere.yaml", "nope")


def test_a_protected_proposals_dir_is_refused_at_construction(repo):
    with pytest.raises(ProtectedPathError, match="itself a protected path"):
        ProposalGate(repo, proposals_dir="configs/proposals")


def test_the_live_file_is_untouched_after_a_proposal(repo):
    gate = ProposalGate(repo)
    before = (repo / "configs" / "live.yaml").read_text()

    gate.propose(
        "raise leverage",
        rationale="Shadow mode shows better fill quality at 3x.",
        configs={"configs/live.yaml": "risk:\n  max_leverage: 3.0\n"},
        evidence={"shadow_days": 20, "net_bps": 1.4},
    )

    assert (repo / "configs" / "live.yaml").read_text() == before


def test_a_proposal_is_a_reviewable_directory(repo):
    gate = ProposalGate(repo)
    proposal = gate.propose(
        "widen quotes in high vpin",
        rationale="Adverse selection exceeds capture above VPIN 0.5.",
        configs={"configs/live.yaml": "signals:\n  max_vpin: 0.45\n"},
        evidence={"decision_horizon_s": 30, "toxicity_ratio": 1.3},
    )

    assert proposal.root.is_dir()
    assert (proposal.root / "proposal.json").exists()
    rationale = (proposal.root / "RATIONALE.md").read_text()
    assert "No live configuration was modified" in rationale
    assert "Reviewer checklist" in rationale
    # The staged config records where a human would apply it, without applying it.
    assert any("configs/live.yaml" in f for f in proposal.files)
    assert (proposal.root / "configs" / "live.yaml").exists()


def test_suggest_only_is_the_default_when_the_mode_is_unset(monkeypatch):
    from titan.agents.governance import in_suggest_only_mode

    monkeypatch.delenv("TITAN_AGENT_MODE", raising=False)
    assert in_suggest_only_mode() is True

    monkeypatch.setenv("TITAN_AGENT_MODE", "garbled")
    assert in_suggest_only_mode() is True

    monkeypatch.setenv("TITAN_AGENT_MODE", "deploy")
    assert in_suggest_only_mode() is False
