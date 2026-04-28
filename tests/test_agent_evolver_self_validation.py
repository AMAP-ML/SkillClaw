from pathlib import Path

from evolve_server.engines import agent_workspace


def test_evolve_agents_md_requires_center_harness_self_validation():
    text = Path("evolve_server/engines/EVOLVE_AGENTS.md").read_text(encoding="utf-8")

    assert "Self-validation before finalizing" in text
    assert "If validation fails" in text
    assert "continue editing" in text
    assert "history/v<N>_evidence.md" in text
    assert "workspace directory" in text
    assert "Do NOT read or write files outside the workspace" in text


def test_agent_workspace_bootstrap_mentions_self_validation():
    text = agent_workspace._EVOLVE_AGENTS_MD

    assert "self-validation" in text
    assert "EVOLVE_AGENTS.md" in text
    assert "Before finalizing" in text
