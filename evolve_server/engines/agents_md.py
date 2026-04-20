"""
Loader for the built-in ``EVOLVE_AGENTS.md`` used by the agent engine.
"""

from __future__ import annotations

from pathlib import Path

_AGENTS_MD_PATH = Path(__file__).resolve().parent / "EVOLVE_AGENTS.md"


def load_agents_md() -> str:
    """Read the built-in evolve guide and return its content."""
    return _AGENTS_MD_PATH.read_text(encoding="utf-8")
