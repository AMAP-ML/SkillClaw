"""Tests that skill-hub and skill-manager recognize a custom Hermes home.

Follow-up to the hermes-home env override: `_is_hermes_skill_root` in both
skill_hub.py and skill_manager.py hardcodes `~/.hermes/skills`. Under
SKILLCLAW_HERMES_HOME / HERMES_HOME, skills written to the real skills dir
must still be treated as hermes-root (category subdirectory layout).
"""
from __future__ import annotations

from skillclaw import skill_hub, skill_manager


def _manager(skills_dir: str) -> skill_manager.SkillManager:
    """Bare instance — bypasses __init__ (we only need _skills_dir)."""
    m = skill_manager.SkillManager.__new__(skill_manager.SkillManager)
    m._skills_dir = skills_dir
    return m


def test_hub_root_recognizes_skillclaw_home(monkeypatch, tmp_path):
    home = tmp_path / "custom-home"
    monkeypatch.setenv("SKILLCLAW_HERMES_HOME", str(home))
    assert skill_hub._is_hermes_skill_root(str(home / "skills")) is True


def test_hub_root_recognizes_hermes_home_env(monkeypatch, tmp_path):
    home = tmp_path / "custom-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert skill_hub._is_hermes_skill_root(str(home / "skills")) is True


def test_manager_root_recognizes_custom_home(monkeypatch, tmp_path):
    home = tmp_path / "custom-home"
    monkeypatch.setenv("SKILLCLAW_HERMES_HOME", str(home))
    assert _manager(str(home / "skills"))._is_hermes_skill_root() is True


def test_hub_root_rejects_unrelated_dir(monkeypatch, tmp_path):
    home = tmp_path / "custom-home"
    monkeypatch.setenv("SKILLCLAW_HERMES_HOME", str(home))
    assert skill_hub._is_hermes_skill_root(str(tmp_path / "elsewhere")) is False


def test_hub_stock_default_preserved(monkeypatch):
    monkeypatch.delenv("SKILLCLAW_HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    import os
    stock = os.path.join(os.path.expanduser("~"), ".hermes", "skills")
    assert skill_hub._is_hermes_skill_root(stock) is True
