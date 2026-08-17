"""Tests for hermes-home resolution in the claw adapter and config store.

SkillClaw hardcodes the hermes home to ``~/.hermes``. On installs where
Hermes itself uses a custom home (e.g. ``HERMES_HOME=G:\\hermes``), the
integration configures and inspects the wrong file. The adapter should
resolve the hermes home from the environment, mirroring how Hermes itself
does:

    SKILLCLAW_HERMES_HOME  (explicit override, wins)
    HERMES_HOME            (matches Hermes' own env var)
    ~/.hermes              (stock default, unchanged)

The skills-dir default in config_store must follow the same rule.

Module constants are evaluated at import time, so each case runs in a
subprocess with a controlled environment.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CASES = [
    # (env_overrides, module, constant, expected)
    (
        {"SKILLCLAW_HERMES_HOME": "/tmp/sc-home"},
        "skillclaw.claw_adapter",
        "_HERMES_HOME",
        "/tmp/sc-home",
    ),
    (
        {"HERMES_HOME": "/tmp/hermes-home"},
        "skillclaw.claw_adapter",
        "_HERMES_HOME",
        "/tmp/hermes-home",
    ),
    (
        {"SKILLCLAW_HERMES_HOME": "/tmp/sc-wins", "HERMES_HOME": "/tmp/loses"},
        "skillclaw.claw_adapter",
        "_HERMES_HOME",
        "/tmp/sc-wins",
    ),
    (
        {"HERMES_HOME": "/tmp/hermes-home"},
        "skillclaw.claw_adapter",
        "_HERMES_SKILLS_DIR",
        "/tmp/hermes-home/skills",
    ),
    (
        {"HERMES_HOME": "/tmp/hermes-home"},
        "skillclaw.config_store",
        "_DEFAULT_HERMES_SKILLS_DIR",
        "/tmp/hermes-home/skills",
    ),
]

_STRIP = ("SKILLCLAW_HERMES_HOME", "HERMES_HOME")


def _read_constant(env_overrides: dict, module: str, constant: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in _STRIP}
    env.update(env_overrides)
    out = subprocess.run(
        [sys.executable, "-c", f"from {module} import {constant}; print({constant})"],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=60,
    )
    assert out.returncode == 0, f"{module} import failed: {out.stderr[-400:]}"
    return out.stdout.strip()


def test_skillclaw_home_override_wins():
    env, module, const, expected = CASES[0]
    assert _read_constant(env, module, const) == str(Path(expected))


def test_hermes_home_env_fallback():
    env, module, const, expected = CASES[1]
    assert _read_constant(env, module, const) == str(Path(expected))


def test_skillclaw_home_beats_hermes_home():
    env, module, const, expected = CASES[2]
    assert _read_constant(env, module, const) == str(Path(expected))


def test_skills_dir_follows_home():
    env, module, const, expected = CASES[3]
    assert _read_constant(env, module, const) == str(Path(expected))


def test_config_store_skills_dir_follows_home():
    env, module, const, expected = CASES[4]
    assert _read_constant(env, module, const) == str(Path(expected))


def test_stock_default_preserved():
    """With no env vars set, the stock ~/.hermes behavior must not change."""
    got = _read_constant({}, "skillclaw.claw_adapter", "_HERMES_HOME")
    assert got == str(Path.home() / ".hermes")
