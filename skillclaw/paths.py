"""Paths belonging to one SkillClaw instance."""

from __future__ import annotations

import os
from pathlib import Path


def has_instance_home() -> bool:
    return bool(os.environ.get("SKILLCLAW_HOME", "").strip())


def skillclaw_home() -> Path:
    value = os.environ.get("SKILLCLAW_HOME", "").strip()
    return Path(value).expanduser().resolve() if value else Path.home() / ".skillclaw"
