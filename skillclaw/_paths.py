"""Shared path resolution for the Hermes integration.

SkillClaw previously hardcoded `~/.hermes` in several modules. Installs
where Hermes uses a custom home (HERMES_HOME env var) were configured and
inspected at the wrong path. All sites now resolve through one helper.
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_hermes_home() -> Path:
    """Resolve the Hermes home directory.

    Order:
      1. SKILLCLAW_HERMES_HOME (explicit override, wins)
      2. HERMES_HOME (matches Hermes' own env var)
      3. ~/.hermes (stock default, unchanged)

    Values are stripped and tilde-expanded; empty values fall through.
    """
    raw = os.environ.get("SKILLCLAW_HERMES_HOME") or os.environ.get("HERMES_HOME")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    return Path.home() / ".hermes"
