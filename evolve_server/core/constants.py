"""
Shared constants and enums for the evolve server.
"""

from __future__ import annotations

import re
from enum import IntEnum

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,}$")


class FailureType(IntEnum):
    """Five-way failure taxonomy for a bad turn."""

    SKILL_CONTENT_STALE = 1
    SKILL_MISSELECT = 2
    SKILL_GAP = 3
    TOOL_ERROR = 4
    MODEL_BASELINE = 5


FAILURE_LABELS: dict[int, str] = {
    FailureType.SKILL_CONTENT_STALE: "Skill content stale",
    FailureType.SKILL_MISSELECT: "Skill misselected",
    FailureType.SKILL_GAP: "Skill gap",
    FailureType.TOOL_ERROR: "Tool usage error",
    FailureType.MODEL_BASELINE: "Model baseline capability",
}


NO_SKILL_KEY = "__no_skill__"


class DecisionAction:
    """Allowed evolution-decision action identifiers."""

    CREATE = "create_skill"
    IMPROVE = "improve_skill"
    OPTIMIZE_DESC = "optimize_description"
    SKIP = "skip"
