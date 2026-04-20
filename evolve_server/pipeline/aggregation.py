"""
Session-level skill aggregation: if ANY interaction in a session
references a skill, the entire session is included in that skill's group.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from ..core.constants import NO_SKILL_KEY

logger = logging.getLogger(__name__)


def aggregate_sessions_by_skill(
    sessions: list[dict],
) -> dict[str, list[dict]]:
    """Group sessions by skill.

    Each session must have ``_skills_referenced`` (a set of skill names)
    attached by the summarization stage.

    - If a session references skill X, it goes into skill X's group.
    - If a session references both A and B, it appears in both groups.
    - If no skill is referenced, it goes into :data:`NO_SKILL_KEY`.
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for session in sessions:
        skills = session.get("_skills_referenced") or set()
        if not skills:
            groups[NO_SKILL_KEY].append(session)
        else:
            for skill_name in skills:
                groups[skill_name].append(session)

    skill_group_count = sum(1 for k in groups if k != NO_SKILL_KEY)
    no_skill_count = len(groups.get(NO_SKILL_KEY, []))
    logger.info(
        "[Aggregation] %d sessions -> %d skill groups + %d no-skill sessions",
        len(sessions),
        skill_group_count,
        no_skill_count,
    )
    return dict(groups)
