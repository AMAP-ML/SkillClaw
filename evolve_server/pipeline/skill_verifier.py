"""
Optional post-generation verifier for workflow-evolved skills.

When enabled, this verifier runs after a candidate skill is generated but
before it is uploaded to shared storage. It is intentionally conservative:
if the verifier cannot confidently approve the candidate, the upload is
blocked and the rejection reason is recorded in the evolve summary.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from ..core.constants import DecisionAction
from ..core.llm_client import AsyncLLMClient

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_SESSIONS = 8
_SUMMARY_MAX_CHARS = 1200
_SKILL_CONTENT_MAX_CHARS = 8000
_CURRENT_SKILL_MAX_CHARS = 4000

_VERIFY_SKILL_SYSTEM = """\
You are the final publication gate for SkillClaw workflow evolution.

You are given:
- the proposed action
- the candidate skill
- optional current skill content
- summarized evidence from the sessions that motivated the change

Your job is NOT to improve the skill. Your job is only to decide whether this
candidate is safe and worthwhile to publish to the shared skill store.

Approve the candidate only if ALL of the following are true:
- it is grounded in the provided evidence
- it does not throw away useful existing environment-specific facts without evidence
- it is specific and reusable rather than generic agent advice
- it is coherent enough to be shared with other users immediately

Reject the candidate if ANY of the following are true:
- it is speculative or weakly supported by the evidence
- it removes useful existing instructions, endpoints, ports, filenames, or payload details without justification
- it mostly adds generic best practices instead of environment-specific knowledge
- it should stay as a local draft or needs more evidence before publication

For `optimize_description`, verify only whether the new description is a safer
and more accurate trigger than the old one.

For `create_skill`, verify that the new skill is genuinely distinct and
generalizable from the provided sessions.

Output EXACTLY one JSON object with:
- "decision": "accept" or "reject"
- "score": number in [0, 1]
- "reason": short explanation
- "checks": object with numeric scores in [0, 1] for:
  - "grounded_in_evidence"
  - "preserves_existing_value"
  - "specificity_and_reusability"
  - "safe_to_publish"

No markdown fences. No extra text.
"""


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    clean = re.sub(r"```(?:json)?\s*", "", str(text or "")).strip().rstrip("`")
    if not clean:
        return None
    try:
        obj = json.loads(clean)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    match = _JSON_BLOCK_RE.search(clean)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _normalize_score(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    score = max(0.0, min(1.0, float(value)))
    return round(score, 3)


def _normalize_checks(raw_checks: Any) -> dict[str, float]:
    if not isinstance(raw_checks, dict):
        return {}
    out: dict[str, float] = {}
    for key in (
        "grounded_in_evidence",
        "preserves_existing_value",
        "specificity_and_reusability",
        "safe_to_publish",
    ):
        score = _normalize_score(raw_checks.get(key))
        if score is not None:
            out[key] = score
    return out


def _compute_score(raw_score: Any, checks: dict[str, float]) -> Optional[float]:
    score = _normalize_score(raw_score)
    if score is not None:
        return score
    if not checks:
        return None
    return round(sum(checks.values()) / len(checks), 3)


def _build_session_evidence(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for session in sessions[:_MAX_SESSIONS]:
        item: dict[str, Any] = {
            "session_id": str(session.get("session_id", "")),
            "summary": _clip_text(session.get("_summary", ""), _SUMMARY_MAX_CHARS),
        }
        skills = session.get("_skills_referenced")
        if skills:
            item["skills_referenced"] = sorted(str(s or "") for s in skills if str(s or ""))
        judge_scores = session.get("_judge_scores")
        if isinstance(judge_scores, dict):
            overall = _normalize_score(judge_scores.get("overall_score"))
            if overall is not None:
                item["judge_overall_score"] = overall
        avg_prm = _normalize_score(session.get("_avg_prm"))
        if avg_prm is not None:
            item["avg_prm"] = avg_prm
        evidence.append(item)
    return evidence


async def verify_skill_candidate(
    llm: AsyncLLMClient,
    skill: dict[str, Any],
    sessions: list[dict[str, Any]],
    action_type: str,
    *,
    current_skill: Optional[dict[str, Any]] = None,
    min_score: float = 0.75,
) -> dict[str, Any]:
    """Verify a candidate skill before publishing it to shared storage."""
    payload = {
        "action": action_type,
        "candidate_skill": {
            "name": str(skill.get("name", "")),
            "description": str(skill.get("description", "")),
            "category": str(skill.get("category", "general")),
            "content": _clip_text(skill.get("content", ""), _SKILL_CONTENT_MAX_CHARS),
        },
        "current_skill": None,
        "session_evidence": _build_session_evidence(sessions),
        "acceptance_threshold": round(float(min_score), 3),
        "notes": {
            "optimize_description_only": action_type == DecisionAction.OPTIMIZE_DESC,
            "create_skill": action_type == DecisionAction.CREATE,
        },
    }
    if current_skill:
        payload["current_skill"] = {
            "name": str(current_skill.get("name", "")),
            "description": str(current_skill.get("description", "")),
            "category": str(current_skill.get("category", "general")),
            "content": _clip_text(current_skill.get("content", ""), _CURRENT_SKILL_MAX_CHARS),
        }

    try:
        raw = await llm.chat(
            [
                {"role": "system", "content": _VERIFY_SKILL_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            max_tokens=2000,
            temperature=0.1,
        )
    except Exception as exc:
        logger.warning("[SkillVerifier] verifier call failed for '%s': %s", skill.get("name", ""), exc)
        return {
            "enabled": True,
            "accepted": False,
            "decision": "reject",
            "score": None,
            "threshold": round(float(min_score), 3),
            "reason": f"Verifier call failed: {exc}",
            "checks": {},
        }

    parsed = _extract_json_object(raw)
    if not parsed:
        logger.warning("[SkillVerifier] invalid verifier output for '%s'", skill.get("name", ""))
        return {
            "enabled": True,
            "accepted": False,
            "decision": "reject",
            "score": None,
            "threshold": round(float(min_score), 3),
            "reason": "Verifier returned invalid JSON.",
            "checks": {},
        }

    checks = _normalize_checks(parsed.get("checks"))
    score = _compute_score(parsed.get("score"), checks)
    decision_raw = str(parsed.get("decision", "") or "").strip().lower()
    reason = str(parsed.get("reason") or parsed.get("rationale") or parsed.get("notes") or "").strip()

    accepted = decision_raw == "accept"
    if score is not None and score < float(min_score):
        accepted = False
    if decision_raw not in {"accept", "reject"}:
        accepted = score is not None and score >= float(min_score)

    return {
        "enabled": True,
        "accepted": bool(accepted),
        "decision": "accept" if accepted else "reject",
        "score": score,
        "threshold": round(float(min_score), 3),
        "reason": reason,
        "checks": checks,
    }
