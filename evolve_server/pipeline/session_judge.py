"""
Session-level LLM judge for classic evolve-server sessions.

This stage runs after summarization so it can reuse the generated
``_trajectory`` and ``_summary`` fields. It only backfills sessions that
do not already have a reliable session-level score from benchmark /
aggregate pipelines.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from ..core.llm_client import AsyncLLMClient

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """\
You are a session-level evaluator for SkillClaw trajectories.

You will receive one session with:
- a lossless trajectory
- an LLM-generated analysis summary
- extracted source artifacts that the agent read
- lightweight metadata such as prior PRM scores and tool-error flags
- extracted final output artifacts when the agent wrote files

Score the session on a 0.0-1.0 scale for:
- task_completion: whether the user's goal was completed
- response_quality: correctness, completeness, and clarity of the final outcome
- efficiency: whether the path avoided unnecessary retries / detours
- tool_usage: whether tool usage was appropriate and effective

Use this weighting for the overall score:
- task_completion: 0.55
- response_quality: 0.30
- efficiency: 0.05
- tool_usage: 0.10

Guidelines:
- 1.0 means clearly excellent on that dimension.
- 0.5 means mixed / uncertain / partially successful.
- 0.0 means clearly failed on that dimension.
- Prefer the trajectory as ground truth; use the summary as supporting analysis.
- Distinguish "missing evidence" from "clear failure". If evidence is weak, be conservative rather than extreme.
- Do not assume benchmark labels exist.
- Prioritize factual correctness and goal completion over polish.
- Do not heavily penalize framework/runtime startup noise (for example benign prologue reads,
  environment initialization, or short non-blocking detours) unless it materially interferes
  with solving the task.
- Use low efficiency scores only for severe wasted effort: repeated failed retries, long
  thrashing loops, or large amounts of irrelevant work.
- Judge tool_usage mainly by whether the core tools chosen were appropriate for reaching a
  correct result; do not over-penalize incidental startup/tooling noise.
- If the session includes concrete output artifacts (for example file contents written by the
  agent), treat those artifacts as strong evidence for task_completion and response_quality.
- If the session includes concrete source artifacts that the agent read from the task workspace,
  use those source artifacts as the primary factual basis for judging whether the final outputs
  are accurate.
- When written outputs match the requested schema/format and are consistent with the available
  evidence, score completion/quality based primarily on correctness of those outputs even if
  earlier exploration was noisy.
- Only lower completion/quality sharply when the final outputs are missing, malformed, clearly
  contradicted by evidence, or unsupported by the available facts.

Return EXACTLY one JSON object with:
{
  "task_completion": <float 0..1>,
  "response_quality": <float 0..1>,
  "efficiency": <float 0..1>,
  "tool_usage": <float 0..1>,
  "overall_score": <float 0..1>,
  "rationale": "<brief explanation>"
}

No markdown fences. No extra text.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_DIMENSION_KEYS = (
    "task_completion",
    "response_quality",
    "efficiency",
    "tool_usage",
)
_WEIGHTS = {
    "task_completion": 0.55,
    "response_quality": 0.30,
    "efficiency": 0.05,
    "tool_usage": 0.10,
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    if not _is_number(value):
        return None
    score = max(0.0, min(1.0, float(value)))
    return round(score, 3)


def _clip_text(value: Any, max_chars: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _compute_weighted_overall(scores: dict[str, float]) -> float:
    total = 0.0
    for key in _DIMENSION_KEYS:
        total += scores[key] * _WEIGHTS[key]
    return round(total, 3)


def _has_benchmark_overall_score(session: dict[str, Any]) -> bool:
    benchmark = session.get("benchmark")
    if isinstance(benchmark, dict) and _is_number(benchmark.get("overall_score")):
        return True
    return False


def _has_aggregate_mean_score(session: dict[str, Any]) -> bool:
    aggregate = session.get("aggregate")
    if isinstance(aggregate, dict) and _is_number(aggregate.get("mean_score")):
        return True
    return False


def _looks_like_existing_session_level_turn_score(session: dict[str, Any]) -> bool:
    turns = session.get("turns")
    if not isinstance(turns, list) or not turns:
        return False

    last_turn = turns[-1] if isinstance(turns[-1], dict) else {}
    last_score = last_turn.get("prm_score")
    if not _is_number(last_score):
        return False
    if not (0.0 <= float(last_score) <= 1.0):
        return False

    earlier_scores = []
    for turn in turns[:-1]:
        if not isinstance(turn, dict):
            continue
        prm = turn.get("prm_score")
        if prm is not None:
            earlier_scores.append(prm)

    # Be conservative: only treat the last-turn score as a benchmark-like
    # session score when the session also carries task/aggregate metadata
    # and there are no earlier PRM scores to suggest per-turn PRM usage.
    has_benchmarkish_context = bool(
        session.get("task_id")
        or session.get("aggregate")
        or session.get("phase")
    )
    return has_benchmarkish_context and not earlier_scores


def _should_skip_judging(session: dict[str, Any]) -> bool:
    turns = session.get("turns")
    if not isinstance(turns, list) or not turns:
        return True

    existing_judge = session.get("_judge_scores")
    if isinstance(existing_judge, dict) and _is_number(existing_judge.get("overall_score")):
        return True

    if _has_benchmark_overall_score(session):
        return True
    if _has_aggregate_mean_score(session):
        return True
    if _looks_like_existing_session_level_turn_score(session):
        return True
    return False


def _build_judge_payload(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session.get("session_id"),
        "num_turns": session.get("num_turns"),
        "skills_referenced": sorted(session.get("_skills_referenced") or []),
        "has_tool_errors": bool(session.get("_has_tool_errors")),
        "prior_prm_scores": list(session.get("_prm_scores") or []),
        "avg_prm_before_judge": session.get("_avg_prm"),
        "source_artifacts": _extract_source_artifacts(session),
        "output_artifacts": _extract_output_artifacts(session),
        "trajectory": session.get("_trajectory") or "",
        "summary": session.get("_summary") or "",
    }


def _extract_output_artifacts(
    session: dict[str, Any],
    *,
    max_artifacts: int = 4,
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for turn in session.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        for tool_call in turn.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            if str(function.get("name") or "").strip() != "write":
                continue
            raw_args = function.get("arguments")
            if not isinstance(raw_args, str):
                continue
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue
            path = str(args.get("path") or "").strip()
            content = args.get("content")
            if not path or content is None:
                continue
            artifacts.append(
                {
                    "path": path,
                    "content": _clip_text(content),
                }
            )
            if len(artifacts) >= max_artifacts:
                return artifacts
    return artifacts


def _extract_source_artifacts(
    session: dict[str, Any],
    *,
    max_artifacts: int = 6,
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for turn in session.get("turns") or []:
        if not isinstance(turn, dict):
            continue

        call_args_by_id: dict[str, dict[str, Any]] = {}
        for tool_call in turn.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            if str(function.get("name") or "").strip() != "read":
                continue
            raw_args = function.get("arguments")
            if not isinstance(raw_args, str):
                continue
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue
            call_id = str(tool_call.get("id") or "").replace("_", "")
            if call_id:
                call_args_by_id[call_id] = parsed_args

        for tool_result in turn.get("tool_results") or []:
            if not isinstance(tool_result, dict):
                continue
            if str(tool_result.get("tool_name") or "").strip() != "read":
                continue
            if bool(tool_result.get("has_error")):
                continue
            result_call_id = str(tool_result.get("tool_call_id") or "").replace("_", "")
            args = call_args_by_id.get(result_call_id)
            if not isinstance(args, dict):
                continue
            path = str(args.get("path") or "").strip()
            if not path or path in seen_paths:
                continue
            if path.startswith("/root/"):
                continue
            content = str(tool_result.get("content") or "").strip()
            if not content or content == "(see attached image)":
                continue
            artifacts.append(
                {
                    "path": path,
                    "content": _clip_text(content),
                }
            )
            seen_paths.add(path)
            if len(artifacts) >= max_artifacts:
                return artifacts
    return artifacts


def _apply_judge_scores(session: dict[str, Any], scores: dict[str, Any]) -> None:
    turns = session.get("turns") or []
    previous_prm_scores = list(session.get("_prm_scores") or [])
    previous_last_prm = None
    if turns and isinstance(turns[-1], dict):
        previous_last_prm = turns[-1].get("prm_score")
        turns[-1]["prm_score"] = scores["overall_score"]

    judge_scores = dict(scores)
    if previous_prm_scores:
        judge_scores["original_prm_scores"] = previous_prm_scores
    if previous_last_prm is not None:
        judge_scores["previous_last_prm_score"] = previous_last_prm

    session["_judge_scores"] = judge_scores
    session["_prm_scores"] = [scores["overall_score"]]
    session["_avg_prm"] = scores["overall_score"]


def _parse_scores(raw: str) -> Optional[dict[str, Any]]:
    payload = _extract_json_object(raw)
    if not payload:
        return None

    scores: dict[str, float] = {}
    for key in _DIMENSION_KEYS:
        normalized = _normalize_score(payload.get(key))
        if normalized is None:
            return None
        scores[key] = normalized

    overall = _compute_weighted_overall(scores)
    result = {
        **scores,
        "overall_score": overall,
        "rationale": str(payload.get("rationale") or "").strip(),
    }
    raw_overall = _normalize_score(payload.get("overall_score"))
    if raw_overall is not None:
        result["model_overall_score"] = raw_overall
    return result


async def judge_session(
    llm: AsyncLLMClient,
    session: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Judge one session and backfill session-level score metadata."""
    if _should_skip_judging(session):
        return None

    payload = _build_judge_payload(session)
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await llm.chat(messages, max_tokens=1200, temperature=0.1)
    except Exception as exc:
        logger.warning(
            "[SessionJudge] LLM call failed for session %s: %s",
            session.get("session_id"),
            exc,
        )
        return None

    scores = _parse_scores(raw)
    if not scores:
        logger.warning(
            "[SessionJudge] could not parse judge output for session %s",
            session.get("session_id"),
        )
        return None

    _apply_judge_scores(session, scores)
    return scores


async def judge_sessions_parallel(
    llm: AsyncLLMClient,
    sessions: list[dict[str, Any]],
) -> int:
    """Judge all sessions that lack a reliable session-level score."""
    if not sessions:
        return 0

    candidates = [session for session in sessions if not _should_skip_judging(session)]
    if not candidates:
        return 0

    results = await asyncio.gather(
        *[judge_session(llm, session) for session in candidates],
        return_exceptions=True,
    )

    judged = 0
    for session, result in zip(candidates, results):
        if isinstance(result, BaseException):
            logger.warning(
                "[SessionJudge] exception while judging session %s: %s",
                session.get("session_id"),
                result,
            )
            continue
        if result is not None:
            judged += 1

    logger.info(
        "[SessionJudge] judged %d/%d candidate sessions (%d total)",
        judged,
        len(candidates),
        len(sessions),
    )
    return judged
