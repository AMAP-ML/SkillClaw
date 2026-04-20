"""
Session-level execution helpers for the current evolve_server pipeline.

The active flow is intentionally small:
- merge same-name conflicts when two evolved versions collide
- evolve an existing skill from aggregated session evidence
- create a brand-new skill from no-skill session groups

Older turn-level attribution / decision / execution prompts were removed so
`evolve_server` matches the session-level pipeline used by `server.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..core.constants import DecisionAction
from ..core.llm_client import AsyncLLMClient
from ..core.utils import parse_single_skill

logger = logging.getLogger(__name__)

_MERGE_SKILL_SYSTEM = """\
You are a skill engineer for SkillClaw.

Two versions of the SAME skill exist because separate evolution actions produced different content under the same name.

Your task: merge the two versions into a single, superior version that combines the best parts of both.

## Merge principles

- Preserve ALL actionable guidance from both versions - do not drop useful content.
- Eliminate redundancy - deduplicate overlapping sections.
- If the two versions contradict each other, prefer the more specific or concrete guidance.
- Preserve the stronger existing structure unless reorganization is clearly beneficial.
- Do not rewrite either version just to make it look more standardized.
- Keep the same name.
- The merged description should cover trigger conditions from both versions.
- Only keep metadata or extra frontmatter that still helps the merged skill.
- The merged content should stay concise, but do not force a rigid section template.

## Output format

Return EXACTLY one JSON object with:
- "name": same name
- "description": merged trigger description
- "content": merged Markdown body only, not a full SKILL.md with frontmatter

Optional fields:
- "metadata": merged metadata when genuinely useful
- "extra_frontmatter": preserved or merged extra frontmatter when justified

No markdown fences. Output ONLY valid JSON.
"""

_EVOLVE_FROM_SESSIONS_SYSTEM = """\
You are a skill engineer for SkillClaw's skill evolution system.

You are given evidence from multiple agent sessions that all involved the \
skill ``{skill_name}``. Each session contains a programmatic trajectory \
(step-by-step tool calls and outcomes) and an LLM-generated analysis.

Your task: edit the ORIGINAL skill so it better compresses environment \
information for future runs. Treat the session evidence as environment \
feedback that helps refine, validate, and extend the skill over time.

Analyze the session evidence alongside the current skill content, then \
decide the best course of action:

1. **improve_skill** - The skill content needs targeted edits based on the \
session evidence (for example missing guidance, outdated information, or \
unclear instructions). Produce the updated skill.

2. **optimize_description** - The skill body content is fine, but its \
description causes it to be matched to wrong tasks. Rewrite ONLY the \
description for more precise triggering. Do NOT change the body content.

3. **create_skill** - The session evidence reveals a recurring pattern, \
capability gap, or reusable strategy that does NOT belong in the current \
skill ``{skill_name}``. A brand-new, separate skill is needed. The current \
skill remains unchanged. Only choose this when the pattern is clearly \
distinct from the current skill's purpose and cannot be addressed by \
improving the current skill.

4. **skip** - The skill is working well enough, or the evidence is too weak \
or ambiguous to justify changes. No action needed.

## Editing principles (for improve_skill)

- Treat the CURRENT skill as the source of truth, not as a rough draft to be rewritten.
- Read the original skill first, then the session evidence.
- Default to targeted edits, not rewrites.
- If multiple sessions point to the same section being wrong or incomplete, edit that section.
- If failures are only corner cases, add the missing checks or clarify constraints without changing unrelated sections.
- Preserve the original structure, heading order, terminology, and effective guidance, especially parts supported by successful sessions.
- Only rewrite an entire section if the evidence shows that section is materially wrong.
- If the skill contains concrete API details (endpoints, ports, payload schemas, tool names) that are factually correct, KEEP them even if the agent did not use them well. These details are the skill's core value.

## Hard constraints

- Do NOT casually change task API contracts, ports, endpoints, output paths, payload formats, or required filenames. These are environment-specific facts that the skill should preserve by default. EXCEPTION: if the session evidence clearly shows that an API endpoint, port, or contract has changed, update the skill to reflect the corrected value.
- Do NOT remove core capabilities, API references, command patterns, or tool-usage examples unrelated to the observed failures.
- Do NOT turn the skill into a different skill with a different purpose.
- Do NOT rewrite the whole skill from scratch.
- Do NOT impose a new template, new mandatory section structure, or a different writing style unless the evidence requires it.
- Do NOT add generic best-practice guidance (for example rate-limit handling, retry logic, state management, or caching) that the agent should handle on its own. Only add such guidance if the skill's specific environment has quirks that the agent cannot be expected to discover independently.

## Conservative editing mode

- Prefer preserving existing section headings and ordering.
- If a successful session supports a section, leave that section untouched unless failure evidence explicitly contradicts it.
- Prefer tightening or clarifying an existing section over adding a brand-new section.
- Do not introduce a new large section unless failure evidence is strong and the existing structure cannot express the fix.
- If you add a new checklist item, keep it short and tied to the observed failure.

## Distinguishing skill problems from agent problems

Not every failure is a skill deficiency. Before editing, consider whether the failure was caused by:
- **The skill** (wrong, missing, or misleading guidance) -> edit the skill.
- **The agent** (subagent misuse, unnecessary restarts, context overflow, or not reading the skill properly) -> these are agent-level issues; do NOT bloat the skill with agent-runtime advice.
- **The environment** (mock API instability, network flakiness, docker quirks) -> if sessions show repeated API failures or timeouts, add a brief note about the instability so the agent knows to expect it. Keep it short; do NOT turn the skill into a retry tutorial.

Critical anti-pattern to avoid: if the skill ALREADY contains correct environment information (API endpoints, ports, payload formats, tool names) and the agent failed because it did NOT use that information, that is an AGENT problem, not a skill problem. Do NOT delete the correct API information from the skill and replace it with instructions like "go read utils.py" or "inspect the mock service code". The whole point of the skill is to save the agent from having to discover those details.

When in doubt, prefer **skip** over a speculative edit.

## Skill-writing principles (for create_skill)

- The new skill must serve a DIFFERENT purpose than ``{skill_name}``.
- Prefer a short, action-oriented name (lowercase-hyphenated slug).
- The name MUST differ from all existing skill names listed below.
- A skill should compress environment information (API endpoints, ports, payload formats, tool-specific quirks, or domain procedures), not generic best practices the agent already knows.
- Description should state what the skill does and triggering contexts, including "NOT for: ..." exclusion conditions. 2-4 sentences.
- Content should be domain-specific, practically useful, and non-obvious.
- Keep it concise, reusable, and evidence-driven.
- Write reusable guidance, not a failure summary or postmortem.

## Output format

Return EXACTLY one JSON object (no markdown fences, no extra text):

If action is improve_skill:
```
{{
  "action": "improve_skill",
  "rationale": "<why, synthesizing the evidence>",
  "skill": {{
    "name": "<keep same name>",
    "description": "<keep or improve>",
    "content": "<full updated Markdown body>",
    "category": "<keep or update>",
    "edit_summary": {{"preserved_sections": [...], "changed_sections": [...], "notes": "..."}}
  }}
}}
```

If action is optimize_description:
```
{{
  "action": "optimize_description",
  "rationale": "<why>",
  "skill": {{
    "name": "<keep same name>",
    "description": "<rewritten description with Use-when and NOT-for conditions>"
  }}
}}
```

If action is create_skill:
```
{{
  "action": "create_skill",
  "rationale": "<why a new skill is needed and why the current skill should not absorb this>",
  "skill": {{
    "name": "<new-lowercase-slug, MUST differ from {skill_name} and all existing names>",
    "description": "<2-4 sentences with triggering contexts and NOT-for conditions>",
    "content": "<skill body in Markdown>"
  }}
}}
```

If action is skip:
```
{{
  "action": "skip",
  "rationale": "<why skipping>"
}}
```
"""

_CREATE_FROM_SESSIONS_SYSTEM = """\
You are a skill engineer for SkillClaw.

You are given summaries of agent sessions where no existing skill was \
referenced. These sessions may reveal patterns that could be captured as a \
reusable skill for future sessions.

Analyze whether these sessions reveal a common pattern, recurring challenge, \
or reusable strategy that would benefit future agent sessions if captured as \
a skill.

1. **create_skill** - A clear, teachable pattern exists that compresses environment-specific knowledge the agent cannot reliably discover on its own. Produce the new skill.
2. **skip** - No actionable or generalizable pattern. The sessions are too diverse, too domain-specific, or the issues are not solvable by skills.

## Skill-writing principles (for create_skill)

- A skill should compress environment information (API endpoints, ports, payload formats, tool-specific quirks, or domain procedures), not generic best practices the agent already knows.
- Prefer a short, action-oriented name (lowercase-hyphenated slug).
- Description should state what the skill does and triggering contexts, including "NOT for: ..." exclusion conditions. 2-4 sentences.
- Content should be domain-specific, practically useful, and non-obvious.
- Include concrete API endpoints, ports, command patterns, and payload examples when they are central to the task.
- Keep it concise, reusable, and evidence-driven.
- Write reusable guidance, not a failure summary or postmortem.
- Use imperative instructions. Organize naturally for the task.
- Do NOT add generic agent-runtime advice (rate-limit handling, retry logic, caching strategies, or state management) unless the environment has specific quirks that require it.

## When to skip

Prefer skip when:
- The failures are caused by agent-level issues (retries, context overflow, or subagent misuse) rather than missing knowledge.
- The sessions are too diverse to extract a single coherent skill.
- The pattern is something the agent should handle via general intelligence.

## Output format

Return EXACTLY one JSON object (no markdown fences, no extra text):

If action is create_skill:
```
{{
  "action": "create_skill",
  "rationale": "<why creating this skill>",
  "skill": {{
    "name": "<lowercase-hyphenated-slug>",
    "description": "<2-4 sentences with triggering contexts and NOT-for>",
    "content": "<skill body in Markdown>"
  }}
}}
```

If action is skip:
```
{{
  "action": "skip",
  "rationale": "<why skipping>"
}}
```
"""

_EVOLVE_DEBUG_DIR = ""


def set_evolve_debug_dir(path: str) -> None:
    """Set the debug dump directory used by session-level evolution calls."""
    global _EVOLVE_DEBUG_DIR
    _EVOLVE_DEBUG_DIR = str(path or "").strip()


def _get_evolve_debug_dir() -> str:
    return _EVOLVE_DEBUG_DIR


async def execute_merge(
    llm: AsyncLLMClient,
    existing_skill: dict,
    incoming_skill: dict,
) -> Optional[dict]:
    """Merge two versions of the same skill into one superior version."""
    user_msg = (
        f"## Version A (currently in shared storage, v{existing_skill.get('_version', '?')})\n\n"
        f"Name: {existing_skill.get('name', '')}\n"
        f"Description: {existing_skill.get('description', '')}\n"
        f"Category: {existing_skill.get('category', 'general')}\n\n"
        f"Content:\n```\n{existing_skill.get('content', '')}\n```\n\n"
        f"---\n\n"
        f"## Version B (newly evolved)\n\n"
        f"Name: {incoming_skill.get('name', '')}\n"
        f"Description: {incoming_skill.get('description', '')}\n"
        f"Category: {incoming_skill.get('category', 'general')}\n\n"
        f"Content:\n```\n{incoming_skill.get('content', '')}\n```"
    )
    messages = [
        {"role": "system", "content": _MERGE_SKILL_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = await llm.chat(messages, max_tokens=8192, temperature=0.3)
    return parse_single_skill(raw)


def _build_skill_block(skill: dict) -> str:
    return (
        f"## Current skill\n\n"
        f"Name: {skill.get('name', '')}\n"
        f"Description: {skill.get('description', '')}\n"
        f"Category: {skill.get('category', 'general')}\n\n"
        f"Content:\n```\n{skill.get('content', '')}\n```\n\n"
    )


def _build_session_evidence(sessions: list[dict], max_sessions: int = 30) -> str:
    """Format session evidence (trajectory + summary) for LLM prompts."""
    blocks: list[str] = []
    for session in sessions[:max_sessions]:
        session_id = session.get("session_id", "?")
        avg_prm = session.get("_avg_prm")
        prm_str = f", avg PRM: {avg_prm}" if avg_prm is not None else ""
        has_errors = session.get("_has_tool_errors", False)
        err_str = ", has tool errors" if has_errors else ""
        skills = session.get("_skills_referenced") or set()
        skill_str = f", skills: {sorted(skills)}" if skills else ""

        aggregate = session.get("aggregate") or {}
        aggregate_str = ""
        if aggregate:
            parts: list[str] = []
            rollout_count = aggregate.get("rollout_count", 0)
            mean_score = aggregate.get("mean_score")
            stability = aggregate.get("stability", "")
            success_count = aggregate.get("success_count", 0)
            fail_count = aggregate.get("fail_count", 0)
            if rollout_count:
                parts.append(f"{rollout_count} rollouts")
            if mean_score is not None:
                parts.append(f"mean ORM={mean_score:.3f}")
            if success_count or fail_count:
                parts.append(f"success={success_count} fail={fail_count}")
            if stability:
                parts.append(f"stability={stability}")
            if parts:
                aggregate_str = f", {', '.join(parts)}"

        trajectory = session.get("_trajectory", "")
        summary = session.get("_summary", "")

        parts = [f"### Session {session_id}{prm_str}{aggregate_str}{err_str}{skill_str}"]
        if trajectory:
            parts.append(f"**Trajectory**:\n{trajectory}")
        if summary:
            parts.append(f"**Analysis**:\n{summary}")
        if not trajectory and not summary:
            parts.append("(no data)")
        blocks.append("\n\n".join(parts))

    if len(sessions) > max_sessions:
        blocks.append(f"\n... and {len(sessions) - max_sessions} more sessions")

    return "\n\n---\n\n".join(blocks)


def _write_debug_dump(stem: str, system: str, user_msg: str, raw: str | None = None) -> None:
    debug_dir = _get_evolve_debug_dir()
    if not debug_dir:
        return

    dump_dir = Path(debug_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / f"{stem}_system.txt").write_text(system, encoding="utf-8")
    (dump_dir / f"{stem}_user.txt").write_text(user_msg, encoding="utf-8")
    if raw is not None:
        (dump_dir / f"{stem}_raw_output.txt").write_text(raw, encoding="utf-8")
    logger.info("[DebugDump] wrote %s prompt artifacts to %s", stem, dump_dir)


async def evolve_skill_from_sessions(
    llm: AsyncLLMClient,
    skill_name: str,
    sessions: list[dict],
    current_skill: Optional[dict],
    existing_skill_names: list[str],
) -> Optional[dict]:
    """Combined decision + execution for one existing-skill session group."""
    system = _EVOLVE_FROM_SESSIONS_SYSTEM.replace("{skill_name}", skill_name)
    skill_section = _build_skill_block(current_skill) if current_skill else ""
    evidence = _build_session_evidence(sessions)
    user_msg = (
        f"{skill_section}"
        f"## Session evidence ({len(sessions)} sessions)\n\n"
        f"{evidence}\n\n"
        f"## Existing skill names in the library\n\n"
        f"{', '.join(existing_skill_names) or '(none)'}\n"
    )

    stem = skill_name.replace("/", "_")
    _write_debug_dump(stem, system, user_msg)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    raw = await llm.chat(messages, max_tokens=8192, temperature=0.4)
    _write_debug_dump(stem, system, user_msg, raw)
    return _parse_evolve_result(raw, skill_name)


async def create_skill_from_sessions(
    llm: AsyncLLMClient,
    sessions: list[dict],
    existing_skill_names: list[str],
) -> Optional[dict]:
    """Combined decision + execution for the no-skill session bucket."""
    evidence = _build_session_evidence(sessions)
    user_msg = (
        f"## Session evidence ({len(sessions)} sessions)\n\n"
        f"{evidence}\n\n"
        f"## Existing skill names in the library\n\n"
        f"{', '.join(existing_skill_names) or '(none)'}\n"
    )

    stem = "no_skill"
    _write_debug_dump(stem, _CREATE_FROM_SESSIONS_SYSTEM, user_msg)

    messages = [
        {"role": "system", "content": _CREATE_FROM_SESSIONS_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = await llm.chat(messages, max_tokens=8192, temperature=0.4)
    _write_debug_dump(stem, _CREATE_FROM_SESSIONS_SYSTEM, user_msg, raw)
    return _parse_evolve_result(raw, "")


def _parse_evolve_result(raw: str, skill_name: str) -> Optional[dict]:
    """Parse the combined decision+execution JSON from the LLM."""
    import json
    import re

    clean = re.sub(r"```(?:json)?\s*", "", raw.strip()).strip().rstrip("`")

    try:
        result = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end <= start:
            logger.warning("[SessionExec] no JSON object found for '%s'", skill_name)
            return None
        try:
            result = json.loads(clean[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            logger.warning("[SessionExec] failed to parse evolve result for '%s'", skill_name)
            return None

    if not isinstance(result, dict):
        return None

    action = result.get("action", DecisionAction.SKIP)
    if action == DecisionAction.SKIP:
        return {"action": DecisionAction.SKIP, "rationale": result.get("rationale", "")}

    skill_data = result.get("skill")
    if not isinstance(skill_data, dict):
        logger.warning("[SessionExec] action '%s' but no skill data for '%s'", action, skill_name)
        return None

    if action == DecisionAction.CREATE:
        if not skill_data.get("name"):
            logger.warning("[SessionExec] create_skill action but no name provided for '%s'", skill_name)
            return None
        if skill_data["name"] == skill_name:
            logger.warning(
                "[SessionExec] create_skill returned same name '%s' - treating as improve",
                skill_name,
            )
            action = DecisionAction.IMPROVE
    elif skill_name and not skill_data.get("name"):
        skill_data["name"] = skill_name

    return {
        "action": action,
        "rationale": result.get("rationale", ""),
        "skill": skill_data,
    }
