"""
Workspace management for the agent engine under ``evolve_server``.

Handles preparing the local workspace directory that OpenClaw operates on,
snapshotting skill state before agent execution, and collecting changes
(new / modified SKILL.md files) after the agent finishes.

Key design note on OpenClaw bootstrap integration:
  OpenClaw's ``ensureAgentWorkspace()`` creates template bootstrap files
  (AGENTS.md, SOUL.md, USER.md, IDENTITY.md, …) using ``writeFileIfMissing``
  with ``flag: 'wx'`` — it will NOT overwrite files that already exist.
  We exploit this by pre-writing our own versions of these files during
  ``prepare()`` so OpenClaw picks them up as-is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from ..core.utils import parse_skill_content

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom bootstrap templates for the evolve-only workspace
# ---------------------------------------------------------------------------

_EVOLVE_AGENTS_MD = """\
# Skill Evolution Agent

You are a **skill evolution engineer**. Your sole task is to analyze agent
session data in this workspace and evolve the skill library.

## First Step — ALWAYS

Read `EVOLVE_AGENTS.md` in this workspace **before doing anything else**.
It contains the full methodology, workspace layout, editing principles,
and all instructions you need.

```
cat EVOLVE_AGENTS.md
```

## Workspace Quick Reference

```
workspace/
├── EVOLVE_AGENTS.md   ← full evolution methodology (READ THIS FIRST)
├── sessions/          ← agent session JSON files to analyze
├── skills/            ← current skill library (read + write)
│   └── <name>/
│       ├── SKILL.md
│       └── history/
├── manifest.json      ← skill manifest (read-only)
└── skill_registry.json
```

## Constraints

- **All file operations** stay within this workspace directory.
- Do NOT modify `sessions/`, `manifest.json`, or `skill_registry.json`.
- Write changes only to `skills/<name>/SKILL.md` (and `history/`).
- If there are no actionable patterns, make no changes — that is fine.

## Memory

You may use `memory/` and `MEMORY.md` for long-term notes across rounds.
Append daily observations to `memory/YYYY-MM-DD.md`. Keep `MEMORY.md` for
curated, high-level summaries of skill evolution decisions.
"""

_EVOLVE_SOUL_MD = """\
You analyze agent session data and evolve reusable skills.
Be methodical: read all sessions, aggregate patterns, then make targeted
changes. Prefer conservative edits over rewrites. Skip when evidence is weak.
"""

_EVOLVE_IDENTITY_MD = """\
Skill Evolution Agent — SkillClaw
"""

_EVOLVE_USER_MD = """\
The operator is the SkillClaw evolve server. Follow EVOLVE_AGENTS.md.
"""


class AgentWorkspace:
    """Manages the filesystem workspace that the OpenClaw agent reads/writes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.skills_dir = self.root / "skills"

    def reset(self) -> None:
        """Completely wipe and recreate the workspace (used in fresh mode)."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("[AgentWorkspace] reset: wiped %s", self.root)

    def prepare(
        self,
        sessions: list[dict],
        existing_skills: dict[str, str],
        manifest: dict[str, dict],
        agents_md: str,
        skill_registry_info: dict[str, Any] | None = None,
    ) -> None:
        """Populate the workspace with input data for the agent.

        Parameters
        ----------
        sessions:
            Raw session dicts drained from storage.
        existing_skills:
            ``{skill_name: SKILL.md content}`` for all current skills.
        manifest:
            Current manifest dict ``{skill_name: metadata}``.
        agents_md:
            Full text of the AGENTS.md to write into the workspace.
        skill_registry_info:
            Optional registry metadata to expose to the agent.
        """
        # Clean sessions dir (always fresh input)
        if self.sessions_dir.exists():
            shutil.rmtree(self.sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Write session files (compact: metadata + trajectory + summary only)
        for s in sessions:
            sid = s.get("session_id", "unknown")
            compact = {
                "session_id": sid,
                "task_id": s.get("task_id", ""),
                "num_turns": s.get("num_turns", len(s.get("turns", []))),
                "aggregate": s.get("aggregate"),
                "_skills_referenced": sorted(s.get("_skills_referenced") or []),
                "_avg_prm": s.get("_avg_prm"),
                "_has_tool_errors": s.get("_has_tool_errors", False),
                "_trajectory": s.get("_trajectory", ""),
                "_summary": s.get("_summary", ""),
            }
            path = self.sessions_dir / f"{sid}.json"
            path.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")

        # Write existing skills
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        for name, content in existing_skills.items():
            skill_dir = self.skills_dir / name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        # Write manifest
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        # Write skill registry info (read-only reference for the agent)
        if skill_registry_info:
            registry_path = self.root / "skill_registry.json"
            registry_path.write_text(
                json.dumps(skill_registry_info, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Write EVOLVE_AGENTS.md (the detailed methodology the agent follows)
        agents_md_path = self.root / "EVOLVE_AGENTS.md"
        agents_md_path.write_text(agents_md, encoding="utf-8")

        # -----------------------------------------------------------
        # Pre-write OpenClaw bootstrap files.
        #
        # OpenClaw's ensureAgentWorkspace() creates these with
        # writeFileIfMissing (flag 'wx'), so it will NOT overwrite
        # files we write here. This lets us:
        #   - Replace AGENTS.md with a focused evolve-only version
        #     that directs the agent to read EVOLVE_AGENTS.md first.
        #   - Replace SOUL.md / IDENTITY.md / USER.md with minimal
        #     content to avoid wasting tokens on default templates.
        #   - Leave TOOLS.md untouched (agent needs tool descriptions).
        #   - Leave MEMORY.md / memory/ untouched (memory-core manages
        #     them; we want to preserve cross-round memory).
        # -----------------------------------------------------------
        bootstrap_files = {
            "AGENTS.md": _EVOLVE_AGENTS_MD,
            "SOUL.md": _EVOLVE_SOUL_MD,
            "IDENTITY.md": _EVOLVE_IDENTITY_MD,
            "USER.md": _EVOLVE_USER_MD,
        }
        for filename, content in bootstrap_files.items():
            fpath = self.root / filename
            fpath.write_text(content, encoding="utf-8")
            logger.debug("[AgentWorkspace] wrote bootstrap %s", filename)

        logger.info(
            "[AgentWorkspace] prepared: %d sessions, %d existing skills, "
            "bootstrap files written in %s",
            len(sessions), len(existing_skills), self.root,
        )

    def snapshot_skills(self) -> dict[str, str]:
        """Return ``{skill_name: sha256_of_content}`` for all skills in the workspace."""
        snapshot: dict[str, str] = {}
        if not self.skills_dir.exists():
            return snapshot
        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.is_file():
                content = skill_md.read_bytes()
                snapshot[skill_dir.name] = hashlib.sha256(content).hexdigest()
        return snapshot

    def collect_changes(
        self, before_snapshot: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Compare current skills against *before_snapshot* and return changed/new skills.

        Returns a list of dicts, each with:
        - ``name``: skill name
        - ``action``: ``"create"`` or ``"improve"``
        - ``skill``: parsed skill dict (name, description, content, ...)
        - ``raw_md``: the raw SKILL.md text
        """
        after_snapshot = self.snapshot_skills()
        changes: list[dict[str, Any]] = []

        for name, after_sha in after_snapshot.items():
            before_sha = before_snapshot.get(name)
            if before_sha == after_sha:
                continue

            skill_md_path = self.skills_dir / name / "SKILL.md"
            raw_md = skill_md_path.read_text(encoding="utf-8")
            parsed = parse_skill_content(name, raw_md)

            action = "create" if before_sha is None else "improve"
            changes.append({
                "name": name,
                "action": action,
                "skill": parsed,
                "raw_md": raw_md,
            })
            logger.info(
                "[AgentWorkspace] detected %s: skill '%s' (sha %s -> %s)",
                action, name,
                (before_sha or "new")[:12],
                after_sha[:12],
            )

        deleted = set(before_snapshot) - set(after_snapshot)
        for name in sorted(deleted):
            logger.warning(
                "[AgentWorkspace] skill '%s' was deleted by agent — ignoring deletion",
                name,
            )

        return changes

    def cleanup_sessions(self) -> None:
        """Remove session files from the workspace after processing."""
        if self.sessions_dir.exists():
            shutil.rmtree(self.sessions_dir)
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
