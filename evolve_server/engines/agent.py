"""
Agent-driven evolve server implementation inside the unified package.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skillclaw.object_store import build_object_store

from ..core.config import EvolveServerConfig
from ..core.constants import SLUG_RE
from ..core.llm_client import AsyncLLMClient
from ..core.skill_registry import SkillIDRegistry
from ..core.utils import build_skill_md
from ..pipeline.summarizer import (
    _extract_session_metadata,
    build_session_trajectory,
    summarize_sessions_parallel,
)
from ..storage.mock_bucket import LocalBucket
from ..storage.oss_helpers import (
    delete_session_keys,
    fetch_skill_content,
    list_session_keys,
    load_manifest,
    read_json_object,
    save_manifest,
)
from .agent_workspace import AgentWorkspace
from .agents_md import load_agents_md
from .openclaw_runner import OpenClawRunner

logger = logging.getLogger(__name__)


class _AnthropicMessagesLLMClient:
    """Minimal Anthropic Messages client for the summarizer path."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        model: str = "gpt-5.4",
        max_tokens: int = 100000,
        temperature: float = 0.4,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _messages_url(self) -> str:
        base = str(self.base_url or "https://api.anthropic.com").rstrip("/")
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import random

        import httpx

        system_parts: list[str] = []
        body_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role in {"user", "assistant"}:
                body_messages.append({"role": role, "content": content})

        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": body_messages or [{"role": "user", "content": ""}],
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        if system_parts:
            request_body["system"] = "\n\n".join(system_parts)
        request_body.update(kwargs)

        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        max_retries = 6
        timeout = httpx.Timeout(600.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(max_retries):
                try:
                    resp = await client.post(
                        self._messages_url(),
                        json=request_body,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    parts: list[str] = []
                    for item in payload.get("content", []) or []:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text")
                            if isinstance(text, str) and text:
                                parts.append(text)
                    return "".join(parts)
                except Exception:
                    if attempt < max_retries - 1:
                        wait = min(2**attempt + random.uniform(0, 1), 30)
                        await asyncio.sleep(wait)
                        continue
                    raise


class AgentEvolveServer:
    """Agent-driven evolution server.

    Parameters
    ----------
    config:
        Server configuration (storage, LLM, scheduling, OpenClaw settings).
    mock:
        If ``True``, use a local directory instead of remote storage.
    mock_root:
        Custom root for mock mode (default: ``evolve_server/mock/``).
    """

    def __init__(
        self,
        config: EvolveServerConfig,
        *,
        mock: bool = False,
        mock_root: str | None = None,
    ) -> None:
        self.config = config
        self._mock = mock

        if mock:
            default_mock = str(Path(__file__).resolve().parent / "mock")
            self._bucket = LocalBucket(root=mock_root or default_mock)
        else:
            self._bucket = build_object_store(
                backend=config.storage_backend,
                endpoint=config.storage_endpoint,
                bucket=config.storage_bucket,
                access_key_id=config.storage_access_key_id,
                secret_access_key=config.storage_secret_access_key,
                region=config.storage_region,
                session_token=config.storage_session_token,
                local_root=config.local_root,
            )

        self._prefix = f"{config.group_id}/"
        self._id_registry = SkillIDRegistry()
        self._running = False

        self._id_registry.load_from_oss(self._bucket, self._prefix)

        self._workspace = AgentWorkspace(config.workspace_root)

        self._runner = OpenClawRunner(
            openclaw_bin=config.openclaw_bin,
            openclaw_home=config.openclaw_home,
            fresh=config.fresh,
            timeout=config.agent_timeout,
            llm_api_key=config.llm_api_key,
            llm_base_url=config.llm_base_url,
            llm_model=config.llm_model,
            llm_api_type=config.llm_api_type,
        )

        # Persistent session ID for multi-round evolution (when fresh=False)
        self._agent_session_id = f"evolve-{config.group_id}"

    async def _summarize_sessions(self, sessions: list[dict]) -> None:
        """Attach compact trajectory/summary fields before workspace prep."""
        api_type = str(self.config.llm_api_type or "").strip().lower()
        if api_type in {"", "openai-completions", "openai-responses", "ollama"}:
            llm: Any = AsyncLLMClient(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            await summarize_sessions_parallel(llm, sessions)
            return

        if api_type == "anthropic-messages":
            llm = _AnthropicMessagesLLMClient(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            await summarize_sessions_parallel(llm, sessions)
            return

        logger.warning(
            "[AgentEvolveServer] summarizer does not support llm_api_type=%s; "
            "falling back to metadata + trajectory only",
            api_type or "(empty)",
        )
        for session in sessions:
            _extract_session_metadata(session)
            session["_trajectory"] = build_session_trajectory(session)
            session["_summary"] = ""

    def _uses_local_storage(self) -> bool:
        backend = str(self.config.storage_backend or "").strip().lower()
        if backend == "local" or self._mock:
            return True
        bucket_type = type(self._bucket).__name__.lower()
        return "local" in bucket_type and bool(self.config.local_root)

    async def _call_storage(self, func, *args):
        if self._uses_local_storage():
            return func(*args)
        return await asyncio.to_thread(func, *args)

    # ================================================================= #
    #  History                                                           #
    # ================================================================= #

    def _append_history(self, record: dict) -> None:
        path = self.config.history_path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[AgentEvolveServer] history write failed: %s", e)

    # ================================================================= #
    #  Storage data access                                               #
    # ================================================================= #

    async def _drain_sessions(self) -> tuple[list[dict], list[str]]:
        keys = await self._call_storage(list_session_keys, self._bucket, self._prefix)
        sessions: list[dict] = []
        consumed_keys: list[str] = []
        for key in keys:
            session = await self._call_storage(read_json_object, self._bucket, key)
            if session:
                sessions.append(session)
                consumed_keys.append(key)
        logger.info(
            "[AgentEvolveServer] drained %d session(s) (%d keys found)",
            len(sessions),
            len(keys),
        )
        return sessions, consumed_keys

    def _load_remote_skills(self) -> dict[str, dict[str, Any]]:
        return load_manifest(self._bucket, self._prefix)

    def _fetch_all_skills(self, manifest: dict[str, dict]) -> dict[str, str]:
        """Fetch SKILL.md content for all skills in the manifest."""
        skills: dict[str, str] = {}
        for name in manifest:
            content = fetch_skill_content(self._bucket, self._prefix, name)
            if content:
                skills[name] = content
        return skills

    # ================================================================= #
    #  Upload evolved skills                                             #
    # ================================================================= #

    def _upload_skill(self, skill: dict, action: str = "create") -> None:
        name = skill.get("name", "")
        if not name:
            return

        skill_id = self._id_registry.get_or_create(name)
        md_content = build_skill_md(skill)
        object_key = f"{self._prefix}skills/{name}/SKILL.md"

        self._bucket.put_object(object_key, md_content.encode("utf-8"))

        content_sha = hashlib.sha256(md_content.encode("utf-8")).hexdigest()
        version = self._id_registry.record_update(name, content_sha, action=action)

        manifest = self._load_remote_skills()
        manifest[name] = {
            "name": name,
            "skill_id": skill_id,
            "version": version,
            "sha256": content_sha,
            "uploaded_by": "evolve_server",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "description": skill.get("description", ""),
            "category": skill.get("category", "general"),
        }
        save_manifest(self._bucket, self._prefix, manifest)
        logger.info(
            "[AgentEvolveServer] uploaded skill %s (id=%s, v%d) to %s",
            name,
            skill_id,
            version,
            object_key,
        )

    # ================================================================= #
    #  Main pipeline                                                     #
    # ================================================================= #

    async def run_once(self) -> dict:
        """Run one full agent-driven evolution cycle.

        1. Drain session files from storage
        2. Prepare workspace (sessions + existing skills)
        3. Snapshot current skills state
        4. Run OpenClaw agent
        5. Collect skill changes from workspace
        6. Upload changed skills to storage
        7. Ack consumed sessions
        """
        logger.info("[AgentEvolveServer] === starting evolution cycle ===")
        t0 = time.monotonic()

        # ---- 1. drain ------------------------------------------------ #
        sessions, session_keys = await self._drain_sessions()
        if not sessions:
            logger.info("[AgentEvolveServer] queue empty — nothing to process")
            return {
                "sessions": 0,
                "skills_evolved": 0,
                "agent_returncode": None,
            }

        # ---- 1.5. summarize sessions -------------------------------- #
        #  Reuse the standard evolve_server summarizer to compress raw
        #  session data before placing it in the workspace.  This adds
        #  _trajectory (programmatic, lossless step-by-step trace) and
        #  _summary (LLM-generated causal analysis) to each session dict,
        #  reducing ~MB-scale raw sessions to ~tens-of-KB each and keeping
        #  the agent well within the context-window budget.
        await self._summarize_sessions(sessions)
        logger.info(
            "[AgentEvolveServer] summarized %d session(s)",
            len(sessions),
        )

        # ---- 2. prepare workspace ------------------------------------ #
        if self.config.fresh:
            self._workspace.reset()

        manifest = await self._call_storage(self._load_remote_skills)
        existing_skills = await self._call_storage(self._fetch_all_skills, manifest)

        agents_md_text = self._load_agents_md()

        registry_info = self._id_registry.all_entries()

        self._workspace.prepare(
            sessions=sessions,
            existing_skills=existing_skills,
            manifest=manifest,
            agents_md=agents_md_text,
            skill_registry_info=registry_info,
        )

        # ---- 3. snapshot --------------------------------------------- #
        before_snapshot = self._workspace.snapshot_skills()

        # ---- 4. run agent -------------------------------------------- #
        session_id = self._agent_session_id if not self.config.fresh else None
        message = (
            f"Process the {len(sessions)} session file(s) in the sessions/ directory. "
            f"Follow the instructions in AGENTS.md to analyze them and evolve "
            f"the skill library in skills/. "
            f"There are currently {len(existing_skills)} existing skill(s)."
        )

        result = await asyncio.to_thread(
            self._runner.run,
            workspace_path=Path(self.config.workspace_root),
            message=message,
            session_id=session_id,
        )

        # ---- 5. collect results -------------------------------------- #
        changes = self._workspace.collect_changes(before_snapshot)

        # ---- 6. upload ----------------------------------------------- #
        skills_evolved = 0
        evolution_records: list[dict] = []

        for change in changes:
            skill = change["skill"]
            action = change["action"]
            name = self._sanitise_name(skill.get("name", change["name"]))
            skill["name"] = name

            try:
                await self._call_storage(self._upload_skill, skill, action)
                skills_evolved += 1
                evolution_records.append(
                    {
                        "action": action,
                        "skill_name": name,
                        "skill_id": self._id_registry.get_or_create(name),
                        "version": self._id_registry.get_version(name),
                        "source": "agent",
                    }
                )
            except Exception as e:
                logger.error(
                    "[AgentEvolveServer] failed to upload skill '%s': %s",
                    name,
                    e,
                )

        # ---- 7. finalize + ack --------------------------------------- #
        await self._call_storage(
            self._id_registry.save_to_oss,
            self._bucket,
            self._prefix,
        )
        await self._call_storage(delete_session_keys, self._bucket, session_keys)

        self._workspace.cleanup_sessions()

        elapsed = time.monotonic() - t0
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "sessions": len(sessions),
            "skills_evolved": skills_evolved,
            "agent_returncode": result.returncode,
            "evolutions": evolution_records,
        }
        self._append_history(summary)
        logger.info(
            "[AgentEvolveServer] === cycle done: %d sessions, %d skills evolved in %.1fs (agent exit=%d) ===",
            len(sessions),
            skills_evolved,
            elapsed,
            result.returncode,
        )
        return summary

    # ================================================================= #
    #  AGENTS.md loading                                                 #
    # ================================================================= #

    def _load_agents_md(self) -> str:
        """Load AGENTS.md from custom path or the built-in default."""
        custom_path = self.config.agents_md_path
        if custom_path and os.path.isfile(custom_path):
            with open(custom_path, encoding="utf-8") as f:
                return f.read()
        return load_agents_md()

    # ================================================================= #
    #  Scheduling                                                        #
    # ================================================================= #

    async def run_periodic(self) -> None:
        self._running = True
        logger.info(
            "[AgentEvolveServer] periodic mode: interval=%ds",
            self.config.interval_seconds,
        )
        while self._running:
            try:
                await self.run_once()
            except Exception as e:
                logger.error("[AgentEvolveServer] cycle error: %s", e, exc_info=True)
            await asyncio.sleep(self.config.interval_seconds)

    def stop(self) -> None:
        self._running = False

    # ================================================================= #
    #  HTTP trigger (optional FastAPI app)                               #
    # ================================================================= #

    def create_http_app(self):
        """Return a FastAPI app with ``/trigger``, ``/status``, ``/health``."""
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI(title="SkillClaw Agent Evolve Server")

        @app.post("/trigger")
        async def trigger_evolve():
            summary = await self.run_once()
            return JSONResponse(content=summary)

        @app.get("/status")
        async def status():
            entries = self._id_registry.all_entries()
            skill_summary = {
                name: {
                    "skill_id": e["skill_id"],
                    "version": e.get("version", 0),
                }
                for name, e in entries.items()
            }
            pending_keys = await self._call_storage(
                list_session_keys,
                self._bucket,
                self._prefix,
            )
            return JSONResponse(
                content={
                    "running": self._running,
                    "pending_sessions": len(pending_keys),
                    "registered_skills": len(entries),
                    "skills": skill_summary,
                    "fresh_mode": self.config.fresh,
                }
            )

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    # ================================================================= #
    #  Internal helpers                                                  #
    # ================================================================= #

    @staticmethod
    def _sanitise_name(raw_name: str) -> str:
        name = raw_name.strip().lower()
        if SLUG_RE.match(name):
            return name
        name = re.sub(r"[^a-z0-9_-]", "-", name).strip("-")
        if not name:
            name = "unnamed-skill"
        return name
