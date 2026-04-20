# Adapted from MetaClaw
"""
SkillClaw service launcher.

Starts the proxy, skill injection, optional PRM side channel,
and client auto-configuration for the selected Claw agent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
from pathlib import Path
from typing import Optional

from .config_store import ConfigStore

logger = logging.getLogger(__name__)

_PID_FILE = Path.home() / ".skillclaw" / "skillclaw.pid"


class SkillClawLauncher:
    """Start/stop SkillClaw services based on ConfigStore."""

    def __init__(self, config_store: ConfigStore):
        self.cs = config_store
        self._api_server = None
        self._stop_event = threading.Event()
        self._validation_worker = None
        self._validation_task = None

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    async def start(self):
        cfg = self.cs.to_skillclaw_config()
        logger.info("[Launcher] Starting SkillClaw …")
        self._write_pid()
        self._setup_signal_handlers()
        await self._run(cfg)

    def stop(self):
        self._stop_event.set()
        if self._validation_worker is not None:
            try:
                self._validation_worker.stop()
            except Exception:
                pass
        if self._api_server is not None:
            try:
                self._api_server.stop()
            except Exception:
                pass
        _PID_FILE.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Core startup                                                         #
    # ------------------------------------------------------------------ #

    async def _run(self, cfg):
        from .api_server import SkillClawAPIServer
        from .prm_scorer import PRMScorer
        from .skill_manager import SkillManager

        skill_manager: Optional[SkillManager] = None
        if cfg.use_skills:
            Path(cfg.skills_dir).mkdir(parents=True, exist_ok=True)
            skill_manager = SkillManager(
                skills_dir=cfg.skills_dir,
                public_skill_root=cfg.skills_public_root,
                retrieval_mode=cfg.retrieval_mode,
                embedding_model_path=cfg.embedding_model_path,
            )
            logger.info("[Launcher] SkillManager loaded: %s skills", skill_manager.get_skill_count())

        prm_scorer = None
        prm_provider = cfg.prm_provider or "openai"
        prm_url = (cfg.prm_url or cfg.llm_api_base or "").strip()
        prm_model = (cfg.prm_model or cfg.llm_model_id or "").strip()
        prm_api_key = (cfg.prm_api_key or cfg.llm_api_key or "").strip()

        if cfg.use_prm and prm_provider == "bedrock" and prm_model:
            from .bedrock_client import BedrockChatClient

            prm_client = BedrockChatClient(
                model_id=prm_model,
                region=cfg.bedrock_region,
            )
            prm_scorer = PRMScorer(
                prm_url=prm_url or "https://api.openai.com/v1",
                prm_model=prm_model,
                api_key=prm_api_key,
                prm_m=cfg.prm_m,
                temperature=cfg.prm_temperature,
                max_new_tokens=cfg.prm_max_new_tokens,
                llm_client=prm_client,
            )
        elif cfg.use_prm and prm_provider == "bedrock":
            logger.warning("[Launcher] PRM enabled but bedrock prm_model is empty; PRM disabled")
        elif cfg.use_prm and prm_url and prm_model:
            prm_scorer = PRMScorer(
                prm_url=prm_url,
                prm_model=prm_model,
                api_key=prm_api_key,
                prm_m=cfg.prm_m,
                temperature=cfg.prm_temperature,
                max_new_tokens=cfg.prm_max_new_tokens,
                llm_client=None,
            )
        elif cfg.use_prm:
            logger.warning(
                "[Launcher] PRM enabled but endpoint/model missing "
                "(prm_url=%r prm_model=%r llm_api_base=%r llm_model_id=%r); PRM disabled",
                cfg.prm_url,
                cfg.prm_model,
                cfg.llm_api_base,
                cfg.llm_model_id,
            )

        # Auto-pull shared skills on startup
        if cfg.sharing_enabled and cfg.sharing_auto_pull_on_start:
            try:
                from .skill_hub import SkillHub

                hub = SkillHub.from_config(cfg)
                result = hub.pull_skills(cfg.skills_dir)
                logger.info(
                    "[Launcher] auto-pull: %d downloaded, %d unchanged, %d deleted",
                    result["downloaded"],
                    result["skipped"],
                    result.get("deleted", 0),
                )
                if skill_manager is not None and (
                    result.get("downloaded", 0) > 0
                    or result.get("deleted", 0) > 0
                    or result.get("restored_from_backup", False)
                ):
                    skill_manager.reload()
            except Exception as e:
                logger.warning("[Launcher] auto-pull failed: %s", e)

        server = SkillClawAPIServer(
            config=cfg,
            sampling_client=None,
            skill_manager=skill_manager,
            prm_scorer=prm_scorer,
        )
        server.start()
        self._api_server = server

        wait_until_ready = getattr(server, "wait_until_ready", None)
        if callable(wait_until_ready) and wait_until_ready(timeout_s=30.0):
            logger.info("[Launcher] proxy ready at http://%s:%d", cfg.proxy_host, cfg.proxy_port)
        elif callable(wait_until_ready):
            logger.warning(
                "[Launcher] proxy did not report ready within timeout on http://%s:%d",
                cfg.proxy_host,
                cfg.proxy_port,
            )
        else:
            logger.info("[Launcher] proxy server does not expose wait_until_ready(); skipping readiness wait")

        from .claw_adapter import configure_claw

        configure_claw(cfg)

        if getattr(cfg, "validation_enabled", False):
            try:
                from .validation_worker import ValidationWorker

                self._validation_worker = ValidationWorker(
                    cfg,
                    idle_provider=server,
                )
                self._validation_task = asyncio.create_task(self._validation_worker.run())
                logger.info("[Launcher] background validation worker started")
            except Exception as e:
                logger.warning("[Launcher] failed to start validation worker: %s", e)

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(1.0)
        finally:
            if self._validation_worker is not None:
                self._validation_worker.stop()
            if self._validation_task is not None:
                await asyncio.gather(self._validation_task, return_exceptions=True)
                self._validation_task = None
            self._validation_worker = None

    # ------------------------------------------------------------------ #
    # PID / signals                                                        #
    # ------------------------------------------------------------------ #

    def _write_pid(self):
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()))

    def _setup_signal_handlers(self):
        def _handler(signum, frame):
            logger.info("[Launcher] signal %s received — stopping …", signum)
            self.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (OSError, ValueError):
                pass
