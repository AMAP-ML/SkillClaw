"""
OpenClaw subprocess wrapper for the agent engine under ``evolve_server``.

Manages the lifecycle of an OpenClaw agent instance:
- Configures ``openclaw.json`` (model provider, workspace, sandbox)
- Controls ``OPENCLAW_HOME`` isolation (fresh vs. persistent)
- Runs ``openclaw agent`` as a subprocess
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpenClawRunner:
    """Manage OpenClaw agent subprocess execution."""

    def __init__(
        self,
        openclaw_bin: str = "openclaw",
        openclaw_home: str | Path = "",
        *,
        fresh: bool = True,
        timeout: int = 600,
        llm_api_key: str = "",
        llm_base_url: str = "https://api.openai.com/v1",
        llm_model: str = "gpt-5.4",
        llm_api_type: str = "openai-completions",
    ) -> None:
        self.openclaw_bin = openclaw_bin
        self.openclaw_home = Path(openclaw_home) if openclaw_home else Path.cwd() / ".openclaw_home"
        self.fresh = fresh
        self.timeout = timeout
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_api_type = llm_api_type

    @property
    def _openclaw_dir(self) -> Path:
        return self.openclaw_home / ".openclaw"

    @property
    def _config_path(self) -> Path:
        return self._openclaw_dir / "openclaw.json"

    def _prepare_home(self) -> None:
        """Prepare OPENCLAW_HOME: wipe if fresh, otherwise keep existing state."""
        if self.fresh:
            if self.openclaw_home.exists():
                shutil.rmtree(self.openclaw_home, ignore_errors=True)
            logger.info("[OpenClawRunner] fresh mode: wiped %s", self.openclaw_home)

        self.openclaw_home.mkdir(parents=True, exist_ok=True)
        self._openclaw_dir.mkdir(parents=True, exist_ok=True)

    def _write_config(self, workspace_path: Path) -> None:
        """Write ``openclaw.json`` with model and workspace settings."""
        config: dict[str, Any] = {}
        if self._config_path.is_file():
            try:
                config = json.loads(self._config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                config = {}

        config.setdefault("gateway", {})
        config["gateway"]["mode"] = "local"
        config["gateway"]["bind"] = "loopback"
        config["gateway"].pop("remote", None)

        providers = config.setdefault("models", {}).setdefault("providers", {})
        providers["evolve-llm"] = {
            "api": self.llm_api_type,
            "baseUrl": self.llm_base_url,
            "apiKey": self.llm_api_key,
            "models": [
                {
                    "id": self.llm_model,
                    "name": self.llm_model,
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 200000,
                    "maxTokens": 16384,
                },
            ],
        }

        agents = config.setdefault("agents", {})
        defaults = agents.setdefault("defaults", {})
        defaults.setdefault("model", {})["primary"] = f"evolve-llm/{self.llm_model}"
        defaults.setdefault("sandbox", {})["mode"] = "off"
        defaults["workspace"] = str(workspace_path.resolve())

        self._config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Ensure agent session directories exist
        agents_dir = self._openclaw_dir / "agents" / "main" / "sessions"
        agents_dir.mkdir(parents=True, exist_ok=True)
        sessions_json = agents_dir / "sessions.json"
        if not sessions_json.exists():
            sessions_json.write_text("{}\n", encoding="utf-8")

    def _build_env(self) -> dict[str, str]:
        """Build the environment dict for the subprocess."""
        env = dict(os.environ)
        env["HOME"] = str(self.openclaw_home)
        env["OPENCLAW_HOME"] = str(self.openclaw_home)
        env["OPENCLAW_CONFIG_PATH"] = str(self._config_path)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    def run(
        self,
        workspace_path: Path,
        message: str,
        session_id: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run an OpenClaw agent with the given message.

        Parameters
        ----------
        workspace_path:
            Path to the prepared workspace directory.
        message:
            The instruction message for the agent.
        session_id:
            Optional session ID. If ``None``, a UUID is generated.
            When ``fresh=False``, reusing the same session_id across
            rounds lets the agent continue from previous context.

        Returns
        -------
        subprocess.CompletedProcess
            The completed process with stdout/stderr captured.
        """
        if session_id is None:
            session_id = f"evolve-{uuid.uuid4().hex[:12]}"

        self._prepare_home()
        self._write_config(workspace_path)

        cmd = [
            self.openclaw_bin,
            "agent",
            "--session-id",
            session_id,
            "--agent",
            "main",
            "--message",
            message,
            "--json",
            "--local",
            "--timeout",
            str(self.timeout),
        ]

        env = self._build_env()

        logger.info(
            "[OpenClawRunner] running: %s (fresh=%s, timeout=%ds, session=%s)",
            " ".join(cmd[:3] + ["..."]),
            self.fresh,
            self.timeout,
            session_id,
        )

        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace_path),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout + 30,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("[OpenClawRunner] agent timed out after %ds", self.timeout)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout=exc.stdout or "",
                stderr=f"TimeoutExpired: {exc}",
            )

        if result.returncode != 0:
            logger.warning(
                "[OpenClawRunner] agent exited with code %d\nstderr: %s",
                result.returncode,
                (result.stderr or "")[:500],
            )
        else:
            logger.info("[OpenClawRunner] agent completed successfully")

        return result
