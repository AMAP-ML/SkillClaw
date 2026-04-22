# Adapted from MetaClaw
"""
User-facing configuration store for SkillClaw.

Reads/writes ~/.skillclaw/config.yaml and bridges to SkillClawConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import SkillClawConfig

CONFIG_DIR = Path.home() / ".skillclaw"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
_DEFAULT_SKILLS_DIR = CONFIG_DIR / "skills"
_DEFAULT_HERMES_SKILLS_DIR = Path.home() / ".hermes" / "skills"
_DEFAULT_CODEX_SKILLS_DIR = Path.home() / ".codex" / "skills"
_DEFAULT_CLAUDE_SKILLS_DIR = Path.home() / ".claude" / "skills"

_DEFAULTS: dict = {
    "llm": {
        "provider": "custom",
        "model_id": "",
        "api_base": "",
        "api_key": "",
    },
    "proxy": {
        "port": 30000,
        "host": "0.0.0.0",
        "api_key": "",
        "served_model_name": "skillclaw-model",
    },
    "claw_type": "openclaw",
    "configure_openclaw": True,
    "skills": {
        "enabled": True,
        "dir": str(_DEFAULT_SKILLS_DIR),
        "retrieval_mode": "template",
        "top_k": 6,
    },
    "openrouter": {
        "app_name": "SkillClaw",
        "app_url": "",
        "route": "fallback",
        "fallback_models": "",
        "data_policy": "",
    },
    "prm": {
        "enabled": True,
        "provider": "openai",
        "url": "",
        "model": "",
        "api_key": "",
    },
    "sharing": {
        "enabled": False,
        "backend": "",
        "endpoint": "",
        "bucket": "",
        "access_key_id": "",
        "secret_access_key": "",
        "region": "",
        "session_token": "",
        "local_root": "",
        "group_id": "default",
        "user_alias": "",
        "auto_pull_on_start": False,
        "push_min_injections": 5,
        "push_min_effectiveness": 0.3,
    },
    "validation": {
        "enabled": False,
        "mode": "replay",
        "idle_after_seconds": 300,
        "poll_interval_seconds": 60,
        "max_jobs_per_day": 5,
        "max_concurrency": 1,
    },
    "dashboard": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 3788,
        "db_path": str(CONFIG_DIR / "dashboard.db"),
        "sync_on_start": True,
        "include_shared": True,
        "evolve_server_url": "",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _coerce(value: Any) -> Any:
    """Auto-coerce string values to bool/int/float where obvious."""
    if not isinstance(value, str):
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _first_non_empty(mapping: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _infer_sharing_backend(sharing: dict[str, Any]) -> str:
    backend = str(sharing.get("backend", "") or "").strip().lower()
    if backend:
        return backend
    if sharing.get("local_root"):
        return "local"
    if any(
        sharing.get(key)
        for key in ("endpoint", "bucket", "access_key_id", "secret_access_key", "region", "session_token")
    ):
        return "s3"
    return ""


def _normalize_validation_mode(value: Any) -> str:
    del value
    return "replay"


def default_skills_dir_for_claw(claw_type: str) -> Path:
    """Return the default local skills directory for the selected agent."""
    normalized = str(claw_type or "").strip().lower()
    if normalized == "hermes":
        return _DEFAULT_HERMES_SKILLS_DIR
    if normalized == "codex":
        return _DEFAULT_CODEX_SKILLS_DIR
    if normalized == "claude":
        return _DEFAULT_CLAUDE_SKILLS_DIR
    return _DEFAULT_SKILLS_DIR


def resolve_skills_dir(skills_dir: Any, *, claw_type: str) -> str:
    """Normalize a configured skills dir, applying agent-native defaults.

    Existing configs that still point at the old generic default are treated as
    "unset" when an agent with a native skill directory is selected so
    SkillClaw follows that agent's own skill library by default.
    """
    raw = str(skills_dir or "").strip()
    generic_default = _DEFAULT_SKILLS_DIR.expanduser()
    normalized_claw = str(claw_type or "").strip().lower()

    if raw:
        expanded = Path(raw).expanduser()
        if normalized_claw in {"hermes", "codex", "claude"} and expanded == generic_default:
            return str(default_skills_dir_for_claw(normalized_claw))
        return str(expanded)

    return str(default_skills_dir_for_claw(claw_type))


def _default_served_model_name(llm_model_id: str) -> str:
    """Return a safe proxy-facing model name for agent integrations.

    GPT-5-style names trigger Hermes' Responses API mode. SkillClaw does not
    want to expose the upstream model name directly because the proxy surface
    may not match the upstream protocol. Use a stable proxy model name instead.
    """
    raw = str(llm_model_id or "").strip()
    if not raw:
        return "skillclaw-model"

    normalized = raw.rsplit("/", 1)[-1].lower()
    if normalized.startswith("gpt-5"):
        return "skillclaw-model"
    return raw


class ConfigStore:
    """Read/write ~/.skillclaw/config.yaml."""

    def __init__(self, config_file: Path = CONFIG_FILE):
        self.config_file = config_file

    def exists(self) -> bool:
        return self.config_file.exists()

    def load(self) -> dict:
        if not self.config_file.exists():
            return _deep_merge({}, _DEFAULTS)
        try:
            import yaml

            with open(self.config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return _deep_merge(_DEFAULTS, data)
        except Exception:
            return _deep_merge({}, _DEFAULTS)

    def save(self, data: dict):
        import yaml

        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def get(self, dotpath: str) -> Any:
        data = self.load()
        for k in dotpath.split("."):
            if not isinstance(data, dict):
                return None
            data = data.get(k)
        return data

    def set(self, dotpath: str, value: Any):
        data = self.load()
        keys = dotpath.split(".")
        d = data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = _coerce(value)
        self.save(data)

    # ------------------------------------------------------------------ #
    # Bridge to SkillClawConfig                                            #
    # ------------------------------------------------------------------ #

    def to_skillclaw_config(self) -> SkillClawConfig:
        data = self.load()
        llm = data.get("llm", {})
        llm_provider = llm.get("provider", "openai")
        llm_api_base = llm.get("api_base", "")
        llm_api_key = llm.get("api_key", "")
        llm_model_id = llm.get("model_id", "")
        proxy = data.get("proxy", {})
        skills = data.get("skills", {})
        orouter = data.get("openrouter", {})
        prm = data.get("prm", {})
        configure_openclaw = bool(data.get("configure_openclaw", True))
        raw_claw_type = str(data.get("claw_type", "openclaw") or "openclaw")
        if not configure_openclaw:
            raw_claw_type = "none"

        sharing = data.get("sharing", {})
        validation = data.get("validation", {})
        dashboard = data.get("dashboard", {})
        sharing_backend = _infer_sharing_backend(sharing)
        sharing_endpoint = _first_non_empty(sharing, "endpoint")
        sharing_bucket = _first_non_empty(sharing, "bucket")
        sharing_access_key_id = _first_non_empty(sharing, "access_key_id")
        sharing_secret_access_key = _first_non_empty(sharing, "secret_access_key")
        sharing_region = _first_non_empty(sharing, "region")
        sharing_session_token = _first_non_empty(sharing, "session_token")
        sharing_local_root = _first_non_empty(sharing, "local_root")

        prm_provider = prm.get("provider", "openai")
        prm_url = str(prm.get("url", "") or llm_api_base)
        prm_model = str(prm.get("model", "") or llm_model_id or "gpt-5.2")
        prm_api_key = str(prm.get("api_key", "") or llm_api_key)

        skills_dir = resolve_skills_dir(
            skills.get("dir", str(_DEFAULT_SKILLS_DIR)),
            claw_type=raw_claw_type,
        )

        return SkillClawConfig(
            # LLM forwarding
            llm_provider=llm_provider,
            llm_api_base=llm_api_base,
            llm_api_key=llm_api_key,
            llm_model_id=llm_model_id,
            bedrock_region=llm.get("bedrock_region") or data.get("bedrock_region", "us-east-1"),
            # OpenRouter
            openrouter_app_name=orouter.get("app_name", "SkillClaw"),
            openrouter_app_url=orouter.get("app_url", ""),
            openrouter_route=orouter.get("route", "fallback"),
            openrouter_fallback_models=orouter.get("fallback_models", ""),
            openrouter_data_policy=orouter.get("data_policy", ""),
            # Proxy
            proxy_port=proxy.get("port", 30000),
            proxy_host=proxy.get("host", "0.0.0.0"),
            proxy_api_key=str(proxy.get("api_key", "") or ""),
            served_model_name=(
                _first_non_empty(proxy, "served_model_name") or _default_served_model_name(llm_model_id)
            ),
            # Skills
            use_skills=bool(skills.get("enabled", True)),
            skills_dir=skills_dir,
            skills_public_root=str(skills.get("public_root", "") or ""),
            retrieval_mode=skills.get("retrieval_mode", "template"),
            skill_top_k=int(skills.get("top_k", 6)),
            max_context_tokens=int(data.get("max_context_tokens", 20000) or 20000),
            # PRM
            use_prm=bool(prm.get("enabled", True)),
            prm_provider=prm_provider,
            prm_url=prm_url,
            prm_model=prm_model,
            prm_api_key=prm_api_key,
            # Model
            model_name=llm.get("model_id") or "Qwen/Qwen3-4B",
            # Claw
            claw_type=raw_claw_type,
            configure_openclaw=configure_openclaw,
            # Sharing
            sharing_enabled=bool(sharing.get("enabled", False)),
            sharing_backend=sharing_backend,
            sharing_endpoint=sharing_endpoint,
            sharing_bucket=sharing_bucket,
            sharing_access_key_id=sharing_access_key_id,
            sharing_secret_access_key=sharing_secret_access_key,
            sharing_region=sharing_region,
            sharing_session_token=sharing_session_token,
            sharing_local_root=sharing_local_root,
            sharing_group_id=str(sharing.get("group_id", "default") or "default"),
            sharing_user_alias=str(sharing.get("user_alias", "") or ""),
            sharing_auto_pull_on_start=bool(sharing.get("auto_pull_on_start", False)),
            sharing_push_min_injections=int(sharing.get("push_min_injections", 5)),
            sharing_push_min_effectiveness=float(sharing.get("push_min_effectiveness", 0.3)),
            validation_enabled=bool(validation.get("enabled", False)),
            validation_mode=_normalize_validation_mode(validation.get("mode", "replay")),
            validation_idle_after_seconds=int(validation.get("idle_after_seconds", 300)),
            validation_poll_interval_seconds=int(validation.get("poll_interval_seconds", 60)),
            validation_max_jobs_per_day=int(validation.get("max_jobs_per_day", 5)),
            validation_max_concurrency=max(1, int(validation.get("max_concurrency", 1))),
            dashboard_enabled=bool(dashboard.get("enabled", False)),
            dashboard_host=str(dashboard.get("host", "127.0.0.1") or "127.0.0.1"),
            dashboard_port=int(dashboard.get("port", 3788) or 3788),
            dashboard_db_path=str(
                dashboard.get("db_path", str(CONFIG_DIR / "dashboard.db"))
                or str(CONFIG_DIR / "dashboard.db")
            ),
            dashboard_sync_on_start=bool(dashboard.get("sync_on_start", True)),
            dashboard_include_shared=bool(dashboard.get("include_shared", True)),
            dashboard_evolve_server_url=str(dashboard.get("evolve_server_url", "") or ""),
        )

    def describe(self) -> str:
        """Return a human-readable summary of the current config."""
        data = self.load()
        llm = data.get("llm", {})
        skills = data.get("skills", {})
        prm = data.get("prm", {})
        dashboard = data.get("dashboard", {})
        claw_type = str(data.get("claw_type", "openclaw") or "openclaw")
        effective_skills_dir = resolve_skills_dir(
            skills.get("dir", str(_DEFAULT_SKILLS_DIR)),
            claw_type=claw_type,
        )
        lines = [
            f"claw_type:       {claw_type}",
            f"llm.provider:    {llm.get('provider', '?')}",
            f"llm.model_id:    {llm.get('model_id', '?')}",
            f"llm.api_base:    {llm.get('api_base', '—') if llm.get('provider') != 'bedrock' else '(n/a)'}",
            *(
                [f"llm.bedrock_region: {llm.get('bedrock_region', 'us-east-1')}"]
                if llm.get("provider") == "bedrock"
                else []
            ),
            *(
                [
                    f"openrouter.route:    {data.get('openrouter', {}).get('route', 'fallback')}",
                    f"openrouter.fallback: {data.get('openrouter', {}).get('fallback_models', '') or '(none)'}",
                    f"openrouter.data:     {data.get('openrouter', {}).get('data_policy', '') or 'allow'}",
                ]
                if llm.get("provider") == "openrouter"
                else []
            ),
            f"proxy.port:      {data.get('proxy', {}).get('port', 30000)}",
            f"skills.enabled:  {skills.get('enabled', True)}",
            f"skills.dir:      {effective_skills_dir}",
            f"prm.enabled:     {prm.get('enabled', False)}",
        ]
        sharing = data.get("sharing", {})
        validation = data.get("validation", {})
        if sharing.get("enabled"):
            backend = _infer_sharing_backend(sharing) or "unknown"
            lines += [
                "sharing.enabled: True",
                f"sharing.backend: {backend}",
            ]
            if backend == "local":
                lines += [
                    f"sharing.local_root: {sharing.get('local_root', '?')}",
                ]
            else:
                lines += [
                    f"sharing.bucket:  {_first_non_empty(sharing, 'bucket', default='?')}",
                    f"sharing.endpoint: {_first_non_empty(sharing, 'endpoint', default='(default)')}",
                ]
            lines += [
                f"sharing.group:   {sharing.get('group_id', 'default')}",
                f"sharing.alias:   {sharing.get('user_alias', '?')}",
                f"sharing.auto_pull: {sharing.get('auto_pull_on_start', False)}",
            ]
        else:
            lines.append("sharing.enabled: False")
        lines += [
            f"validation.enabled: {validation.get('enabled', False)}",
            f"validation.mode: {_normalize_validation_mode(validation.get('mode', 'replay'))}",
            f"validation.idle_after: {validation.get('idle_after_seconds', 300)}",
            f"validation.poll_interval: {validation.get('poll_interval_seconds', 60)}",
            f"dashboard.enabled: {dashboard.get('enabled', False)}",
            f"dashboard.host: {dashboard.get('host', '127.0.0.1')}",
            f"dashboard.port: {dashboard.get('port', 3788)}",
            f"dashboard.db_path: {dashboard.get('db_path', str(CONFIG_DIR / 'dashboard.db'))}",
            f"dashboard.include_shared: {dashboard.get('include_shared', True)}",
            f"dashboard.evolve_server_url: {dashboard.get('evolve_server_url', '') or '(not set)'}",
        ]
        return "\n".join(lines)
