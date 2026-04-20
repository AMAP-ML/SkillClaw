# Adapted from MetaClaw
"""
Interactive first-time setup wizard for SkillClaw.
"""

from __future__ import annotations

from pathlib import Path

from .claw_adapter import CLAW_TYPES
from .config_store import CONFIG_DIR, ConfigStore, resolve_skills_dir

_PROVIDER_PRESETS = {
    "kimi": {
        "api_base": "https://api.moonshot.cn/v1",
        "model_id": "moonshotai/Kimi-K2.5",
    },
    "qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_id": "qwen-plus",
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "model_id": "gpt-4o",
    },
    "minimax": {
        "api_base": "https://api.minimax.io/v1",
        "model_id": "MiniMax-M2.7",
    },
    "novita": {
        "api_base": "https://api.novita.ai/openai",
        "model_id": "moonshotai/kimi-k2.5",
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "model_id": "google/gemini-2.5-pro",
    },
    "bedrock": {
        "api_base": "",
        "model_id": "us.anthropic.claude-sonnet-4-6",
    },
    "custom": {
        "api_base": "",
        "model_id": "",
    },
}


def _prompt(msg: str, default: str = "", hide: bool = False) -> str:
    import getpass

    if default:
        display_default = "***" if hide else default
        full_msg = f"{msg} [{display_default}]: "
    else:
        full_msg = f"{msg}: "
    try:
        if hide:
            val = getpass.getpass(full_msg)
        else:
            val = input(full_msg)
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return val.strip() or default


def _prompt_bool(msg: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"
    val = _prompt(f"{msg} ({default_str})")
    if not val:
        return default
    return val.lower() in {"y", "yes", "true", "1"}


def _prompt_int(msg: str, default: int = 0) -> int:
    while True:
        val = _prompt(msg, str(default))
        try:
            return int(val)
        except ValueError:
            print(f"  Please enter an integer (got: {val!r})")


def _prompt_choice(msg: str, choices: list[str], default: str = "") -> str:
    choices_str = "/".join(f"[{c}]" if c == default else c for c in choices)
    while True:
        val = _prompt(f"{msg} ({choices_str})", default)
        if val in choices:
            return val
        print(f"  Invalid choice. Pick one of: {choices}")


def _infer_existing_sharing_backend(current_sharing: dict) -> str:
    backend = str(current_sharing.get("backend", "") or "").strip().lower()
    if backend:
        return backend
    if current_sharing.get("local_root"):
        return "local"
    if any(current_sharing.get(key) for key in ("endpoint", "bucket", "access_key_id", "secret_access_key")):
        return "s3"
    return "s3"


class SetupWizard:
    """Interactive configuration wizard."""

    def run(self):
        print("\n" + "=" * 60)
        print("  SkillClaw Setup")
        print("=" * 60)
        print("\nThis wizard will create ~/.skillclaw/config.yaml")
        print("You can re-run 'skillclaw setup' at any time to reconfigure.\n")

        cs = ConfigStore()
        existing = cs.load() if cs.exists() else {}

        # ---- CLI agent (claw type) ----
        print("\n--- CLI Agent ---")
        print("SkillClaw will auto-configure the chosen agent to route its LLM\ncalls through the SkillClaw proxy.")
        current_claw = existing.get("claw_type", "openclaw")
        claw_type = _prompt_choice(
            "CLI agent to configure",
            CLAW_TYPES,
            default=current_claw,
        )

        # ---- LLM provider ----
        print("\n--- LLM Configuration ---")
        current_llm = existing.get("llm", {})
        current_provider = current_llm.get("provider", "custom")
        provider = _prompt_choice(
            "LLM provider",
            ["kimi", "qwen", "openai", "minimax", "novita", "openrouter", "bedrock", "custom"],
            default=current_provider,
        )
        preset = _PROVIDER_PRESETS[provider]
        openrouter_config: dict = existing.get("openrouter", {})
        if provider == "bedrock":
            api_base = ""
            api_key = ""
            model_id = _prompt(
                "Bedrock model ID (inference profile)",
                default=current_llm.get("model_id") or preset["model_id"],
            )
            bedrock_region = _prompt(
                "AWS region",
                default=current_llm.get("bedrock_region", "us-east-1"),
            )
        else:
            bedrock_region = ""
            api_base = _prompt(
                "API base URL [http://api.example.com/v1]",
                default=current_llm.get("api_base") or preset["api_base"],
            )
            model_id = _prompt(
                "Model ID",
                default=current_llm.get("model_id") or preset["model_id"],
            )
            api_key = _prompt(
                "API key",
                default=current_llm.get("api_key", ""),
                hide=True,
            )

        # OpenRouter-specific options
        if provider == "openrouter":
            print("\n--- OpenRouter Options ---")
            or_route = _prompt_choice(
                "Routing strategy",
                ["fallback", "price", "throughput", "latency"],
                default=openrouter_config.get("route", "fallback"),
            )
            or_fallback = _prompt(
                "Fallback models (comma-separated, optional)",
                default=openrouter_config.get("fallback_models", ""),
            )
            or_data_policy = _prompt_choice(
                "Data collection policy",
                ["allow", "deny"],
                default="deny" if openrouter_config.get("data_policy") == "deny" else "allow",
            )
            openrouter_config = {
                "app_name": "SkillClaw",
                "route": or_route,
                "fallback_models": or_fallback,
                "data_policy": "" if or_data_policy == "allow" else "deny",
            }
        else:
            openrouter_config = {}

        # ---- Skills ----
        print("\n--- Skills Configuration ---")
        current_skills = existing.get("skills", {})
        default_skills_dir = resolve_skills_dir(
            current_skills.get("dir", str(CONFIG_DIR / "skills")),
            claw_type=claw_type,
        )
        if claw_type == "hermes":
            print(
                "Hermes shares its local skill library with SkillClaw by default.\n"
                f"Recommended directory: {default_skills_dir}"
            )
        elif claw_type == "codex":
            print(f"Codex reads native skills from ~/.codex/skills.\nRecommended directory: {default_skills_dir}")
        elif claw_type == "claude":
            print(
                f"Claude Code reads native skills from ~/.claude/skills.\nRecommended directory: {default_skills_dir}"
            )
        skills_enabled = _prompt_bool("Enable skill injection", default=current_skills.get("enabled", True))
        skills_dir = _prompt(
            "Skills directory",
            default=default_skills_dir,
        )
        skills_dir = str(Path(skills_dir).expanduser())

        # ---- PRM ----
        print("\n--- PRM (Quality Scoring) ---")
        current_prm = existing.get("prm", {})
        prm_enabled = _prompt_bool(
            "Enable PRM response quality scoring",
            default=current_prm.get("enabled", True),
        )
        prm_config: dict = {"enabled": False}
        if prm_enabled:
            default_prm_url = current_prm.get("url") or api_base or "https://api.openai.com/v1"
            default_prm_model = current_prm.get("model") or model_id or "gpt-5.2"
            default_prm_api_key = current_prm.get("api_key") or api_key
            prm_url = _prompt(
                "PRM API URL",
                default=default_prm_url,
            )
            prm_model = _prompt(
                "PRM model ID",
                default=default_prm_model,
            )
            prm_api_key = _prompt(
                "PRM API key",
                default=default_prm_api_key,
                hide=True,
            )
            prm_config = {
                "enabled": True,
                "provider": current_prm.get("provider", "openai"),
                "url": prm_url,
                "model": prm_model,
                "api_key": prm_api_key,
            }

        # ---- Sharing ----
        print("\n--- Skill Sharing (Optional) ---")
        current_sharing = existing.get("sharing", {})
        sharing_enabled = _prompt_bool(
            "Enable shared skill storage",
            default=current_sharing.get("enabled", False),
        )
        sharing_config: dict = {"enabled": False}
        if sharing_enabled:
            sharing_backend = _prompt_choice(
                "Storage backend",
                ["local", "s3", "oss"],
                default=_infer_existing_sharing_backend(current_sharing),
            )
            group_id = _prompt(
                "Group ID (namespace for shared skills)",
                default=current_sharing.get("group_id", "default"),
            )
            user_alias = _prompt(
                "Your alias (shown as skill author)",
                default=current_sharing.get("user_alias", ""),
            )
            auto_pull = _prompt_bool(
                "Auto-pull shared skills on startup",
                default=current_sharing.get("auto_pull_on_start", False),
            )
            sharing_config = {
                "enabled": True,
                "backend": sharing_backend,
                "group_id": group_id,
                "user_alias": user_alias,
                "auto_pull_on_start": auto_pull,
            }
            if sharing_backend == "local":
                local_root = _prompt(
                    "Local shared storage root",
                    default=current_sharing.get("local_root", ""),
                )
                sharing_config["local_root"] = local_root
            else:
                endpoint = _prompt(
                    "Storage endpoint",
                    default=current_sharing.get("endpoint", ""),
                )
                bucket = _prompt(
                    "Bucket name",
                    default=current_sharing.get("bucket", ""),
                )
                access_key_id = _prompt(
                    "Access key ID",
                    default=current_sharing.get("access_key_id", ""),
                    hide=True,
                )
                secret_access_key = _prompt(
                    "Secret access key",
                    default=current_sharing.get("secret_access_key", ""),
                    hide=True,
                )
                sharing_config.update(
                    {
                        "endpoint": endpoint,
                        "bucket": bucket,
                        "access_key_id": access_key_id,
                        "secret_access_key": secret_access_key,
                    }
                )
                if sharing_backend == "s3":
                    region = _prompt(
                        "Region (optional)",
                        default=current_sharing.get("region", ""),
                    )
                    session_token = _prompt(
                        "Session token (optional)",
                        default=current_sharing.get("session_token", ""),
                        hide=True,
                    )
                    if region:
                        sharing_config["region"] = region
                    if session_token:
                        sharing_config["session_token"] = session_token

        # ---- Proxy port ----
        print("\n--- Proxy Configuration ---")
        current_proxy = existing.get("proxy", {})
        default_served_model_name = str(current_proxy.get("served_model_name") or "skillclaw-model")
        served_model_name = _prompt(
            "Proxy model name exposed to agents",
            default=default_served_model_name,
        )
        proxy_port = _prompt_int("Proxy port", default=current_proxy.get("port", 30000))

        # ---- Write config ----
        proxy_config = dict(current_proxy)
        proxy_config["port"] = proxy_port
        proxy_config.setdefault("host", "0.0.0.0")
        proxy_config["served_model_name"] = served_model_name or "skillclaw-model"
        data = {
            "claw_type": claw_type,
            "llm": {
                "provider": provider,
                "model_id": model_id,
                "api_base": api_base,
                "api_key": api_key,
                "bedrock_region": bedrock_region,
            },
            "openrouter": openrouter_config,
            "proxy": proxy_config,
            "skills": {
                "enabled": skills_enabled,
                "dir": skills_dir,
                "retrieval_mode": current_skills.get("retrieval_mode", "template"),
                "top_k": current_skills.get("top_k", 6),
            },
            "prm": prm_config,
            "sharing": sharing_config,
        }

        cs.save(data)
        Path(skills_dir).expanduser().mkdir(parents=True, exist_ok=True)
        if sharing_config.get("enabled") and sharing_config.get("backend") == "local":
            local_root = sharing_config.get("local_root")
            if local_root:
                Path(str(local_root)).expanduser().mkdir(parents=True, exist_ok=True)

        print(f"\nConfig saved to: {cs.config_file}")
        print("\nRun 'skillclaw start' to launch SkillClaw.")
        print("=" * 60 + "\n")
