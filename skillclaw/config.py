# Adapted from MetaClaw
"""
Unified configuration for SkillClaw.
"""

from dataclasses import dataclass


@dataclass
class SkillClawConfig:
    # ------------------------------------------------------------------ #
    # Model                                                               #
    # ------------------------------------------------------------------ #
    model_name: str = "Qwen/Qwen3-4B"

    # ------------------------------------------------------------------ #
    # Reward / PRM                                                        #
    # ------------------------------------------------------------------ #
    use_prm: bool = True
    prm_provider: str = "openai"
    prm_url: str = "https://api.openai.com/v1"
    prm_model: str = "gpt-5.2"
    prm_api_key: str = ""
    prm_m: int = 3
    prm_temperature: float = 0.6
    prm_max_new_tokens: int = 1024

    # ------------------------------------------------------------------ #
    # Skills                                                              #
    # ------------------------------------------------------------------ #
    use_skills: bool = False
    skills_dir: str = "memory_data/skills"
    skills_public_root: str = ""
    retrieval_mode: str = "template"
    embedding_model_path: str = "Qwen/Qwen3-Embedding-0.6B"
    skill_top_k: int = 6
    max_skills_prompt_chars: int = 30000

    # ------------------------------------------------------------------ #
    # Context window                                                       #
    # ------------------------------------------------------------------ #
    max_context_tokens: int = 20000

    # ------------------------------------------------------------------ #
    # API Server                                                          #
    # ------------------------------------------------------------------ #
    proxy_port: int = 30000
    proxy_host: str = "0.0.0.0"
    served_model_name: str = "skillclaw-model"
    proxy_api_key: str = ""
    record_enabled: bool = True
    record_dir: str = "records/"

    # Which CLI agent to auto-configure on startup.
    claw_type: str = "openclaw"
    configure_openclaw: bool = True

    # ------------------------------------------------------------------ #
    # LLM forwarding                                                      #
    # ------------------------------------------------------------------ #
    llm_provider: str = "openai"
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model_id: str = ""

    # ------------------------------------------------------------------ #
    # OpenRouter-specific (ignored for other providers)                    #
    # ------------------------------------------------------------------ #
    openrouter_app_name: str = "SkillClaw"
    openrouter_app_url: str = ""
    openrouter_route: str = "fallback"
    openrouter_fallback_models: str = ""
    openrouter_data_policy: str = ""

    # ------------------------------------------------------------------ #
    # Skill sharing (generic object storage)                              #
    # ------------------------------------------------------------------ #
    sharing_enabled: bool = False
    sharing_backend: str = ""
    sharing_endpoint: str = ""
    sharing_bucket: str = ""
    sharing_access_key_id: str = ""
    sharing_secret_access_key: str = ""
    sharing_region: str = ""
    sharing_session_token: str = ""
    sharing_local_root: str = ""

    sharing_group_id: str = "default"
    sharing_user_alias: str = ""
    sharing_auto_pull_on_start: bool = False
    sharing_push_min_injections: int = 5
    sharing_push_min_effectiveness: float = 0.3

    # ------------------------------------------------------------------ #
    # Background validation                                               #
    # ------------------------------------------------------------------ #
    validation_enabled: bool = False
    validation_mode: str = "replay"
    validation_idle_after_seconds: int = 300
    validation_poll_interval_seconds: int = 60
    validation_max_jobs_per_day: int = 5
    validation_max_concurrency: int = 1

    # ------------------------------------------------------------------ #
    # Cloud / Bedrock                                                      #
    # ------------------------------------------------------------------ #
    bedrock_region: str = "us-east-1"
