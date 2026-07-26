import os

from evolve_server.core.config import EvolveServerConfig
from skillclaw.config import SkillClawConfig


def test_from_skillclaw_config_respects_evolve_llm_max_tokens(monkeypatch):
    monkeypatch.setenv("EVOLVE_LLM_MAX_TOKENS", "8192")
    cfg = SkillClawConfig(
        sharing_enabled=True,
        sharing_backend="local",
        sharing_local_root="/tmp/skillclaw-share",
        sharing_group_id="test-group",
        llm_api_key="test-key",
        llm_api_base="https://api.openai.com/v1",
        llm_model_id="gpt-4o",
    )

    evolve_cfg = EvolveServerConfig.from_skillclaw_config(cfg)

    assert evolve_cfg.llm_max_tokens == 8192
