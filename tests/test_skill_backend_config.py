from __future__ import annotations

from evolve_server.core.config import EvolveServerConfig
from skillclaw.config import SkillClawConfig
from skillclaw.config_store import ConfigStore


def test_skill_backend_overrides_skill_storage_without_changing_session_storage(monkeypatch) -> None:
    monkeypatch.delenv("EVOLVE_STORAGE_BACKEND", raising=False)
    cfg = SkillClawConfig(
        sharing_backend="oss",
        sharing_skill_backend="nacos",
        sharing_endpoint="https://oss-cn-hangzhou.aliyuncs.com",
        sharing_bucket="skillclaw-sessions",
        sharing_access_key_id="ak",
        sharing_secret_access_key="sk",
        sharing_nacos_server="http://nacos.test",
        sharing_group_id="team-a",
    )

    evolve_config = EvolveServerConfig.from_skillclaw_config(cfg)

    assert evolve_config.skill_storage_backend == "nacos"
    assert evolve_config.nacos_server == "http://nacos.test"
    assert evolve_config.storage_backend == "oss"
    assert evolve_config.storage_endpoint == "https://oss-cn-hangzhou.aliyuncs.com"
    assert evolve_config.storage_bucket == "skillclaw-sessions"


def test_skill_backend_empty_keeps_legacy_nacos_backend_behavior(monkeypatch) -> None:
    monkeypatch.delenv("EVOLVE_STORAGE_BACKEND", raising=False)
    cfg = SkillClawConfig(
        sharing_backend="nacos",
        sharing_endpoint="http://legacy-nacos.test",
        sharing_group_id="team-a",
    )

    evolve_config = EvolveServerConfig.from_skillclaw_config(cfg)

    assert evolve_config.skill_storage_backend == "nacos"
    assert evolve_config.nacos_server == "http://legacy-nacos.test"
    assert evolve_config.storage_backend == ""
    assert evolve_config.storage_endpoint == ""


def test_config_store_reads_skill_backend() -> None:
    class InlineConfigStore(ConfigStore):
        def load(self) -> dict:
            return {
                "sharing": {
                    "enabled": True,
                    "backend": "oss",
                    "skill_backend": "nacos",
                    "endpoint": "https://oss-cn-hangzhou.aliyuncs.com",
                    "bucket": "skillclaw-sessions",
                    "nacos_server": "http://nacos.test",
                }
            }

    cfg = InlineConfigStore().to_skillclaw_config()

    assert cfg.sharing_backend == "oss"
    assert cfg.sharing_skill_backend == "nacos"
    assert cfg.sharing_nacos_server == "http://nacos.test"
