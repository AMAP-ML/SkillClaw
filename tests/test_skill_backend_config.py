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
        sharing_nacos_publish_mode="direct",
        sharing_skill_reload_mode="callback",
        evolve_proxy_reload_url="http://proxy.test",
        proxy_api_key="proxy-secret",
        sharing_group_id="team-a",
    )

    evolve_config = EvolveServerConfig.from_skillclaw_config(cfg)

    assert evolve_config.skill_storage_backend == "nacos"
    assert evolve_config.nacos_server == "http://nacos.test"
    assert evolve_config.nacos_publish_mode == "direct"
    assert evolve_config.skill_reload_mode == "callback"
    assert evolve_config.proxy_reload_url == "http://proxy.test"
    assert evolve_config.proxy_reload_api_key == "proxy-secret"
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
    assert evolve_config.nacos_publish_mode == "review"
    assert evolve_config.skill_reload_mode == "poll"
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
                    "nacos_publish_mode": "direct",
                    "session_upload_interval": "3",
                    "skill_reload_mode": "callback",
                    "skill_reload_interval_seconds": "10",
                },
                "evolve": {
                    "server_url": "http://evolve.test",
                    "proxy_reload_url": "http://proxy.test",
                },
            }

    cfg = InlineConfigStore().to_skillclaw_config()

    assert cfg.sharing_backend == "oss"
    assert cfg.sharing_skill_backend == "nacos"
    assert cfg.sharing_nacos_server == "http://nacos.test"
    assert cfg.sharing_nacos_publish_mode == "direct"
    assert cfg.sharing_session_upload_interval == 3
    assert cfg.sharing_skill_reload_mode == "callback"
    assert cfg.sharing_skill_reload_interval_seconds == 10
    assert cfg.evolve_server_url == "http://evolve.test"
    assert cfg.evolve_proxy_reload_url == "http://proxy.test"


def test_config_store_normalizes_new_nacos_and_reload_options() -> None:
    class InlineConfigStore(ConfigStore):
        def load(self) -> dict:
            return {
                "sharing": {
                    "enabled": True,
                    "backend": "nacos",
                    "endpoint": "http://nacos.test",
                    "nacos_publish_mode": "bad-mode",
                    "session_upload_interval": "-2",
                    "skill_reload_mode": "bad-mode",
                    "skill_reload_interval_seconds": "1",
                }
            }

    cfg = InlineConfigStore().to_skillclaw_config()

    assert cfg.sharing_nacos_publish_mode == "review"
    assert cfg.sharing_session_upload_interval == 0
    assert cfg.sharing_skill_reload_mode == "poll"
    assert cfg.sharing_skill_reload_interval_seconds == 5
