from __future__ import annotations

from pathlib import Path

from skillclaw import claw_adapter
from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig
from skillclaw.config_store import ConfigStore


class FakeSkillManager:
    def __init__(self) -> None:
        self.injected = []

    def refresh_if_changed(self) -> None:
        return None

    def build_injection_prompt(self, max_chars: int = 30_000) -> str:
        return "<available_skills><skill><name>demo</name></skill></available_skills>"

    def get_all_skills(self) -> list[dict]:
        return [{"name": "demo"}]

    def record_injection(self, names: list[str]) -> None:
        self.injected.append(list(names))


def test_configure_codex_registers_profile_without_replacing_global_defaults(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    profile_config_path = tmp_path / ".codex" / "skillclaw.config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "gpt-5.5"\nmodel_provider = "openai"\n\n[profiles.default]\nmodel = "gpt-5.5"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(claw_adapter, "_CODEX_CONFIG_PATH", config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_PROFILE_CONFIG_PATH", profile_config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_SKILLS_DIR", tmp_path / ".codex" / "skills")
    monkeypatch.setattr(claw_adapter, "_CODEX_BACKUP_DIR", tmp_path / "backups")

    claw_adapter._configure_codex(
        SkillClawConfig(
            served_model_name="skillclaw-model",
            proxy_api_key="skillclaw-key",
            proxy_port=31000,
        )
    )

    text = config_path.read_text(encoding="utf-8")
    assert 'model = "gpt-5.5"' in text
    assert 'model_provider = "openai"' in text
    assert "[profiles.skillclaw]" not in text
    assert "[model_providers.skillclaw]" not in text
    profile_text = profile_config_path.read_text(encoding="utf-8")
    assert 'model = "skillclaw-model"' in profile_text
    assert 'model_provider = "skillclaw"' in profile_text
    assert "[model_providers.skillclaw]" in profile_text
    assert 'base_url = "http://127.0.0.1:31000/v1"' in profile_text
    assert 'wire_api = "responses"' in profile_text
    assert 'experimental_bearer_token = "skillclaw-key"' in profile_text
    assert (tmp_path / ".codex" / "skills").is_dir()


def test_configure_codex_removes_legacy_global_skillclaw_defaults(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    profile_config_path = tmp_path / ".codex" / "skillclaw.config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        (
            'model = "skillclaw-model"\n'
            'model_provider = "skillclaw"\n\n'
            "[model_providers.skillclaw]\n"
            'base_url = "http://127.0.0.1:30000/v1"\n\n'
            "[profiles.skillclaw]\n"
            'model = "skillclaw-model"\n'
            'model_provider = "skillclaw"\n\n'
            "[profiles.default]\n"
            'model = "gpt-5.5"\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(claw_adapter, "_CODEX_CONFIG_PATH", config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_PROFILE_CONFIG_PATH", profile_config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_SKILLS_DIR", tmp_path / ".codex" / "skills")
    monkeypatch.setattr(claw_adapter, "_CODEX_BACKUP_DIR", tmp_path / "backups")

    claw_adapter._configure_codex(SkillClawConfig(served_model_name="skillclaw-model"))

    top_level = config_path.read_text(encoding="utf-8").split("[", 1)[0]
    assert "model_provider" not in top_level
    assert "model =" not in top_level
    text = config_path.read_text(encoding="utf-8")
    assert "[profiles.skillclaw]" not in text
    assert "[model_providers.skillclaw]" not in text
    assert "[profiles.default]" in text
    assert "[model_providers.skillclaw]" in profile_config_path.read_text(encoding="utf-8")


def test_inspect_codex_config_reads_split_profile_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    profile_config_path = tmp_path / ".codex" / "skillclaw.config.toml"
    skills_dir = tmp_path / ".codex" / "skills"
    config_path.parent.mkdir(parents=True)
    skills_dir.mkdir()
    config_path.write_text('model = "gpt-5.5"\nmodel_provider = "openai"\n', encoding="utf-8")
    profile_config_path.write_text(
        (
            'model = "skillclaw-model"\n'
            'model_provider = "skillclaw"\n\n'
            "[model_providers.skillclaw]\n"
            'name = "SkillClaw"\n'
            'base_url = "http://127.0.0.1:31000/v1"\n'
            'wire_api = "responses"\n'
            'experimental_bearer_token = "skillclaw-key"\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(claw_adapter, "_CODEX_CONFIG_PATH", config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_PROFILE_CONFIG_PATH", profile_config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_SKILLS_DIR", skills_dir)
    monkeypatch.setattr(claw_adapter, "_CODEX_BACKUP_DIR", tmp_path / "backups")

    report = claw_adapter.inspect_codex_config(
        SkillClawConfig(
            served_model_name="skillclaw-model",
            proxy_api_key="skillclaw-key",
            proxy_port=31000,
            skills_dir=str(skills_dir),
        )
    )

    assert report["status"] == "ok"
    assert report["proxy_match"] is True
    assert report["configured_profile_model"] == "skillclaw-model"
    assert report["configured_base_url"] == "http://127.0.0.1:31000/v1"


def test_restore_codex_config_removes_split_profile_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    profile_config_path = tmp_path / ".codex" / "skillclaw.config.toml"
    backup_path = tmp_path / "backups" / "config.latest.toml"
    config_path.parent.mkdir(parents=True)
    backup_path.parent.mkdir(parents=True)
    config_path.write_text('model_provider = "skillclaw"\n', encoding="utf-8")
    profile_config_path.write_text('model_provider = "skillclaw"\n', encoding="utf-8")
    backup_path.write_text('model_provider = "openai"\n', encoding="utf-8")
    monkeypatch.setattr(claw_adapter, "_CODEX_CONFIG_PATH", config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_PROFILE_CONFIG_PATH", profile_config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_BACKUP_DIR", backup_path.parent)

    result = claw_adapter.restore_codex_config()

    assert config_path.read_text(encoding="utf-8") == 'model_provider = "openai"\n'
    assert not profile_config_path.exists()
    assert result["removed_profile"] == "True"


def test_codex_config_defaults_to_responses_mode_and_codex_skills(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.yaml")
    store.save(
        {
            "claw_type": "codex",
            "llm": {"provider": "openai", "api_base": "http://upstream.test/v1", "model_id": "upstream"},
            "proxy": {"served_model_name": "skillclaw-model"},
            "skills": {"enabled": True},
            "prm": {"enabled": False},
        }
    )

    cfg = store.to_skillclaw_config()

    assert cfg.llm_api_mode == "responses"
    assert cfg.skills_dir.endswith(".codex/skills")


def test_native_responses_body_injects_skills_without_dropping_codex_tools() -> None:
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(max_skills_prompt_chars=10_000)
    server.skill_manager = FakeSkillManager()
    custom_tool = {"type": "custom", "name": "apply_patch"}
    namespace_tool = {"type": "namespace", "name": "mcp__github__"}
    body = {
        "instructions": "base instructions",
        "input": "hi",
        "tools": [custom_tool, namespace_tool],
        "tool_choice": {"type": "custom", "name": "apply_patch"},
    }

    prepared = server._prepare_native_responses_body(body, turn_type="main")

    assert prepared is not body
    assert prepared["tools"] == [custom_tool, namespace_tool]
    assert prepared["tool_choice"] == {"type": "custom", "name": "apply_patch"}
    assert prepared["instructions"].startswith("base instructions")
    assert "<available_skills>" in prepared["instructions"]
    assert server.skill_manager.injected == [["demo"]]
