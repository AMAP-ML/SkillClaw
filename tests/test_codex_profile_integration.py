from __future__ import annotations

from pathlib import Path

from skillclaw import claw_adapter
from skillclaw.config import SkillClawConfig


def test_configure_codex_registers_profile_without_replacing_global_defaults(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "gpt-5.5"\nmodel_provider = "openai"\n\n[profiles.default]\nmodel = "gpt-5.5"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(claw_adapter, "_CODEX_CONFIG_PATH", config_path)
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
    assert "[model_providers.skillclaw]" in text
    assert 'base_url = "http://127.0.0.1:31000/v1"' in text
    assert 'wire_api = "responses"' in text
    assert 'experimental_bearer_token = "skillclaw-key"' in text
    assert "[profiles.skillclaw]" in text
    assert 'model = "skillclaw-model"' in text
    assert 'model_provider = "skillclaw"' in text
    assert (tmp_path / ".codex" / "skills").is_dir()


def test_configure_codex_removes_legacy_global_skillclaw_defaults(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "skillclaw-model"\nmodel_provider = "skillclaw"\n\n[profiles.default]\nmodel = "gpt-5.5"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(claw_adapter, "_CODEX_CONFIG_PATH", config_path)
    monkeypatch.setattr(claw_adapter, "_CODEX_SKILLS_DIR", tmp_path / ".codex" / "skills")
    monkeypatch.setattr(claw_adapter, "_CODEX_BACKUP_DIR", tmp_path / "backups")

    claw_adapter._configure_codex(SkillClawConfig(served_model_name="skillclaw-model"))

    top_level = config_path.read_text(encoding="utf-8").split("[", 1)[0]
    assert "model_provider" not in top_level
    assert "model =" not in top_level
    assert "[profiles.skillclaw]" in config_path.read_text(encoding="utf-8")
