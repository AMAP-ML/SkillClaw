from pathlib import Path

import pytest
import yaml

from skillclaw import claw_adapter as adapter
from skillclaw.config import SkillClawConfig
from skillclaw.config_store import ConfigStore
from skillclaw.hermes_config import import_hermes_upstream, resolve_hermes_config


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    a, b = tmp_path / "a" / "config.yaml", tmp_path / "b" / "config.yaml"
    for path, name in [(a, "model-a"), (b, "model-b")]:
        path.parent.mkdir()
        path.write_text(
            yaml.safe_dump(
                {
                    "model": {"base_url": "https://upstream.test/v1", "default": name, "api_key": "example-key"},
                    "unrelated": {"keep": True},
                }
            )
        )
    monkeypatch.setattr(adapter, "_HERMES_BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(adapter, "_LEGACY_SKILLCLAW_SKILLS_DIR", tmp_path / "legacy")
    return a, b


def config(path, **kwargs):
    return SkillClawConfig(
        llm_api_base="https://upstream.test/v1",
        llm_model_id="model-a",
        claw_type="hermes",
        hermes_config_path=str(path),
        skills_dir=str(path.parent / "skills"),
        **kwargs,
    )


def test_minimal_config_does_not_modify_hermes(profiles):
    a, _ = profiles
    before = a.read_bytes()
    adapter._configure_hermes(config(a))
    assert a.read_bytes() == before
    assert not (a.parent / "skills").exists()


def test_enabled_without_explicit_target_does_not_write(profiles, monkeypatch):
    a, _ = profiles
    before = a.read_bytes()
    monkeypatch.setenv("HERMES_CONFIG", str(a))
    adapter._configure_hermes(SkillClawConfig(configure_hermes=True))
    assert a.read_bytes() == before


def test_selected_profile_backup_restore_and_restart_idempotence(profiles):
    a, b = profiles
    original_a, original_b = a.read_bytes(), b.read_bytes()
    cfg = config(a, configure_hermes=True)
    adapter._configure_hermes(cfg)
    assert b.read_bytes() == original_b
    assert yaml.safe_load(a.read_text())["unrelated"] == {"keep": True}
    changed = a.stat().st_mtime_ns
    adapter._configure_hermes(cfg)
    assert a.stat().st_mtime_ns == changed
    adapter._configure_hermes(config(b, configure_hermes=True))
    backup_a = adapter._latest_hermes_backup_path(a)
    with pytest.raises(ValueError, match="different"):
        adapter.restore_hermes_config(backup_a, config_path=str(b))
    adapter.restore_hermes_config(config_path=str(a))
    assert a.read_bytes() == original_a
    assert b.read_bytes() != original_b


def test_legacy_global_backup_never_automatically_restored(profiles):
    a, _ = profiles
    adapter._HERMES_BACKUP_DIR.mkdir()
    (adapter._HERMES_BACKUP_DIR / "config.latest.yaml").write_text("wrong-profile: true")
    with pytest.raises(FileNotFoundError, match="profile-scoped"):
        adapter.restore_hermes_config(config_path=str(a))


def test_backup_failure_and_invalid_yaml_preserve_source(profiles, monkeypatch):
    a, _ = profiles
    before = a.read_bytes()
    monkeypatch.setattr(adapter, "_backup_hermes_config_if_changed", lambda *args: None)
    with pytest.raises(OSError, match="refusing"):
        adapter._configure_hermes(config(a, configure_hermes=True))
    assert a.read_bytes() == before
    a.write_text("model: [invalid")
    with pytest.raises(ValueError, match="Invalid Hermes YAML"):
        adapter._configure_hermes(config(a, configure_hermes=True))
    assert a.read_text() == "model: [invalid"


def test_persisted_path_overrides_environment_and_controls_skills(profiles, monkeypatch, tmp_path):
    a, b = profiles
    monkeypatch.setenv("HERMES_HOME", str(b.parent))
    assert resolve_hermes_config() == b
    monkeypatch.setenv("HERMES_CONFIG", str(a))
    assert resolve_hermes_config() == a
    path = tmp_path / "skillclaw.yaml"
    path.write_text(yaml.safe_dump({"claw_type": "hermes", "hermes": {"config_path": str(b), "configure": True}}))
    cfg = ConfigStore(path).to_skillclaw_config()
    assert Path(cfg.skills_dir) == b.parent / "skills"
    assert cfg.configure_hermes
    assert adapter.inspect_hermes_config(cfg)["config_path"] == str(b)


def test_literal_upstream_import(profiles):
    a, _ = profiles
    imported = import_hermes_upstream(a)
    assert imported == {
        "provider": "custom",
        "api_base": "https://upstream.test/v1",
        "model_id": "model-a",
        "api_key": "example-key",
        "api_mode": "chat",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_key", "${SECRET}"),
        ("api_key", "skillclaw"),
        ("base_url", "http://127.0.0.1:30000/v1"),
        ("default", ""),
        ("api_mode", "oauth"),
    ],
)
def test_import_refuses_ambiguous_or_proxy_values(profiles, field, value):
    a, _ = profiles
    data = yaml.safe_load(a.read_text())
    data["model"][field] = value
    a.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        import_hermes_upstream(a)
