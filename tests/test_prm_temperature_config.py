"""``prm.temperature`` in config.yaml must reach ``SkillClawConfig.prm_temperature``.

Regression coverage for the config bridge: the field existed but was never
wired from YAML, and the first wiring attempt silently rewrote an explicit
``temperature: 0`` to the default via an ``or``-guard. These cases pin the
correct behavior: configured values pass through (including falsy-but-valid
0), and the default only applies when the key is absent.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from skillclaw.config_store import ConfigStore


def _store(tmp_path: Path, prm: dict) -> ConfigStore:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "llm": {"provider": "openai", "model_id": "k3", "api_base": "https://example.invalid/v1"},
                "prm": prm,
            }
        )
    )
    return ConfigStore(cfg)


def test_prm_temperature_from_config(tmp_path):
    store = _store(tmp_path, {"enabled": True, "model": "k3", "temperature": 1})
    assert store.to_skillclaw_config().prm_temperature == 1


def test_prm_temperature_fractional_value(tmp_path):
    store = _store(tmp_path, {"enabled": True, "model": "gpt-5.2", "temperature": 0.2})
    assert store.to_skillclaw_config().prm_temperature == 0.2


def test_prm_temperature_zero_is_honored(tmp_path):
    """An explicit 0 (deterministic scoring) must not be rewritten to 0.6."""
    store = _store(tmp_path, {"enabled": True, "model": "gpt-5.2", "temperature": 0})
    assert store.to_skillclaw_config().prm_temperature == 0


def test_prm_temperature_default_when_unset(tmp_path):
    store = _store(tmp_path, {"enabled": True, "model": "gpt-5.2"})
    assert store.to_skillclaw_config().prm_temperature == 0.6
