"""Explicit Hermes profile selection and conservative upstream import."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml


def resolve_hermes_config(config_path: str = "") -> Path:
    selected = str(config_path or os.environ.get("HERMES_CONFIG", "")).strip()
    if selected:
        return Path(selected).expanduser().resolve()
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    return (home / "config.yaml").resolve()


def load_hermes_mapping(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid Hermes YAML; file was not modified") from exc
    if not isinstance(data, dict):
        raise ValueError("Hermes config must contain a YAML mapping")
    return data


def import_hermes_upstream(path: Path, proxy_port: int = 30000) -> dict:
    """Import literal custom-endpoint values only; never resolve secret references."""
    model = load_hermes_mapping(path).get("model")
    if not isinstance(model, dict):
        raise ValueError("Hermes model settings must include an explicit base_url and default model")
    values = {key: model.get(key, "") for key in ("base_url", "default", "api_key")}
    if any(not isinstance(value, str) or "$" in value or "{{" in value for value in values.values()):
        raise ValueError("Enter upstream settings manually for environment/secret references")
    base, name, key = (values[k].strip() for k in ("base_url", "default", "api_key"))
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not name:
        raise ValueError("Hermes needs a literal HTTP(S) base_url and non-empty default model")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Enter a plain upstream base URL without embedded credentials or query")
    if key == "skillclaw" or (parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.port == proxy_port):
        raise ValueError("Hermes already points at SkillClaw; import from the original upstream settings instead")
    mode = model.get("api_mode") or "chat"
    if mode not in {"chat", "responses"}:
        raise ValueError("Unsupported Hermes API mode; enter upstream settings manually")
    return {"provider": "custom", "api_base": base, "model_id": name, "api_key": key, "api_mode": mode}
