# Adapted from MetaClaw
"""
Claw adapter: auto-configures the active CLI agent to use the SkillClaw proxy.

Supported agents:
  openclaw  — runs `openclaw config set …` + `openclaw gateway restart`
  hermes    — patches ~/.hermes/config.yaml to point model traffic at SkillClaw
  codex     — patches ~/.codex/config.toml to register SkillClaw as a provider
  claude    — patches ~/.claude/settings.json to route Anthropic traffic via SkillClaw
  qwenpaw   — patches QwenPaw model config, selects SkillClaw as active model
  ironclaw  — patches ~/.ironclaw/.env, runs `ironclaw service restart`
  picoclaw  — patches ~/.picoclaw/config.json model_list, runs `picoclaw gateway restart`
  zeroclaw  — patches ~/.zeroclaw/config.toml, runs `zeroclaw service restart`
  nanoclaw  — patches nanoclaw's .env (ANTHROPIC_BASE_URL), restarts via launchd/systemd
  nemoclaw  — registers skillclaw provider in OpenShell, sets inference route
  none      — skip auto-configuration entirely

Add more claws by implementing a `_configure_<name>` function and registering
it in ``_ADAPTERS``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import yaml

if TYPE_CHECKING:
    from .config import SkillClawConfig

logger = logging.getLogger(__name__)
_LEGACY_SKILLCLAW_SKILLS_DIR = Path.home() / ".skillclaw" / "skills"
_HERMES_HOME = Path.home() / ".hermes"
_HERMES_SKILLS_DIR = _HERMES_HOME / "skills"
_HERMES_BACKUP_DIR = Path.home() / ".skillclaw" / "backups" / "hermes"
_CODEX_HOME = Path.home() / ".codex"
_CODEX_CONFIG_PATH = _CODEX_HOME / "config.toml"
_CODEX_SKILLS_DIR = _CODEX_HOME / "skills"
_CODEX_BACKUP_DIR = Path.home() / ".skillclaw" / "backups" / "codex"
_CLAUDE_HOME = Path.home() / ".claude"
_CLAUDE_SETTINGS_PATH = _CLAUDE_HOME / "settings.json"
_CLAUDE_SKILLS_DIR = _CLAUDE_HOME / "skills"
_CLAUDE_BACKUP_DIR = Path.home() / ".skillclaw" / "backups" / "claude"


# ------------------------------------------------------------------ #
# Dispatcher                                                          #
# ------------------------------------------------------------------ #

def configure_claw(cfg: "SkillClawConfig") -> None:
    """Dispatch to the appropriate claw adapter based on cfg.claw_type."""
    claw = getattr(cfg, "claw_type", "openclaw")

    # Backward-compat: configure_openclaw=False → treat as "none"
    configure_flag = getattr(cfg, "configure_openclaw", True)
    if not configure_flag:
        claw = "none"

    adapter = _ADAPTERS.get(claw)
    if adapter is None:
        logger.warning(
            "[ClawAdapter] Unknown claw_type=%r — skipping auto-configuration", claw
        )
        return
    adapter(cfg)


# ------------------------------------------------------------------ #
# OpenClaw adapter                                                    #
# ------------------------------------------------------------------ #

def _configure_openclaw(cfg: "SkillClawConfig") -> None:
    """Auto-configure OpenClaw to use the SkillClaw proxy."""
    model_id = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"
    provider_json = json.dumps({
        "api": "openai-completions",
        "baseUrl": f"http://127.0.0.1:{cfg.proxy_port}/v1",
        "apiKey": cfg.proxy_api_key or "skillclaw",
        "models": [{
            "id": model_id,
            "name": model_id,
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 32768,
            "maxTokens": 8192,
        }],
    })

    commands = [
        ["openclaw", "config", "set", "models.providers.skillclaw",
         "--json", provider_json],
        ["openclaw", "config", "set", "agents.defaults.model.primary",
         f"skillclaw/{model_id}"],
        ["openclaw", "config", "set", "agents.defaults.sandbox.mode", "off"],
        ["openclaw", "gateway", "restart"],
    ]
    _run_commands("openclaw", commands)


# ------------------------------------------------------------------ #
# Hermes adapter                                                      #
# ------------------------------------------------------------------ #

def _load_yaml_mapping(path: Path, label: str) -> dict:
    """Load a YAML mapping, falling back to an empty mapping."""
    if not path.exists():
        return {}

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("[ClawAdapter] Failed to read %s config %s: %s", label, path, e)
        return {}

    if isinstance(loaded, dict):
        return loaded

    logger.warning(
        "[ClawAdapter] %s config %s is not a mapping; replacing it",
        label,
        path,
    )
    return {}


def _load_json_mapping(path: Path, label: str) -> dict:
    """Load a JSON mapping, falling back to an empty mapping."""
    if not path.exists():
        return {}

    try:
        loaded = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("[ClawAdapter] Failed to read %s config %s: %s", label, path, e)
        return {}

    if isinstance(loaded, dict):
        return loaded

    logger.warning(
        "[ClawAdapter] %s config %s is not a mapping; replacing it",
        label,
        path,
    )
    return {}


def _write_yaml_mapping_atomic(path: Path, data: dict, label: str) -> None:
    """Atomically write a YAML mapping to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(_yaml_mapping_to_text(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        logger.info("[ClawAdapter] %s config updated: %s", label, path)
    except Exception as e:
        logger.error("[ClawAdapter] Failed to write %s config %s: %s", label, path, e)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _yaml_mapping_to_text(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _write_text_atomic(path: Path, text: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        logger.info("[ClawAdapter] %s updated: %s", label, path)
    except Exception as e:
        logger.error("[ClawAdapter] Failed to write %s %s: %s", label, path, e)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _backup_text_file_if_changed(
    path: Path,
    new_text: str,
    *,
    backup_dir: Path,
    backup_stem: str,
    backup_suffix: str,
    label: str,
) -> Path | None:
    """Save a timestamped backup before overwriting a text file."""
    if not path.exists():
        return None

    try:
        current_text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("[ClawAdapter] Failed to read %s for backup: %s", path, e)
        return None

    if current_text == new_text:
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{backup_stem}.{timestamp}.{backup_suffix}"
    latest_path = backup_dir / f"{backup_stem}.latest.{backup_suffix}"
    try:
        backup_path.write_text(current_text, encoding="utf-8")
        latest_path.write_text(current_text, encoding="utf-8")
        logger.info("[ClawAdapter] %s backup saved: %s", label, backup_path)
        return backup_path
    except Exception as e:
        logger.warning("[ClawAdapter] Failed to save %s backup: %s", label, e)
        return None


def _latest_backup_path(backup_dir: Path, backup_stem: str, backup_suffix: str) -> Path | None:
    latest_path = backup_dir / f"{backup_stem}.latest.{backup_suffix}"
    if latest_path.exists():
        return latest_path
    if not backup_dir.is_dir():
        return None
    backups = sorted(backup_dir.glob(f"{backup_stem}.*.{backup_suffix}"))
    return backups[-1] if backups else None


def _format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def _parse_toml_value(raw: str) -> object:
    value = raw.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except Exception:
            return value[1:-1]
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def _upsert_top_level_toml_keys(text: str, updates: dict[str, object]) -> str:
    """Update simple top-level TOML assignments before the first table."""
    lines = text.splitlines()
    first_table_index = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            first_table_index = idx
            break

    preamble = lines[:first_table_index]
    remainder = lines[first_table_index:]
    seen: set[str] = set()
    updated_preamble: list[str] = []

    for line in preamble:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            updated_preamble.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            updated_preamble.append(f"{key} = {_format_toml_value(updates[key])}")
            seen.add(key)
            continue
        updated_preamble.append(line)

    missing_keys = [key for key in updates if key not in seen]
    if missing_keys:
        if updated_preamble and updated_preamble[-1].strip():
            updated_preamble.append("")
        for key in missing_keys:
            updated_preamble.append(f"{key} = {_format_toml_value(updates[key])}")
        if remainder:
            updated_preamble.append("")

    merged = updated_preamble + remainder
    return "\n".join(merged).rstrip() + "\n"


def _remove_toml_table(text: str, table_name: str) -> str:
    """Remove a TOML table and its body, if present."""
    lines = text.splitlines()
    kept: list[str] = []
    skipping = False
    target_header = f"[{table_name}]"

    for line in lines:
        stripped = line.strip()
        is_header = stripped.startswith("[") and stripped.endswith("]")
        if is_header:
            if skipping:
                skipping = False
            if stripped == target_header:
                skipping = True
                continue
        if skipping:
            continue
        kept.append(line)

    return "\n".join(kept).rstrip() + "\n"


def _extract_toml_table(text: str, table_name: str) -> dict[str, object]:
    """Extract simple key/value pairs from a TOML table."""
    result: dict[str, object] = {}
    lines = text.splitlines()
    target_header = f"[{table_name}]"
    inside = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if inside:
                break
            inside = stripped == target_header
            continue
        if not inside or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        result[key.strip()] = _parse_toml_value(raw_value)
    return result


def _extract_top_level_toml_value(text: str, key: str) -> object | None:
    """Read a simple top-level TOML assignment before the first table."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        if stripped.startswith("#") or "=" not in stripped:
            continue
        current_key, raw_value = stripped.split("=", 1)
        if current_key.strip() == key:
            return _parse_toml_value(raw_value)
    return None


def _prepare_external_skills_dir(target_dir: Path, label: str) -> None:
    """Prepare an agent-native skill directory without overwriting existing skills."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if not _LEGACY_SKILLCLAW_SKILLS_DIR.is_dir():
        return

    migrated = _copy_missing_skill_dirs(_LEGACY_SKILLCLAW_SKILLS_DIR, target_dir)
    if migrated > 0:
        logger.info(
            "[ClawAdapter] migrated %d legacy SkillClaw skill(s) into %s skills dir",
            migrated,
            label,
        )


def _backup_hermes_config_if_changed(config_path: Path, new_text: str) -> Path | None:
    """Save the current Hermes config before overwriting it, if it changed."""
    return _backup_text_file_if_changed(
        config_path,
        new_text,
        backup_dir=_HERMES_BACKUP_DIR,
        backup_stem="config",
        backup_suffix="yaml",
        label="Hermes config",
    )


def _latest_hermes_backup_path() -> Path | None:
    return _latest_backup_path(_HERMES_BACKUP_DIR, "config", "yaml")


def _write_json_mapping_atomic(path: Path, data: dict, label: str) -> None:
    """Atomically write a JSON mapping to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        logger.info("[ClawAdapter] %s config updated: %s", label, path)
    except Exception as e:
        logger.error("[ClawAdapter] Failed to write %s config %s: %s", label, path, e)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _configure_hermes(cfg: "SkillClawConfig") -> None:
    """Auto-configure Hermes to route model traffic through SkillClaw."""
    config_path = _HERMES_HOME / "config.yaml"
    model_id = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"
    api_key = cfg.proxy_api_key or "skillclaw"
    base_url = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    _prepare_hermes_skills_dir(cfg)

    data = _load_yaml_mapping(config_path, "Hermes")
    model = data.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if isinstance(model, str) and model.strip() else {}

    model["provider"] = "custom"
    model["base_url"] = base_url
    model["default"] = model_id
    model["api_key"] = api_key
    # Clear stale provider-specific mode so Hermes auto-detects from the proxy URL.
    model["api_mode"] = ""

    data["model"] = model
    _backup_hermes_config_if_changed(config_path, _yaml_mapping_to_text(data))
    _write_yaml_mapping_atomic(config_path, data, "Hermes")


def inspect_hermes_config(cfg: "SkillClawConfig") -> dict[str, object]:
    """Return a diagnostic snapshot of the local Hermes integration state."""
    config_path = _HERMES_HOME / "config.yaml"
    expected_model = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"
    expected_base_url = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    expected_api_key = cfg.proxy_api_key or "skillclaw"
    expected_skills_dir = Path(str(getattr(cfg, "skills_dir", "") or _HERMES_SKILLS_DIR)).expanduser()

    data = _load_yaml_mapping(config_path, "Hermes")
    model = data.get("model") if isinstance(data, dict) else {}
    if not isinstance(model, dict):
        model = {"default": model} if isinstance(model, str) and model else {}

    configured_provider = str(model.get("provider", "") or "")
    configured_base_url = str(model.get("base_url", "") or "")
    configured_default = str(model.get("default", "") or "")
    configured_api_key = str(model.get("api_key", "") or "")

    backup_path = _latest_hermes_backup_path()
    proxy_match = (
        configured_provider == "custom"
        and configured_base_url == expected_base_url
        and configured_default == expected_model
        and configured_api_key == expected_api_key
    )
    legacy_present = _LEGACY_SKILLCLAW_SKILLS_DIR.is_dir()
    uses_default_skills_dir = expected_skills_dir == _HERMES_SKILLS_DIR
    issues: list[str] = []
    notes: list[str] = [
        "This integration only rewrites Hermes-local config and does not touch other claw adapters.",
        "Hermes session capture still relies on explicit session headers when available, with proxy-side heuristics as the fallback.",
    ]
    next_steps: list[str] = []

    if not config_path.exists():
        issues.append("Hermes config is missing: ~/.hermes/config.yaml")
    if not proxy_match:
        issues.append("Hermes model routing is not pointing at the local SkillClaw proxy.")
        next_steps.append("Start SkillClaw once so it can rewrite ~/.hermes/config.yaml.")
    if not expected_skills_dir.is_dir():
        issues.append(f"Hermes skills directory is missing: {expected_skills_dir}")
        next_steps.append(f"Create or prepare the Hermes skills directory: {expected_skills_dir}")
    if legacy_present:
        notes.append(
            f"Legacy SkillClaw skills were found at {_LEGACY_SKILLCLAW_SKILLS_DIR}; missing skills are copied into the Hermes library on startup."
        )
    if not backup_path:
        next_steps.append("Run SkillClaw once before relying on `skillclaw restore hermes`, so a backup can be created.")

    return {
        "status": "ok" if not issues else "warning",
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "integration_scope": "hermes-only",
        "expected_model": expected_model,
        "expected_base_url": expected_base_url,
        "configured_provider": configured_provider or "(unset)",
        "configured_base_url": configured_base_url or "(unset)",
        "configured_model": configured_default or "(unset)",
        "proxy_match": proxy_match,
        "expected_skills_dir": str(expected_skills_dir),
        "skills_dir_exists": expected_skills_dir.is_dir(),
        "skills_dir_mode": "hermes-default" if uses_default_skills_dir else "custom",
        "legacy_skillclaw_skills_dir": str(_LEGACY_SKILLCLAW_SKILLS_DIR),
        "legacy_skillclaw_skills_present": legacy_present,
        "latest_backup": str(backup_path) if backup_path else "(none)",
        "session_boundary_mode": "explicit headers if provided, proxy heuristics otherwise",
        "issues": issues,
        "notes": notes,
        "next_steps": next_steps,
    }


def restore_hermes_config(backup_path: Path | None = None) -> dict[str, str]:
    """Restore ~/.hermes/config.yaml from the latest or a specified backup."""
    source = Path(backup_path).expanduser() if backup_path is not None else _latest_hermes_backup_path()
    if source is None or not source.exists():
        raise FileNotFoundError("No Hermes backup found")

    text = source.read_text(encoding="utf-8")
    target = _HERMES_HOME / "config.yaml"
    _write_text_atomic(target, text, "Hermes config restore")
    return {"source": str(source), "target": str(target)}


def _prepare_hermes_skills_dir(cfg: "SkillClawConfig") -> None:
    """Prepare the Hermes-local skill directory without touching other agents."""
    target_dir = Path(str(getattr(cfg, "skills_dir", "") or _HERMES_SKILLS_DIR)).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    if target_dir != _HERMES_SKILLS_DIR:
        logger.info(
            "[ClawAdapter] Hermes uses custom skills dir: %s",
            target_dir,
        )
        return

    if not _LEGACY_SKILLCLAW_SKILLS_DIR.is_dir():
        return

    migrated = _copy_missing_skill_dirs(_LEGACY_SKILLCLAW_SKILLS_DIR, target_dir)
    if migrated > 0:
        logger.info(
            "[ClawAdapter] migrated %d legacy SkillClaw skill(s) into Hermes skills dir",
            migrated,
        )


def _copy_missing_skill_dirs(src_root: Path, dst_root: Path) -> int:
    """Copy only skill folders that do not already exist in the destination."""
    copied = 0
    for entry in sorted(src_root.iterdir()):
        if not entry.is_dir():
            continue
        src_skill_md = entry / "SKILL.md"
        if not src_skill_md.is_file():
            continue
        dst_dir = dst_root / entry.name
        dst_skill_md = dst_dir / "SKILL.md"
        if dst_skill_md.exists():
            continue
        shutil.copytree(entry, dst_dir)
        copied += 1
    return copied


# ------------------------------------------------------------------ #
# Codex adapter                                                       #
# ------------------------------------------------------------------ #

def _backup_codex_config_if_changed(config_path: Path, new_text: str) -> Path | None:
    return _backup_text_file_if_changed(
        config_path,
        new_text,
        backup_dir=_CODEX_BACKUP_DIR,
        backup_stem="config",
        backup_suffix="toml",
        label="Codex config",
    )


def _latest_codex_backup_path() -> Path | None:
    return _latest_backup_path(_CODEX_BACKUP_DIR, "config", "toml")


def _build_codex_provider_block(base_url: str, api_key: str) -> str:
    lines = [
        "[model_providers.skillclaw]",
        'name = "SkillClaw"',
        f"base_url = {_format_toml_value(base_url)}",
        'wire_api = "responses"',
        f"experimental_bearer_token = {_format_toml_value(api_key)}",
    ]
    return "\n".join(lines) + "\n"


def _configure_codex(cfg: "SkillClawConfig") -> None:
    """Auto-configure Codex CLI to use the SkillClaw proxy."""
    model_id = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"
    api_key = cfg.proxy_api_key or "skillclaw"
    base_url = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    config_path = _CODEX_CONFIG_PATH
    _prepare_external_skills_dir(_CODEX_SKILLS_DIR, "Codex")

    existing_text = ""
    if config_path.exists():
        try:
            existing_text = config_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("[ClawAdapter] Failed to read Codex config %s: %s", config_path, e)

    updated = _upsert_top_level_toml_keys(
        existing_text,
        {
            "model": model_id,
            "model_provider": "skillclaw",
        },
    )
    updated = _remove_toml_table(updated, "model_providers.skillclaw").rstrip() + "\n\n"
    updated += _build_codex_provider_block(base_url, api_key)

    _backup_codex_config_if_changed(config_path, updated)
    _write_text_atomic(config_path, updated, "Codex config")


def inspect_codex_config(cfg: "SkillClawConfig") -> dict[str, object]:
    """Return a diagnostic snapshot of the local Codex integration state."""
    config_path = _CODEX_CONFIG_PATH
    expected_model = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"
    expected_base_url = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    expected_api_key = cfg.proxy_api_key or "skillclaw"
    expected_skills_dir = _CODEX_SKILLS_DIR
    configured_skillclaw_skills_dir = Path(
        str(getattr(cfg, "skills_dir", "") or expected_skills_dir),
    ).expanduser()

    text = ""
    if config_path.exists():
        try:
            text = config_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("[ClawAdapter] Failed to read Codex config %s: %s", config_path, e)

    configured_model = str(_extract_top_level_toml_value(text, "model") or "")
    configured_provider = str(_extract_top_level_toml_value(text, "model_provider") or "")
    provider_cfg = _extract_toml_table(text, "model_providers.skillclaw")
    configured_base_url = str(provider_cfg.get("base_url") or "")
    configured_wire_api = str(provider_cfg.get("wire_api") or "")
    configured_token = str(provider_cfg.get("experimental_bearer_token") or "")

    proxy_match = (
        configured_model == expected_model
        and configured_provider == "skillclaw"
        and configured_base_url == expected_base_url
        and configured_wire_api == "responses"
        and configured_token == expected_api_key
    )

    backup_path = _latest_codex_backup_path()
    skills_dir_match = configured_skillclaw_skills_dir == expected_skills_dir
    issues: list[str] = []
    notes: list[str] = [
        "Codex uses the OpenAI Responses-compatible SkillClaw endpoint via `model_providers.skillclaw`.",
        "Codex session boundaries fall back to proxy-side heuristics because Codex does not send SkillClaw session headers.",
    ]
    next_steps: list[str] = []

    if not config_path.exists():
        issues.append("Codex config is missing: ~/.codex/config.toml")
    if not proxy_match:
        issues.append("Codex model routing is not pointing at the local SkillClaw proxy.")
        next_steps.append("Start SkillClaw once with `claw_type=codex` so it can rewrite ~/.codex/config.toml.")
    if not expected_skills_dir.is_dir():
        issues.append(f"Codex skills directory is missing: {expected_skills_dir}")
        next_steps.append(f"Create or prepare the Codex skills directory: {expected_skills_dir}")
    if not skills_dir_match:
        issues.append(
            f"SkillClaw is configured to evolve skills in {configured_skillclaw_skills_dir}, "
            f"but Codex reads skills from {expected_skills_dir}."
        )
        next_steps.append(f"Set `skills.dir` to {expected_skills_dir} when using the Codex integration.")
    if not backup_path:
        next_steps.append("Run SkillClaw once before relying on `skillclaw restore codex`, so a backup can be created.")

    return {
        "status": "ok" if not issues else "warning",
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "integration_scope": "codex-only",
        "expected_model": expected_model,
        "configured_model": configured_model or "(unset)",
        "expected_base_url": expected_base_url,
        "configured_base_url": configured_base_url or "(unset)",
        "configured_provider": configured_provider or "(unset)",
        "proxy_match": proxy_match,
        "expected_skills_dir": str(expected_skills_dir),
        "skills_dir_exists": expected_skills_dir.is_dir(),
        "skills_dir_mode": "codex-default" if skills_dir_match else "custom",
        "configured_skillclaw_skills_dir": str(configured_skillclaw_skills_dir),
        "configured_wire_api": configured_wire_api or "(unset)",
        "latest_backup": str(backup_path) if backup_path else "(none)",
        "session_boundary_mode": "proxy heuristics",
        "issues": issues,
        "notes": notes,
        "next_steps": next_steps,
    }


def restore_codex_config(backup_path: Path | None = None) -> dict[str, str]:
    """Restore ~/.codex/config.toml from the latest or a specified backup."""
    source = Path(backup_path).expanduser() if backup_path is not None else _latest_codex_backup_path()
    if source is None or not source.exists():
        raise FileNotFoundError("No Codex backup found")

    text = source.read_text(encoding="utf-8")
    target = _CODEX_CONFIG_PATH
    _write_text_atomic(target, text, "Codex config restore")
    return {"source": str(source), "target": str(target)}


# ------------------------------------------------------------------ #
# Claude Code adapter                                                 #
# ------------------------------------------------------------------ #

def _backup_claude_settings_if_changed(settings_path: Path, new_text: str) -> Path | None:
    return _backup_text_file_if_changed(
        settings_path,
        new_text,
        backup_dir=_CLAUDE_BACKUP_DIR,
        backup_stem="settings",
        backup_suffix="json",
        label="Claude Code settings",
    )


def _latest_claude_backup_path() -> Path | None:
    return _latest_backup_path(_CLAUDE_BACKUP_DIR, "settings", "json")


def _configure_claude(cfg: "SkillClawConfig") -> None:
    """Auto-configure Claude Code to use the SkillClaw proxy."""
    settings_path = _CLAUDE_SETTINGS_PATH
    api_key = cfg.proxy_api_key or "skillclaw"
    base_url = f"http://127.0.0.1:{cfg.proxy_port}"
    _prepare_external_skills_dir(_CLAUDE_SKILLS_DIR, "Claude Code")

    data = _load_json_mapping(settings_path, "Claude Code")
    env = data.get("env")
    if not isinstance(env, dict):
        env = {}
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = api_key
    data["env"] = env

    new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _backup_claude_settings_if_changed(settings_path, new_text)
    _write_text_atomic(settings_path, new_text, "Claude Code settings")


def inspect_claude_config(cfg: "SkillClawConfig") -> dict[str, object]:
    """Return a diagnostic snapshot of the local Claude Code integration state."""
    settings_path = _CLAUDE_SETTINGS_PATH
    expected_base_url = f"http://127.0.0.1:{cfg.proxy_port}"
    expected_api_key = cfg.proxy_api_key or "skillclaw"
    expected_skills_dir = _CLAUDE_SKILLS_DIR
    configured_skillclaw_skills_dir = Path(
        str(getattr(cfg, "skills_dir", "") or expected_skills_dir),
    ).expanduser()

    data = _load_json_mapping(settings_path, "Claude Code")
    env = data.get("env") if isinstance(data.get("env"), dict) else {}
    configured_base_url = str(env.get("ANTHROPIC_BASE_URL", "") or "")
    configured_token = str(env.get("ANTHROPIC_AUTH_TOKEN", "") or "")
    configured_model = str(data.get("model", "") or "")
    proxy_match = (
        configured_base_url == expected_base_url
        and configured_token == expected_api_key
    )

    backup_path = _latest_claude_backup_path()
    skills_dir_match = configured_skillclaw_skills_dir == expected_skills_dir
    issues: list[str] = []
    notes: list[str] = [
        "Claude Code uses SkillClaw through `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` in ~/.claude/settings.json.",
        "Claude Code session boundaries fall back to proxy-side heuristics because Claude Code does not send SkillClaw session headers.",
    ]
    next_steps: list[str] = []

    if not settings_path.exists():
        issues.append("Claude Code settings are missing: ~/.claude/settings.json")
    if not proxy_match:
        issues.append("Claude Code is not pointing at the local SkillClaw proxy.")
        next_steps.append("Start SkillClaw once with `claw_type=claude` so it can rewrite ~/.claude/settings.json.")
    if not expected_skills_dir.is_dir():
        issues.append(f"Claude Code skills directory is missing: {expected_skills_dir}")
        next_steps.append(f"Create or prepare the Claude Code skills directory: {expected_skills_dir}")
    if not skills_dir_match:
        issues.append(
            f"SkillClaw is configured to evolve skills in {configured_skillclaw_skills_dir}, "
            f"but Claude Code reads skills from {expected_skills_dir}."
        )
        next_steps.append(f"Set `skills.dir` to {expected_skills_dir} when using the Claude Code integration.")
    if not backup_path:
        next_steps.append("Run SkillClaw once before relying on `skillclaw restore claude`, so a backup can be created.")

    return {
        "status": "ok" if not issues else "warning",
        "config_path": str(settings_path),
        "config_exists": settings_path.exists(),
        "integration_scope": "claude-only",
        "configured_model": configured_model or "(unset)",
        "expected_base_url": expected_base_url,
        "configured_base_url": configured_base_url or "(unset)",
        "configured_provider": "anthropic-env",
        "proxy_match": proxy_match,
        "expected_skills_dir": str(expected_skills_dir),
        "skills_dir_exists": expected_skills_dir.is_dir(),
        "skills_dir_mode": "claude-default" if skills_dir_match else "custom",
        "configured_skillclaw_skills_dir": str(configured_skillclaw_skills_dir),
        "latest_backup": str(backup_path) if backup_path else "(none)",
        "session_boundary_mode": "proxy heuristics",
        "issues": issues,
        "notes": notes,
        "next_steps": next_steps,
    }


def restore_claude_config(backup_path: Path | None = None) -> dict[str, str]:
    """Restore ~/.claude/settings.json from the latest or a specified backup."""
    source = Path(backup_path).expanduser() if backup_path is not None else _latest_claude_backup_path()
    if source is None or not source.exists():
        raise FileNotFoundError("No Claude Code backup found")

    text = source.read_text(encoding="utf-8")
    target = _CLAUDE_SETTINGS_PATH
    _write_text_atomic(target, text, "Claude Code settings restore")
    return {"source": str(source), "target": str(target)}


# ------------------------------------------------------------------ #
# QwenPaw adapter                                                     #
# ------------------------------------------------------------------ #

def _get_qwenpaw_env(key: str, default: str = "") -> str:
    """Look up a QwenPaw env var."""
    if key in os.environ:
        return str(os.environ[key])
    return default


def _resolve_qwenpaw_dirs() -> tuple[Path, Path]:
    """Resolve QwenPaw working/secret directories."""
    working_dir = Path(
        _get_qwenpaw_env("QWENPAW_WORKING_DIR", "~/.qwenpaw"),
    ).expanduser().resolve()
    secret_dir = Path(
        _get_qwenpaw_env("QWENPAW_SECRET_DIR", f"{working_dir}.secret"),
    ).expanduser().resolve()
    return working_dir, secret_dir


def _upsert_model_info(models: object, model_id: str) -> list[dict[str, object]]:
    """Ensure a model list contains the SkillClaw proxy model."""
    normalized: list[dict[str, object]] = []
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                normalized.append(dict(item))

    for model in normalized:
        if str(model.get("id", "")).strip() == model_id:
            model["name"] = model.get("name") or model_id
            return normalized

    normalized.append({"id": model_id, "name": model_id})
    return normalized


def _configure_qwenpaw(cfg: "SkillClawConfig") -> None:
    """Auto-configure QwenPaw to use the SkillClaw proxy.

    QwenPaw stores model provider state under ``<secret>/providers`` while
    its app config lives in ``<working>/config.json``. Update both shapes so
    SkillClaw can point QwenPaw at the local proxy in one step.
    """
    working_dir, secret_dir = _resolve_qwenpaw_dirs()
    config_path = working_dir / "config.json"
    provider_path = secret_dir / "providers" / "builtin" / "qwenpaw-local.json"
    active_model_path = secret_dir / "providers" / "active_model.json"
    model_id = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"
    api_key = cfg.proxy_api_key or "skillclaw"
    base_url = f"http://127.0.0.1:{cfg.proxy_port}/v1"

    # Keep the app config aligned with the provider selection.
    config_data = _load_json_mapping(config_path, "QwenPaw")
    if not isinstance(config_data.get("models"), dict):
        config_data["models"] = {}
    config_data["models"]["default"] = {
        "provider": "openai_compatible",
        "model": model_id,
        "api_key": api_key,
        "base_url": base_url,
    }
    _write_json_mapping_atomic(config_path, config_data, "QwenPaw")

    # Current QwenPaw provider storage.
    provider_data = _load_json_mapping(provider_path, "QwenPaw provider")
    provider_data["id"] = "qwenpaw-local"
    provider_data["name"] = "QwenPaw Local"
    provider_data["chat_model"] = str(
        provider_data.get("chat_model") or "OpenAIChatModel",
    )
    provider_data["base_url"] = base_url
    provider_data["api_key"] = api_key
    provider_data["api_key_prefix"] = str(provider_data.get("api_key_prefix") or "")
    provider_data["is_local"] = True
    provider_data["freeze_url"] = False
    provider_data["require_api_key"] = False
    provider_data["is_custom"] = False
    provider_data["support_model_discovery"] = bool(
        provider_data.get("support_model_discovery", False),
    )
    provider_data["support_connection_check"] = bool(
        provider_data.get("support_connection_check", True),
    )
    provider_data["generate_kwargs"] = (
        provider_data["generate_kwargs"]
        if isinstance(provider_data.get("generate_kwargs"), dict)
        else {}
    )
    provider_data["meta"] = (
        provider_data["meta"] if isinstance(provider_data.get("meta"), dict) else {}
    )
    provider_data["models"] = (
        provider_data["models"] if isinstance(provider_data.get("models"), list) else []
    )
    provider_data["extra_models"] = _upsert_model_info(
        provider_data.get("extra_models"),
        model_id,
    )
    _write_json_mapping_atomic(provider_path, provider_data, "QwenPaw provider")

    active_model = {"provider_id": "qwenpaw-local", "model": model_id}
    _write_json_mapping_atomic(active_model_path, active_model, "QwenPaw active model")

    # Best-effort reload for the current CLI.
    _run_commands("qwenpaw", [["qwenpaw", "daemon", "restart"]], ignore_missing=True)


# ------------------------------------------------------------------ #
# IronClaw adapter                                                    #
# ------------------------------------------------------------------ #

def _configure_ironclaw(cfg: "SkillClawConfig") -> None:
    """Auto-configure IronClaw to use the SkillClaw proxy.

    Patches ~/.ironclaw/.env to set LLM_BACKEND=openai_compatible and
    point LLM_BASE_URL at the SkillClaw proxy port.  Triggers a service
    restart so the new env vars take effect immediately.
    """
    env_path = Path.home() / ".ironclaw" / ".env"
    model_id = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"

    new_vars = {
        "LLM_BACKEND": "openai_compatible",
        "LLM_BASE_URL": f"http://127.0.0.1:{cfg.proxy_port}/v1",
        "LLM_MODEL": model_id,
        "LLM_API_KEY": cfg.proxy_api_key or "skillclaw",
    }

    _patch_dotenv(env_path, new_vars)

    # IronClaw reads .env at startup, so a service restart is required.
    _run_commands(
        "ironclaw",
        [["ironclaw", "service", "restart"]],
        ignore_missing=True,
    )


def _patch_dotenv(env_path: Path, new_vars: dict[str, str], label: str = "IronClaw") -> None:
    """Update or insert KEY=VALUE lines in a .env file (preserves comments)."""
    lines: list[str] = []
    if env_path.exists():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            logger.warning("[ClawAdapter] Failed to read %s: %s", env_path, e)

    updated: set[str] = set()
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in new_vars:
            new_lines.append(f"{key}={new_vars[key]}")
            updated.add(key)
        else:
            new_lines.append(line)

    # Append any keys that were not already in the file
    for key, val in new_vars.items():
        if key not in updated:
            new_lines.append(f"{key}={val}")

    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info("[ClawAdapter] %s .env updated: %s", label, env_path)
    except Exception as e:
        logger.error("[ClawAdapter] Failed to write %s: %s", env_path, e)


# ------------------------------------------------------------------ #
# PicoClaw adapter                                                     #
# ------------------------------------------------------------------ #

def _configure_picoclaw(cfg: "SkillClawConfig") -> None:
    """Auto-configure PicoClaw to use the SkillClaw proxy.

    Injects a ``skillclaw`` entry into the ``model_list`` array in
    ``~/.picoclaw/config.json`` and sets it as the default model via
    ``agents.defaults.model_name``.
    """
    config_path = Path.home() / ".picoclaw" / "config.json"
    model_id = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"

    data: dict = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[ClawAdapter] Failed to read %s: %s", config_path, e)

    # Build the SkillClaw model entry
    skillclaw_entry = {
        "model_name": "skillclaw",
        "model": f"openai/{model_id}",
        "api_key": cfg.proxy_api_key or "skillclaw",
        "api_base": f"http://127.0.0.1:{cfg.proxy_port}/v1",
    }

    # Ensure model_list exists and upsert the skillclaw entry
    model_list = data.get("model_list")
    if not isinstance(model_list, list):
        model_list = []
    # Remove any previous skillclaw entry
    model_list = [m for m in model_list if m.get("model_name") != "skillclaw"]
    model_list.append(skillclaw_entry)
    data["model_list"] = model_list

    # Set skillclaw as the active default model
    if not isinstance(data.get("agents"), dict):
        data["agents"] = {}
    if not isinstance(data["agents"].get("defaults"), dict):
        data["agents"]["defaults"] = {}
    data["agents"]["defaults"]["model_name"] = "skillclaw"

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("[ClawAdapter] PicoClaw config updated: %s", config_path)
    except Exception as e:
        logger.error("[ClawAdapter] Failed to write %s: %s", config_path, e)
        return

    _run_commands(
        "picoclaw",
        [["picoclaw", "gateway", "restart"]],
        ignore_missing=True,
    )


# ------------------------------------------------------------------ #
# ZeroClaw adapter                                                     #
# ------------------------------------------------------------------ #

def _configure_zeroclaw(cfg: "SkillClawConfig") -> None:
    """Auto-configure ZeroClaw to use the SkillClaw proxy.

    Patches ``~/.zeroclaw/config.toml`` to set the provider to
    ``openai-compatible`` pointing at the SkillClaw proxy.  Falls back to
    a simple line-based patcher to avoid a hard dependency on a TOML
    write library.
    """
    config_path = Path.home() / ".zeroclaw" / "config.toml"
    model_id = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"

    new_vars = {
        "provider": "openai-compatible",
        "model": model_id,
        "api_key": cfg.proxy_api_key or "skillclaw",
        "base_url": f"http://127.0.0.1:{cfg.proxy_port}/v1",
    }

    _patch_toml(config_path, new_vars)

    _run_commands(
        "zeroclaw",
        [["zeroclaw", "service", "restart"]],
        ignore_missing=True,
    )


def _patch_toml(toml_path: Path, new_vars: dict[str, str]) -> None:
    """Update or insert key = "value" lines in a TOML file.

    This is a minimal line-based patcher (no full TOML parser required).
    It handles simple ``key = "value"`` pairs at the top level.  Existing
    keys are updated in-place; missing keys are appended.
    """
    lines: list[str] = []
    if toml_path.exists():
        try:
            lines = toml_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            logger.warning("[ClawAdapter] Failed to read %s: %s", toml_path, e)

    updated: set[str] = set()
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in new_vars:
            new_lines.append(f'{key} = "{new_vars[key]}"')
            updated.add(key)
        else:
            new_lines.append(line)

    for key, val in new_vars.items():
        if key not in updated:
            new_lines.append(f'{key} = "{val}"')

    try:
        toml_path.parent.mkdir(parents=True, exist_ok=True)
        toml_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info("[ClawAdapter] ZeroClaw config.toml updated: %s", toml_path)
    except Exception as e:
        logger.error("[ClawAdapter] Failed to write %s: %s", toml_path, e)


# ------------------------------------------------------------------ #
# NanoClaw adapter                                                    #
# ------------------------------------------------------------------ #

def _configure_nanoclaw(cfg: "SkillClawConfig") -> None:
    """Auto-configure NanoClaw to route API calls through SkillClaw proxy.

    NanoClaw uses an Anthropic-compatible credential proxy (credential-proxy.ts)
    that forwards container API calls to ANTHROPIC_BASE_URL.  SkillClaw exposes
    a /v1/messages Anthropic-compatible endpoint, so we point ANTHROPIC_BASE_URL
    at the SkillClaw proxy and restart the service.

    Config file location is discovered in priority order:
      1. WorkingDirectory from ~/Library/LaunchAgents/com.nanoclaw.plist (macOS)
      2. WorkingDirectory from ~/.config/systemd/user/nanoclaw.service (Linux)
      3. Common install locations: ~/nanoclaw, ~/code/nanoclaw, ~/.nanoclaw
    """
    env_path = _find_nanoclaw_env()
    if env_path is None:
        logger.warning(
            "[ClawAdapter] Could not locate nanoclaw .env — "
            "set ANTHROPIC_BASE_URL=http://127.0.0.1:%d and "
            "ANTHROPIC_API_KEY=%s in nanoclaw's .env manually.",
            cfg.proxy_port,
            cfg.proxy_api_key or "skillclaw",
        )
        return

    new_vars = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{cfg.proxy_port}",
        "ANTHROPIC_API_KEY": cfg.proxy_api_key or "skillclaw",
    }
    _patch_dotenv(env_path, new_vars, label="NanoClaw")

    # Restart nanoclaw via launchd (macOS) or systemd --user (Linux)
    if platform.system() == "Darwin":
        import os
        uid = os.getuid()
        _run_commands(
            "nanoclaw",
            [["launchctl", "kickstart", "-k", f"gui/{uid}/com.nanoclaw"]],
            ignore_missing=True,
        )
    else:
        _run_commands(
            "nanoclaw",
            [["systemctl", "--user", "restart", "nanoclaw"]],
            ignore_missing=True,
        )


def _find_nanoclaw_env() -> Path | None:
    """Locate nanoclaw's project-root .env file by probing known locations."""
    home = Path.home()

    # 1. macOS launchd plist → WorkingDirectory
    plist_path = home / "Library" / "LaunchAgents" / "com.nanoclaw.plist"
    if plist_path.exists():
        try:
            import plistlib
            with open(plist_path, "rb") as f:
                plist = plistlib.load(f)
            work_dir = plist.get("WorkingDirectory")
            if work_dir:
                candidate = Path(work_dir) / ".env"
                if candidate.parent.is_dir():
                    return candidate
        except Exception as e:
            logger.debug("[ClawAdapter] failed to parse nanoclaw plist: %s", e)

    # 2. Linux systemd service → WorkingDirectory=
    service_path = home / ".config" / "systemd" / "user" / "nanoclaw.service"
    if service_path.exists():
        try:
            for line in service_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("WorkingDirectory="):
                    work_dir = stripped.split("=", 1)[1].strip()
                    candidate = Path(work_dir) / ".env"
                    if candidate.parent.is_dir():
                        return candidate
        except Exception as e:
            logger.debug("[ClawAdapter] failed to parse nanoclaw service file: %s", e)

    # 3. Common install locations (check for existing .env or at least the dir)
    for candidate in [
        home / "nanoclaw" / ".env",
        home / "code" / "nanoclaw" / ".env",
        home / ".nanoclaw" / ".env",
    ]:
        if candidate.exists() or candidate.parent.is_dir():
            return candidate

    return None


# ------------------------------------------------------------------ #
# NemoClaw adapter                                                    #
# ------------------------------------------------------------------ #

def _configure_nemoclaw(cfg: "SkillClawConfig") -> None:
    """Auto-configure NemoClaw to route inference through SkillClaw proxy.

    NemoClaw runs OpenClaw inside an OpenShell sandbox with a pluggable
    inference provider.  We register (or update) a 'skillclaw' OpenAI-compatible
    provider via the openshell CLI and set it as the active inference route.
    The config is also persisted to ~/.nemoclaw/config.json so that
    `openclaw nemoclaw status` reflects the current state.
    """
    model_id = cfg.served_model_name or cfg.llm_model_id or "skillclaw-model"
    api_key = cfg.proxy_api_key or "skillclaw"
    base_url = f"http://127.0.0.1:{cfg.proxy_port}/v1"

    # Step 1: Register (or update) the skillclaw provider in OpenShell
    create_cmd = [
        "openshell", "provider", "create",
        "--name", "skillclaw",
        "--type", "openai",
        "--credential", f"OPENAI_API_KEY={api_key}",
        "--config", f"OPENAI_BASE_URL={base_url}",
    ]
    try:
        result = subprocess.run(
            create_cmd, capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            stderr = result.stderr or ""
            if "AlreadyExists" in stderr or "already exists" in stderr.lower():
                logger.info("[ClawAdapter] openshell provider 'skillclaw' exists — updating")
                _run_commands("nemoclaw", [[
                    "openshell", "provider", "update", "skillclaw",
                    "--credential", f"OPENAI_API_KEY={api_key}",
                    "--config", f"OPENAI_BASE_URL={base_url}",
                ]])
            else:
                logger.warning(
                    "[ClawAdapter] openshell provider create failed: %s",
                    stderr.strip(),
                )
                return
        else:
            logger.info("[ClawAdapter] openshell provider create skillclaw → ok")
    except FileNotFoundError:
        logger.warning(
            "[ClawAdapter] 'openshell' not found in PATH — configure NemoClaw manually."
        )
        return
    except Exception as e:
        logger.warning("[ClawAdapter] openshell provider create error: %s", e)
        return

    # Step 2: Set inference route to the skillclaw provider
    _run_commands("nemoclaw", [[
        "openshell", "inference", "set",
        "--provider", "skillclaw",
        "--model", model_id,
    ]])

    # Step 3: Persist ~/.nemoclaw/config.json
    _write_nemoclaw_config(base_url, model_id, api_key)

    # Step 4: Restart openclaw gateway inside the sandbox (best-effort)
    _run_commands("nemoclaw", [["openclaw", "gateway", "restart"]], ignore_missing=True)


def _write_nemoclaw_config(endpoint_url: str, model: str, api_key: str) -> None:
    """Write ~/.nemoclaw/config.json to record SkillClaw as the active provider."""
    config_path = Path.home() / ".nemoclaw" / "config.json"
    data = {
        "endpointType": "custom",
        "endpointUrl": endpoint_url,
        "ncpPartner": None,
        "model": model,
        "profile": "ncp",
        "credentialEnv": "OPENAI_API_KEY",
        "onboardedAt": datetime.datetime.utcnow().isoformat() + "Z",
    }
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("[ClawAdapter] NemoClaw config updated: %s", config_path)
    except Exception as e:
        logger.error("[ClawAdapter] Failed to write %s: %s", config_path, e)


# ------------------------------------------------------------------ #
# Noop adapter                                                        #
# ------------------------------------------------------------------ #

def _configure_none(cfg: "SkillClawConfig") -> None:
    logger.info("[ClawAdapter] claw_type=none — skipping auto-configuration")


# ------------------------------------------------------------------ #
# Registry                                                            #
# ------------------------------------------------------------------ #

_ADAPTERS: dict[str, Callable[["SkillClawConfig"], None]] = {
    "openclaw": _configure_openclaw,
    "hermes": _configure_hermes,
    "codex": _configure_codex,
    "claude": _configure_claude,
    "qwenpaw": _configure_qwenpaw,
    "ironclaw": _configure_ironclaw,
    "picoclaw": _configure_picoclaw,
    "zeroclaw": _configure_zeroclaw,
    "nanoclaw": _configure_nanoclaw,
    "nemoclaw": _configure_nemoclaw,
    "none": _configure_none,
}

# Canonical list of supported claw types (for CLI choices / wizard).
CLAW_TYPES: list[str] = list(_ADAPTERS)


# ------------------------------------------------------------------ #
# Shared helper                                                       #
# ------------------------------------------------------------------ #

def _run_commands(
    agent_name: str,
    commands: list[list[str]],
    ignore_missing: bool = False,
) -> None:
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                logger.warning(
                    "[ClawAdapter] %s command failed: %s\n  stderr: %s",
                    agent_name,
                    " ".join(cmd),
                    result.stderr.strip(),
                )
            else:
                logger.info("[ClawAdapter] %s → ok", " ".join(cmd[:4]))
        except FileNotFoundError:
            if ignore_missing:
                logger.debug(
                    "[ClawAdapter] '%s' not found in PATH — skipping restart step",
                    cmd[0],
                )
            else:
                logger.warning(
                    "[ClawAdapter] '%s' not found in PATH — skipping auto-config. "
                    "Configure %s manually.",
                    cmd[0],
                    agent_name,
                )
            break
        except Exception as e:
            logger.warning("[ClawAdapter] %s command error: %s", agent_name, e)
