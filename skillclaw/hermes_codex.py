"""Resolve Hermes-managed Codex OAuth credentials without taking ownership of them."""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable


class HermesInstallationError(RuntimeError):
    """Hermes runtime modules cannot be imported from the active installation."""


def _runtime_imports() -> tuple[Callable, Callable]:
    from agent.auxiliary_client import _codex_cloudflare_headers
    from hermes_cli.auth import resolve_codex_runtime_credentials

    return resolve_codex_runtime_credentials, _codex_cloudflare_headers


def _candidate_sources() -> list[Path]:
    hermes_home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    candidates: list[Path] = []
    legacy_home = os.environ.get("SKILLCLAW_HERMES_HOME")
    if legacy_home:
        candidates.append(Path(legacy_home).expanduser() / "hermes-agent")
    candidates.extend(
        [
            hermes_home / "hermes-agent",
            Path.home() / ".hermes" / "hermes-agent",
        ]
    )
    executable = shutil.which("hermes")
    if executable:
        resolved = Path(executable).resolve()
        candidates.extend(parent for parent in resolved.parents if (parent / "hermes_cli" / "auth.py").is_file())
    return list(dict.fromkeys(path for path in candidates if path is not None))


@lru_cache(maxsize=1)
def _resolver() -> tuple[Callable, Callable]:
    try:
        return _runtime_imports()
    except ImportError:
        pass

    checked: list[str] = []
    for source in _candidate_sources():
        checked.append(str(source))
        if not (source / "hermes_cli" / "auth.py").is_file():
            continue
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        try:
            return _runtime_imports()
        except ImportError:
            continue

    raise HermesInstallationError("Hermes runtime modules were not importable; checked: " + ", ".join(checked))


def _pool_runtime():
    _resolver()  # Ensure Hermes source/package is importable before pool imports.
    from agent.credential_pool import load_pool
    from hermes_cli.auth import DEFAULT_CODEX_BASE_URL

    return load_pool("openai-codex"), DEFAULT_CODEX_BASE_URL


def recover_upstream(
    failed_token: str,
    *,
    status_code: int,
) -> tuple[str, str, dict[str, str]] | None:
    """Refresh/rotate the failed pooled credential and return a retry credential."""
    try:
        pool, default_base_url = _pool_runtime()
        next_entry = pool.try_refresh_matching(api_key_hint=failed_token) if status_code == 401 else None
        if next_entry is None:
            next_entry = pool.mark_exhausted_and_rotate(
                status_code=status_code,
                error_context={"message": f"HTTP {status_code}"},
                api_key_hint=failed_token,
            )
        if next_entry is None or not next_entry.runtime_api_key:
            return None
        token = next_entry.runtime_api_key
        base_url = str(next_entry.base_url or next_entry.inference_base_url or default_base_url).rstrip("/")
        _, header_builder = _resolver()
        return base_url, token, header_builder(token)
    except Exception:
        return None


def _select_pool_credential(base_url: str, token: str) -> tuple[str, str]:
    """Prefer Hermes's active pooled account while retaining auth-store fallback."""
    try:
        pool, _ = _pool_runtime()
    except Exception:
        return base_url, token
    if not pool.has_credentials():
        return base_url, token
    entry = pool.select()
    if entry is None or not entry.runtime_api_key:
        raise RuntimeError("Hermes Codex credential pool has no available credential")
    selected_base = entry.base_url or entry.inference_base_url or base_url
    return str(selected_base).rstrip("/"), entry.runtime_api_key


def resolve_upstream(*, force_refresh: bool = False) -> tuple[str, str, dict[str, str]]:
    """Return the Codex endpoint, transient bearer, and required transport headers."""
    credential_resolver, header_builder = _resolver()
    credentials = credential_resolver(force_refresh=force_refresh)
    base_url = str(credentials.get("base_url") or "").rstrip("/")
    api_key = str(credentials.get("api_key") or "")
    if not base_url or not api_key:
        raise RuntimeError("Hermes has no usable Codex OAuth credential")
    base_url, api_key = _select_pool_credential(base_url, api_key)
    return base_url, api_key, header_builder(api_key)
