"""Version selection helpers for Nacos skill lifecycle operations."""

from __future__ import annotations

import re
from typing import Any

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_V_VERSION_RE = re.compile(r"^v(\d+)$")


def _collect_versions(summary: dict[str, Any], detail: dict[str, Any] | None = None) -> list[str]:
    versions: list[str] = []
    labels = summary.get("labels")
    if isinstance(labels, dict):
        latest = labels.get("latest")
        if latest:
            versions.append(str(latest))
        versions.extend(str(v) for k, v in labels.items() if k != "latest" and v)
    for field in ("editingVersion", "reviewingVersion", "version"):
        if summary.get(field):
            versions.append(str(summary[field]))
    for item in (detail or {}).get("versions") or []:
        if isinstance(item, dict) and item.get("version"):
            versions.append(str(item["version"]))
    return versions


def _parse_v_version(version: str | None) -> int | None:
    raw = str(version or "").strip()
    match = _V_VERSION_RE.fullmatch(raw)
    return int(match.group(1)) if match else None


def _parse_semver(version: str | None) -> tuple[int, int, int] | None:
    match = _SEMVER_RE.fullmatch(str(version or "").strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _next_version(summary: dict[str, Any], detail: dict[str, Any] | None = None) -> str:
    versions = _collect_versions(summary, detail)
    selected_format = ""
    for version in versions:
        if _parse_semver(version) is not None:
            selected_format = "semver"
            break
        if _parse_v_version(version) is not None:
            selected_format = "v"
            break

    if selected_format == "v":
        next_num = max(_parse_v_version(version) or 0 for version in versions) + 1
        return f"v{next_num}"

    semver_versions = [parsed for version in versions if (parsed := _parse_semver(version)) is not None]
    if not semver_versions:
        return "0.0.1"
    major, minor, patch = max(semver_versions)
    return f"{major}.{minor}.{patch + 1}"
