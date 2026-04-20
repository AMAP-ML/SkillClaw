"""
Persistent skill-name → skill-id registry with version tracking.

IDs are the first 12 hex characters of SHA-256(name), consistent with the
convention used in ``SkillManager._parse_skill_md``.  The registry is
persisted to shared storage as ``{prefix}evolve_skill_registry.json`` so that IDs
remain stable across restarts.

Each entry also tracks:
  - ``version``: incremented on every content-changing update
  - ``content_sha``: SHA-256 of the latest SKILL.md content
  - ``history``: list of ``{version, content_sha, timestamp, action}``
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SkillIDRegistry:
    """Maintains a persistent mapping  skill_name -> {skill_id, version, ...}."""

    def __init__(self) -> None:
        self._map: dict[str, dict[str, Any]] = {}

    # -- persistence -------------------------------------------------- #

    def load_from_oss(self, bucket, prefix: str) -> None:
        key = f"{prefix}evolve_skill_registry.json"
        try:
            data = bucket.get_object(key).read().decode("utf-8")
            raw = json.loads(data)
            if isinstance(raw, dict):
                self._map = self._normalise(raw)
            logger.info("[SkillIDRegistry] loaded %d entries from storage", len(self._map))
        except Exception:
            logger.info("[SkillIDRegistry] no existing registry in storage — starting fresh")

    def save_to_oss(self, bucket, prefix: str) -> None:
        key = f"{prefix}evolve_skill_registry.json"
        content = json.dumps(self._map, ensure_ascii=False, indent=2)
        bucket.put_object(key, content.encode("utf-8"))
        logger.info("[SkillIDRegistry] saved %d entries to storage", len(self._map))

    # -- backward compat: old format was {name: id_str} --------------- #

    @staticmethod
    def _normalise(raw: dict) -> dict[str, dict[str, Any]]:
        """Accept both old ``{name: id_str}`` and new ``{name: {...}}`` formats."""
        out: dict[str, dict[str, Any]] = {}
        for name, val in raw.items():
            if isinstance(val, str):
                out[name] = {
                    "skill_id": val,
                    "version": 1,
                    "content_sha": "",
                    "history": [],
                }
            elif isinstance(val, dict):
                out[name] = val
            else:
                out[name] = {
                    "skill_id": hashlib.sha256(name.encode()).hexdigest()[:12],
                    "version": 1,
                    "content_sha": "",
                    "history": [],
                }
        return out

    # -- lookup / generate -------------------------------------------- #

    def get_or_create(self, skill_name: str) -> str:
        """Return the stable skill_id (create entry if missing)."""
        entry = self._map.get(skill_name)
        if entry:
            return entry["skill_id"]
        sid = hashlib.sha256(skill_name.encode()).hexdigest()[:12]
        self._map[skill_name] = {
            "skill_id": sid,
            "version": 0,
            "content_sha": "",
            "history": [],
        }
        return sid

    def get(self, skill_name: str) -> Optional[str]:
        entry = self._map.get(skill_name)
        return entry["skill_id"] if entry else None

    def get_version(self, skill_name: str) -> int:
        entry = self._map.get(skill_name)
        return entry.get("version", 0) if entry else 0

    def get_content_sha(self, skill_name: str) -> str:
        entry = self._map.get(skill_name)
        return entry.get("content_sha", "") if entry else ""

    def record_update(
        self,
        skill_name: str,
        content_sha: str,
        action: str = "create",
    ) -> int:
        """Record a content-changing update. Returns the new version number.

        Increments ``version``, updates ``content_sha``, and appends to
        ``history`` (capped at 20 entries to bound storage).
        """
        entry = self._map.get(skill_name)
        if not entry:
            self.get_or_create(skill_name)
            entry = self._map[skill_name]

        new_version = entry.get("version", 0) + 1
        entry["version"] = new_version
        entry["content_sha"] = content_sha
        history: list = entry.setdefault("history", [])
        history.append(
            {
                "version": new_version,
                "content_sha": content_sha,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
            }
        )
        if len(history) > 20:
            entry["history"] = history[-20:]

        return new_version

    def all_ids(self) -> dict[str, str]:
        return {name: entry["skill_id"] for name, entry in self._map.items()}

    def all_entries(self) -> dict[str, dict[str, Any]]:
        return dict(self._map)
