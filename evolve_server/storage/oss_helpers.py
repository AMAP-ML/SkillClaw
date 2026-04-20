"""Shared storage helper functions."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from skillclaw.object_store import build_object_store

logger = logging.getLogger(__name__)


def make_bucket(endpoint: str, bucket_name: str, key_id: str, key_secret: str):
    """Backward-compatible helper that builds an OSS-backed object store."""
    return build_object_store(
        backend="oss",
        endpoint=endpoint,
        bucket=bucket_name,
        access_key_id=key_id,
        secret_access_key=key_secret,
    )


def list_session_keys(bucket, prefix: str) -> list[str]:
    """List all ``*.json`` objects under ``{prefix}sessions/``."""
    if hasattr(bucket, "iter_objects"):
        iterator = bucket.iter_objects(prefix=f"{prefix}sessions/")
    else:
        from .mock_bucket import LocalBucket, LocalObjectIterator

        if isinstance(bucket, LocalBucket):
            iterator = LocalObjectIterator(bucket, prefix=f"{prefix}sessions/")
        else:
            import oss2

            iterator = oss2.ObjectIterator(bucket, prefix=f"{prefix}sessions/")
    keys: list[str] = []
    for obj in iterator:
        if obj.key.endswith(".json"):
            keys.append(obj.key)
    return keys


def read_json_object(bucket, key: str) -> Optional[dict]:
    """Download and parse a single JSON object from storage."""
    try:
        data = bucket.get_object(key).read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        logger.warning("[Storage] failed to read %s: %s", key, e)
        return None


def load_manifest(bucket, prefix: str) -> dict[str, dict[str, Any]]:
    """Load ``manifest.jsonl`` from storage. Returns ``{skill_name: record}``."""
    key = f"{prefix}manifest.jsonl"
    try:
        data = bucket.get_object(key).read().decode("utf-8")
    except Exception:
        return {}

    skills: dict[str, dict[str, Any]] = {}
    for line in data.strip().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            name = rec.get("name", "")
            if name:
                skills[name] = rec
        except json.JSONDecodeError:
            continue
    return skills


def save_manifest(bucket, prefix: str, manifest: dict[str, dict[str, Any]]) -> None:
    """Write the full manifest back to storage."""
    lines = [json.dumps(rec, ensure_ascii=False) for rec in manifest.values()]
    content = "\n".join(lines) + "\n" if lines else ""
    bucket.put_object(f"{prefix}manifest.jsonl", content.encode("utf-8"))


def delete_session_keys(bucket, keys: list[str]) -> int:
    """Delete session objects from the bucket (OSS or local).

    Returns the number of successfully deleted keys.
    """
    deleted = 0
    for key in keys:
        try:
            bucket.delete_object(key)
            deleted += 1
        except Exception as e:
            logger.warning("[OSS] failed to delete %s: %s", key, e)
    if deleted:
        logger.info("[OSS] deleted %d/%d session keys", deleted, len(keys))
    return deleted


def fetch_skill_content(bucket, prefix: str, skill_name: str) -> Optional[str]:
    """Download a single ``SKILL.md`` from storage."""
    key = f"{prefix}skills/{skill_name}/SKILL.md"
    try:
        return bucket.get_object(key).read().decode("utf-8")
    except Exception:
        return None
