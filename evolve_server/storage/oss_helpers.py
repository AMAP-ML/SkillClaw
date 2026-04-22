"""Shared storage helper functions."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from skillclaw.object_store import build_object_store
from skillclaw.skill_bundle import (
    bundle_entrypoint_text,
    bundle_file_records,
    bundle_tree_sha256,
    coerce_skill_bundle,
)

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


def list_object_keys(bucket, prefix: str) -> list[str]:
    """List all object keys under *prefix* across local/OSS backends."""
    if hasattr(bucket, "iter_objects"):
        iterator = bucket.iter_objects(prefix=prefix)
    else:
        from .mock_bucket import LocalBucket, LocalObjectIterator

        if isinstance(bucket, LocalBucket):
            iterator = LocalObjectIterator(bucket, prefix=prefix)
        else:
            import oss2

            iterator = oss2.ObjectIterator(bucket, prefix=prefix)
    return [obj.key for obj in iterator]


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


def fetch_skill_bundle(
    bucket,
    prefix: str,
    skill_name: str,
    record: Optional[dict[str, Any]] = None,
) -> dict[str, bytes]:
    """Download a full skill bundle from storage.

    Backward compatibility:
      - bundle-aware records read nested files from ``skills/<name>/files/...``
      - legacy records fall back to a single ``SKILL.md``
    """
    bundle: dict[str, bytes] = {}
    file_entries = (record or {}).get("files")
    if isinstance(file_entries, list) and file_entries:
        for item in file_entries:
            rel_path = str((item or {}).get("path") or "").strip().replace("\\", "/")
            if not rel_path:
                continue
            if rel_path == "SKILL.md":
                key = f"{prefix}skills/{skill_name}/SKILL.md"
            else:
                key = f"{prefix}skills/{skill_name}/files/{rel_path}"
            bundle[rel_path] = bucket.get_object(key).read()
        return bundle

    content = fetch_skill_content(bucket, prefix, skill_name)
    if content is None:
        return {}
    bundle["SKILL.md"] = content.encode("utf-8")
    return bundle


def fetch_skill_bundle_text(
    bucket,
    prefix: str,
    skill_name: str,
    record: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Download the bundle and return its ``SKILL.md`` entrypoint text."""
    bundle = fetch_skill_bundle(bucket, prefix, skill_name, record)
    if not bundle:
        return None
    try:
        return bundle_entrypoint_text(bundle)
    except Exception:
        return None


def skill_version_prefix(prefix: str, skill_name: str, version: int) -> str:
    return f"{prefix}skills/{skill_name}/versions/v{max(1, int(version or 1))}/"


def skill_version_bundle_key(prefix: str, skill_name: str, version: int, rel_path: str) -> str:
    clean = str(rel_path or "").strip().replace("\\", "/")
    base = skill_version_prefix(prefix, skill_name, version)
    if clean == "SKILL.md":
        return f"{base}SKILL.md"
    return f"{base}files/{clean}"


def skill_version_record_key(prefix: str, skill_name: str, version: int) -> str:
    return f"{skill_version_prefix(prefix, skill_name, version)}bundle.json"


def save_version_bundle(
    bucket,
    prefix: str,
    skill_name: str,
    version: int,
    bundle_files: dict[str, bytes],
) -> dict[str, Any]:
    bundle = coerce_skill_bundle(bundle_files)
    record = {
        "format": "bundle_v1",
        "entrypoint": "SKILL.md",
        "tree_sha256": bundle_tree_sha256(bundle),
        "files": bundle_file_records(bundle),
    }
    keep_keys: set[str] = set()
    for rel_path, data in sorted(bundle.items()):
        key = skill_version_bundle_key(prefix, skill_name, version, rel_path)
        keep_keys.add(key)
        bucket.put_object(key, data)
    for key in list_object_keys(bucket, f"{skill_version_prefix(prefix, skill_name, version)}files/"):
        if key not in keep_keys:
            bucket.delete_object(key)
    bucket.put_object(
        skill_version_record_key(prefix, skill_name, version),
        json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return record


def load_version_bundle_record(
    bucket,
    prefix: str,
    skill_name: str,
    version: int,
) -> Optional[dict[str, Any]]:
    try:
        payload = bucket.get_object(skill_version_record_key(prefix, skill_name, version)).read().decode("utf-8")
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def fetch_version_bundle(
    bucket,
    prefix: str,
    skill_name: str,
    version: int,
    record: Optional[dict[str, Any]] = None,
) -> dict[str, bytes]:
    bundle: dict[str, bytes] = {}
    version_record = record or load_version_bundle_record(bucket, prefix, skill_name, version) or {}
    file_entries = version_record.get("files")
    if isinstance(file_entries, list) and file_entries:
        for item in file_entries:
            rel_path = str((item or {}).get("path") or "").strip().replace("\\", "/")
            if not rel_path:
                continue
            key = skill_version_bundle_key(prefix, skill_name, version, rel_path)
            bundle[rel_path] = bucket.get_object(key).read()
        return bundle

    try:
        bundle["SKILL.md"] = bucket.get_object(skill_version_bundle_key(prefix, skill_name, version, "SKILL.md")).read()
    except Exception:
        return {}
    return bundle
