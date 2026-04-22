"""
Skill Hub: shared skill sync via pluggable object storage.

Provides bidirectional sync between local skill directories and a shared
object store, enabling group-wide skill sharing with incremental (sha256-based)
transfers. Default pull behavior mirrors the cloud snapshot into the local
skills directory with backup and rollback safety.

Usage:
    hub = SkillHub(config)
    hub.pull_skills("/path/to/local/skills")   # mirror cloud snapshot locally
    hub.push_skills("/path/to/local/skills")   # upload new/updated skills
    hub.list_remote()                          # list skills on cloud
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Collection, Optional

from evolve_server.core.skill_registry import SkillIDRegistry

from .object_store import build_object_store, is_not_found_error
from .skill_bundle import (
    bundle_entrypoint_bytes,
    bundle_file_records,
    bundle_has_only_entrypoint,
    bundle_tree_sha256,
    read_skill_bundle_with_meta,
    write_skill_bundle,
)

logger = logging.getLogger(__name__)


def _is_hermes_skill_root(skills_dir: str) -> bool:
    return os.path.realpath(skills_dir) == os.path.realpath(os.path.join(os.path.expanduser("~"), ".hermes", "skills"))


def _skill_dir_for_root(skills_dir: str, skill_name: str, category: str = "general") -> str:
    if _is_hermes_skill_root(skills_dir) and category and category != "general":
        return os.path.join(skills_dir, category, skill_name)
    return os.path.join(skills_dir, skill_name)


def _category_from_skill_path(skills_dir: str, skill_md_path: str) -> str:
    if not _is_hermes_skill_root(skills_dir):
        return "general"
    rel = os.path.relpath(skill_md_path, skills_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        return str(parts[0] or "general")
    return "general"


class SkillHub:
    """Sync skills between a local directory and a shared object store."""

    def __init__(
        self,
        *,
        backend: str,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "",
        session_token: str = "",
        local_root: str = "",
        group_id: str = "default",
        user_alias: str = "",
    ):
        self._bucket = build_object_store(
            backend=backend,
            endpoint=endpoint,
            bucket=bucket,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
            session_token=session_token,
            local_root=local_root,
        )
        self._group_id = group_id
        self._user_alias = user_alias or os.environ.get("USER", "anonymous")

    @classmethod
    def from_config(cls, config) -> "SkillHub":
        backend = str(getattr(config, "sharing_backend", "") or "").strip().lower()
        endpoint = str(getattr(config, "sharing_endpoint", "") or "")
        bucket = str(getattr(config, "sharing_bucket", "") or "")
        access_key_id = str(getattr(config, "sharing_access_key_id", "") or "")
        secret_access_key = str(getattr(config, "sharing_secret_access_key", "") or "")
        local_root = str(getattr(config, "sharing_local_root", "") or "")
        return cls(
            backend=backend or ("local" if local_root else "s3" if (bucket or endpoint) else "oss"),
            endpoint=endpoint,
            bucket=bucket,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=str(getattr(config, "sharing_region", "") or ""),
            session_token=str(getattr(config, "sharing_session_token", "") or ""),
            local_root=local_root,
            group_id=getattr(config, "sharing_group_id", "default"),
            user_alias=getattr(config, "sharing_user_alias", ""),
        )

    # ------------------------------------------------------------------ #
    # Remote key helpers                                                   #
    # ------------------------------------------------------------------ #

    def _prefix(self) -> str:
        return f"{self._group_id}/"

    def _manifest_key(self) -> str:
        return f"{self._prefix()}manifest.jsonl"

    def _skill_key(self, skill_name: str) -> str:
        return f"{self._prefix()}skills/{skill_name}/SKILL.md"

    def _skill_files_prefix(self, skill_name: str) -> str:
        return f"{self._prefix()}skills/{skill_name}/files/"

    def _skill_bundle_key(self, skill_name: str, rel_path: str) -> str:
        clean = str(rel_path or "").strip().replace("\\", "/")
        if clean == "SKILL.md":
            return self._skill_key(skill_name)
        return f"{self._skill_files_prefix(skill_name)}{clean}"

    def _iter_remote_keys(self, prefix: str):
        return self._bucket.iter_objects(prefix=prefix)

    def _delete_remote_bundle_extras(self, skill_name: str, keep_paths: Collection[str]) -> None:
        keep_keys = {
            self._skill_bundle_key(skill_name, rel_path)
            for rel_path in keep_paths
            if rel_path != "SKILL.md"
        }
        for obj in self._iter_remote_keys(self._skill_files_prefix(skill_name)):
            key = str(getattr(obj, "key", "") or "")
            if key and key not in keep_keys:
                self._bucket.delete_object(key)

    def _download_skill_bundle(
        self,
        skill_name: str,
        record: dict[str, Any],
    ) -> dict[str, bytes]:
        bundle: dict[str, bytes] = {}
        file_entries = record.get("files")
        if isinstance(file_entries, list) and file_entries:
            for item in file_entries:
                rel_path = str((item or {}).get("path") or "").strip().replace("\\", "/")
                if not rel_path:
                    continue
                key = self._skill_bundle_key(skill_name, rel_path)
                bundle[rel_path] = self._bucket.get_object(key).read()
        else:
            bundle["SKILL.md"] = self._bucket.get_object(self._skill_key(skill_name)).read()
        return bundle

    def _skill_version_prefix(self, skill_name: str, version: int) -> str:
        return f"{self._prefix()}skills/{skill_name}/versions/v{max(1, int(version or 1))}/"

    def _skill_version_bundle_key(self, skill_name: str, version: int, rel_path: str) -> str:
        clean = str(rel_path or "").strip().replace("\\", "/")
        if clean == "SKILL.md":
            return f"{self._skill_version_prefix(skill_name, version)}SKILL.md"
        return f"{self._skill_version_prefix(skill_name, version)}files/{clean}"

    def _skill_version_record_key(self, skill_name: str, version: int) -> str:
        return f"{self._skill_version_prefix(skill_name, version)}bundle.json"

    def _save_version_bundle(self, skill_name: str, version: int, bundle_files: dict[str, bytes]) -> dict[str, Any]:
        record = {
            "format": "bundle_v1",
            "entrypoint": "SKILL.md",
            "tree_sha256": bundle_tree_sha256(bundle_files),
            "files": bundle_file_records(bundle_files),
        }
        keep_keys: set[str] = set()
        for rel_path, data in sorted(bundle_files.items()):
            key = self._skill_version_bundle_key(skill_name, version, rel_path)
            keep_keys.add(key)
            self._bucket.put_object(key, data)
        for obj in self._iter_remote_keys(f"{self._skill_version_prefix(skill_name, version)}files/"):
            key = str(getattr(obj, "key", "") or "")
            if key and key not in keep_keys:
                self._bucket.delete_object(key)
        self._bucket.put_object(
            self._skill_version_record_key(skill_name, version),
            json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return record

    @staticmethod
    def _local_bundle_matches_record(skill_dir: str, record: dict[str, Any]) -> bool:
        bundle, _records, tree_sha = read_skill_bundle_with_meta(skill_dir)
        if not bundle:
            return False
        if record.get("format") == "bundle_v1":
            return str(record.get("tree_sha256") or "") == tree_sha

        try:
            skill_md = bundle_entrypoint_bytes(bundle)
        except Exception:
            return False
        skill_sha = hashlib.sha256(skill_md).hexdigest()
        return bundle_has_only_entrypoint(bundle) and str(record.get("sha256") or "") == skill_sha

    # ------------------------------------------------------------------ #
    # Manifest operations                                                  #
    # ------------------------------------------------------------------ #

    def _load_remote_manifest(self) -> dict[str, dict[str, Any]]:
        """Load manifest.jsonl from storage. Returns {skill_name: record}."""
        key = self._manifest_key()
        try:
            result = self._bucket.get_object(key)
            content = result.read().decode("utf-8")
        except Exception as e:
            if is_not_found_error(e):
                return {}
            logger.warning("[SkillHub] failed to load manifest: %s", e)
            return {}

        manifest: dict[str, dict[str, Any]] = {}
        for line in content.strip().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                name = rec.get("name", "")
                if name:
                    manifest[name] = rec
            except json.JSONDecodeError:
                continue
        return manifest

    def _save_remote_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        """Write the full manifest back to storage."""
        lines = [json.dumps(rec, ensure_ascii=False) for rec in manifest.values()]
        content = "\n".join(lines) + "\n" if lines else ""
        self._bucket.put_object(self._manifest_key(), content.encode("utf-8"))

    # ------------------------------------------------------------------ #
    # Push (local -> cloud)                                                #
    # ------------------------------------------------------------------ #

    def push_skills(
        self,
        skills_dir: str,
        skill_filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        """Upload new/changed skills from local directory to shared storage.

        Parameters
        ----------
        skills_dir:
            Path to the local skills directory.
        skill_filter:
            Optional quality gate. When provided, must contain:
              - ``"stats"``: dict mapping skill_name → stats record
              - ``"min_injections"``: int — skills with fewer injections are
                in a "probation" period and are not uploaded yet
              - ``"min_effectiveness"``: float — skills below this threshold
                after the probation period are blocked from upload

            Skills that have *never* been injected (no stats entry) are
            treated as brand-new and are allowed through (they need cloud
            exposure to gather data from other group members).

        Returns {"uploaded": N, "skipped": M, "filtered": F, "total_local": T}.
        """
        if _is_hermes_skill_root(skills_dir):
            paths = sorted(glob.glob(os.path.join(skills_dir, "**", "SKILL.md"), recursive=True))
        else:
            paths = sorted(glob.glob(os.path.join(skills_dir, "*", "SKILL.md")))
        if not paths:
            logger.info("[SkillHub] no local skills to push")
            return {"uploaded": 0, "skipped": 0, "filtered": 0, "total_local": 0}

        manifest = self._load_remote_manifest()
        registry = SkillIDRegistry()
        registry.load_from_oss(self._bucket, self._prefix())
        uploaded = 0
        skipped = 0
        filtered = 0

        stats = (skill_filter or {}).get("stats", {})
        min_inj = (skill_filter or {}).get("min_injections", 0)
        min_eff = (skill_filter or {}).get("min_effectiveness", 0.0)
        use_filter = skill_filter is not None

        for path in paths:
            skill_name = os.path.basename(os.path.dirname(path))
            skill_dir = os.path.dirname(path)

            if use_filter and skill_name in stats:
                entry = stats[skill_name]
                inj = entry.get("inject_count", 0)
                eff = entry.get("effectiveness", 0.5)
                if inj >= min_inj and eff < min_eff:
                    logger.info(
                        "[SkillHub] filtered out skill %s (effectiveness=%.2f < %.2f, injections=%d)",
                        skill_name,
                        eff,
                        min_eff,
                        inj,
                    )
                    filtered += 1
                    continue

            bundle_files, bundle_records, tree_sha = read_skill_bundle_with_meta(skill_dir)
            skill_md = bundle_entrypoint_bytes(bundle_files)
            local_sha = hashlib.sha256(skill_md).hexdigest()

            remote_rec = manifest.get(skill_name)
            if remote_rec and self._local_bundle_matches_record(skill_dir, remote_rec):
                skipped += 1
                continue

            self._bucket.put_object(self._skill_key(skill_name), skill_md)
            for rel_path, data in sorted(bundle_files.items()):
                if rel_path == "SKILL.md":
                    continue
                self._bucket.put_object(self._skill_bundle_key(skill_name, rel_path), data)
            self._delete_remote_bundle_extras(skill_name, bundle_files.keys())

            bundle_record = {
                "format": "bundle_v1",
                "entrypoint": "SKILL.md",
                "tree_sha256": tree_sha,
                "files": bundle_records,
            }
            version = registry.record_update(
                skill_name,
                local_sha,
                action="push",
                bundle_record=bundle_record,
            )
            self._save_version_bundle(skill_name, version, bundle_files)

            manifest[skill_name] = {
                **(remote_rec or {}),
                "name": skill_name,
                "skill_id": registry.get_or_create(skill_name),
                "version": version,
                "sha256": local_sha,
                "tree_sha256": tree_sha,
                "format": "bundle_v1",
                "entrypoint": "SKILL.md",
                "files": bundle_records,
                "uploaded_by": self._user_alias,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            self._enrich_manifest_entry(manifest[skill_name], path)
            manifest[skill_name].setdefault("category", _category_from_skill_path(skills_dir, path))
            uploaded += 1
            logger.info("[SkillHub] pushed skill: %s", skill_name)

        if uploaded > 0:
            self._save_remote_manifest(manifest)
            registry.save_to_oss(self._bucket, self._prefix())

        logger.info(
            "[SkillHub] push complete: %d uploaded, %d skipped, %d filtered, %d total",
            uploaded,
            skipped,
            filtered,
            len(paths),
        )
        return {"uploaded": uploaded, "skipped": skipped, "filtered": filtered, "total_local": len(paths)}

    @staticmethod
    def _enrich_manifest_entry(entry: dict, skill_path: str) -> None:
        """Parse SKILL.md YAML frontmatter to fill description/category in manifest."""
        try:
            with open(skill_path, encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return

        if not raw.startswith("---"):
            return
        end_idx = raw.find("\n---", 3)
        if end_idx == -1:
            return

        try:
            import yaml

            fm = yaml.safe_load(raw[3:end_idx].strip()) or {}
        except Exception:
            fm = {}

        if not isinstance(fm, dict):
            return

        desc = fm.get("description")
        if desc:
            entry["description"] = str(desc).strip()

        metadata = fm.get("metadata")
        sc_meta = (metadata or {}).get("skillclaw", {}) if isinstance(metadata, dict) else {}
        if isinstance(sc_meta, dict) and sc_meta.get("category"):
            entry["category"] = str(sc_meta["category"]).strip()
        elif fm.get("category"):
            entry["category"] = str(fm["category"]).strip()

    # ------------------------------------------------------------------ #
    # Pull (cloud -> local)                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _list_local_skill_dirs(skills_dir: str) -> dict[str, list[str]]:
        """Return {skill_name: [skill_dir, ...]} for local `<name>/SKILL.md` entries."""
        out: dict[str, list[str]] = {}
        if not os.path.isdir(skills_dir):
            return out
        if _is_hermes_skill_root(skills_dir):
            for path in sorted(glob.glob(os.path.join(skills_dir, "**", "SKILL.md"), recursive=True)):
                skill_dir = os.path.dirname(path)
                name = os.path.basename(skill_dir)
                out.setdefault(name, []).append(skill_dir)
            return out
        for entry in os.scandir(skills_dir):
            if not entry.is_dir():
                continue
            skill_md = os.path.join(entry.path, "SKILL.md")
            if os.path.isfile(skill_md):
                out.setdefault(entry.name, []).append(entry.path)
        return out

    @classmethod
    def _list_local_skills(cls, skills_dir: str) -> dict[str, str]:
        """Return {skill_name: skill_dir} for local `<name>/SKILL.md` entries."""
        grouped = cls._list_local_skill_dirs(skills_dir)
        return {name: dirs[-1] for name, dirs in grouped.items() if dirs}

    @staticmethod
    def _resolve_pull_target_dir(
        skills_dir: str,
        skill_name: str,
        category: str,
        local_dirs_by_name: dict[str, list[str]],
    ) -> str:
        target = _skill_dir_for_root(skills_dir, skill_name, category)
        if not _is_hermes_skill_root(skills_dir):
            return target

        existing_dirs = local_dirs_by_name.get(skill_name) or []
        if not existing_dirs:
            return target

        if str(category or "general").strip() == "general":
            nested = [path for path in existing_dirs if len(os.path.relpath(path, skills_dir).split(os.sep)) >= 2]
            if len(nested) == 1:
                return nested[0]

        target_real = os.path.realpath(target)
        existing_real = [os.path.realpath(path) for path in existing_dirs]
        if target_real in existing_real:
            return target

        return target

    @staticmethod
    def _remove_duplicate_local_skill_dirs(
        skill_name: str,
        keep_dir: str,
        local_dirs_by_name: dict[str, list[str]],
    ) -> None:
        keep_real = os.path.realpath(keep_dir)
        for skill_dir in local_dirs_by_name.get(skill_name) or []:
            if os.path.realpath(skill_dir) == keep_real:
                continue
            if not os.path.isdir(skill_dir):
                continue
            shutil.rmtree(skill_dir)
            logger.info("[SkillHub] removed duplicate local skill dir: %s", skill_dir)

    @staticmethod
    def _prune_backups(backup_root: str, prefix: str, keep: int = 3) -> None:
        """Keep only newest `keep` backups for current skills dir."""
        try:
            names = sorted(n for n in os.listdir(backup_root) if n.startswith(prefix))
        except Exception:
            return
        to_delete = names[:-keep] if keep > 0 else names
        for name in to_delete:
            path = os.path.join(backup_root, name)
            try:
                shutil.rmtree(path)
            except Exception:
                pass

    def pull_skills(
        self,
        skills_dir: str,
        mirror: bool = True,
        skip_names: Optional[Collection[str]] = None,
        include_names: Optional[Collection[str]] = None,
    ) -> dict[str, Any]:
        """Mirror cloud skills to local directory with backup + rollback safety.

        Parameters
        ----------
        skills_dir:
            Local skills directory.
        mirror:
            When ``True`` (default), do full mirror: local stale skill folders
            not present in remote manifest are deleted. When ``False``, perform
            incremental pull only (download/update remote skills, never delete
            local extras).
        skip_names:
            Skill names to preserve from local disk during this pull.
        include_names:
            Optional subset of remote skill names to download. When provided,
            pull is forced into incremental mode to avoid deleting unrelated
            local skills.

        Returns:
            {
                "downloaded": N,
                "skipped": M,
                "deleted": D,
                "total_remote": T,
                "restored_from_backup": bool,
                "backup_dir": "...",
            }
        """
        os.makedirs(skills_dir, exist_ok=True)
        local_dirs_by_name = self._list_local_skill_dirs(skills_dir)
        local_skills = {name: dirs[-1] for name, dirs in local_dirs_by_name.items() if dirs}
        manifest = self._load_remote_manifest()
        skip_set = {str(name or "").strip() for name in (skip_names or []) if str(name or "").strip()}
        include_set = {str(name or "").strip() for name in (include_names or []) if str(name or "").strip()}
        if include_set and mirror:
            mirror = False
        if include_set:
            manifest = {name: rec for name, rec in manifest.items() if name in include_set}
        missing_names = sorted(include_set - set(manifest))
        requested_count = len(include_set)
        matched_remote = len(manifest)

        def _result(
            *,
            downloaded: int,
            skipped: int,
            deleted: int,
            total_remote: int,
            restored_from_backup: bool,
            backup_dir: str,
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "downloaded": downloaded,
                "skipped": skipped,
                "deleted": deleted,
                "total_remote": total_remote,
                "restored_from_backup": restored_from_backup,
                "backup_dir": backup_dir,
            }
            if include_set:
                payload.update(
                    {
                        "requested": requested_count,
                        "matched_remote": matched_remote,
                        "missing": len(missing_names),
                        "missing_names": missing_names,
                    }
                )
            return payload

        if not manifest:
            # Empty/failed manifest is treated as no-op to avoid accidental wipe.
            if include_set:
                logger.info(
                    "[SkillHub] none of the requested remote skills matched the manifest: %s",
                    ", ".join(missing_names) or "(empty request)",
                )
            else:
                logger.warning("[SkillHub] remote manifest empty; skip mirror pull (downloaded=0 skipped=0 deleted=0)")
            return _result(
                downloaded=0,
                skipped=0,
                deleted=0,
                total_remote=0,
                restored_from_backup=False,
                backup_dir="",
            )

        downloaded = 0
        skipped = 0
        deleted = 0
        restored_from_backup = False

        if not mirror:
            for name, rec in manifest.items():
                category = str(rec.get("category", "general") or "general")
                local_dir = self._resolve_pull_target_dir(
                    skills_dir,
                    name,
                    category,
                    local_dirs_by_name,
                )
                local_path = os.path.join(local_dir, "SKILL.md")

                if name in skip_set and os.path.exists(local_path):
                    skipped += 1
                    self._remove_duplicate_local_skill_dirs(name, local_dir, local_dirs_by_name)
                    logger.info("[SkillHub] preserved local skill during pull: %s", name)
                    continue

                if os.path.isdir(local_dir) and self._local_bundle_matches_record(local_dir, rec):
                    skipped += 1
                    self._remove_duplicate_local_skill_dirs(name, local_dir, local_dirs_by_name)
                    continue

                try:
                    bundle = self._download_skill_bundle(name, rec)
                except Exception as e:
                    logger.warning("[SkillHub] failed to download skill %s: %s", name, e)
                    continue

                write_skill_bundle(local_dir, bundle, clean=True)
                downloaded += 1
                self._remove_duplicate_local_skill_dirs(name, local_dir, local_dirs_by_name)
                logger.info("[SkillHub] pulled skill: %s", name)

            logger.info(
                "[SkillHub] incremental pull complete: %d downloaded, %d skipped, %d total remote",
                downloaded,
                skipped,
                len(manifest),
            )
            return _result(
                downloaded=downloaded,
                skipped=skipped,
                deleted=0,
                total_remote=len(manifest),
                restored_from_backup=False,
                backup_dir="",
            )

        parent_dir = os.path.dirname(os.path.abspath(skills_dir))
        base_name = os.path.basename(os.path.abspath(skills_dir))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_root = os.path.join(parent_dir, ".skillclaw_backups")
        os.makedirs(backup_root, exist_ok=True)
        backup_prefix = f"{base_name}_"
        backup_dir = os.path.join(backup_root, f"{backup_prefix}{stamp}")
        staging_dir = os.path.join(parent_dir, f".skillclaw_pull_stage_{base_name}_{stamp}")

        try:
            shutil.copytree(skills_dir, backup_dir)
        except Exception as e:
            logger.warning("[SkillHub] backup before pull failed: %s", e)
            return _result(
                downloaded=0,
                skipped=0,
                deleted=0,
                total_remote=len(manifest),
                restored_from_backup=False,
                backup_dir="",
            )

        os.makedirs(staging_dir, exist_ok=True)
        resolved_targets: dict[str, str] = {}

        try:
            for name, rec in manifest.items():
                category = str(rec.get("category", "general") or "general")
                target_dir = self._resolve_pull_target_dir(
                    skills_dir,
                    name,
                    category,
                    local_dirs_by_name,
                )
                resolved_targets[name] = target_dir
                local_path = os.path.join(target_dir, "SKILL.md")
                staged_dir = os.path.join(staging_dir, os.path.relpath(target_dir, skills_dir))

                if name in skip_set and os.path.exists(local_path):
                    skipped += 1
                    if os.path.isdir(target_dir):
                        shutil.copytree(target_dir, staged_dir, dirs_exist_ok=True)
                    logger.info("[SkillHub] preserved local skill during pull: %s", name)
                    continue

                if os.path.isdir(target_dir) and self._local_bundle_matches_record(target_dir, rec):
                    skipped += 1
                    shutil.copytree(target_dir, staged_dir, dirs_exist_ok=True)
                    continue

                try:
                    bundle = self._download_skill_bundle(name, rec)
                except Exception as e:
                    raise RuntimeError(f"failed to download skill {name}: {e}") from e

                write_skill_bundle(staged_dir, bundle, clean=True)
                downloaded += 1
                logger.info("[SkillHub] pulled skill: %s", name)

            remote_names = set(manifest.keys())
            local_names = set(local_skills.keys())
            for stale in sorted(local_names - remote_names):
                shutil.rmtree(local_skills[stale], ignore_errors=False)
                deleted += 1

            for name in sorted(remote_names):
                rec = manifest.get(name, {})
                category = str(rec.get("category", "general") or "general")
                dst_dir = resolved_targets.get(name) or self._resolve_pull_target_dir(
                    skills_dir,
                    name,
                    category,
                    local_dirs_by_name,
                )
                src_dir = os.path.join(staging_dir, os.path.relpath(dst_dir, skills_dir))
                if os.path.isdir(dst_dir):
                    shutil.rmtree(dst_dir)
                os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                shutil.move(src_dir, dst_dir)
                self._remove_duplicate_local_skill_dirs(name, dst_dir, local_dirs_by_name)

        except Exception as e:
            logger.warning("[SkillHub] mirror pull failed, restoring backup: %s", e)
            try:
                if os.path.isdir(skills_dir):
                    shutil.rmtree(skills_dir)
                shutil.copytree(backup_dir, skills_dir)
                restored_from_backup = True
                logger.info("[SkillHub] local skills restored from backup: %s", backup_dir)
            except Exception as restore_err:
                logger.error("[SkillHub] backup restore failed: %s", restore_err)

            return _result(
                downloaded=0,
                skipped=0,
                deleted=0,
                total_remote=len(manifest),
                restored_from_backup=restored_from_backup,
                backup_dir=backup_dir,
            )
        finally:
            if os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)

        logger.info(
            "[SkillHub] pull complete: %d downloaded, %d skipped, %d deleted, %d total remote",
            downloaded,
            skipped,
            deleted,
            len(manifest),
        )
        self._prune_backups(backup_root, backup_prefix, keep=3)
        return _result(
            downloaded=downloaded,
            skipped=skipped,
            deleted=deleted,
            total_remote=len(manifest),
            restored_from_backup=False,
            backup_dir=backup_dir,
        )

    # ------------------------------------------------------------------ #
    # List remote skills                                                   #
    # ------------------------------------------------------------------ #

    def list_remote(self) -> list[dict[str, Any]]:
        """Return a list of skill metadata dicts from the remote manifest."""
        manifest = self._load_remote_manifest()
        return list(manifest.values())

    # ------------------------------------------------------------------ #
    # Sync (pull then push)                                                #
    # ------------------------------------------------------------------ #

    def sync_skills(self, skills_dir: str) -> dict[str, dict[str, Any]]:
        """Bidirectional sync: pull first, then push."""
        # Sync keeps bidirectional semantics: pull updates from cloud without
        # deleting local-only skills, then push local deltas to cloud.
        pull_result = self.pull_skills(skills_dir, mirror=False)
        push_result = self.push_skills(skills_dir)
        return {"pull": pull_result, "push": push_result}
