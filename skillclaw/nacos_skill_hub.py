"""Nacos-backed skill sharing for SkillClaw.

This adapter intentionally uses Nacos' skill lifecycle API instead of mapping
Nacos onto the generic object-store protocol. Local skills remain normal
directories; Nacos owns remote versions, review submission, publish, labels,
and online state.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Optional

import httpx

from .skill_bundle import (
    bundle_entrypoint_text,
    bundle_file_records,
    bundle_tree_sha256,
    read_skill_bundle_with_meta,
    write_skill_bundle,
)
from .skill_hub import _is_hermes_skill_root, _skill_dir_for_root

logger = logging.getLogger(__name__)


class NacosSkillClient:
    """Small synchronous client for Nacos Skill admin/client endpoints."""

    def __init__(
        self,
        *,
        server: str,
        namespace_id: str = "public",
        access_token: str = "",
        username: str = "",
        password: str = "",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.server = str(server or "").rstrip("/")
        if not self.server:
            raise ValueError("Nacos sharing requires sharing.nacos_server or sharing.endpoint.")
        self.namespace_id = str(namespace_id or "public")
        self.access_token = access_token
        self.username = username
        self.password = password
        self._client = httpx.Client(timeout=timeout, transport=transport)
        if not self.access_token and self.username and self.password:
            self._login()

    def close(self) -> None:
        self._client.close()

    def _login(self) -> None:
        response = self._client.post(
            f"{self.server}/v1/auth/login",
            data={"username": self.username, "password": self.password},
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("accessToken") or payload.get("data", {}).get("accessToken", "")
        if not token:
            raise RuntimeError(f"Nacos login failed: no accessToken in response: {payload}")
        self.access_token = token
        logger.info("[NacosSkillClient] logged in as %s", self.username)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.access_token:
            headers["accessToken"] = self.access_token
        return headers

    def _url(self, path: str) -> str:
        return f"{self.server}{path}"

    @staticmethod
    def _unwrap_json(response: httpx.Response) -> Any:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return payload
        if "data" in payload:
            code = payload.get("code")
            if code not in (None, 0, 200, "0", "200"):
                raise RuntimeError(payload.get("message") or payload.get("msg") or f"Nacos request failed: {payload}")
            return payload.get("data")
        return payload

    def list_skills(self, *, page_size: int = 100) -> list[dict[str, Any]]:
        page_no = 1
        out: list[dict[str, Any]] = []
        while True:
            data = self._unwrap_json(
                self._client.get(
                    self._url("/v3/admin/ai/skills/list"),
                    params={
                        "namespaceId": self.namespace_id,
                        "pageNo": page_no,
                        "pageSize": page_size,
                    },
                    headers=self._headers(),
                )
            )
            if isinstance(data, list):
                out.extend(item for item in data if isinstance(item, dict))
                return out
            if not isinstance(data, dict):
                return out
            items = data.get("pageItems") or data.get("items") or []
            out.extend(item for item in items if isinstance(item, dict))
            total = int(data.get("totalCount") or len(out) or 0)
            if len(out) >= total or not items:
                return out
            page_no += 1

    def get_skill(self, name: str) -> dict[str, Any]:
        data = self._unwrap_json(
            self._client.get(
                self._url("/v3/admin/ai/skills"),
                params={"namespaceId": self.namespace_id, "skillName": name},
                headers=self._headers(),
            )
        )
        return data if isinstance(data, dict) else {}

    def download_skill_zip(
        self,
        name: str,
        *,
        version: str | None = None,
        label: str = "latest",
        admin: bool = False,
    ) -> bytes:
        if admin and version:
            response = self._client.get(
                self._url("/v3/admin/ai/skills/version/download"),
                params={"namespaceId": self.namespace_id, "skillName": name, "version": version},
                headers=self._headers(),
            )
        else:
            params = {"namespaceId": self.namespace_id, "name": name}
            if version:
                params["version"] = version
            elif label:
                params["label"] = label
            response = self._client.get(
                self._url("/v3/client/ai/skills"),
                params=params,
                headers=self._headers(),
            )
        response.raise_for_status()
        return response.content

    def upload_skill_zip(
        self,
        *,
        zip_bytes: bytes,
        filename: str,
        overwrite: bool,
        target_version: str | None,
    ) -> str:
        data: dict[str, str] = {
            "namespaceId": self.namespace_id,
            "overwrite": "true" if overwrite else "false",
        }
        if target_version:
            data["targetVersion"] = target_version
        response = self._client.post(
            self._url("/v3/admin/ai/skills/upload"),
            data=data,
            files={"file": (filename, zip_bytes, "application/zip")},
            headers=self._headers(),
        )
        result = self._unwrap_json(response)
        return str(result or "")

    def submit(self, name: str, version: str) -> str:
        result = self._unwrap_json(
            self._client.post(
                self._url("/v3/admin/ai/skills/submit"),
                data={"namespaceId": self.namespace_id, "skillName": name, "version": version},
                headers=self._headers(),
            )
        )
        return str(result or "")

    def publish(self, name: str, version: str, *, update_latest_label: bool = True) -> None:
        self._unwrap_json(
            self._client.post(
                self._url("/v3/admin/ai/skills/publish"),
                data={
                    "namespaceId": self.namespace_id,
                    "skillName": name,
                    "version": version,
                    "updateLatestLabel": "true" if update_latest_label else "false",
                },
                headers=self._headers(),
            )
        )


def _bundle_to_nacos_zip(skill_name: str, bundle_files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path, data in sorted(bundle_files.items()):
            zf.writestr(f"{skill_name}/{rel_path}", data)
    return buf.getvalue()


def _nacos_zip_to_bundle(zip_bytes: bytes) -> dict[str, bytes]:
    bundle: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = [name for name in zf.namelist() if name and not name.endswith("/")]
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        strip_root = len(roots) == 1 and all("/" in name for name in names)
        for name in names:
            rel = name.split("/", 1)[1] if strip_root else name
            if not rel:
                continue
            bundle[rel] = zf.read(name)
    return bundle


def _parse_version_number(version: str | None) -> int:
    raw = str(version or "").strip()
    if raw.startswith("v"):
        raw = raw[1:]
    try:
        return int(raw)
    except ValueError:
        return 0


def _next_version(summary: dict[str, Any], detail: dict[str, Any] | None = None) -> str:
    versions: list[str] = []
    labels = summary.get("labels")
    if isinstance(labels, dict):
        versions.extend(str(v) for v in labels.values() if v)
    for field in ("editingVersion", "reviewingVersion", "version"):
        if summary.get(field):
            versions.append(str(summary[field]))
    for item in (detail or {}).get("versions") or []:
        if isinstance(item, dict) and item.get("version"):
            versions.append(str(item["version"]))
    next_num = max([_parse_version_number(v) for v in versions] or [0]) + 1
    return f"v{next_num}"


class NacosSkillHub:
    """SkillHub-compatible facade backed by Nacos Skill APIs."""

    def __init__(
        self,
        *,
        client: NacosSkillClient,
        user_alias: str = "",
        label: str = "latest",
    ) -> None:
        self._client = client
        self._user_alias = user_alias or os.environ.get("USER", "anonymous")
        self._label = str(label or "latest")

    @classmethod
    def from_config(cls, config) -> "NacosSkillHub":
        client = NacosSkillClient(
            server=str(getattr(config, "sharing_nacos_server", "") or getattr(config, "sharing_endpoint", "") or ""),
            namespace_id=str(getattr(config, "sharing_nacos_namespace_id", "") or "public"),
            access_token=str(getattr(config, "sharing_nacos_access_token", "") or ""),
            username=str(getattr(config, "sharing_nacos_username", "") or ""),
            password=str(getattr(config, "sharing_nacos_password", "") or ""),
        )
        return cls(
            client=client,
            user_alias=str(getattr(config, "sharing_user_alias", "") or ""),
            label=str(getattr(config, "sharing_nacos_label", "") or "latest"),
        )

    def _remote_by_name(self) -> dict[str, dict[str, Any]]:
        return {str(item.get("name") or ""): item for item in self._client.list_skills() if item.get("name")}

    @staticmethod
    def _local_bundle_matches_remote(skill_dir: str, remote_bundle: dict[str, bytes]) -> bool:
        local_bundle, _records, local_sha = read_skill_bundle_with_meta(skill_dir)
        return bool(local_bundle) and local_sha == bundle_tree_sha256(remote_bundle)

    def _download_skill_bundle(self, name: str, rec: dict[str, Any]) -> dict[str, bytes]:
        labels = rec.get("labels") if isinstance(rec.get("labels"), dict) else {}
        version = labels.get(self._label)
        zip_bytes = self._client.download_skill_zip(name, version=version, label=self._label)
        return _nacos_zip_to_bundle(zip_bytes)

    def list_remote(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._client.list_skills():
            labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
            out.append({
                **item,
                "version": labels.get(self._label) or item.get("version") or item.get("reviewingVersion") or "",
                "uploaded_by": item.get("owner") or item.get("from") or "nacos",
                "uploaded_at": item.get("updatedAt") or item.get("updateTime") or "",
                "category": item.get("category") or "general",
            })
        return out

    def push_skills(
        self,
        skills_dir: str,
        skill_filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if _is_hermes_skill_root(skills_dir):
            paths = sorted(Path(skills_dir).glob("**/SKILL.md"))
        else:
            paths = sorted(Path(skills_dir).glob("*/SKILL.md"))
        if not paths:
            return {"uploaded": 0, "skipped": 0, "filtered": 0, "submitted": 0, "published": 0, "total_local": 0}

        remote = self._remote_by_name()
        stats = (skill_filter or {}).get("stats", {})
        min_inj = (skill_filter or {}).get("min_injections", 0)
        min_eff = (skill_filter or {}).get("min_effectiveness", 0.0)
        use_filter = skill_filter is not None
        uploaded = skipped = filtered = submitted = 0

        for path in paths:
            skill_name = path.parent.name
            skill_dir = str(path.parent)
            if use_filter and skill_name in stats:
                entry = stats[skill_name]
                inj = entry.get("inject_count", 0)
                eff = entry.get("effectiveness", 0.5)
                if inj >= min_inj and eff < min_eff:
                    filtered += 1
                    continue

            bundle_files, _bundle_records, _tree_sha = read_skill_bundle_with_meta(skill_dir)
            remote_rec = remote.get(skill_name)
            if remote_rec:
                try:
                    remote_bundle = self._download_skill_bundle(skill_name, remote_rec)
                    if self._local_bundle_matches_remote(skill_dir, remote_bundle):
                        skipped += 1
                        continue
                except Exception as exc:
                    logger.info("[NacosSkillHub] remote comparison skipped for %s: %s", skill_name, exc)

            detail = self._client.get_skill(skill_name) if remote_rec else {}
            target_version = _next_version(remote_rec or {}, detail)
            zip_bytes = _bundle_to_nacos_zip(skill_name, bundle_files)
            self._client.upload_skill_zip(
                zip_bytes=zip_bytes,
                filename=f"{skill_name}-{target_version}.zip",
                overwrite=True,
                target_version=target_version,
            )
            uploaded += 1

            self._client.submit(skill_name, target_version)
            submitted += 1

            logger.info("[NacosSkillHub] pushed skill %s as %s", skill_name, target_version)

        return {
            "uploaded": uploaded,
            "skipped": skipped,
            "filtered": filtered,
            "submitted": submitted,
            "published": 0,
            "total_local": len(paths),
        }

    def publish_skill(self, name: str, version: str, *, update_latest_label: bool = True) -> dict[str, Any]:
        self._client.publish(name, version, update_latest_label=update_latest_label)
        return {
            "skill_name": name,
            "version": version,
            "published": 1,
            "updated_latest_label": bool(update_latest_label),
        }

    def pull_skills(
        self,
        skills_dir: str,
        mirror: bool = True,
        skip_names: Optional[Collection[str]] = None,
        include_names: Optional[Collection[str]] = None,
    ) -> dict[str, Any]:
        os.makedirs(skills_dir, exist_ok=True)
        remote = self._remote_by_name()
        include_set = {str(name or "").strip() for name in (include_names or []) if str(name or "").strip()}
        skip_set = {str(name or "").strip() for name in (skip_names or []) if str(name or "").strip()}
        if include_set:
            remote = {name: rec for name, rec in remote.items() if name in include_set}
            mirror = False
        if not remote:
            return {
                "downloaded": 0,
                "skipped": 0,
                "deleted": 0,
                "total_remote": 0,
                "restored_from_backup": False,
                "backup_dir": "",
            }

        local_dirs_by_name = {}
        for skill_md in Path(skills_dir).glob("**/SKILL.md" if _is_hermes_skill_root(skills_dir) else "*/SKILL.md"):
            local_dirs_by_name.setdefault(skill_md.parent.name, []).append(str(skill_md.parent))

        downloaded = skipped = deleted = failed = 0
        failed_names: list[str] = []
        for name, rec in remote.items():
            category = str(rec.get("category", "general") or "general")
            target_dir = _skill_dir_for_root(skills_dir, name, category)
            if name in skip_set and os.path.exists(os.path.join(target_dir, "SKILL.md")):
                skipped += 1
                continue
            try:
                bundle = self._download_skill_bundle(name, rec)
            except Exception as exc:
                failed += 1
                failed_names.append(name)
                logger.warning("[NacosSkillHub] failed to download skill %s: %s", name, exc)
                continue
            if os.path.isdir(target_dir) and self._local_bundle_matches_remote(target_dir, bundle):
                skipped += 1
                continue
            write_skill_bundle(target_dir, bundle, clean=True)
            downloaded += 1

        if mirror:
            for name, dirs in local_dirs_by_name.items():
                if name in remote:
                    continue
                for skill_dir in dirs:
                    shutil.rmtree(skill_dir)
                    deleted += 1

        return {
            "downloaded": downloaded,
            "skipped": skipped,
            "deleted": deleted,
            "failed": failed,
            "failed_names": failed_names,
            "total_remote": len(remote),
            "restored_from_backup": False,
            "backup_dir": "",
        }

    def sync_skills(self, skills_dir: str) -> dict[str, dict[str, Any]]:
        pull_result = self.pull_skills(skills_dir, mirror=False)
        push_result = self.push_skills(skills_dir)
        return {"pull": pull_result, "push": push_result}

    def build_manifest_record(self, skill_name: str, bundle_files: dict[str, bytes]) -> dict[str, Any]:
        skill_md = bundle_entrypoint_text(bundle_files)
        content_sha = hashlib.sha256(skill_md.encode("utf-8")).hexdigest()
        return {
            "name": skill_name,
            "sha256": content_sha,
            "tree_sha256": bundle_tree_sha256(bundle_files),
            "format": "bundle_v1",
            "entrypoint": "SKILL.md",
            "files": bundle_file_records(bundle_files),
            "uploaded_by": self._user_alias,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
