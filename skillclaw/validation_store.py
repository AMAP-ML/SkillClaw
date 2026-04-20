"""
Shared storage helpers for distributed client-side validation.

The validation flow uses the same object store boundary as the rest of
SkillClaw. Jobs are produced by the evolve server, validated by opted-in
clients, and later finalized by the evolve server.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from evolve_server.core.utils import build_skill_md

from .object_store import build_object_store, is_not_found_error

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ValidationStore:
    """Persist validation jobs/results/decisions in shared storage."""

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
    ) -> None:
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

    @classmethod
    def from_config(cls, config) -> "ValidationStore":
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
            group_id=str(getattr(config, "sharing_group_id", "default") or "default"),
        )

    def _prefix(self) -> str:
        return f"{self._group_id}/"

    def _job_key(self, job_id: str) -> str:
        return f"{self._prefix()}validation_jobs/{job_id}.json"

    def _candidate_skill_key(self, job_id: str) -> str:
        return f"{self._prefix()}candidate_skills/{job_id}/SKILL.md"

    def _result_key(self, job_id: str, user_alias: str) -> str:
        return f"{self._prefix()}validation_results/{job_id}/{user_alias}.json"

    def _decision_key(self, job_id: str) -> str:
        return f"{self._prefix()}validation_decisions/{job_id}.json"

    def make_job_id(self, skill_name: str) -> str:
        slug = str(skill_name or "candidate").strip().lower().replace("_", "-")
        slug = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in slug).strip("-") or "candidate"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{timestamp}-{slug}-{uuid.uuid4().hex[:8]}"

    def save_job(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("job_id", "") or "")
        if not job_id:
            raise ValueError("validation job requires job_id")
        payload = dict(job)
        payload.setdefault("created_at", _utc_now_iso())
        self._bucket.put_object(
            self._job_key(job_id),
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        candidate_skill = payload.get("candidate_skill")
        if isinstance(candidate_skill, dict) and candidate_skill.get("name"):
            self._bucket.put_object(
                self._candidate_skill_key(job_id),
                build_skill_md(candidate_skill).encode("utf-8"),
            )

    def load_job(self, job_id: str) -> Optional[dict[str, Any]]:
        try:
            return json.loads(self._bucket.get_object(self._job_key(job_id)).read().decode("utf-8"))
        except Exception as exc:
            if not is_not_found_error(exc):
                logger.warning("[ValidationStore] failed to load job %s: %s", job_id, exc)
            return None

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        prefix = f"{self._prefix()}validation_jobs/"
        for obj in self._bucket.iter_objects(prefix=prefix):
            if not obj.key.endswith(".json"):
                continue
            try:
                jobs.append(json.loads(self._bucket.get_object(obj.key).read().decode("utf-8")))
            except Exception as exc:
                logger.warning("[ValidationStore] failed to parse %s: %s", obj.key, exc)
        jobs.sort(key=lambda item: str(item.get("created_at", "")))
        return jobs

    def save_result(self, job_id: str, user_alias: str, result: dict[str, Any]) -> None:
        payload = dict(result)
        payload["job_id"] = job_id
        payload["user_alias"] = user_alias
        payload.setdefault("created_at", _utc_now_iso())
        self._bucket.put_object(
            self._result_key(job_id, user_alias),
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    def load_result(self, job_id: str, user_alias: str) -> Optional[dict[str, Any]]:
        try:
            return json.loads(self._bucket.get_object(self._result_key(job_id, user_alias)).read().decode("utf-8"))
        except Exception as exc:
            if not is_not_found_error(exc):
                logger.warning(
                    "[ValidationStore] failed to load result for %s/%s: %s",
                    job_id,
                    user_alias,
                    exc,
                )
            return None

    def list_results(self, job_id: str) -> list[dict[str, Any]]:
        prefix = f"{self._prefix()}validation_results/{job_id}/"
        results: list[dict[str, Any]] = []
        for obj in self._bucket.iter_objects(prefix=prefix):
            if not obj.key.endswith(".json"):
                continue
            try:
                results.append(json.loads(self._bucket.get_object(obj.key).read().decode("utf-8")))
            except Exception as exc:
                logger.warning("[ValidationStore] failed to parse %s: %s", obj.key, exc)
        results.sort(key=lambda item: (str(item.get("created_at", "")), str(item.get("user_alias", ""))))
        return results

    def save_decision(self, job_id: str, decision: dict[str, Any]) -> None:
        payload = dict(decision)
        payload["job_id"] = job_id
        payload.setdefault("decided_at", _utc_now_iso())
        self._bucket.put_object(
            self._decision_key(job_id),
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    def load_decision(self, job_id: str) -> Optional[dict[str, Any]]:
        try:
            return json.loads(self._bucket.get_object(self._decision_key(job_id)).read().decode("utf-8"))
        except Exception as exc:
            if not is_not_found_error(exc):
                logger.warning("[ValidationStore] failed to load decision %s: %s", job_id, exc)
            return None

    def list_open_jobs(self, *, user_alias: str = "") -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for job in self.list_jobs():
            job_id = str(job.get("job_id", "") or "")
            if not job_id:
                continue
            if self.load_decision(job_id):
                continue
            if user_alias and self.load_result(job_id, user_alias):
                continue
            jobs.append(job)
        return jobs
