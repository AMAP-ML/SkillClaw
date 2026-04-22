"""
Core orchestrator for the current session-level evolve_server pipeline.

Active flow:
1. Drain pending sessions from shared storage.
2. Summarize sessions and extract metadata.
3. Optionally backfill a session-level score with session_judge.
4. Aggregate sessions by referenced skill.
5. Evolve existing-skill groups or create new skills from no-skill groups.
6. Upload skills, persist registry state, and ack processed sessions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from skillclaw.object_store import build_object_store
from skillclaw.skill_bundle import bundle_tree_sha256
from skillclaw.validation_store import ValidationStore

from ..core.config import EvolveServerConfig
from ..core.constants import NO_SKILL_KEY, SLUG_RE, DecisionAction
from ..core.llm_client import AsyncLLMClient
from ..core.skill_registry import SkillIDRegistry
from ..core.utils import build_skill_md, parse_skill_content
from ..pipeline.aggregation import aggregate_sessions_by_skill
from ..pipeline.execution import (
    create_skill_from_sessions,
    evolve_skill_from_sessions,
    execute_merge,
    set_evolve_debug_dir,
)
from ..pipeline.session_judge import judge_sessions_parallel
from ..pipeline.skill_verifier import verify_skill_candidate
from ..pipeline.summarizer import set_summarizer_debug_dir, summarize_sessions_parallel
from ..storage.oss_helpers import (
    delete_session_keys,
    fetch_skill_content,
    list_session_keys,
    load_manifest,
    read_json_object,
    save_manifest,
    save_version_bundle,
)

logger = logging.getLogger(__name__)


class EvolveServer:
    """Session-level evolve server backed by shared object storage."""

    def __init__(
        self,
        config: EvolveServerConfig,
        *,
        mock: bool = False,
        mock_root: str | None = None,
    ) -> None:
        self.config = config
        self._mock = mock
        self._bucket = self._build_bucket(mock=mock, mock_root=mock_root)
        self._prefix = f"{config.group_id}/"
        self._llm = AsyncLLMClient(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
        self._validation_store = ValidationStore(
            backend=self.config.storage_backend,
            endpoint=self.config.storage_endpoint,
            bucket=self.config.storage_bucket,
            access_key_id=self.config.storage_access_key_id,
            secret_access_key=self.config.storage_secret_access_key,
            region=self.config.storage_region,
            session_token=self.config.storage_session_token,
            local_root=self.config.local_root,
            group_id=self.config.group_id,
        )
        self._id_registry = SkillIDRegistry()
        self._running = False

        set_evolve_debug_dir(config.debug_dump_dir)
        set_summarizer_debug_dir(config.debug_dump_dir)
        self._id_registry.load_from_oss(self._bucket, self._prefix)

    def _build_bucket(self, *, mock: bool, mock_root: str | None):
        if mock:
            from ..storage.mock_bucket import LocalBucket

            return LocalBucket(root=mock_root)
        return build_object_store(
            backend=self.config.storage_backend,
            endpoint=self.config.storage_endpoint,
            bucket=self.config.storage_bucket,
            access_key_id=self.config.storage_access_key_id,
            secret_access_key=self.config.storage_secret_access_key,
            region=self.config.storage_region,
            session_token=self.config.storage_session_token,
            local_root=self.config.local_root,
        )

    def _uses_local_storage(self) -> bool:
        backend = str(self.config.storage_backend or "").strip().lower()
        if backend == "local" or self._mock:
            return True
        bucket_type = type(self._bucket).__name__.lower()
        return "local" in bucket_type and bool(self.config.local_root)

    async def _call_storage(self, func, *args):
        if self._uses_local_storage():
            return func(*args)
        return await asyncio.to_thread(func, *args)

    def _append_history(self, record: dict) -> None:
        path = self.config.history_path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[EvolveServer] history write failed: %s", exc)

    async def _drain_sessions(self) -> tuple[list[dict], list[str]]:
        keys = await self._call_storage(list_session_keys, self._bucket, self._prefix)
        sessions: list[dict] = []
        consumed_keys: list[str] = []
        for key in keys:
            session = await self._call_storage(read_json_object, self._bucket, key)
            if session:
                sessions.append(session)
                consumed_keys.append(key)
        logger.info(
            "[EvolveServer] drained %d session(s) from queue (%d keys found)",
            len(sessions),
            len(keys),
        )
        return sessions, consumed_keys

    def _load_remote_skills(self) -> dict[str, dict[str, Any]]:
        return load_manifest(self._bucket, self._prefix)

    def _load_remote_skill_record(self, name: str) -> Optional[dict[str, Any]]:
        rec = self._load_remote_skills().get(name)
        return rec if isinstance(rec, dict) else None

    @staticmethod
    def _overlay_manifest_metadata(
        skill: Optional[dict[str, Any]],
        manifest_record: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not skill or not manifest_record:
            return skill
        category = str(manifest_record.get("category", "") or "").strip()
        if category and str(skill.get("category", "general") or "general").strip() == "general":
            skill["category"] = category
        if not str(skill.get("description", "") or "").strip():
            description = str(manifest_record.get("description", "") or "").strip()
            if description:
                skill["description"] = description
        return skill

    def _fetch_skill(self, name: str) -> Optional[str]:
        return fetch_skill_content(self._bucket, self._prefix, name)

    def _upload_skill(self, skill: dict, action: str) -> None:
        name = skill.get("name", "")
        if not name:
            return

        skill_id = self._id_registry.get_or_create(name)
        md_content = build_skill_md(skill)
        object_key = f"{self._prefix}skills/{name}/SKILL.md"
        md_bytes = md_content.encode("utf-8")
        self._bucket.put_object(object_key, md_bytes)

        content_sha = hashlib.sha256(md_bytes).hexdigest()
        tree_sha = bundle_tree_sha256({"SKILL.md": md_bytes})
        bundle_record = {
            "format": "bundle_v1",
            "entrypoint": "SKILL.md",
            "tree_sha256": tree_sha,
            "files": [{"path": "SKILL.md", "sha256": content_sha, "size": len(md_bytes)}],
        }
        version = self._id_registry.record_update(
            name,
            content_sha,
            action=action,
            bundle_record=bundle_record,
        )
        save_version_bundle(self._bucket, self._prefix, name, version, {"SKILL.md": md_bytes})

        manifest = self._load_remote_skills()
        manifest[name] = {
            "name": name,
            "skill_id": skill_id,
            "version": version,
            "sha256": content_sha,
            "tree_sha256": tree_sha,
            "format": "bundle_v1",
            "entrypoint": "SKILL.md",
            "files": bundle_record["files"],
            "uploaded_by": "evolve_server",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "description": skill.get("description", ""),
            "category": skill.get("category", "general"),
        }
        save_manifest(self._bucket, self._prefix, manifest)
        logger.info(
            "[EvolveServer] uploaded skill %s (id=%s, v%d) to %s",
            name,
            skill_id,
            version,
            object_key,
        )

    def _detect_conflict(self, name: str, incoming_skill: dict) -> bool:
        existing_sha = self._id_registry.get_content_sha(name)
        if not existing_sha:
            return False
        incoming_md = build_skill_md(incoming_skill)
        incoming_sha = hashlib.sha256(incoming_md.encode("utf-8")).hexdigest()
        return existing_sha != incoming_sha

    async def _resolve_and_upload(self, skill: dict, action_type: str) -> str:
        name = skill.get("name", "")
        has_conflict = await self._call_storage(self._detect_conflict, name, skill)
        if not has_conflict:
            await self._call_storage(self._upload_skill, skill, action_type)
            return action_type

        logger.info("[EvolveServer] conflict detected for '%s' - merging", name)
        existing_md = await self._call_storage(self._fetch_skill, name)
        if not existing_md:
            await self._call_storage(self._upload_skill, skill, action_type)
            return action_type

        existing_skill = parse_skill_content(name, existing_md)
        existing_skill = self._overlay_manifest_metadata(
            existing_skill,
            self._load_remote_skill_record(name),
        )
        existing_skill["_version"] = self._id_registry.get_version(name)
        merged = await execute_merge(self._llm, existing_skill, skill)
        if merged and merged.get("name"):
            merged["name"] = name
            await self._call_storage(self._upload_skill, merged, "merge")
            return "merge"

        logger.warning("[EvolveServer] merge failed for '%s' - keeping incoming version", name)
        await self._call_storage(self._upload_skill, skill, action_type)
        return action_type

    def _empty_judge_summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.use_session_judge),
            "judged_sessions": 0,
            "scored_sessions": 0,
            "mean_score": None,
            "min_score": None,
            "max_score": None,
        }

    async def _run_session_judge(self, sessions: list[dict]) -> dict[str, Any]:
        summary = self._empty_judge_summary()
        if not self.config.use_session_judge or not sessions:
            return summary

        judged = await judge_sessions_parallel(self._llm, sessions)
        scores = [
            float(judge_scores["overall_score"])
            for session in sessions
            for judge_scores in [session.get("_judge_scores")]
            if isinstance(judge_scores, dict) and isinstance(judge_scores.get("overall_score"), (int, float))
        ]
        summary["judged_sessions"] = judged
        summary["scored_sessions"] = len(scores)
        if scores:
            summary["mean_score"] = round(sum(scores) / len(scores), 3)
            summary["min_score"] = round(min(scores), 3)
            summary["max_score"] = round(max(scores), 3)
        logger.info("[EvolveServer] judged %d sessions without benchmark scores", judged)
        return summary

    def _empty_skill_verifier_summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.use_skill_verifier),
            "verified_skills": 0,
            "accepted": 0,
            "rejected": 0,
            "mean_score": None,
            "min_score": None,
            "max_score": None,
        }

    def _collect_skill_verifier_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        summary = self._empty_skill_verifier_summary()
        if not self.config.use_skill_verifier:
            return summary

        scores: list[float] = []
        for record in records:
            verification = record.get("verification")
            if not isinstance(verification, dict) or not verification.get("enabled"):
                continue
            summary["verified_skills"] += 1
            if verification.get("accepted"):
                summary["accepted"] += 1
            else:
                summary["rejected"] += 1
            score = verification.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                scores.append(float(score))

        if scores:
            summary["mean_score"] = round(sum(scores) / len(scores), 3)
            summary["min_score"] = round(min(scores), 3)
            summary["max_score"] = round(max(scores), 3)
        return summary

    def _empty_validation_publish_summary(self) -> dict[str, Any]:
        return {
            "publish_mode": self.config.publish_mode,
            "jobs_scanned": 0,
            "pending": 0,
            "published": 0,
            "rejected": 0,
        }

    def _build_validation_evidence(self, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for session in sessions[:8]:
            item: dict[str, Any] = {
                "session_id": str(session.get("session_id", "")),
                "summary": str(session.get("_summary", "")),
            }
            skills = session.get("_skills_referenced")
            if skills:
                item["skills_referenced"] = sorted(str(s or "") for s in skills if str(s or ""))
            judge_scores = session.get("_judge_scores")
            if isinstance(judge_scores, dict) and isinstance(judge_scores.get("overall_score"), (int, float)):
                item["judge_overall_score"] = float(judge_scores["overall_score"])
            if isinstance(session.get("_avg_prm"), (int, float)):
                item["avg_prm"] = float(session["_avg_prm"])
            evidence.append(item)
        return evidence

    def _build_replay_cases(self, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        preferred: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []

        for session in sessions[:6]:
            session_id = str(session.get("session_id", "") or "")
            turns = session.get("turns") or []
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                instruction = str(turn.get("prompt_text", "") or "").strip()
                reference_response = str(turn.get("response_text", "") or "").strip()
                if not instruction or not reference_response:
                    continue
                case = {
                    "session_id": session_id,
                    "turn_num": int(turn.get("turn_num", 0) or 0),
                    "instruction": instruction[:3000],
                    "reference_response": reference_response[:4000],
                    "had_tool_calls": bool(turn.get("tool_calls")),
                    "had_tool_results": bool(turn.get("tool_results") or turn.get("tool_observations")),
                }
                if not case["had_tool_calls"] and not case["had_tool_results"]:
                    preferred.append(case)
                else:
                    fallback.append(case)
                if len(preferred) >= 3:
                    return preferred[:3]

        if preferred:
            return preferred[:3]
        return fallback[:3]

    def _queue_validation_job(
        self,
        skill: dict[str, Any],
        action_type: str,
        sessions: list[dict[str, Any]],
        rationale: str,
        source: str,
        *,
        current_skill: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        name = str(skill.get("name", "") or "")
        skill_id = self._id_registry.get_or_create(name)
        job_id = self._validation_store.make_job_id(name)
        job = {
            "job_id": job_id,
            "status": "pending_validation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_skill_name": name,
            "candidate_skill_id": skill_id,
            "candidate_skill": skill,
            "current_skill": current_skill,
            "proposed_action": action_type,
            "source": source,
            "rationale": rationale,
            "session_ids": [session.get("session_id", "") for session in sessions],
            "session_evidence": self._build_validation_evidence(sessions),
            "replay_cases": self._build_replay_cases(sessions),
            "min_results": self.config.validation_required_results,
            "min_approvals": self.config.validation_required_approvals,
            "min_score": self.config.validation_min_mean_score,
            "max_rejections": self.config.validation_max_rejections,
        }
        self._validation_store.save_job(job)
        logger.info(
            "[EvolveServer] queued validation job %s for skill '%s' (publish_mode=validated)",
            job_id,
            name,
        )
        return {
            "action": "queued_for_validation",
            "proposed_action": action_type,
            "skill_name": name,
            "skill_id": skill_id,
            "version": None,
            "session_ids": job["session_ids"],
            "rationale": rationale,
            "source": source,
            "edit_summary": skill.get("edit_summary"),
            "uploaded": False,
            "validation_job_id": job_id,
        }

    async def _finalize_validation_jobs(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        summary = self._empty_validation_publish_summary()
        if self.config.publish_mode != "validated":
            return [], summary

        records: list[dict[str, Any]] = []
        for job in self._validation_store.list_jobs():
            summary["jobs_scanned"] += 1
            job_id = str(job.get("job_id", "") or "")
            if not job_id:
                continue
            if self._validation_store.load_decision(job_id):
                continue

            results = self._validation_store.list_results(job_id)
            if not results:
                summary["pending"] += 1
                continue

            accepted = 0
            rejected = 0
            scores: list[float] = []
            for result in results:
                if result.get("accepted") is True:
                    accepted += 1
                else:
                    rejected += 1
                score = result.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    scores.append(float(score))

            mean_score = round(sum(scores) / len(scores), 3) if scores else None
            publish_ready = (
                len(results) >= self.config.validation_required_results
                and accepted >= self.config.validation_required_approvals
                and mean_score is not None
                and mean_score >= self.config.validation_min_mean_score
            )
            reject_ready = rejected >= self.config.validation_max_rejections

            if publish_ready:
                candidate_skill = job.get("candidate_skill")
                if not isinstance(candidate_skill, dict) or not candidate_skill.get("name"):
                    self._validation_store.save_decision(
                        job_id,
                        {
                            "status": "rejected",
                            "reason": "candidate skill payload missing",
                            "result_count": len(results),
                            "accepted_count": accepted,
                            "rejected_count": rejected,
                            "mean_score": mean_score,
                        },
                    )
                    summary["rejected"] += 1
                    continue
                action_type = str(job.get("proposed_action", DecisionAction.CREATE) or DecisionAction.CREATE)
                actual_action = await self._resolve_and_upload(candidate_skill, action_type)
                self._validation_store.save_decision(
                    job_id,
                    {
                        "status": "published",
                        "published_action": actual_action,
                        "result_count": len(results),
                        "accepted_count": accepted,
                        "rejected_count": rejected,
                        "mean_score": mean_score,
                    },
                )
                summary["published"] += 1
                records.append(
                    {
                        "action": "published_after_validation",
                        "published_action": actual_action,
                        "skill_name": str(candidate_skill.get("name", "")),
                        "skill_id": self._id_registry.get_or_create(str(candidate_skill.get("name", ""))),
                        "version": self._id_registry.get_version(str(candidate_skill.get("name", ""))),
                        "session_ids": list(job.get("session_ids") or []),
                        "rationale": str(job.get("rationale", "") or ""),
                        "source": "validation_publish",
                        "uploaded": True,
                        "validation_job_id": job_id,
                        "validation_results": {
                            "result_count": len(results),
                            "accepted_count": accepted,
                            "rejected_count": rejected,
                            "mean_score": mean_score,
                        },
                    }
                )
                continue

            if reject_ready:
                self._validation_store.save_decision(
                    job_id,
                    {
                        "status": "rejected",
                        "reason": "client validation rejected candidate",
                        "result_count": len(results),
                        "accepted_count": accepted,
                        "rejected_count": rejected,
                        "mean_score": mean_score,
                    },
                )
                summary["rejected"] += 1
                records.append(
                    {
                        "action": "validation_rejected",
                        "proposed_action": str(job.get("proposed_action", "")),
                        "skill_name": str(job.get("candidate_skill_name", "")),
                        "skill_id": str(job.get("candidate_skill_id", "")),
                        "version": None,
                        "session_ids": list(job.get("session_ids") or []),
                        "rationale": str(job.get("rationale", "") or ""),
                        "source": "validation_publish",
                        "uploaded": False,
                        "validation_job_id": job_id,
                        "validation_results": {
                            "result_count": len(results),
                            "accepted_count": accepted,
                            "rejected_count": rejected,
                            "mean_score": mean_score,
                        },
                    }
                )
                continue

            summary["pending"] += 1

        return records, summary

    async def _run_skill_verifier(
        self,
        skill: dict[str, Any],
        action_type: str,
        sessions: list[dict[str, Any]],
        *,
        current_skill: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.config.use_skill_verifier:
            return {"enabled": False}
        return await verify_skill_candidate(
            self._llm,
            skill,
            sessions,
            action_type,
            current_skill=current_skill,
            min_score=self.config.skill_verifier_min_score,
        )

    def _inherit_current_skill(
        self,
        evolved_skill: Optional[dict[str, Any]],
        current_skill: Optional[dict[str, Any]],
        *,
        overwrite_body: bool = False,
    ) -> None:
        if not evolved_skill or not current_skill:
            return
        if overwrite_body:
            evolved_skill["content"] = current_skill.get("content", "")
            evolved_skill["category"] = current_skill.get("category", "general")
        else:
            evolved_skill.setdefault("content", current_skill.get("content", ""))
            evolved_skill.setdefault("category", current_skill.get("category", "general"))
        evolved_skill.setdefault("extra_frontmatter", current_skill.get("extra_frontmatter") or {})

    async def _materialize_skill(
        self,
        evolved_skill: Optional[dict],
        action_type: str,
        sessions: list[dict[str, Any]],
        rationale: str,
        source: str,
        *,
        current_skill: Optional[dict[str, Any]] = None,
    ) -> Optional[dict]:
        if not evolved_skill or not evolved_skill.get("name"):
            return None

        if action_type == DecisionAction.IMPROVE and current_skill and current_skill.get("name"):
            name = current_skill["name"]
        else:
            name = self._sanitise_name(evolved_skill["name"])
        evolved_skill["name"] = name

        verification = await self._run_skill_verifier(
            evolved_skill,
            action_type,
            sessions,
            current_skill=current_skill,
        )
        session_ids = [session.get("session_id", "") for session in sessions]
        if verification.get("enabled") and not verification.get("accepted"):
            logger.info(
                "[EvolveServer] verifier rejected skill '%s': %s",
                name,
                verification.get("reason", "no reason provided"),
            )
            return {
                "action": "verification_rejected",
                "proposed_action": action_type,
                "skill_name": name,
                "skill_id": None,
                "version": None,
                "session_ids": session_ids,
                "rationale": rationale,
                "source": source,
                "edit_summary": evolved_skill.get("edit_summary"),
                "uploaded": False,
                "verification": verification,
            }

        skill_id = self._id_registry.get_or_create(name)
        evolved_skill["skill_id"] = skill_id
        if self.config.publish_mode == "validated":
            record = self._queue_validation_job(
                evolved_skill,
                action_type,
                sessions,
                rationale,
                source,
                current_skill=current_skill,
            )
            record["verification"] = verification
            return record

        actual_action = await self._resolve_and_upload(evolved_skill, action_type)
        logger.info(
            "[EvolveServer] %s skill '%s' (id=%s, v%d)",
            actual_action,
            name,
            skill_id,
            self._id_registry.get_version(name),
        )
        return {
            "action": actual_action,
            "skill_name": name,
            "skill_id": skill_id,
            "version": self._id_registry.get_version(name),
            "session_ids": session_ids,
            "rationale": rationale,
            "source": source,
            "edit_summary": evolved_skill.get("edit_summary"),
            "uploaded": True,
            "verification": verification,
        }

    async def _evolve_skill_group(
        self,
        skill_name: str,
        sessions: list[dict],
        existing_skill_names: list[str],
    ) -> Optional[dict]:
        current_md = await self._call_storage(self._fetch_skill, skill_name)
        current_skill = parse_skill_content(skill_name, current_md) if current_md else None
        current_skill = self._overlay_manifest_metadata(
            current_skill,
            await self._call_storage(self._load_remote_skill_record, skill_name),
        )

        result = await evolve_skill_from_sessions(
            self._llm,
            skill_name,
            sessions,
            current_skill,
            existing_skill_names,
        )
        if not result or result.get("action") == DecisionAction.SKIP:
            logger.info("[EvolveServer] skill '%s': LLM decided to skip", skill_name)
            return None

        action_type = result.get("action", DecisionAction.IMPROVE)
        evolved_skill = result.get("skill")
        if not evolved_skill:
            return None

        if action_type == DecisionAction.OPTIMIZE_DESC and current_skill:
            self._inherit_current_skill(evolved_skill, current_skill, overwrite_body=True)
        elif current_skill:
            self._inherit_current_skill(evolved_skill, current_skill)

        return await self._materialize_skill(
            evolved_skill,
            action_type,
            sessions,
            result.get("rationale", ""),
            "skill_group",
            current_skill=current_skill,
        )

    async def _handle_no_skill_sessions(
        self,
        sessions: list[dict],
        existing_skill_names: list[str],
    ) -> list[dict]:
        result = await create_skill_from_sessions(self._llm, sessions, existing_skill_names)
        if not result or result.get("action") == DecisionAction.SKIP:
            logger.info("[EvolveServer] no-skill sessions: LLM decided to skip")
            return []

        evolved_skill = result.get("skill")
        if not evolved_skill:
            return []

        record = await self._materialize_skill(
            evolved_skill,
            DecisionAction.CREATE,
            sessions,
            result.get("rationale", ""),
            "no_skill",
        )
        return [record] if record else []

    async def run_once(self) -> dict:
        logger.info("[EvolveServer] === starting evolution cycle ===")
        started_at = time.monotonic()

        sessions, session_keys = await self._drain_sessions()
        judge_summary = self._empty_judge_summary()
        skill_group_count = 0
        no_skill_sessions: list[dict] = []
        evolution_records: list[dict] = []
        had_processing_error = False

        if sessions:
            logger.info("[EvolveServer] summarizing %d sessions", len(sessions))
            await summarize_sessions_parallel(self._llm, sessions)
            judge_summary = await self._run_session_judge(sessions)

            grouped_sessions = aggregate_sessions_by_skill(sessions)
            no_skill_sessions = grouped_sessions.pop(NO_SKILL_KEY, [])
            skill_group_count = len(grouped_sessions)

            manifest = await self._call_storage(self._load_remote_skills)
            existing_skill_names = [item.get("name", "") for item in manifest.values()]

            if grouped_sessions:
                logger.info("[EvolveServer] evolving %d skill group(s)", skill_group_count)
            for skill_name, skill_sessions in grouped_sessions.items():
                try:
                    record = await self._evolve_skill_group(skill_name, skill_sessions, existing_skill_names)
                except Exception as exc:
                    logger.error("[EvolveServer] skill '%s' evolve failed: %s", skill_name, exc)
                    had_processing_error = True
                    continue
                if record:
                    evolution_records.append(record)

            if no_skill_sessions:
                logger.info("[EvolveServer] processing %d no-skill sessions", len(no_skill_sessions))
                try:
                    evolution_records.extend(
                        await self._handle_no_skill_sessions(no_skill_sessions, existing_skill_names)
                    )
                except Exception as exc:
                    logger.error("[EvolveServer] no-skill evolve failed: %s", exc)
                    had_processing_error = True
        else:
            logger.info("[EvolveServer] queue empty - checking pending validation publish jobs")

        published_records, validation_publish_summary = await self._finalize_validation_jobs()
        all_records = evolution_records + published_records

        await self._call_storage(self._id_registry.save_to_oss, self._bucket, self._prefix)
        if session_keys and not had_processing_error:
            await self._call_storage(delete_session_keys, self._bucket, session_keys)
        elif session_keys and had_processing_error:
            logger.warning(
                "[EvolveServer] retaining %d session(s) in queue because this cycle had processing errors",
                len(session_keys),
            )

        elapsed = round(time.monotonic() - started_at, 1)
        uploaded_skills = sum(1 for record in all_records if record.get("uploaded"))
        queued_candidates = sum(1 for record in all_records if record.get("action") == "queued_for_validation")
        published_after_validation = sum(
            1 for record in all_records if record.get("action") == "published_after_validation"
        )
        skill_verifier_summary = self._collect_skill_verifier_summary(all_records)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "sessions": len(sessions),
            "skill_groups": skill_group_count,
            "no_skill_sessions": len(no_skill_sessions),
            "actions": len(all_records),
            "skills_evolved": uploaded_skills,
            "uploaded_skills": uploaded_skills,
            "candidates_queued": queued_candidates,
            "published_after_validation": published_after_validation,
            "evolutions": all_records,
            "session_judge": judge_summary,
            "skill_verifier": skill_verifier_summary,
            "validation_publish": validation_publish_summary,
            "had_processing_error": had_processing_error,
        }
        self._append_history(summary)
        logger.info(
            "[EvolveServer] === cycle done: %d sessions, %d skill groups, %d uploaded, %d queued in %.1fs ===",
            len(sessions),
            skill_group_count,
            uploaded_skills,
            queued_candidates,
            elapsed,
        )
        return summary

    async def run_periodic(self) -> None:
        self._running = True
        logger.info("[EvolveServer] periodic mode: interval=%ds", self.config.interval_seconds)
        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("[EvolveServer] cycle error: %s", exc, exc_info=True)
            await asyncio.sleep(self.config.interval_seconds)

    def stop(self) -> None:
        self._running = False

    def create_http_app(self):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI(title="SkillClaw Evolve Server")

        @app.post("/trigger")
        async def trigger_evolve():
            return JSONResponse(content=await self.run_once())

        @app.get("/status")
        async def status():
            entries = self._id_registry.all_entries()
            pending_keys = await self._call_storage(list_session_keys, self._bucket, self._prefix)
            return JSONResponse(
                content={
                    "running": self._running,
                    "pending_sessions": len(pending_keys),
                    "registered_skills": len(entries),
                    "skills": {
                        name: {
                            "skill_id": item["skill_id"],
                            "version": item.get("version", 0),
                        }
                        for name, item in entries.items()
                    },
                }
            )

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    @staticmethod
    def _sanitise_name(raw_name: str) -> str:
        name = raw_name.strip().lower()
        if SLUG_RE.match(name):
            return name
        name = re.sub(r"[^a-z0-9_-]", "-", name).strip("-")
        return name or "unnamed-skill"
