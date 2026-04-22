"""
FastAPI dashboard service for SkillClaw.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import SkillClawConfig
from .dashboard_ingest import build_dashboard_snapshot
from .dashboard_store import DashboardStore
from .skill_hub import SkillHub

logger = logging.getLogger(__name__)


def _assets_dir() -> Path:
    return Path(__file__).with_name("dashboard_assets")


def _build_skill_filter(config: SkillClawConfig, *, no_filter: bool = False) -> dict[str, Any] | None:
    if no_filter:
        return None
    stats_path = Path(config.skills_dir).expanduser() / "skill_stats.json"
    if not stats_path.is_file():
        return None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(stats, dict):
        return None
    return {
        "stats": stats,
        "min_injections": int(config.sharing_push_min_injections or 0),
        "min_effectiveness": float(config.sharing_push_min_effectiveness or 0.0),
    }


def _sharing_backend(config: SkillClawConfig) -> str:
    backend = str(config.sharing_backend or "").strip().lower()
    if backend:
        return backend
    if config.sharing_local_root:
        return "local"
    if config.sharing_bucket or config.sharing_endpoint:
        return "s3"
    return ""


def _sharing_target(config: SkillClawConfig) -> str:
    backend = _sharing_backend(config)
    if backend == "local":
        return f"local:{config.sharing_local_root}/{config.sharing_group_id}"
    if config.sharing_bucket:
        return f"{backend}:{config.sharing_bucket}/{config.sharing_group_id}"
    return f"{backend}:{config.sharing_group_id}"


def _require_sharing_hub(config: SkillClawConfig) -> SkillHub:
    if not config.sharing_enabled:
        raise ValueError("skill sharing is not enabled in the current config")
    backend = _sharing_backend(config)
    if backend == "local" and not config.sharing_local_root:
        raise ValueError("local sharing backend requires sharing_local_root")
    if backend == "s3" and not config.sharing_bucket:
        raise ValueError("s3 sharing backend requires sharing_bucket")
    if backend == "oss" and (not config.sharing_bucket or not config.sharing_endpoint):
        raise ValueError("oss sharing backend requires sharing_bucket and sharing_endpoint")
    if not backend:
        raise ValueError("sharing backend is not configured")
    return SkillHub.from_config(config)


def _local_sessions_from_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in (snapshot.get("sessions") or []) if str(item.get("source", "") or "") == "local"]


def _normalize_selection(items: Any, *, field_name: str) -> list[str] | None:
    if items is None:
        return None
    if not isinstance(items, list):
        raise ValueError(f"'{field_name}' must be a list of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


class DashboardService:
    """Owns dashboard snapshot sync, queries, and operations."""

    def __init__(self, config: SkillClawConfig) -> None:
        self.config = config
        self.store = DashboardStore(config.dashboard_db_path)

    def sync(self) -> dict[str, Any]:
        snapshot = build_dashboard_snapshot(self.config)
        summary = self.store.replace_snapshot(snapshot)
        return {
            "summary": summary,
            "overview": self.store.get_overview(),
        }

    def _embedded_evolve_server(self):
        from evolve_server.core.config import EvolveServerConfig
        from evolve_server.engines.workflow import EvolveServer

        from .validation_store import ValidationStore

        evolve_config = EvolveServerConfig.from_skillclaw_config(self.config)
        try:
            validation_store = ValidationStore.from_config(self.config)
            if validation_store.list_jobs():
                evolve_config.publish_mode = "validated"
                evolve_config.__post_init__()
        except Exception:
            pass
        return EvolveServer(evolve_config)

    def pull_skills(self, *, skill_names: list[str] | None = None) -> dict[str, Any]:
        hub = _require_sharing_hub(self.config)
        selection = list(skill_names or [])
        if selection:
            result = hub.pull_skills(
                self.config.skills_dir,
                mirror=False,
                include_names=selection,
            )
        else:
            result = hub.pull_skills(self.config.skills_dir)
        sync_result = self.sync()
        return {
            "operation": "pull",
            "target": _sharing_target(self.config),
            "selection": {
                "mode": "selected" if selection else "all",
                "requested": selection,
                "count": len(selection),
            },
            "result": result,
            "sync": sync_result["summary"],
        }

    def push_skills(self, *, no_filter: bool = False) -> dict[str, Any]:
        hub = _require_sharing_hub(self.config)
        result = hub.push_skills(
            self.config.skills_dir,
            skill_filter=_build_skill_filter(self.config, no_filter=no_filter),
        )
        sync_result = self.sync()
        return {
            "operation": "push",
            "target": _sharing_target(self.config),
            "result": result,
            "sync": sync_result["summary"],
        }

    def sync_skills(self) -> dict[str, Any]:
        hub = _require_sharing_hub(self.config)
        result = hub.sync_skills(self.config.skills_dir)
        sync_result = self.sync()
        return {
            "operation": "sync",
            "target": _sharing_target(self.config),
            "result": result,
            "sync": sync_result["summary"],
        }

    def export_local_sessions(self, *, session_ids: list[str] | None = None) -> dict[str, Any]:
        hub = _require_sharing_hub(self.config)
        snapshot = build_dashboard_snapshot(self.config)
        sessions = _local_sessions_from_snapshot(snapshot)
        total_local_sessions = len(sessions)
        session_lookup = {
            str(item.get("session_id", "") or ""): item for item in sessions if str(item.get("session_id", "") or "")
        }
        selection = list(session_ids or [])
        missing_ids: list[str] = []
        if selection:
            selected_sessions: list[dict[str, Any]] = []
            for session_id in selection:
                session = session_lookup.get(session_id)
                if session is None:
                    missing_ids.append(session_id)
                    continue
                selected_sessions.append(session)
            sessions = selected_sessions

        exported = 0
        skipped = 0
        for session in sessions:
            session_id = str(session.get("session_id", "") or "")
            if not session_id:
                continue
            payload = {
                "session_id": session_id,
                "timestamp": str(session.get("timestamp", "") or ""),
                "user_alias": str(session.get("user_alias", "") or "local"),
                "num_turns": int(session.get("num_turns", 0) or 0),
                "turns": list(session.get("turns") or []),
                "source": "local-dashboard-export",
                "outcome": str(session.get("outcome", "") or ""),
                "outcome_reasons": list(session.get("outcome_reasons") or []),
            }
            key = f"{hub._prefix()}sessions/{session_id}.json"
            content = json.dumps(payload, ensure_ascii=False, sort_keys=True)

            try:
                existing = hub._bucket.get_object(key).read().decode("utf-8")
                if existing == content:
                    skipped += 1
                    continue
            except Exception:
                pass

            hub._bucket.put_object(key, content.encode("utf-8"))
            exported += 1

        sync_result = self.sync()
        return {
            "operation": "export-local-sessions",
            "target": _sharing_target(self.config),
            "selection": {
                "mode": "selected" if selection else "all",
                "requested": selection,
                "count": len(selection),
            },
            "result": {
                "exported": exported,
                "skipped": skipped,
                "matched": len(sessions),
                "requested": len(selection) if selection else len(sessions),
                "missing": len(missing_ids),
                "missing_ids": missing_ids,
                "total_local_sessions": total_local_sessions,
            },
            "sync": sync_result["summary"],
        }

    def activate_skill_version(self, skill_id: str, *, target: str) -> dict[str, Any]:
        skill = self.store.get_skill(skill_id)
        if not isinstance(skill, dict):
            raise ValueError("skill not found")

        skill_name = str(skill.get("name", "") or "").strip()
        if not skill_name:
            raise ValueError("skill name is missing")

        selected_target = str(target or "").strip()
        if not selected_target:
            raise ValueError("'target' is required")

        document = ""
        label = ""
        if selected_target == "local-current":
            document = str(skill.get("skill_md") or skill.get("content") or "").strip()
            label = "本地当前版本"
        elif selected_target == "shared-current":
            document = str(skill.get("remote_skill_md") or skill.get("remote_content") or "").strip()
            label = "共享当前版本"
        elif selected_target.startswith("shared-version:"):
            raw_version = selected_target.split(":", 1)[1].strip()
            try:
                version_num = int(raw_version)
            except ValueError as exc:
                raise ValueError("invalid shared version target") from exc
            versions = skill.get("versions") or []
            version_payload = next(
                (
                    item
                    for item in versions
                    if isinstance(item, dict) and int(item.get("version", 0) or 0) == version_num
                ),
                None,
            )
            if not isinstance(version_payload, dict):
                raise ValueError(f"shared version not found: v{version_num}")
            document = str(version_payload.get("skill_md") or version_payload.get("content") or "").strip()
            label = f"共享 v{version_num}"
        else:
            raise ValueError(f"unsupported activation target: {selected_target}")

        if not document:
            raise ValueError("selected version does not include a document snapshot")

        local_path = Path(str(skill.get("local_path", "") or "")).expanduser()
        if not str(local_path).strip() or local_path.name != "SKILL.md":
            local_path = Path(self.config.skills_dir).expanduser() / skill_name / "SKILL.md"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(document.rstrip() + "\n", encoding="utf-8")

        sync_result = self.sync()
        return {
            "operation": "activate-skill-version",
            "skill_id": skill_id,
            "skill_name": skill_name,
            "target": selected_target,
            "label": label,
            "local_path": str(local_path),
            "sync": sync_result["summary"],
        }

    async def submit_validation_review(
        self,
        job_id: str,
        *,
        accepted: bool,
        score: float | None = None,
        notes: str = "",
        auto_finalize: bool = True,
    ) -> dict[str, Any]:
        if not self.config.sharing_enabled:
            raise ValueError("skill sharing is not enabled in the current config")

        from .validation_store import ValidationStore

        validation_store = ValidationStore.from_config(self.config)
        job = validation_store.load_job(job_id)
        if not isinstance(job, dict):
            raise ValueError(f"validation job not found: {job_id}")

        raw_alias = str(self.config.sharing_user_alias or "").strip()
        user_alias = raw_alias or "dashboard-review"
        normalized_score = score
        if normalized_score is None:
            normalized_score = 0.95 if accepted else 0.05
        normalized_score = max(0.0, min(1.0, float(normalized_score)))
        note_text = str(notes or "").strip()

        result_payload = {
            "validator_mode": "manual",
            "decision": "accept" if accepted else "reject",
            "accepted": bool(accepted),
            "score": normalized_score,
            "threshold": float(job.get("min_score", 0.75) or 0.75),
            "reason": note_text or f"Manual review submitted by {user_alias}.",
            "notes": note_text,
            "review_source": "dashboard",
        }
        validation_store.save_result(job_id, user_alias, result_payload)

        response = {
            "operation": "submit-validation-review",
            "job_id": job_id,
            "user_alias": user_alias,
            "result": result_payload,
        }

        if auto_finalize and str(self.config.dashboard_evolve_server_url or "").strip():
            response["finalize"] = await self.trigger_evolve()
            return response

        sync_result = self.sync()
        response["sync"] = sync_result["summary"]
        return response

    async def get_evolve_status(self) -> dict[str, Any]:
        base_url = str(self.config.dashboard_evolve_server_url or "").strip()
        if not base_url:
            if not self.config.sharing_enabled:
                return {
                    "configured": False,
                    "url": "",
                }
            try:
                from evolve_server.storage.oss_helpers import list_session_keys

                server = self._embedded_evolve_server()
                pending_keys = await server._call_storage(list_session_keys, server._bucket, server._prefix)
                entries = server._id_registry.all_entries()
                return {
                    "configured": True,
                    "url": "embedded://local-evolve",
                    "healthy": True,
                    "status": {
                        "running": False,
                        "pending_sessions": len(pending_keys),
                        "registered_skills": len(entries),
                        "skills": {
                            name: {
                                "skill_id": item.get("skill_id", ""),
                                "version": item.get("version", 0),
                            }
                            for name, item in entries.items()
                        },
                    },
                }
            except Exception as exc:
                return {
                    "configured": True,
                    "url": "embedded://local-evolve",
                    "healthy": False,
                    "error": str(exc),
                }
        status_url = base_url.rstrip("/") + "/status"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(status_url)
                response.raise_for_status()
            payload = response.json()
            return {
                "configured": True,
                "url": base_url,
                "healthy": True,
                "status": payload,
            }
        except Exception as exc:
            return {
                "configured": True,
                "url": base_url,
                "healthy": False,
                "error": str(exc),
            }

    async def trigger_evolve(self) -> dict[str, Any]:
        base_url = str(self.config.dashboard_evolve_server_url or "").strip()
        if not base_url:
            if not self.config.sharing_enabled:
                raise ValueError("skill sharing is not enabled in the current config")

            server = self._embedded_evolve_server()
            result = await server.run_once()
            sync_result = self.sync()
            return {
                "operation": "trigger-evolve",
                "url": "embedded://local-evolve",
                "result": result,
                "sync": sync_result["summary"],
            }
        trigger_url = base_url.rstrip("/") + "/trigger"
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(trigger_url)
            response.raise_for_status()
        sync_result = self.sync()
        return {
            "operation": "trigger-evolve",
            "url": trigger_url,
            "result": response.json(),
            "sync": sync_result["summary"],
        }


def create_dashboard_app(config: SkillClawConfig) -> FastAPI:
    service = DashboardService(config)
    assets_dir = _assets_dir()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.store.initialize()
        if config.dashboard_sync_on_start:
            try:
                service.sync()
            except Exception:
                logger.exception("[Dashboard] initial sync failed")
        app.state.dashboard_service = service
        yield

    app = FastAPI(title="SkillClaw Dashboard", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def dashboard_index():
        return FileResponse(assets_dir / "index.html")

    @app.get("/api/v1/health")
    async def health():
        return {
            "status": "ok",
            "db_path": service.store.db_path,
            "meta": service.store.get_meta(),
        }

    @app.get("/api/v1/overview")
    async def overview():
        return service.store.get_overview()

    @app.get("/api/v1/skills")
    async def list_skills(
        search: str = "",
        category: str = "",
        source: str = "",
        limit: int = 500,
    ):
        return {
            "items": service.store.list_skills(
                search=search.strip(),
                category=category.strip(),
                source=source.strip(),
                limit=limit,
            )
        }

    @app.get("/api/v1/skills/{skill_id}")
    async def get_skill(skill_id: str):
        payload = service.store.get_skill(skill_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="skill not found")
        return payload

    @app.post("/api/v1/skills/{skill_id}/activate")
    async def activate_skill(skill_id: str, payload: dict[str, Any] | None = Body(default=None)):
        try:
            body = payload or {}
            target = str(body.get("target", "") or "").strip()
            if not target:
                raise ValueError("'target' is required")
            return service.activate_skill_version(skill_id, target=target)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/sessions")
    async def list_sessions(
        skill_id: str = "",
        search: str = "",
        limit: int = 200,
    ):
        return {
            "items": service.store.list_sessions(
                skill_id=skill_id.strip(),
                search=search.strip(),
                limit=limit,
            )
        }

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        payload = service.store.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="session not found")
        return payload

    @app.get("/api/v1/validation/jobs")
    async def validation_jobs(status: str = "", limit: int = 200):
        return {
            "items": service.store.list_validation_jobs(
                status=status.strip(),
                limit=limit,
            )
        }

    @app.get("/api/v1/evolve/status")
    async def evolve_status():
        return await service.get_evolve_status()

    @app.post("/api/v1/sync")
    async def sync_projection():
        return service.sync()

    @app.post("/api/v1/ops/pull")
    async def pull_skills(payload: dict[str, Any] | None = Body(default=None)):
        try:
            body = payload or {}
            raw_skill_names = body.get("skill_names")
            skill_names = _normalize_selection(raw_skill_names, field_name="skill_names")
            if raw_skill_names is not None and not skill_names:
                raise ValueError("'skill_names' must contain at least one non-empty value")
            return service.pull_skills(skill_names=skill_names)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/ops/push")
    async def push_skills(payload: dict[str, Any] | None = Body(default=None)):
        try:
            return service.push_skills(no_filter=bool((payload or {}).get("no_filter", False)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/ops/sync")
    async def sync_skills():
        try:
            return service.sync_skills()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/ops/export-sessions")
    async def export_sessions(payload: dict[str, Any] | None = Body(default=None)):
        try:
            body = payload or {}
            raw_session_ids = body.get("session_ids")
            session_ids = _normalize_selection(raw_session_ids, field_name="session_ids")
            if raw_session_ids is not None and not session_ids:
                raise ValueError("'session_ids' must contain at least one non-empty value")
            return service.export_local_sessions(session_ids=session_ids)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/ops/trigger-evolve")
    async def trigger_evolve():
        try:
            return await service.trigger_evolve()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/validation/jobs/{job_id}/review")
    async def submit_review(job_id: str, payload: dict[str, Any] | None = Body(default=None)):
        body = payload or {}
        accepted = body.get("accepted")
        if not isinstance(accepted, bool):
            raise HTTPException(status_code=400, detail="'accepted' must be a boolean")

        raw_score = body.get("score")
        score: float | None = None
        if raw_score is not None and raw_score != "":
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="'score' must be a number in [0, 1]") from exc
            if score < 0.0 or score > 1.0:
                raise HTTPException(status_code=400, detail="'score' must be a number in [0, 1]")

        try:
            return await service.submit_validation_review(
                job_id,
                accepted=accepted,
                score=score,
                notes=str(body.get("notes", "") or ""),
                auto_finalize=bool(body.get("auto_finalize", True)),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def serve_dashboard(config: SkillClawConfig) -> None:
    """Run the dashboard HTTP service."""
    app = create_dashboard_app(config)
    uvicorn.run(
        app,
        host=str(config.dashboard_host or "127.0.0.1"),
        port=int(config.dashboard_port or 3788),
        log_level="info",
    )
