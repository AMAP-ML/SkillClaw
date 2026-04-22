"""
SQLite-backed projection store for the SkillClaw dashboard.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


class DashboardStore:
    """Materialized dashboard snapshot stored in SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).expanduser())

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT 'general',
                    source TEXT NOT NULL DEFAULT 'local',
                    has_local INTEGER NOT NULL DEFAULT 0,
                    has_remote INTEGER NOT NULL DEFAULT 0,
                    local_path TEXT NOT NULL DEFAULT '',
                    uploaded_at TEXT NOT NULL DEFAULT '',
                    uploaded_by TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    current_version INTEGER NOT NULL DEFAULT 0,
                    current_sha TEXT NOT NULL DEFAULT '',
                    local_inject_count INTEGER NOT NULL DEFAULT 0,
                    observed_injection_count INTEGER NOT NULL DEFAULT 0,
                    read_count INTEGER NOT NULL DEFAULT 0,
                    modified_count INTEGER NOT NULL DEFAULT 0,
                    session_count INTEGER NOT NULL DEFAULT 0,
                    effectiveness REAL NOT NULL DEFAULT 0.0,
                    positive_count INTEGER NOT NULL DEFAULT 0,
                    negative_count INTEGER NOT NULL DEFAULT 0,
                    neutral_count INTEGER NOT NULL DEFAULT 0,
                    content TEXT NOT NULL DEFAULT '',
                    raw_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
                CREATE INDEX IF NOT EXISTS idx_skills_category ON skills(category);
                CREATE INDEX IF NOT EXISTS idx_skills_source ON skills(source);
                CREATE INDEX IF NOT EXISTS idx_skills_sessions
                    ON skills(session_count DESC, observed_injection_count DESC);

                CREATE TABLE IF NOT EXISTS skill_versions (
                    skill_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content_sha TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL DEFAULT '',
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (skill_id, version)
                );

                CREATE INDEX IF NOT EXISTS idx_skill_versions_timestamp ON skill_versions(skill_id, timestamp DESC);

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL DEFAULT '',
                    user_alias TEXT NOT NULL DEFAULT '',
                    num_turns INTEGER NOT NULL DEFAULT 0,
                    avg_prm_score REAL,
                    skill_names_json TEXT NOT NULL DEFAULT '[]',
                    prompt_preview TEXT NOT NULL DEFAULT '',
                    response_preview TEXT NOT NULL DEFAULT '',
                    raw_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_user_alias ON sessions(user_alias);

                CREATE TABLE IF NOT EXISTS session_skill_links (
                    session_id TEXT NOT NULL,
                    skill_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL DEFAULT '',
                    relation TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (session_id, skill_id, relation)
                );

                CREATE INDEX IF NOT EXISTS idx_session_skill_links_skill ON session_skill_links(skill_id, relation);
                CREATE INDEX IF NOT EXISTS idx_session_skill_links_session ON session_skill_links(session_id);

                CREATE TABLE IF NOT EXISTS validation_jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT '',
                    skill_name TEXT NOT NULL DEFAULT '',
                    proposed_action TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT '',
                    result_count INTEGER NOT NULL DEFAULT 0,
                    accepted_count INTEGER NOT NULL DEFAULT 0,
                    rejected_count INTEGER NOT NULL DEFAULT 0,
                    mean_score REAL,
                    raw_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_validation_jobs_status ON validation_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_validation_jobs_created ON validation_jobs(created_at DESC);
                """
            )

    def replace_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        with self._connect() as conn:
            conn.execute("DELETE FROM meta")
            conn.execute("DELETE FROM skill_versions")
            conn.execute("DELETE FROM session_skill_links")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM validation_jobs")
            conn.execute("DELETE FROM skills")

            for skill in snapshot.get("skills") or []:
                conn.execute(
                    """
                    INSERT INTO skills (
                        skill_id,
                        name,
                        description,
                        category,
                        source,
                        has_local,
                        has_remote,
                        local_path,
                        uploaded_at,
                        uploaded_by,
                        updated_at,
                        current_version,
                        current_sha,
                        local_inject_count,
                        observed_injection_count,
                        read_count,
                        modified_count,
                        session_count,
                        effectiveness,
                        positive_count,
                        negative_count,
                        neutral_count,
                        content,
                        raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(skill.get("skill_id", "") or ""),
                        str(skill.get("name", "") or ""),
                        str(skill.get("description", "") or ""),
                        str(skill.get("category", "general") or "general"),
                        str(skill.get("source", "local") or "local"),
                        1 if skill.get("has_local") else 0,
                        1 if skill.get("has_remote") else 0,
                        str(skill.get("local_path", "") or ""),
                        str(skill.get("uploaded_at", "") or ""),
                        str(skill.get("uploaded_by", "") or ""),
                        str(skill.get("updated_at", "") or ""),
                        int(skill.get("current_version", 0) or 0),
                        str(skill.get("current_sha", "") or ""),
                        int(skill.get("local_inject_count", 0) or 0),
                        int(skill.get("observed_injection_count", 0) or 0),
                        int(skill.get("read_count", 0) or 0),
                        int(skill.get("modified_count", 0) or 0),
                        int(skill.get("session_count", 0) or 0),
                        float(skill.get("effectiveness", 0.0) or 0.0),
                        int(skill.get("positive_count", 0) or 0),
                        int(skill.get("negative_count", 0) or 0),
                        int(skill.get("neutral_count", 0) or 0),
                        str(skill.get("skill_md", "") or ""),
                        _json_dumps(skill),
                    ),
                )
                for version in skill.get("versions") or []:
                    conn.execute(
                        """
                        INSERT INTO skill_versions (
                            skill_id,
                            version,
                            content_sha,
                            action,
                            timestamp,
                            raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(skill.get("skill_id", "") or ""),
                            int(version.get("version", 0) or 0),
                            str(version.get("content_sha", "") or ""),
                            str(version.get("action", "") or ""),
                            str(version.get("timestamp", "") or ""),
                            _json_dumps(version),
                        ),
                    )

            for session in snapshot.get("sessions") or []:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id,
                        timestamp,
                        user_alias,
                        num_turns,
                        avg_prm_score,
                        skill_names_json,
                        prompt_preview,
                        response_preview,
                        raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(session.get("session_id", "") or ""),
                        str(session.get("timestamp", "") or ""),
                        str(session.get("user_alias", "") or ""),
                        int(session.get("num_turns", 0) or 0),
                        float(session["avg_prm_score"]) if session.get("avg_prm_score") is not None else None,
                        _json_dumps(session.get("skill_names") or []),
                        str(session.get("prompt_preview", "") or ""),
                        str(session.get("response_preview", "") or ""),
                        _json_dumps(session),
                    ),
                )

            for link in snapshot.get("session_skill_links") or []:
                conn.execute(
                    """
                    INSERT INTO session_skill_links (
                        session_id,
                        skill_id,
                        skill_name,
                        relation,
                        count
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(link.get("session_id", "") or ""),
                        str(link.get("skill_id", "") or ""),
                        str(link.get("skill_name", "") or ""),
                        str(link.get("relation", "") or ""),
                        int(link.get("count", 0) or 0),
                    ),
                )

            for job in snapshot.get("validation_jobs") or []:
                conn.execute(
                    """
                    INSERT INTO validation_jobs (
                        job_id,
                        created_at,
                        skill_name,
                        proposed_action,
                        status,
                        result_count,
                        accepted_count,
                        rejected_count,
                        mean_score,
                        raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(job.get("job_id", "") or ""),
                        str(job.get("created_at", "") or ""),
                        str(job.get("skill_name", "") or ""),
                        str(job.get("proposed_action", "") or ""),
                        str(job.get("status", "") or ""),
                        int(job.get("result_count", 0) or 0),
                        int(job.get("accepted_count", 0) or 0),
                        int(job.get("rejected_count", 0) or 0),
                        float(job["mean_score"]) if job.get("mean_score") is not None else None,
                        _json_dumps(job),
                    ),
                )

            meta = dict(snapshot.get("meta") or {})
            meta["generated_at"] = str(snapshot.get("generated_at", "") or "")
            meta["skill_count"] = len(snapshot.get("skills") or [])
            meta["session_count"] = len(snapshot.get("sessions") or [])
            meta["validation_job_count"] = len(snapshot.get("validation_jobs") or [])
            for key, value in meta.items():
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?)",
                    (str(key), _json_dumps(value)),
                )

        return {
            "generated_at": str(snapshot.get("generated_at", "") or ""),
            "skills": len(snapshot.get("skills") or []),
            "sessions": len(snapshot.get("sessions") or []),
            "validation_jobs": len(snapshot.get("validation_jobs") or []),
            "warnings": list((snapshot.get("meta") or {}).get("warnings") or []),
        }

    def get_meta(self) -> dict[str, Any]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        return {str(row["key"]): _json_loads(row["value"], row["value"]) for row in rows}

    def _skill_summary_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "skill_id": row["skill_id"],
            "name": row["name"],
            "description": row["description"],
            "category": row["category"],
            "source": row["source"],
            "has_local": bool(row["has_local"]),
            "has_remote": bool(row["has_remote"]),
            "local_path": row["local_path"],
            "uploaded_at": row["uploaded_at"],
            "uploaded_by": row["uploaded_by"],
            "updated_at": row["updated_at"],
            "current_version": row["current_version"],
            "current_sha": row["current_sha"],
            "local_inject_count": row["local_inject_count"],
            "observed_injection_count": row["observed_injection_count"],
            "read_count": row["read_count"],
            "modified_count": row["modified_count"],
            "session_count": row["session_count"],
            "effectiveness": row["effectiveness"],
            "positive_count": row["positive_count"],
            "negative_count": row["negative_count"],
            "neutral_count": row["neutral_count"],
        }

    def list_skills(
        self,
        *,
        search: str = "",
        category: str = "",
        source: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        self.initialize()
        clauses: list[str] = []
        params: list[Any] = []
        if search:
            clauses.append("(name LIKE ? OR description LIKE ?)")
            needle = f"%{search}%"
            params.extend([needle, needle])
        if category:
            clauses.append("category = ?")
            params.append(category)
        if source:
            clauses.append("source = ?")
            params.append(source)

        query = """
            SELECT * FROM skills
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += """
            ORDER BY
                session_count DESC,
                observed_injection_count DESC,
                local_inject_count DESC,
                name ASC
            LIMIT ?
        """
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._skill_summary_from_row(row) for row in rows]

    def get_skill(self, skill_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM skills WHERE skill_id = ?", (skill_id,)).fetchone()
            if row is None:
                return None
            payload = _json_loads(row["raw_json"], {})
            payload.update(self._skill_summary_from_row(row))
            version_rows = conn.execute(
                """
                SELECT raw_json
                FROM skill_versions
                WHERE skill_id = ?
                ORDER BY version DESC, timestamp DESC
                """,
                (skill_id,),
            ).fetchall()
            related_sessions = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.timestamp,
                    s.user_alias,
                    s.num_turns,
                    s.avg_prm_score,
                    s.prompt_preview,
                    s.response_preview,
                    SUM(CASE WHEN l.relation = 'injected' THEN l.count ELSE 0 END) AS injected_count,
                    SUM(CASE WHEN l.relation = 'read' THEN l.count ELSE 0 END) AS read_count,
                    SUM(CASE WHEN l.relation = 'modified' THEN l.count ELSE 0 END) AS modified_count
                FROM session_skill_links AS l
                JOIN sessions AS s
                  ON s.session_id = l.session_id
                WHERE l.skill_id = ?
                GROUP BY
                    s.session_id,
                    s.timestamp,
                    s.user_alias,
                    s.num_turns,
                    s.avg_prm_score,
                    s.prompt_preview,
                    s.response_preview
                ORDER BY s.timestamp DESC, s.session_id DESC
                LIMIT 50
                """,
                (skill_id,),
            ).fetchall()

        payload["versions"] = [_json_loads(item["raw_json"], {}) for item in version_rows]
        payload["related_sessions"] = [
            {
                "session_id": item["session_id"],
                "timestamp": item["timestamp"],
                "user_alias": item["user_alias"],
                "num_turns": item["num_turns"],
                "avg_prm_score": item["avg_prm_score"],
                "prompt_preview": item["prompt_preview"],
                "response_preview": item["response_preview"],
                "injected_count": item["injected_count"],
                "read_count": item["read_count"],
                "modified_count": item["modified_count"],
            }
            for item in related_sessions
        ]
        return payload

    def list_sessions(
        self,
        *,
        skill_id: str = "",
        search: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        self.initialize()
        clauses: list[str] = []
        params: list[Any] = []
        query = "SELECT DISTINCT s.* FROM sessions AS s"
        if skill_id:
            query += " JOIN session_skill_links AS l ON l.session_id = s.session_id"
            clauses.append("l.skill_id = ?")
            params.append(skill_id)
        if search:
            needle = f"%{search}%"
            clauses.append("(s.session_id LIKE ? OR s.user_alias LIKE ? OR s.prompt_preview LIKE ?)")
            params.extend([needle, needle, needle])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY s.timestamp DESC, s.session_id DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            payload = _json_loads(row["raw_json"], {})
            items.append(
                {
                    "session_id": row["session_id"],
                    "timestamp": row["timestamp"],
                    "user_alias": row["user_alias"],
                    "num_turns": row["num_turns"],
                    "avg_prm_score": row["avg_prm_score"],
                    "skill_names": _json_loads(row["skill_names_json"], []),
                    "prompt_preview": row["prompt_preview"],
                    "response_preview": row["response_preview"],
                    "source": str(payload.get("source", "") or ""),
                    "outcome": str(payload.get("outcome", "") or ""),
                }
            )
        return items

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            payload = _json_loads(row["raw_json"], {})
            payload.update(
                {
                    "session_id": row["session_id"],
                    "timestamp": row["timestamp"],
                    "user_alias": row["user_alias"],
                    "num_turns": row["num_turns"],
                    "avg_prm_score": row["avg_prm_score"],
                    "skill_names": _json_loads(row["skill_names_json"], []),
                    "prompt_preview": row["prompt_preview"],
                    "response_preview": row["response_preview"],
                }
            )
            link_rows = conn.execute(
                """
                SELECT skill_id, skill_name, relation, count
                FROM session_skill_links
                WHERE session_id = ?
                ORDER BY skill_name ASC, relation ASC
                """,
                (session_id,),
            ).fetchall()
        payload["links"] = [
            {
                "skill_id": item["skill_id"],
                "skill_name": item["skill_name"],
                "relation": item["relation"],
                "count": item["count"],
            }
            for item in link_rows
        ]
        return payload

    def list_validation_jobs(
        self,
        *,
        status: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        self.initialize()
        query = "SELECT * FROM validation_jobs"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, job_id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "job_id": row["job_id"],
                "created_at": row["created_at"],
                "skill_name": row["skill_name"],
                "proposed_action": row["proposed_action"],
                "status": row["status"],
                "result_count": row["result_count"],
                "accepted_count": row["accepted_count"],
                "rejected_count": row["rejected_count"],
                "mean_score": row["mean_score"],
                "details": _json_loads(row["raw_json"], {}),
            }
            for row in rows
        ]

    def get_overview(self) -> dict[str, Any]:
        self.initialize()
        with self._connect() as conn:
            counts = {
                "skills": int(conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]),
                "local_skills": int(conn.execute("SELECT COUNT(*) FROM skills WHERE has_local = 1").fetchone()[0]),
                "shared_skills": int(conn.execute("SELECT COUNT(*) FROM skills WHERE has_remote = 1").fetchone()[0]),
                "sessions": int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]),
                "validation_jobs": int(conn.execute("SELECT COUNT(*) FROM validation_jobs").fetchone()[0]),
                "open_validation_jobs": int(
                    conn.execute(
                        "SELECT COUNT(*) FROM validation_jobs WHERE status IN ('pending', 'review')"
                    ).fetchone()[0]
                ),
                "local_injections": int(
                    conn.execute("SELECT COALESCE(SUM(local_inject_count), 0) FROM skills").fetchone()[0]
                ),
                "observed_injections": int(
                    conn.execute("SELECT COALESCE(SUM(observed_injection_count), 0) FROM skills").fetchone()[0]
                ),
                "observed_reads": int(conn.execute("SELECT COALESCE(SUM(read_count), 0) FROM skills").fetchone()[0]),
                "observed_modifications": int(
                    conn.execute("SELECT COALESCE(SUM(modified_count), 0) FROM skills").fetchone()[0]
                ),
            }
            top_skills = conn.execute(
                """
                SELECT *
                FROM skills
                ORDER BY
                    session_count DESC,
                    observed_injection_count DESC,
                    local_inject_count DESC,
                    name ASC
                LIMIT 8
                """
            ).fetchall()
            recent_sessions = conn.execute(
                """
                SELECT *
                FROM sessions
                ORDER BY timestamp DESC, session_id DESC
                LIMIT 8
                """
            ).fetchall()
            categories = conn.execute(
                """
                SELECT category, COUNT(*) AS count
                FROM skills
                GROUP BY category
                ORDER BY count DESC, category ASC
                """
            ).fetchall()

        return {
            "counts": counts,
            "top_skills": [self._skill_summary_from_row(row) for row in top_skills],
            "recent_sessions": [
                {
                    "session_id": row["session_id"],
                    "timestamp": row["timestamp"],
                    "user_alias": row["user_alias"],
                    "num_turns": row["num_turns"],
                    "avg_prm_score": row["avg_prm_score"],
                    "skill_names": _json_loads(row["skill_names_json"], []),
                    "prompt_preview": row["prompt_preview"],
                    "source": str(_json_loads(row["raw_json"], {}).get("source", "") or ""),
                    "outcome": str(_json_loads(row["raw_json"], {}).get("outcome", "") or ""),
                }
                for row in recent_sessions
            ],
            "categories": [
                {
                    "category": row["category"],
                    "count": row["count"],
                }
                for row in categories
            ],
            "meta": self.get_meta(),
        }
