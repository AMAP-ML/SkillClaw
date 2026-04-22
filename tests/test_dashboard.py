from __future__ import annotations

import hashlib
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from skillclaw.config import SkillClawConfig
from skillclaw.dashboard_ingest import build_dashboard_snapshot
from skillclaw.dashboard_server import DashboardService, create_dashboard_app
from skillclaw.dashboard_store import DashboardStore


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _skill_id(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def _skill_doc(name: str, description: str, body: str, *, category: str = "general") -> str:
    return textwrap.dedent(
        f"""\
        ---
        name: {name}
        description: "{description}"
        category: {category}
        ---

        # {name}

        {body}
        """
    )


def _history_entry(version: int, document: str, timestamp: str, action: str) -> dict[str, object]:
    return {
        "version": version,
        "content_sha": _sha256_text(document),
        "timestamp": timestamp,
        "action": action,
        "skill_md": document,
        "content": document,
    }


def _transcript_record(role: str, text: str) -> dict[str, object]:
    payload = text
    if role == "user":
        payload = f"<user_query>\n{text}\n</user_query>"
    return {
        "role": role,
        "message": {
            "content": [
                {
                    "type": "text",
                    "text": payload,
                }
            ]
        },
    }


class DashboardFixture:
    def __init__(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.skills_dir = self.root / "skills"
        self.share_root = self.root / "share"
        self.group_dir = self.share_root / "team-alpha"
        self.db_path = self.root / "dashboard.sqlite3"

        self.local_docs = self._build_local_docs()
        self.shared_docs = self._build_shared_docs()

        self._create_local_skills()
        self._create_local_state()
        self._create_local_records()
        self._create_shared_snapshot()

        self.config = SkillClawConfig(
            use_skills=True,
            skills_dir=str(self.skills_dir),
            record_dir=str(self.root / "records"),
            sharing_enabled=True,
            sharing_backend="local",
            sharing_local_root=str(self.share_root),
            sharing_group_id="team-alpha",
            sharing_user_alias="tester",
            dashboard_enabled=True,
            dashboard_db_path=str(self.db_path),
            dashboard_sync_on_start=True,
            dashboard_include_shared=True,
            dashboard_evolve_server_url="",
        )

    def cleanup(self) -> None:
        self.tempdir.cleanup()

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_local_docs(self) -> dict[str, str]:
        return {
            "debug-notes": _skill_doc(
                "debug-notes",
                "Keep a compact running log while debugging.",
                """\
                ## When to use
                - When the failure mode keeps changing between retries.
                - When you need a short breadcrumb trail before editing code.

                ## Workflow
                1. Record the failed assumption.
                2. Capture the last observable fact.
                3. State the next probe before making a patch.
                """,
                category="coding",
            ),
            "api-contract-checklist": _skill_doc(
                "api-contract-checklist",
                "Verify request, auth, and schema assumptions before patching API tests.",
                """\
                ## Checklist
                - Confirm auth headers and tenant routing.
                - Check request and response schema drift.
                - Verify fixture defaults before changing handlers.
                """,
                category="backend",
            ),
            "release-rollback-runbook": _skill_doc(
                "release-rollback-runbook",
                "Coordinate rollback checks during incident mitigation.",
                """\
                ## Rollback guardrails
                - Identify blast radius and freeze new writes.
                - Verify migration compatibility before rollback.
                - Keep a short operator handoff note after mitigation.
                """,
                category="ops",
            ),
            "incident-timeline": _skill_doc(
                "incident-timeline",
                "Summarize incident progression, impact window, and mitigation sequence.",
                """\
                ## Timeline template
                - Start with first customer-visible symptom.
                - Keep timestamps in chronological order.
                - Separate hypothesis, action, and observed outcome.
                """,
                category="ops",
            ),
        }

    def _build_shared_docs(self) -> dict[str, str]:
        return {
            "debug-notes": _skill_doc(
                "debug-notes",
                "Keep a compact running log while debugging.",
                """\
                ## Shared practice
                - Capture the failed assumption before editing files.
                - After each retry, summarize what changed and what stayed invariant.
                - End with one concrete next step instead of a generic note.
                """,
                category="coding",
            ),
            "incident-timeline": _skill_doc(
                "incident-timeline",
                "Summarize incident progression, impact window, and mitigation sequence.",
                """\
                ## Shared practice
                - Anchor the timeline on customer impact and service restoration.
                - Record mitigation checkpoints, not every shell command.
                - Close with one unresolved question for the next responder.
                """,
                category="ops",
            ),
            "release-rollback-runbook": _skill_doc(
                "release-rollback-runbook",
                "Coordinate rollback checks during incident mitigation.",
                """\
                ## Shared rollback path
                - Confirm release identifier, migration window, and affected tenants.
                - Execute rollback in the least-coupled order.
                - Validate alarms, dashboards, and customer traffic after recovery.
                """,
                category="ops",
            ),
            "sql-trace": _skill_doc(
                "sql-trace",
                "Trace SQL state transitions during debugging.",
                """\
                ## Trace format
                - Log query, bind parameters, row count, and transaction scope.
                - Mark where state diverges from expectation.
                - Keep application log references next to query traces.
                """,
                category="data_analysis",
            ),
            "prompt-risk-screener": _skill_doc(
                "prompt-risk-screener",
                "Screen prompts for policy, jailbreak, and ambiguity risks before execution.",
                """\
                ## Screening loop
                - Classify policy-sensitive intent first.
                - Separate ambiguity from deliberate jailbreak behavior.
                - Recommend the smallest safe rewrite when blocking is not required.
                """,
                category="governance",
            ),
            "handoff-brief": _skill_doc(
                "handoff-brief",
                "Prepare a concise operator handoff after long debugging sessions.",
                """\
                ## Handoff format
                - Problem statement in one sentence.
                - What changed, what remains risky, and what to verify next.
                - Include owners, timestamps, and the next blocking question.
                """,
                category="operations",
            ),
        }

    def _create_local_skills(self) -> None:
        local_stats = {
            "debug-notes": {
                "inject_count": 18,
                "positive_count": 10,
                "negative_count": 2,
                "neutral_count": 6,
                "last_injected_at": "2026-04-21T01:20:00Z",
                "effectiveness": 0.79,
            },
            "api-contract-checklist": {
                "inject_count": 12,
                "positive_count": 7,
                "negative_count": 1,
                "neutral_count": 4,
                "last_injected_at": "2026-04-21T02:00:00Z",
                "effectiveness": 0.82,
            },
            "release-rollback-runbook": {
                "inject_count": 5,
                "positive_count": 2,
                "negative_count": 1,
                "neutral_count": 2,
                "last_injected_at": "2026-04-20T23:30:00Z",
                "effectiveness": 0.58,
            },
            "incident-timeline": {
                "inject_count": 9,
                "positive_count": 5,
                "negative_count": 1,
                "neutral_count": 3,
                "last_injected_at": "2026-04-20T23:10:00Z",
                "effectiveness": 0.74,
            },
        }

        for name, document in self.local_docs.items():
            skill_dir = self.skills_dir / name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(document, encoding="utf-8")

        (self.skills_dir / "skill_stats.json").write_text(
            json.dumps(local_stats, indent=2),
            encoding="utf-8",
        )

    def _create_shared_snapshot(self) -> None:
        skills_dir = self.group_dir / "skills"
        sessions_dir = self.group_dir / "sessions"
        validation_jobs_dir = self.group_dir / "validation_jobs"
        validation_results_dir = self.group_dir / "validation_results"
        validation_decisions_dir = self.group_dir / "validation_decisions"

        for path in (skills_dir, sessions_dir, validation_jobs_dir, validation_results_dir, validation_decisions_dir):
            path.mkdir(parents=True, exist_ok=True)

        for name, document in self.shared_docs.items():
            skill_dir = skills_dir / name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(document, encoding="utf-8")

        manifest = [
            {
                "name": "debug-notes",
                "description": "Keep a compact running log while debugging.",
                "category": "coding",
                "sha256": _sha256_text(self.shared_docs["debug-notes"]),
                "uploaded_by": "alice",
                "uploaded_at": "2026-04-20T09:30:00Z",
            },
            {
                "name": "incident-timeline",
                "description": "Summarize incident progression, impact window, and mitigation sequence.",
                "category": "ops",
                "sha256": _sha256_text(self.shared_docs["incident-timeline"]),
                "uploaded_by": "carol",
                "uploaded_at": "2026-04-20T16:25:00Z",
            },
            {
                "name": "release-rollback-runbook",
                "description": "Coordinate rollback checks during incident mitigation.",
                "category": "ops",
                "sha256": _sha256_text(self.shared_docs["release-rollback-runbook"]),
                "uploaded_by": "dan",
                "uploaded_at": "2026-04-20T17:20:00Z",
            },
            {
                "name": "sql-trace",
                "description": "Trace SQL state transitions during debugging.",
                "category": "data_analysis",
                "sha256": _sha256_text(self.shared_docs["sql-trace"]),
                "uploaded_by": "bob",
                "uploaded_at": "2026-04-20T09:40:00Z",
            },
            {
                "name": "prompt-risk-screener",
                "description": "Screen prompts for policy, jailbreak, and ambiguity risks before execution.",
                "category": "governance",
                "sha256": _sha256_text(self.shared_docs["prompt-risk-screener"]),
                "uploaded_by": "mia",
                "uploaded_at": "2026-04-20T20:10:00Z",
            },
            {
                "name": "handoff-brief",
                "description": "Prepare a concise operator handoff after long debugging sessions.",
                "category": "operations",
                "sha256": _sha256_text(self.shared_docs["handoff-brief"]),
                "uploaded_by": "erin",
                "uploaded_at": "2026-04-20T21:10:00Z",
            },
        ]
        (self.group_dir / "manifest.jsonl").write_text(
            "\n".join(json.dumps(item) for item in manifest) + "\n",
            encoding="utf-8",
        )

        registry = {
            "debug-notes": {
                "skill_id": _skill_id("debug-notes"),
                "version": 3,
                "content_sha": _sha256_text(self.shared_docs["debug-notes"]),
                "history": [
                    _history_entry(
                        1,
                        _skill_doc(
                            "debug-notes",
                            "Keep a compact running log while debugging.",
                            "Capture the failing assumption before editing any file.",
                            category="coding",
                        ),
                        "2026-04-18T08:00:00Z",
                        "create",
                    ),
                    _history_entry(
                        2,
                        _skill_doc(
                            "debug-notes",
                            "Keep a compact running log while debugging.",
                            "Capture the failing assumption and note what changed after each retry.",
                            category="coding",
                        ),
                        "2026-04-19T08:15:00Z",
                        "improve",
                    ),
                    _history_entry(3, self.shared_docs["debug-notes"], "2026-04-20T09:30:00Z", "improve"),
                ],
            },
            "incident-timeline": {
                "skill_id": _skill_id("incident-timeline"),
                "version": 2,
                "content_sha": _sha256_text(self.shared_docs["incident-timeline"]),
                "history": [
                    _history_entry(
                        1,
                        _skill_doc(
                            "incident-timeline",
                            "Summarize incident progression, impact window, and mitigation sequence.",
                            "Track first impact, mitigation, and restore time.",
                            category="ops",
                        ),
                        "2026-04-19T12:00:00Z",
                        "create",
                    ),
                    _history_entry(2, self.shared_docs["incident-timeline"], "2026-04-20T16:25:00Z", "improve"),
                ],
            },
            "release-rollback-runbook": {
                "skill_id": _skill_id("release-rollback-runbook"),
                "version": 2,
                "content_sha": _sha256_text(self.shared_docs["release-rollback-runbook"]),
                "history": [
                    _history_entry(
                        1,
                        _skill_doc(
                            "release-rollback-runbook",
                            "Coordinate rollback checks during incident mitigation.",
                            "Confirm rollback owner and migration compatibility.",
                            category="ops",
                        ),
                        "2026-04-19T19:30:00Z",
                        "create",
                    ),
                    _history_entry(2, self.shared_docs["release-rollback-runbook"], "2026-04-20T17:20:00Z", "improve"),
                ],
            },
            "sql-trace": {
                "skill_id": _skill_id("sql-trace"),
                "version": 2,
                "content_sha": _sha256_text(self.shared_docs["sql-trace"]),
                "history": [
                    _history_entry(
                        1,
                        _skill_doc(
                            "sql-trace",
                            "Trace SQL state transitions during debugging.",
                            "Log query text, parameters, and row counts.",
                            category="data_analysis",
                        ),
                        "2026-04-19T07:50:00Z",
                        "create",
                    ),
                    _history_entry(2, self.shared_docs["sql-trace"], "2026-04-20T09:40:00Z", "improve"),
                ],
            },
            "prompt-risk-screener": {
                "skill_id": _skill_id("prompt-risk-screener"),
                "version": 4,
                "content_sha": _sha256_text(self.shared_docs["prompt-risk-screener"]),
                "history": [
                    _history_entry(
                        1,
                        _skill_doc(
                            "prompt-risk-screener",
                            "Screen prompts for policy, jailbreak, and ambiguity risks before execution.",
                            "Classify unsafe requests before suggesting changes.",
                            category="governance",
                        ),
                        "2026-04-18T06:15:00Z",
                        "create",
                    ),
                    _history_entry(
                        2,
                        _skill_doc(
                            "prompt-risk-screener",
                            "Screen prompts for policy, jailbreak, and ambiguity risks before execution.",
                            "Separate policy risk from simple ambiguity.",
                            category="governance",
                        ),
                        "2026-04-19T10:45:00Z",
                        "improve",
                    ),
                    _history_entry(
                        3,
                        _skill_doc(
                            "prompt-risk-screener",
                            "Screen prompts for policy, jailbreak, and ambiguity risks before execution.",
                            "Add a small safe-rewrite suggestion when blocking is not needed.",
                            category="governance",
                        ),
                        "2026-04-20T08:10:00Z",
                        "improve",
                    ),
                    _history_entry(4, self.shared_docs["prompt-risk-screener"], "2026-04-20T20:10:00Z", "improve"),
                ],
            },
            "handoff-brief": {
                "skill_id": _skill_id("handoff-brief"),
                "version": 1,
                "content_sha": _sha256_text(self.shared_docs["handoff-brief"]),
                "history": [
                    _history_entry(1, self.shared_docs["handoff-brief"], "2026-04-20T21:10:00Z", "create"),
                ],
            },
        }
        (self.group_dir / "evolve_skill_registry.json").write_text(
            json.dumps(registry, indent=2),
            encoding="utf-8",
        )

        shared_sessions = [
            {
                "session_id": "sess-104",
                "source": "shared",
                "timestamp": "2026-04-21T01:30:00Z",
                "user_alias": "jane",
                "num_turns": 3,
                "outcome": "success",
                "outcome_reasons": [
                    "staging contract mismatch reproduced",
                    "shared handoff prepared for oncall",
                ],
                "turns": [
                    {
                        "turn_num": 1,
                        "prompt_text": "Trace the flaky partner API contract failure.",
                        "response_text": "I will validate auth headers and collect a compact trace.",
                        "injected_skills": ["debug-notes", "api-contract-checklist"],
                        "read_skills": [{"skill_name": "handoff-brief"}],
                        "modified_skills": [],
                        "prm_score": 0.83,
                    },
                    {
                        "turn_num": 2,
                        "prompt_text": "Patch the auth fixture and re-run the failing contract test.",
                        "response_text": (
                            "Header casing was wrong in the fixture; "
                            "I updated the checklist and reran the contract path."
                        ),
                        "injected_skills": [],
                        "read_skills": [{"skill_name": "api-contract-checklist"}],
                        "modified_skills": [{"skill_name": "api-contract-checklist"}],
                        "prm_score": 0.78,
                    },
                    {
                        "turn_num": 3,
                        "prompt_text": "Summarize the fix for the next responder.",
                        "response_text": "Recorded the mismatch, patch scope, and verification steps.",
                        "injected_skills": ["debug-notes"],
                        "read_skills": [],
                        "modified_skills": [],
                        "prm_score": 0.8,
                    },
                ],
            },
            {
                "session_id": "sess-103",
                "source": "shared",
                "timestamp": "2026-04-20T23:10:00Z",
                "user_alias": "nora",
                "num_turns": 2,
                "outcome": "review",
                "outcome_reasons": [
                    "prompt policy boundary refined",
                    "candidate improvement queued for human review",
                ],
                "turns": [
                    {
                        "turn_num": 1,
                        "prompt_text": (
                            "Screen a customer prompt that mixes benign analytics with policy-sensitive content."
                        ),
                        "response_text": "I will separate ambiguity from policy-sensitive intent first.",
                        "injected_skills": ["prompt-risk-screener"],
                        "read_skills": [],
                        "modified_skills": [],
                        "prm_score": 0.76,
                    },
                    {
                        "turn_num": 2,
                        "prompt_text": "Rewrite the screening guidance to reduce false positives.",
                        "response_text": "Added a safe-rewrite suggestion path and explicit jailbreak detection notes.",
                        "injected_skills": [],
                        "read_skills": [{"skill_name": "prompt-risk-screener"}],
                        "modified_skills": [{"skill_name": "prompt-risk-screener"}],
                        "prm_score": 0.71,
                    },
                ],
            },
            {
                "session_id": "sess-102",
                "source": "shared",
                "timestamp": "2026-04-20T18:40:00Z",
                "user_alias": "bob",
                "num_turns": 2,
                "outcome": "success",
                "outcome_reasons": [
                    "row count drift isolated to transaction scope",
                ],
                "turns": [
                    {
                        "turn_num": 1,
                        "prompt_text": "Investigate why the SQL update did not persist.",
                        "response_text": "I will trace transaction boundaries and compare row counts.",
                        "injected_skills": ["sql-trace"],
                        "read_skills": [{"skill_name": "debug-notes"}],
                        "modified_skills": [],
                        "prm_score": 0.81,
                    },
                    {
                        "turn_num": 2,
                        "prompt_text": "Patch the query logging and retry.",
                        "response_text": (
                            "Added query logging, confirmed the transaction scope, and updated the trace skill."
                        ),
                        "injected_skills": ["debug-notes"],
                        "read_skills": [{"skill_name": "sql-trace"}],
                        "modified_skills": [{"skill_name": "sql-trace"}],
                        "prm_score": 0.67,
                    },
                ],
            },
            {
                "session_id": "sess-101",
                "source": "shared",
                "timestamp": "2026-04-20T15:20:00Z",
                "user_alias": "carol",
                "num_turns": 2,
                "outcome": "rollback",
                "outcome_reasons": [
                    "release toggled off after tenant write amplification",
                    "postmortem timeline requested",
                ],
                "turns": [
                    {
                        "turn_num": 1,
                        "prompt_text": "Assemble an incident timeline for the tenant write spike.",
                        "response_text": "I will collect customer impact, mitigation steps, and restore time.",
                        "injected_skills": ["incident-timeline"],
                        "read_skills": [{"skill_name": "release-rollback-runbook"}],
                        "modified_skills": [],
                        "prm_score": 0.74,
                    },
                    {
                        "turn_num": 2,
                        "prompt_text": "Prepare a rollback note for the release manager.",
                        "response_text": (
                            "Captured the rollback sequence and annotated the timeline with mitigation checkpoints."
                        ),
                        "injected_skills": ["release-rollback-runbook"],
                        "read_skills": [],
                        "modified_skills": [{"skill_name": "incident-timeline"}],
                        "prm_score": 0.69,
                    },
                ],
            },
            {
                "session_id": "sess-100",
                "source": "shared",
                "timestamp": "2026-04-20T11:00:00Z",
                "user_alias": "alice",
                "num_turns": 2,
                "outcome": "success",
                "outcome_reasons": [
                    "query logging patch validated",
                ],
                "turns": [
                    {
                        "turn_num": 1,
                        "prompt_text": "Investigate why the SQL update did not persist.",
                        "response_text": "I will inspect the transaction boundaries first.",
                        "injected_skills": ["debug-notes"],
                        "read_skills": [{"skill_name": "sql-trace"}],
                        "modified_skills": [],
                        "prm_score": 0.81,
                    },
                    {
                        "turn_num": 2,
                        "prompt_text": "Patch the query logging and retry.",
                        "response_text": "Added query logging and replayed the failing path.",
                        "injected_skills": ["sql-trace"],
                        "read_skills": [],
                        "modified_skills": [{"skill_name": "sql-trace"}],
                        "prm_score": 0.67,
                    },
                ],
            },
        ]
        for payload in shared_sessions:
            (sessions_dir / f"{payload['session_id']}.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

        prompt_candidate_doc = _skill_doc(
            "prompt-risk-screener",
            "Screen prompts for policy, jailbreak, and ambiguity risks before execution.",
            """\
            ## Candidate change
            - Add a small safe-rewrite path before hard blocking ambiguous prompts.
            - Require an explicit jailbreak note when instructions conflict.
            - Preserve a short rationale that a human reviewer can inspect later.
            """,
            category="governance",
        )
        api_candidate_doc = _skill_doc(
            "api-contract-checklist",
            "Verify request, auth, and schema assumptions before patching API tests.",
            """\
            ## Candidate checklist
            - Verify auth, tenant routing, and fixture defaults before touching handlers.
            - Compare contract fixtures against the latest generated schema.
            - Record one rollback-safe verification step after the patch lands.
            """,
            category="backend",
        )
        sql_candidate_doc = _skill_doc(
            "sql-trace",
            "Trace SQL state transitions during debugging.",
            """\
            ## Candidate improvement
            - Include transaction scope and row count deltas in the trace.
            - Link the trace to the failing application log span.
            - Mark the exact point where persistence diverged.
            """,
            category="data_analysis",
        )
        rollback_candidate_doc = _skill_doc(
            "release-rollback-runbook",
            "Coordinate rollback checks during incident mitigation.",
            """\
            ## Candidate rollback path
            - Roll back immediately after migration lock acquisition.
            - Skip secondary dashboard validation until after customer traffic stabilizes.
            - Keep a terse operator note only if the incident stays open for more than 30 minutes.
            """,
            category="ops",
        )

        validation_jobs = [
            {
                "job_id": "job-pending",
                "created_at": "2026-04-21T01:40:00Z",
                "candidate_skill_name": "prompt-risk-screener",
                "proposed_action": "improve",
                "session_ids": ["sess-103"],
                "session_evidence": [
                    {
                        "session_id": "sess-103",
                        "summary": (
                            "Reviewer flagged false positives when ambiguous prompts were screened too aggressively."
                        ),
                        "judge_overall_score": 0.71,
                        "avg_prm": 0.735,
                    }
                ],
                "candidate_skill": {
                    "name": "prompt-risk-screener",
                    "description": "Screen prompts for policy, jailbreak, and ambiguity risks before execution.",
                    "category": "governance",
                    "skill_md": prompt_candidate_doc,
                },
                "current_skill": {
                    "name": "prompt-risk-screener",
                    "description": "Screen prompts for policy, jailbreak, and ambiguity risks before execution.",
                    "category": "governance",
                    "skill_md": self.shared_docs["prompt-risk-screener"],
                },
                "min_results": 2,
                "min_approvals": 1,
                "min_score": 0.7,
                "max_rejections": 1,
                "rationale": (
                    "The current skill blocks too early on ambiguous prompts. "
                    "A reviewer should confirm the safer rewrite path."
                ),
            },
            {
                "job_id": "job-review",
                "created_at": "2026-04-20T22:10:00Z",
                "candidate_skill_name": "api-contract-checklist",
                "proposed_action": "create",
                "session_ids": ["local-004", "sess-104"],
                "session_evidence": [
                    {
                        "session_id": "local-004",
                        "summary": "Local run found an auth header casing mismatch before modifying handlers.",
                        "judge_overall_score": 0.84,
                        "avg_prm": 0.0,
                    },
                    {
                        "session_id": "sess-104",
                        "summary": "Shared session confirmed the checklist generalized to a partner contract failure.",
                        "judge_overall_score": 0.8,
                        "avg_prm": 0.803,
                    },
                ],
                "candidate_skill": {
                    "name": "api-contract-checklist",
                    "description": "Verify request, auth, and schema assumptions before patching API tests.",
                    "category": "backend",
                    "skill_md": api_candidate_doc,
                },
                "min_results": 2,
                "min_approvals": 2,
                "min_score": 0.75,
                "max_rejections": 0,
                "rationale": (
                    "The checklist looks reusable, but reviewers disagree on whether it is still too API-specific."
                ),
            },
            {
                "job_id": "job-rejected",
                "created_at": "2026-04-20T19:45:00Z",
                "candidate_skill_name": "release-rollback-runbook",
                "proposed_action": "improve",
                "session_ids": ["local-003", "sess-101"],
                "session_evidence": [
                    {
                        "session_id": "local-003",
                        "summary": "Local operator notes suggested a faster rollback shortcut.",
                        "judge_overall_score": 0.42,
                        "avg_prm": 0.0,
                    },
                    {
                        "session_id": "sess-101",
                        "summary": (
                            "Shared incident replay showed the shortcut skipped validation steps needed by oncall."
                        ),
                        "judge_overall_score": 0.35,
                        "avg_prm": 0.715,
                    },
                ],
                "candidate_skill": {
                    "name": "release-rollback-runbook",
                    "description": "Coordinate rollback checks during incident mitigation.",
                    "category": "ops",
                    "skill_md": rollback_candidate_doc,
                },
                "current_skill": {
                    "name": "release-rollback-runbook",
                    "description": "Coordinate rollback checks during incident mitigation.",
                    "category": "ops",
                    "skill_md": self.shared_docs["release-rollback-runbook"],
                },
                "min_results": 2,
                "min_approvals": 2,
                "min_score": 0.7,
                "max_rejections": 0,
                "rationale": "The candidate removed validation checks that responders still need during rollback.",
            },
            {
                "job_id": "job-published",
                "created_at": "2026-04-20T11:05:00Z",
                "candidate_skill_name": "sql-trace",
                "proposed_action": "improve",
                "session_ids": ["sess-100", "sess-102"],
                "session_evidence": [
                    {
                        "session_id": "sess-100",
                        "summary": "Initial trace isolated the missing transaction boundary.",
                        "judge_overall_score": 0.83,
                        "avg_prm": 0.74,
                    },
                    {
                        "session_id": "sess-102",
                        "summary": "Improved trace format captured row count drift and application log correlation.",
                        "judge_overall_score": 0.88,
                        "avg_prm": 0.74,
                    },
                ],
                "candidate_skill": {
                    "name": "sql-trace",
                    "description": "Trace SQL state transitions during debugging.",
                    "category": "data_analysis",
                    "skill_md": sql_candidate_doc,
                },
                "current_skill": {
                    "name": "sql-trace",
                    "description": "Trace SQL state transitions during debugging.",
                    "category": "data_analysis",
                    "skill_md": self.shared_docs["sql-trace"],
                },
                "min_results": 2,
                "min_approvals": 2,
                "min_score": 0.75,
                "max_rejections": 0,
                "rationale": (
                    "The new trace format consistently improved debugging quality across two SQL persistence failures."
                ),
            },
        ]
        for payload in validation_jobs:
            (validation_jobs_dir / f"{payload['job_id']}.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

        validation_results = {
            "job-review": [
                {
                    "job_id": "job-review",
                    "user_alias": "qa-ops",
                    "accepted": True,
                    "decision": "accept",
                    "score": 0.82,
                    "created_at": "2026-04-20T22:18:00Z",
                    "notes": "The checklist abstracts the contract debugging flow well.",
                    "validator_mode": "manual",
                },
                {
                    "job_id": "job-review",
                    "user_alias": "sec-review",
                    "accepted": False,
                    "decision": "reject",
                    "score": 0.38,
                    "created_at": "2026-04-20T22:24:00Z",
                    "notes": "Still too tied to one partner auth flow.",
                    "validator_mode": "manual",
                },
            ],
            "job-rejected": [
                {
                    "job_id": "job-rejected",
                    "user_alias": "oncall-sre",
                    "accepted": False,
                    "decision": "reject",
                    "score": 0.41,
                    "created_at": "2026-04-20T19:52:00Z",
                    "notes": "The shortcut removes rollback validation that the operator still needs.",
                    "validator_mode": "manual",
                },
                {
                    "job_id": "job-rejected",
                    "user_alias": "incident-commander",
                    "accepted": False,
                    "decision": "reject",
                    "score": 0.33,
                    "created_at": "2026-04-20T19:58:00Z",
                    "notes": "Too risky for a shared runbook.",
                    "validator_mode": "review",
                },
            ],
            "job-published": [
                {
                    "job_id": "job-published",
                    "user_alias": "alice",
                    "accepted": True,
                    "decision": "accept",
                    "score": 0.88,
                    "created_at": "2026-04-20T11:12:00Z",
                    "notes": "The trace format is clearer and reusable.",
                    "validator_mode": "manual",
                },
                {
                    "job_id": "job-published",
                    "user_alias": "bob",
                    "accepted": True,
                    "decision": "accept",
                    "score": 0.91,
                    "created_at": "2026-04-20T11:13:30Z",
                    "notes": "Captures transaction scope and row count drift without adding noise.",
                    "validator_mode": "review",
                },
            ],
        }
        for job_id, results in validation_results.items():
            result_dir = validation_results_dir / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            for result in results:
                (result_dir / f"{result['user_alias']}.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

        validation_decisions = {
            "job-published": {
                "job_id": "job-published",
                "status": "published",
                "published_action": "improve",
                "decided_at": "2026-04-20T11:14:00Z",
                "reason": "Published after two positive reviews with mean score above threshold.",
            },
            "job-rejected": {
                "job_id": "job-rejected",
                "status": "rejected",
                "published_action": "improve",
                "decided_at": "2026-04-20T20:05:00Z",
                "reason": "Rejected because the rollback shortcut removed mandatory verification.",
            },
        }
        for job_id, decision in validation_decisions.items():
            (validation_decisions_dir / f"{job_id}.json").write_text(
                json.dumps(decision, indent=2),
                encoding="utf-8",
            )

    def _create_local_state(self) -> None:
        state_dir = self.root / "state"
        trajectories_dir = state_dir / "trajectories"
        transcripts_root = self.root / "cursor_transcripts"

        trajectories_dir.mkdir(parents=True, exist_ok=True)
        transcripts_root.mkdir(parents=True, exist_ok=True)

        conversations = [
            {
                "session_id": "local-004",
                "start_time": "2026-04-21T01:40:00Z",
                "end_time": "2026-04-21T02:10:00Z",
                "outcome": "success",
                "active_skills": ["api-contract-checklist", "debug-notes"],
                "outcome_reasons": [
                    "auth header mismatch isolated before handler edits",
                    "schema fixture regenerated and verified",
                ],
                "turns": [
                    (
                        "Investigate why the partner API test still returns 401 after the fixture refactor.",
                        "I will validate the auth contract, capture the current "
                        "request shape, and keep a short debug log before editing "
                        "code.",
                    ),
                    (
                        "Patch the fixture and summarize the exact verification steps.",
                        "The header casing was wrong in the fixture. I updated the "
                        "contract checklist, reran the failing test, and recorded "
                        "the verification path.",
                    ),
                ],
            },
            {
                "session_id": "local-003",
                "start_time": "2026-04-20T22:50:00Z",
                "end_time": "2026-04-20T23:30:00Z",
                "outcome": "rollback",
                "active_skills": ["incident-timeline", "release-rollback-runbook"],
                "outcome_reasons": [
                    "tenant write amplification required rollback",
                    "operator notes needed a cleaner incident timeline",
                ],
                "turns": [
                    (
                        "Prepare a rollback note for the release that amplified tenant writes.",
                        "I will build a timeline first, then turn the rollback into a short runbook for the operator.",
                    ),
                    (
                        "Summarize what remains risky after the rollback.",
                        "The rollback stabilized traffic, but secondary validation "
                        "is still required for delayed jobs and dashboard lag.",
                    ),
                ],
            },
            {
                "session_id": "local-002",
                "start_time": "2026-04-20T20:50:00Z",
                "end_time": "2026-04-20T21:25:00Z",
                "outcome": "failure",
                "active_skills": ["debug-notes"],
                "outcome_reasons": [
                    "state drift still not reproducible locally",
                    "next step is to instrument the retry worker",
                ],
                "turns": [
                    (
                        "Debug why the retry worker stops after one transient failure.",
                        "I will keep a compact note of each assumption and check "
                        "whether the worker state changes between retries.",
                    ),
                    (
                        "Summarize what is still unknown.",
                        "We know the retry lease expires too early, but we still "
                        "need one more probe around worker heartbeats.",
                    ),
                ],
            },
            {
                "session_id": "local-001",
                "start_time": "2026-04-20T08:00:00Z",
                "end_time": "2026-04-20T08:30:00Z",
                "outcome": "failure",
                "active_skills": ["debug-notes"],
                "outcome_reasons": [
                    "command failed: pytest tests/test_api.py",
                    "user correction: tests/conftest.py",
                ],
                "turns": [
                    (
                        "Please debug the failing API test.",
                        "I will inspect the failing test and trace the request path.",
                    ),
                    (
                        "Now summarize the fix.",
                        "The issue was an auth header mismatch in the test fixture.",
                    ),
                ],
            },
        ]

        conv_offsets: dict[str, int] = {}
        for conversation in conversations:
            session_id = str(conversation["session_id"])
            transcript_dir = transcripts_root / session_id
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_dir / f"{session_id}.jsonl"

            transcript_lines: list[dict[str, object]] = []
            for user_text, assistant_text in conversation["turns"]:
                transcript_lines.append(_transcript_record("user", user_text))
                transcript_lines.append(_transcript_record("assistant", assistant_text))
            transcript_path.write_text(
                "\n".join(json.dumps(item) for item in transcript_lines) + "\n",
                encoding="utf-8",
            )
            conv_offsets[str(transcript_path)] = 0

            self._write_json(
                trajectories_dir / f"{session_id}.json",
                {
                    "conversation_id": session_id,
                    "active_skills": conversation["active_skills"],
                    "outcome": conversation["outcome"],
                    "start_time": conversation["start_time"],
                    "end_time": conversation["end_time"],
                    "outcome_reasons": conversation["outcome_reasons"],
                },
            )

        self._write_json(state_dir / "conv_offsets.json", conv_offsets)

    def _create_local_records(self) -> None:
        records_dir = self.root / "records"
        records_dir.mkdir(parents=True, exist_ok=True)

        conversations = [
            {
                "session_id": "local-005",
                "turn": 1,
                "timestamp": "2026-04-21 03:15:00",
                "messages": [
                    {"role": "system", "content": "Skill catalog injected"},
                    {
                        "role": "user",
                        "content": "Summarize the retry-worker regression and keep the answer brief.",
                    },
                ],
                "instruction_text": "Summarize the retry-worker regression and keep the answer brief.",
                "prompt_text": (
                    "system: Skill catalog injected\n\nuser: Summarize the "
                    "retry-worker regression and keep the answer brief."
                ),
                "response_text": (
                    "The worker exits after the first transient failure because "
                    "the lease heartbeat stops renewing after the retry path "
                    "resets its state."
                ),
                "tool_calls": [],
            },
            {
                "session_id": "local-005",
                "turn": 2,
                "timestamp": "2026-04-21 03:16:30",
                "messages": [
                    {"role": "system", "content": "Skill catalog injected"},
                    {
                        "role": "user",
                        "content": "Now list the next two verification steps.",
                    },
                ],
                "instruction_text": "Now list the next two verification steps.",
                "prompt_text": "system: Skill catalog injected\n\nuser: Now list the next two verification steps.",
                "response_text": (
                    "1. Capture lease-heartbeat timestamps across the retry "
                    "boundary.\n2. Verify whether the worker resets its retry "
                    "state before the heartbeat loop restarts."
                ),
                "tool_calls": [],
            },
        ]
        (records_dir / "conversations.jsonl").write_text(
            "\n".join(json.dumps(item) for item in conversations) + "\n",
            encoding="utf-8",
        )

        prm_scores = [
            {
                "session_id": "local-005",
                "turn": 1,
                "score": 0.61,
                "votes": [0.6, 0.61, 0.62],
            },
            {
                "session_id": "local-005",
                "turn": 2,
                "score": 0.67,
                "votes": [0.65, 0.67, 0.69],
            },
        ]
        (records_dir / "prm_scores.jsonl").write_text(
            "\n".join(json.dumps(item) for item in prm_scores) + "\n",
            encoding="utf-8",
        )


class DashboardSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = DashboardFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def test_snapshot_and_store_queries(self) -> None:
        snapshot = build_dashboard_snapshot(self.fixture.config)
        self.assertEqual(len(snapshot["skills"]), 7)
        self.assertEqual(len(snapshot["sessions"]), 10)
        self.assertEqual(len(snapshot["validation_jobs"]), 4)
        self.assertEqual(snapshot["sessions"][0]["session_id"], "local-005")
        self.assertEqual(snapshot["sessions"][0]["num_turns"], 2)

        store = DashboardStore(str(self.fixture.db_path))
        summary = store.replace_snapshot(snapshot)
        self.assertEqual(summary["skills"], 7)
        self.assertEqual(summary["sessions"], 10)

        overview = store.get_overview()
        self.assertEqual(overview["counts"]["skills"], 7)
        self.assertEqual(overview["counts"]["sessions"], 10)
        self.assertEqual(overview["counts"]["validation_jobs"], 4)
        self.assertEqual(overview["counts"]["open_validation_jobs"], 2)

        skills = store.list_skills()
        self.assertEqual(
            {item["name"] for item in skills},
            {
                "api-contract-checklist",
                "debug-notes",
                "handoff-brief",
                "incident-timeline",
                "prompt-risk-screener",
                "release-rollback-runbook",
                "sql-trace",
            },
        )
        debug_skill = next(item for item in skills if item["name"] == "debug-notes")
        self.assertEqual(debug_skill["source"], "both")
        self.assertEqual(debug_skill["local_inject_count"], 18)
        self.assertEqual(debug_skill["session_count"], 6)
        self.assertEqual(debug_skill["observed_injection_count"], 7)

        debug_detail = store.get_skill(debug_skill["skill_id"])
        self.assertIsNotNone(debug_detail)
        self.assertGreaterEqual(len(debug_detail["versions"]), 3)
        self.assertEqual(debug_detail["related_sessions"][0]["session_id"], "local-004")

        session_detail = store.get_session("sess-100")
        self.assertIsNotNone(session_detail)
        self.assertEqual(session_detail["num_turns"], 2)
        self.assertEqual(len(session_detail["links"]), 4)

        local_session = store.get_session("local-001")
        self.assertIsNotNone(local_session)
        self.assertEqual(local_session["source"], "local")
        self.assertEqual(local_session["outcome"], "failure")
        self.assertEqual(len(local_session["turns"]), 2)

        record_session = store.get_session("local-005")
        self.assertIsNotNone(record_session)
        self.assertEqual(record_session["source"], "local")
        self.assertEqual(record_session["num_turns"], 2)
        self.assertEqual(
            record_session["turns"][0]["prompt_text"],
            "Summarize the retry-worker regression and keep the answer brief.",
        )
        self.assertEqual(record_session["turns"][1]["prm_score"], 0.67)

        validation_jobs = store.list_validation_jobs()
        statuses = {item["job_id"]: item["status"] for item in validation_jobs}
        self.assertEqual(statuses["job-published"], "published")
        self.assertEqual(statuses["job-pending"], "pending")
        self.assertEqual(statuses["job-review"], "review")
        self.assertEqual(statuses["job-rejected"], "rejected")

    def test_local_sessions_are_visible_without_sharing(self) -> None:
        config = SkillClawConfig(
            use_skills=True,
            skills_dir=str(self.fixture.skills_dir),
            record_dir=str(self.fixture.root / "records"),
            sharing_enabled=False,
            dashboard_enabled=True,
            dashboard_db_path=str(self.fixture.root / "dashboard-local-only.sqlite3"),
            dashboard_sync_on_start=True,
            dashboard_include_shared=True,
        )

        snapshot = build_dashboard_snapshot(config)
        self.assertEqual(len(snapshot["skills"]), 4)
        self.assertEqual(len(snapshot["sessions"]), 5)
        self.assertEqual(snapshot["sessions"][0]["session_id"], "local-005")
        self.assertEqual(snapshot["sessions"][0]["source"], "local")

    def test_export_local_sessions_to_shared_storage(self) -> None:
        service = DashboardService(self.fixture.config)
        result = service.export_local_sessions()
        self.assertEqual(result["result"]["exported"], 5)
        exported_path = self.fixture.share_root / "team-alpha" / "sessions" / "local-001.json"
        self.assertTrue(exported_path.exists())
        exported_payload = json.loads(exported_path.read_text(encoding="utf-8"))
        self.assertEqual(exported_payload["session_id"], "local-001")
        self.assertEqual(exported_payload["source"], "local-dashboard-export")

    def test_export_selected_local_sessions_to_shared_storage(self) -> None:
        service = DashboardService(self.fixture.config)
        result = service.export_local_sessions(session_ids=["local-001", "missing-002"])
        self.assertEqual(result["selection"]["mode"], "selected")
        self.assertEqual(result["result"]["requested"], 2)
        self.assertEqual(result["result"]["matched"], 1)
        self.assertEqual(result["result"]["missing"], 1)
        self.assertEqual(result["result"]["missing_ids"], ["missing-002"])
        self.assertEqual(result["result"]["exported"], 1)
        exported_path = self.fixture.share_root / "team-alpha" / "sessions" / "local-001.json"
        self.assertTrue(exported_path.exists())


class DashboardApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = DashboardFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def test_dashboard_api_and_ui(self) -> None:
        app = create_dashboard_app(self.fixture.config)
        with TestClient(app) as client:
            index_resp = client.get("/")
            self.assertEqual(index_resp.status_code, 200)
            self.assertIn("技能演化看板", index_resp.text)
            self.assertNotIn("触发 evolve", index_resp.text)
            self.assertNotIn("推送本地 skill", index_resp.text)
            self.assertNotIn("人工审核", index_resp.text)

            health_resp = client.get("/api/v1/health")
            self.assertEqual(health_resp.status_code, 200)
            self.assertEqual(health_resp.json()["status"], "ok")

            overview_resp = client.get("/api/v1/overview")
            self.assertEqual(overview_resp.status_code, 200)
            overview = overview_resp.json()
            self.assertEqual(overview["counts"]["skills"], 7)
            self.assertEqual(overview["counts"]["sessions"], 10)

            skills_resp = client.get("/api/v1/skills")
            self.assertEqual(skills_resp.status_code, 200)
            skills = skills_resp.json()["items"]
            self.assertEqual(len(skills), 7)

            debug_skill = next(item for item in skills if item["name"] == "debug-notes")
            detail_resp = client.get(f"/api/v1/skills/{debug_skill['skill_id']}")
            self.assertEqual(detail_resp.status_code, 200)
            debug_detail = detail_resp.json()
            self.assertEqual(debug_detail["name"], "debug-notes")
            debug_v2 = next(item for item in debug_detail["versions"] if int(item["version"]) == 2)

            activate_resp = client.post(
                f"/api/v1/skills/{debug_skill['skill_id']}/activate",
                json={"target": "shared-version:2"},
            )
            self.assertEqual(activate_resp.status_code, 200)
            self.assertEqual(activate_resp.json()["target"], "shared-version:2")
            local_debug_path = self.fixture.skills_dir / "debug-notes" / "SKILL.md"
            self.assertEqual(local_debug_path.read_text(encoding="utf-8").strip(), debug_v2["skill_md"].strip())

            sessions_resp = client.get("/api/v1/sessions")
            self.assertEqual(sessions_resp.status_code, 200)
            self.assertEqual(len(sessions_resp.json()["items"]), 10)

            session_resp = client.get("/api/v1/sessions/sess-100")
            self.assertEqual(session_resp.status_code, 200)
            self.assertEqual(session_resp.json()["session_id"], "sess-100")

            validation_resp = client.get("/api/v1/validation/jobs")
            self.assertEqual(validation_resp.status_code, 200)
            self.assertEqual(len(validation_resp.json()["items"]), 4)

            export_selected_resp = client.post(
                "/api/v1/ops/export-sessions",
                json={"session_ids": ["local-001"]},
            )
            self.assertEqual(export_selected_resp.status_code, 200)
            export_selected = export_selected_resp.json()
            self.assertEqual(export_selected["selection"]["mode"], "selected")
            self.assertEqual(export_selected["result"]["requested"], 1)
            self.assertEqual(export_selected["result"]["matched"], 1)
            self.assertEqual(export_selected["result"]["exported"], 1)

            pull_selected_resp = client.post(
                "/api/v1/ops/pull",
                json={"skill_names": ["sql-trace"]},
            )
            self.assertEqual(pull_selected_resp.status_code, 200)
            pull_selected = pull_selected_resp.json()
            self.assertEqual(pull_selected["selection"]["mode"], "selected")
            self.assertEqual(pull_selected["result"]["requested"], 1)
            self.assertEqual(pull_selected["result"]["matched_remote"], 1)
            self.assertTrue((self.fixture.skills_dir / "sql-trace" / "SKILL.md").exists())

            skills_after_pull = client.get("/api/v1/skills")
            self.assertEqual(skills_after_pull.status_code, 200)
            sql_trace = next(item for item in skills_after_pull.json()["items"] if item["name"] == "sql-trace")
            self.assertEqual(sql_trace["source"], "both")

            review_resp = client.post(
                "/api/v1/validation/jobs/job-pending/review",
                json={
                    "accepted": True,
                    "score": 0.91,
                    "notes": "Looks reusable and grounded in the session evidence.",
                    "auto_finalize": False,
                },
            )
            self.assertEqual(review_resp.status_code, 200)
            review_payload = review_resp.json()
            self.assertEqual(review_payload["user_alias"], "tester")
            self.assertEqual(review_payload["result"]["accepted"], True)
            saved_result_path = self.fixture.group_dir / "validation_results" / "job-pending" / "tester.json"
            self.assertTrue(saved_result_path.exists())
            saved_result = json.loads(saved_result_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_result["decision"], "accept")
            self.assertEqual(saved_result["notes"], "Looks reusable and grounded in the session evidence.")

            validation_after_review = client.get("/api/v1/validation/jobs")
            self.assertEqual(validation_after_review.status_code, 200)
            reviewed_job = next(
                item for item in validation_after_review.json()["items"] if item["job_id"] == "job-pending"
            )
            self.assertEqual(reviewed_job["status"], "review")
            self.assertEqual(reviewed_job["result_count"], 1)
            self.assertEqual(reviewed_job["accepted_count"], 1)

            evolve_status_resp = client.get("/api/v1/evolve/status")
            self.assertEqual(evolve_status_resp.status_code, 200)
            evolve_status = evolve_status_resp.json()
            self.assertTrue(evolve_status["configured"])
            self.assertEqual(evolve_status["url"], "embedded://local-evolve")
            self.assertIn("registered_skills", evolve_status["status"])

            sync_resp = client.post("/api/v1/sync")
            self.assertEqual(sync_resp.status_code, 200)
            self.assertEqual(sync_resp.json()["summary"]["skills"], 7)


if __name__ == "__main__":
    unittest.main()
