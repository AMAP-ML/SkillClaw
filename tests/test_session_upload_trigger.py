from __future__ import annotations

import httpx
import pytest

from skillclaw.api_server import (
    _PROTOCOL_ANTHROPIC_MESSAGES,
    SkillClawAPIServer,
    _classify_raw_turn_kind,
    _is_user_turn_boundary,
)
from skillclaw.config import SkillClawConfig


def _server_for_snapshot_tests() -> SkillClawAPIServer:
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        sharing_enabled=True,
        sharing_session_upload_interval=2,
        evolve_server_url="http://evolve.test",
    )
    server._user_turn_counts = {}
    server._session_turns = {
        "session-a": [
            {"turn_num": 1, "prompt_text": "one"},
            {"turn_num": 2, "prompt_text": "two"},
        ]
    }
    return server


def test_session_snapshot_upload_only_queues_on_configured_interval() -> None:
    server = _server_for_snapshot_tests()
    queued = []

    def fake_create_task(coro):
        queued.append(coro)
        return None

    server._safe_create_task = fake_create_task

    server._maybe_upload_session_snapshot("session-a", 1)
    assert queued == []

    server._maybe_upload_session_snapshot("session-a", 2)
    assert len(queued) == 1

    queued[0].close()


def test_session_snapshot_upload_uses_stable_deep_copy() -> None:
    server = _server_for_snapshot_tests()
    queued = []
    captured = {}

    class DummyCoro:
        def close(self):
            return None

    def fake_create_task(coro):
        queued.append(coro)
        return None

    def fake_upload_snapshot(session_id, turns):
        captured["session_id"] = session_id
        captured["turns"] = turns
        return DummyCoro()

    server._safe_create_task = fake_create_task
    server._upload_session_snapshot_and_trigger = fake_upload_snapshot
    server._session_turns["session-a"][1]["tool_results"] = [{"text": "before"}]

    server._maybe_upload_session_snapshot("session-a", 2)
    server._session_turns["session-a"][1]["tool_results"][0]["text"] = "after"

    assert len(queued) == 1
    assert queued[0] is not None
    queued[0].close()
    assert captured["session_id"] == "session-a"
    assert captured["turns"][1]["tool_results"][0]["text"] == "before"


@pytest.mark.anyio
async def test_session_snapshot_triggers_evolve_only_after_successful_upload() -> None:
    server = _server_for_snapshot_tests()
    calls = {"upload": 0, "trigger": 0}

    async def fake_upload(_session_id, _turns):
        calls["upload"] += 1
        return calls["upload"] == 1

    async def fake_trigger():
        calls["trigger"] += 1

    server._upload_session_data = fake_upload
    server._trigger_evolve = fake_trigger

    await server._upload_session_snapshot_and_trigger("session-a", [{"turn_num": 1}])
    await server._upload_session_snapshot_and_trigger("session-a", [{"turn_num": 2}])

    assert calls == {"upload": 2, "trigger": 1}


def test_user_turn_cadence_counts_visible_turns_not_raw_turns() -> None:
    server = _server_for_snapshot_tests()
    queued = []

    def fake_create_task(coro):
        queued.append(coro)
        return None

    server._safe_create_task = fake_create_task
    server._session_turns["session-a"].append({"turn_num": 3, "raw_turn_kind": "tool_use"})

    assert server._advance_user_turn_and_maybe_upload("session-a") == 1
    assert queued == []

    assert server._advance_user_turn_and_maybe_upload("session-a") == 2
    assert len(queued) == 1

    queued[0].close()


def test_claude_internal_turns_do_not_advance_user_turn_boundary() -> None:
    assert (
        _classify_raw_turn_kind(
            _PROTOCOL_ANTHROPIC_MESSAGES,
            '{"title":"Weekly report"}',
            [],
        )
        == "session_title"
    )
    assert (
        _classify_raw_turn_kind(
            _PROTOCOL_ANTHROPIC_MESSAGES,
            "",
            [{"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
        )
        == "tool_use"
    )
    assert not _is_user_turn_boundary("session_title")
    assert not _is_user_turn_boundary("tool_use")
    assert _is_user_turn_boundary(_classify_raw_turn_kind(_PROTOCOL_ANTHROPIC_MESSAGES, "final answer", []))


def test_openclaw_and_hermes_main_turns_use_user_turn_upload_cadence() -> None:
    server = _server_for_snapshot_tests()
    captured = {}

    class DummyCoro:
        def close(self):
            return None

    def fake_create_task(coro):
        return None

    def fake_upload_snapshot(session_id, turns):
        captured["session_id"] = session_id
        captured["turns"] = turns
        return DummyCoro()

    server._safe_create_task = fake_create_task
    server._upload_session_snapshot_and_trigger = fake_upload_snapshot
    server._session_turns["session-a"] = []

    for raw_turn_num in range(1, 4):
        turn_record = {
            "turn_num": raw_turn_num,
            "raw_turn_kind": _classify_raw_turn_kind("", f"answer {raw_turn_num}", []),
            "prompt_text": f"user {raw_turn_num}",
        }
        server._session_turns["session-a"].append(turn_record)
        user_turn_num = server._next_user_turn_num("session-a")
        turn_record["user_turn_num"] = user_turn_num
        server._maybe_upload_session_snapshot("session-a", user_turn_num)

    assert server._user_turn_counts["session-a"] == 3
    assert captured["session_id"] == "session-a"
    assert [t["turn_num"] for t in captured["turns"]] == [1, 2]
    assert [t["user_turn_num"] for t in captured["turns"]] == [1, 2]


def test_skill_reload_polling_starts_only_in_poll_mode(monkeypatch) -> None:
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(sharing_enabled=True, sharing_skill_reload_mode="poll")
    server._skill_reload_task = None
    server._skill_reload_interval_seconds = 30
    created = []

    class FakeTask:
        def done(self):
            return False

        def add_done_callback(self, _callback):
            return None

    def fake_create_task(coro):
        created.append(coro)
        return FakeTask()

    import asyncio

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    server._start_skill_reload_polling()

    assert len(created) == 1
    created[0].close()


def test_skill_reload_polling_does_not_start_when_disabled_or_callback(monkeypatch) -> None:
    created = []

    class FakeTask:
        def done(self):
            return False

    def fake_create_task(coro):
        created.append(coro)
        return FakeTask()

    import asyncio

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    for mode in ("off", "callback"):
        server = object.__new__(SkillClawAPIServer)
        server.config = SkillClawConfig(sharing_enabled=True, sharing_skill_reload_mode=mode)
        server._skill_reload_task = None
        server._skill_reload_interval_seconds = 30
        server._start_skill_reload_polling()

    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(sharing_enabled=False, sharing_skill_reload_mode="poll")
    server._skill_reload_task = None
    server._skill_reload_interval_seconds = 30
    server._start_skill_reload_polling()

    assert created == []


@pytest.mark.anyio
async def test_internal_reload_skills_endpoint_requires_auth_and_pulls(tmp_path) -> None:
    server = SkillClawAPIServer(
        SkillClawConfig(
            proxy_api_key="secret",
            record_enabled=False,
            record_dir=str(tmp_path),
        )
    )
    calls = {"pull": 0}

    async def fake_pull(skip_names=None):
        assert skip_names is None
        calls["pull"] += 1

    class FakeSkillManager:
        def get_all_skills(self):
            return [{"name": "weekly-report"}, {"name": "demo"}]

    server._pull_skills_from_cloud = fake_pull
    server.skill_manager = FakeSkillManager()

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")
    try:
        unauthorized = await client.post("/internal/reload-skills")
        authorized = await client.post(
            "/internal/reload-skills",
            headers={"Authorization": "Bearer secret"},
        )
    finally:
        await client.aclose()

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json() == {"ok": True, "skills": 2}
    assert calls == {"pull": 1}
