from __future__ import annotations

import pytest

from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig


def _server_for_snapshot_tests() -> SkillClawAPIServer:
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        sharing_enabled=True,
        sharing_session_upload_interval=2,
        evolve_server_url="http://evolve.test",
    )
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
