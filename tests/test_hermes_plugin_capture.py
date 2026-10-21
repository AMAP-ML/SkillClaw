import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

from evolve_server.core.constants import NO_SKILL_KEY
from evolve_server.pipeline.aggregation import aggregate_sessions_by_skill
from evolve_server.pipeline.summarizer import _extract_session_metadata

_PLUGIN = Path(__file__).resolve().parents[1] / "integrations" / "hermes_plugin" / "skillclaw_capture" / "__init__.py"
_spec = importlib.util.spec_from_file_location("skillclaw_capture_plugin", _PLUGIN)
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


def _call(name, args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def _turn(n, calls, session="sess-A"):
    return {
        "session_id": session,
        "turn_id": f"t{n}",
        "user_message": f"q{n}",
        "assistant_response": f"a{n}",
        "model": "m",
        "conversation_history": [
            {"role": "system", "content": "SYSTEM_PROMPT"},
            {"role": "user", "content": f"q{n}"},
            {"role": "assistant", "tool_calls": calls},
            {"role": "tool", "content": "TOOL_RESULT"},
        ],
    }


def _queue(root, group="g"):
    return sorted((root / group / "sessions").glob("*.json"))


def test_turns_of_one_session_form_one_record_with_skill_refs(tmp_path):
    cap = plugin.SessionCapture(tmp_path, group_id="g", user_alias="alice")
    cap.on_turn(**_turn(1, [_call("skill_view", {"name": "web/blocked-page", "file_path": "references/x.md"})]))
    cap.on_turn(**_turn(2, [_call("terminal", {"command": "SECRET_ARG"})]))
    cap.on_turn(**_turn(3, [_call("skill_manage", {"action": "patch", "name": "debug-helper", "content": "BODY"})]))

    assert cap.flush_idle(idle_seconds=3600) == []  # still open
    assert cap.flush_idle(idle_seconds=0)
    [path] = _queue(tmp_path)
    rec = json.loads(path.read_text())
    assert rec["num_turns"] == 3
    assert [t["prompt_text"] for t in rec["turns"]] == ["q1", "q2", "q3"]
    assert rec["turns"][0]["read_skills"] == [{"skill_id": "", "skill_name": "blocked-page", "path": "references/x.md"}]
    assert rec["turns"][2]["modified_skills"] == [{"skill_id": "", "skill_name": "debug-helper"}]
    text = path.read_text()
    for leaked in ("SECRET_ARG", "BODY", "SYSTEM_PROMPT", "TOOL_RESULT"):
        assert leaked not in text


def test_record_is_grouped_by_skill_by_the_evolve_pipeline(tmp_path):
    cap = plugin.SessionCapture(tmp_path, group_id="g", user_alias="alice")
    cap.on_turn(**_turn(1, [_call("skill_view", {"name": "api-helper"})]))
    cap.flush_idle(idle_seconds=0)
    [path] = _queue(tmp_path)
    session = json.loads(path.read_text())
    _extract_session_metadata(session)
    grouped = aggregate_sessions_by_skill([session])
    assert "api-helper" in grouped and NO_SKILL_KEY not in grouped


def test_duplicate_hook_delivery_is_ignored(tmp_path):
    cap = plugin.SessionCapture(tmp_path, group_id="g", user_alias="alice")
    cap.on_turn(**_turn(1, []))
    cap.on_turn(**_turn(1, []))
    cap.flush_idle(idle_seconds=0)
    [path] = _queue(tmp_path)
    assert json.loads(path.read_text())["num_turns"] == 1


def test_finalize_publishes_and_long_sessions_split_with_unique_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin, "MAX_TURNS", 2)
    cap = plugin.SessionCapture(tmp_path, group_id="g", user_alias="alice")
    for n in range(5):
        cap.on_turn(**_turn(n, []))
    cap.on_session_closed(session_id="sess-A")
    recs = [json.loads(p.read_text()) for p in _queue(tmp_path)]
    assert sorted(r["num_turns"] for r in recs) == [1, 2, 2]
    assert len({r["session_id"] for r in recs}) == 3


def test_internal_and_empty_turns_are_skipped(tmp_path):
    cap = plugin.SessionCapture(tmp_path, group_id="g", user_alias="alice")
    cap.on_turn(**_turn(1, [], session="skillclaw-evolve-x"))
    cap.on_turn(session_id="s", turn_id="t", assistant_response="")
    cap.flush_idle(idle_seconds=0)
    assert _queue(tmp_path) == []


def test_register_is_inert_without_root(monkeypatch, tmp_path):
    monkeypatch.delenv("SKILLCLAW_CAPTURE_ROOT", raising=False)
    hooks = []

    class Ctx:
        def register_hook(self, name, fn):
            hooks.append(name)

        def on_unload(self, fn):
            self.stop = fn

    ctx = Ctx()
    plugin.register(ctx)
    assert hooks == []
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("SKILLCLAW_CAPTURE_ROOT", str(tmp_path))
    plugin.register(ctx)
    try:
        assert hooks == ["post_llm_call", "on_session_finalize", "on_session_reset"]
    finally:
        ctx.stop()


def test_scrub_removes_tokens():
    out = plugin.scrub("Authorization: Bearer abc.def-ghi and sk-" + "a" * 30)
    assert "abc.def-ghi" not in out and "a" * 30 not in out


@pytest.mark.parametrize("action", [[], {}])
def test_malformed_operation_does_not_drop_finished_turn(tmp_path, action):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    cap.on_turn(**_turn(1, [_call("skill_manage", {"operations": [{"action": action, "name": "x"}]})]))
    cap.flush_idle(0)
    [path] = _queue(tmp_path)
    turn = json.loads(path.read_text())["turns"][0]
    assert turn["response_text"] == "a1"
    assert turn["modified_skills"] == []


def test_idle_worker_publishes_without_later_turn_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLCLAW_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv("SKILLCLAW_GROUP_ID", "g")
    monkeypatch.setenv("SKILLCLAW_CAPTURE_IDLE_SECONDS", "0.02")
    hooks = {}

    class Ctx:
        def register_hook(self, name, fn):
            hooks[name] = fn

        def on_unload(self, fn):
            self.stop = fn

    ctx = Ctx()
    plugin.register(ctx)
    cap = hooks["post_llm_call"].__self__
    try:
        hooks["post_llm_call"](**_turn(1, []))
        deadline = time.monotonic() + 3
        while not _queue(tmp_path) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(_queue(tmp_path)) == 1
    finally:
        ctx.stop()
    assert not cap._worker.is_alive()


def test_operations_schema_and_parallel_calls_are_attributed(tmp_path):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    calls = [
        _call("skill_view", {"name": "plugin:reader"}),
        _call("skill_view", {"name": "category/other", "file_path": "references/a.md"}),
        _call(
            "skill_manage",
            {
                "operations": [
                    {"action": "patch", "name": "category/writer", "old_string": "PRIVATE", "new_string": "BODY"},
                    {"action": "write_file", "name": "writer2", "file_content": "SECRET"},
                ]
            },
        ),
    ]
    cap.on_turn(**_turn(1, calls))
    cap.flush_idle(0)
    [path] = _queue(tmp_path)
    turn = json.loads(path.read_text())["turns"][0]
    assert {r["skill_name"] for r in turn["read_skills"]} == {"plugin:reader", "other"}
    assert {r["skill_name"] for r in turn["modified_skills"]} == {"writer", "writer2"}
    assert not any(s in path.read_text() for s in ("PRIVATE", "BODY", "SECRET"))


@pytest.mark.parametrize("bad", ["{broken", "[]", '{"turns": []}', '{"turns": "invalid"}'])
def test_corrupt_state_is_preserved_on_append_and_flush(tmp_path, bad):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    cap.on_turn(**_turn(1, []))
    [path] = cap.open_dir.glob("*.json")
    path.write_text(bad)
    with pytest.raises(ValueError):
        cap.on_turn(**_turn(2, []))
    assert path.read_text() == bad
    assert cap.flush_idle(0) == []
    assert path.read_text() == bad
    assert not _queue(tmp_path)


def test_profile_boundary_includes_custom_home_not_only_alias(tmp_path):
    a = plugin.SessionCapture(tmp_path, group_id="g", user_alias="same", profile_id="/homes/a")
    b = plugin.SessionCapture(tmp_path, group_id="g", user_alias="same", profile_id="/homes/b")
    a.on_turn(**_turn(1, []))
    assert b.flush_idle(0) == []
    b.on_turn(**_turn(1, []))
    a.flush_idle(0)
    b.flush_idle(0)
    assert len(_queue(tmp_path)) == 2
    assert a.state != b.state


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_id": "../escape"},
        {"group_id": "/absolute"},
        {"user_alias": "../../escape"},
        {"idle_seconds": -1},
        {"idle_seconds": float("nan")},
    ],
)
def test_invalid_config_cannot_escape_capture_root(tmp_path, kwargs):
    with pytest.raises(ValueError):
        plugin.SessionCapture(tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


def test_retry_after_record_before_receipt_crash_does_not_duplicate(tmp_path, monkeypatch):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    original = plugin._atomic_json

    def crash_after_record(path, value):
        original(path, value)
        if path.parent == cap.open_dir:
            raise OSError("simulated death after record commit")

    with monkeypatch.context() as m:
        m.setattr(plugin, "_atomic_json", crash_after_record)
        with pytest.raises(OSError):
            cap.on_turn(**_turn(1, []))
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    cap.on_turn(**_turn(1, []))
    cap.flush_idle(0)
    [path] = _queue(tmp_path)
    assert json.loads(path.read_text())["num_turns"] == 1
    path.unlink()  # consumer has removed the queue file
    cap.on_turn(**_turn(1, []))
    cap.flush_idle(0)
    assert not _queue(tmp_path)


def test_publish_moves_atomically_and_receipts_survive_consumption(tmp_path, monkeypatch):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    cap.on_turn(**_turn(1, []))
    original = os.replace

    def crash_after_move(src, dst):
        original(src, dst)
        if Path(dst).parent == cap.queue:
            raise OSError("simulated death after queue rename")

    with monkeypatch.context() as m:
        m.setattr(plugin.os, "replace", crash_after_move)
        with pytest.raises(OSError):
            cap.on_session_closed(session_id="sess-A")
    [queued] = _queue(tmp_path)
    queued.unlink()
    recovered = plugin.SessionCapture(tmp_path, group_id="g")
    recovered.flush_idle(0)
    recovered.on_turn(**_turn(1, []))
    recovered.flush_idle(0)
    assert not _queue(tmp_path)
    assert not list(recovered.open_dir.glob("*.json"))


def test_stale_pid_temp_does_not_block_atomic_write(tmp_path):
    path = tmp_path / "record.json"
    stale = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    stale.write_text("left after crash")
    plugin._atomic_json(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    assert path.stat().st_mode & 0o777 == 0o600


def test_reset_closes_only_old_session(tmp_path):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    cap.on_turn(**_turn(1, [], session="old"))
    cap.on_turn(**_turn(1, [], session="new"))
    cap.on_session_closed(old_session_id="old", new_session_id="new")
    assert len(_queue(tmp_path)) == 1
    assert len(list(cap.open_dir.glob("*.json"))) == 1


def test_recovery_flush_repairs_receipts_without_hook_redelivery(tmp_path, monkeypatch):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    original = plugin._atomic_json

    def missing_receipt(path, record):
        if path.parent == cap.seen_dir:
            raise OSError("receipt unavailable")
        original(path, record)

    with monkeypatch.context() as m:
        m.setattr(plugin, "_atomic_json", missing_receipt)
        with pytest.raises(OSError):
            cap.on_turn(**_turn(1, []))
    recovered = plugin.SessionCapture(tmp_path, group_id="g")
    recovered.flush_idle(0)
    [queued] = _queue(tmp_path)
    queued.unlink()
    recovered.on_turn(**_turn(1, []))
    recovered.flush_idle(0)
    assert not _queue(tmp_path)


def test_failed_handoff_keeps_original_until_retry(tmp_path, monkeypatch):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    cap.on_turn(**_turn(1, []))
    [opened] = cap.open_dir.glob("*.json")
    original = opened.read_bytes()
    with monkeypatch.context() as m:
        m.setattr(plugin.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("unavailable")))
        assert cap.flush_idle(0) == []
    assert opened.read_bytes() == original
    assert not _queue(tmp_path)
    assert len(cap.flush_idle(0)) == 1


def test_bad_record_does_not_block_other_sessions(tmp_path):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    cap.on_turn(**_turn(1, [], session="bad"))
    [bad] = cap.open_dir.glob("*.json")
    bad.write_text("{broken")
    cap.on_turn(**_turn(1, [], session="good"))
    assert len(cap.flush_idle(0)) == 1
    assert bad.read_text() == "{broken"


def test_current_turn_boundary_and_malformed_calls():
    history = [
        {"role": "assistant", "tool_calls": [_call("skill_view", {"name": "old"})]},
        {"role": "user", "content": "new"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": "bad"},
                {"function": {"name": "skill_manage", "arguments": "broken"}},
                _call("skill_manage", {"operations": [None, {"action": "unknown", "name": "skip"}]}),
                _call("skill_view", {"name": "current"}),
            ],
        },
    ]
    names, read, modified = plugin.current_turn_tools(history)
    assert names == ["skill_manage", "skill_view"]
    assert read == [{"skill_id": "", "skill_name": "current"}]
    assert modified == []


def test_size_split_preserves_turns_and_rejects_oversized_single_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin, "MAX_RECORD_BYTES", 1200)
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    for n in range(3):
        cap.on_turn(**{**_turn(n, []), "assistant_response": "x" * 300})
    cap.flush_idle(0)
    records = [json.loads(p.read_text()) for p in _queue(tmp_path)]
    assert sum(r["num_turns"] for r in records) == 3
    assert len(records) == 3
    with pytest.raises(ValueError):
        cap.on_turn(**{**_turn(9, []), "assistant_response": "x" * 2000})
    assert not list(cap.open_dir.glob("*.json"))
    assert len(_queue(tmp_path)) == 3


def test_same_session_multiple_instances_serialize_and_dedupe(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    captures = [plugin.SessionCapture(tmp_path, group_id="g") for _ in range(4)]
    ready = threading.Event()
    acquired = threading.Event()

    def contender():
        ready.set()
        with captures[1]._lock():
            acquired.set()

    worker = threading.Thread(target=contender, daemon=True)
    try:
        with captures[0]._lock():
            worker.start()
            assert ready.wait(2)
            assert not acquired.wait(0.1), "second capture entered an already-held state lock"
    finally:
        worker.join(2)
    assert acquired.is_set()
    assert not worker.is_alive()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(captures[n % 4].on_turn, **_turn(n // 2, [])) for n in range(16)]
        for future in futures:
            future.result()
    captures[0].flush_idle(0)
    [queued] = _queue(tmp_path)
    assert json.loads(queued.read_text())["num_turns"] == 8


def test_stopped_capture_does_not_accept_or_publish_turns(tmp_path):
    cap = plugin.SessionCapture(tmp_path, group_id="g")
    cap.on_turn(**_turn(1, []))
    cap.stop()
    cap.on_turn(**_turn(2, []))
    cap.on_session_closed(session_id="sess-A")
    assert cap.flush_idle(0) == []
    assert not _queue(tmp_path)
    [opened] = cap.open_dir.glob("*.json")
    assert json.loads(opened.read_text())["num_turns"] == 1


def test_registration_binds_custom_profile_home(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "custom-profile"))
    monkeypatch.setenv("SKILLCLAW_CAPTURE_ROOT", str(tmp_path / "capture"))
    hooks = {}

    class Ctx:
        def register_hook(self, name, fn):
            hooks[name] = fn

        def on_unload(self, fn):
            self.stop = fn

    ctx = Ctx()
    plugin.register(ctx)
    try:
        assert plugin._home() == (tmp_path / "custom-profile").resolve()
        cap = hooks["post_llm_call"].__self__
        expected = plugin.SessionCapture(tmp_path / "capture", user_alias="default", profile_id=str(plugin._home()))
        assert cap.state == expected.state
    finally:
        ctx.stop()
