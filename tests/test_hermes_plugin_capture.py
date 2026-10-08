import importlib.util
import json
from pathlib import Path

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


def test_register_is_inert_without_root(monkeypatch):
    monkeypatch.delenv("SKILLCLAW_CAPTURE_ROOT", raising=False)
    hooks = []

    class Ctx:
        def register_hook(self, name, fn):
            hooks.append(name)

    plugin.register(Ctx())
    assert hooks == []
    monkeypatch.setenv("SKILLCLAW_CAPTURE_ROOT", "/tmp/x")
    plugin.register(Ctx())
    assert hooks == ["post_llm_call", "on_session_finalize"]


def test_scrub_removes_tokens():
    out = plugin.scrub("Authorization: Bearer abc.def-ghi and sk-" + "a" * 30)
    assert "abc.def-ghi" not in out and "a" * 30 not in out
