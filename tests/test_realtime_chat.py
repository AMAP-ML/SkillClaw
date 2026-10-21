import asyncio
import json

import httpx
import pytest

from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig


def chunk(delta=None, finish=None):
    return (
        "data: "
        + json.dumps({"id": "chat-1", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]})
        + "\n\n"
    ).encode()


class GatedStream(httpx.AsyncByteStream):
    def __init__(self, tail=None, error=False):
        self.release = asyncio.Event()
        self.closed = False
        self.error = error
        self.tail = tail or [chunk({"content": " world"}, "stop"), b"data: [DONE]\n\n"]

    async def __aiter__(self):
        yield chunk({"role": "assistant", "content": "hello", "reasoning_content": "separate"})
        await self.release.wait()
        if self.error:
            raise httpx.ReadError("private upstream details")
        for part in self.tail:
            yield part

    async def aclose(self):
        self.closed = True


def setup(monkeypatch, tmp_path, stream, first_status=None):
    server = SkillClawAPIServer(
        SkillClawConfig(
            llm_api_base="http://primary.test/v1",
            llm_model_id="primary",
            llm_fallbacks=[{"api_base": "http://backup.test/v1", "model_id": "backup"}],
            claw_type="none",
            use_skills=False,
            use_prm=False,
            record_enabled=False,
            record_dir=str(tmp_path),
        )
    )
    requests = []

    def handler(request):
        requests.append(request)
        assert json.loads(request.content)["stream"] is True
        if len(requests) == 1 and first_status:
            return httpx.Response(first_status, json={"error": "unavailable"})
        return httpx.Response(200, stream=stream)

    client_cls = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client_cls(transport=httpx.MockTransport(handler), **kw))
    generator = server._stream_chat_and_track(
        {"model": "public-alias", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        "session-1",
        "main",
        False,
    )
    return server, generator, requests


@pytest.mark.asyncio
async def test_first_delta_arrives_before_generation_finishes_and_records_once(monkeypatch, tmp_path):
    upstream = GatedStream()
    server, stream, requests = setup(monkeypatch, tmp_path, upstream)
    first = await asyncio.wait_for(anext(stream), 1)
    payload = json.loads(first.removeprefix("data: "))
    assert payload["choices"][0]["delta"]["content"] == "hello"
    assert payload["choices"][0]["delta"]["reasoning_content"] == "separate"
    assert payload["model"] == "public-alias"
    assert server._session_turns.get("session-1", []) == []
    upstream.release.set()
    rest = [part async for part in stream]
    assert rest[-1] == "data: [DONE]\n\n"
    turns = server._session_turns["session-1"]
    assert len(turns) == 1
    assert turns[0]["response_text"] == "hello world"
    assert turns[0]["reasoning_content"] == "separate"
    assert len(requests) == 1 and upstream.closed


@pytest.mark.asyncio
async def test_tool_call_fragments_are_preserved_and_assembled_once(monkeypatch, tmp_path):
    upstream = GatedStream(
        tail=[
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":'},
                        }
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"a"}'}}]}, "tool_calls"),
            b"data: [DONE]\n\n",
        ]
    )
    upstream.release.set()
    server, stream, _ = setup(monkeypatch, tmp_path, upstream)
    parts = [part async for part in stream]
    assert len(parts) == 4
    call = server._session_turns["session-1"][0]["tool_calls"][0]
    assert call["id"] == "call-a"
    assert call["function"] == {"name": "read", "arguments": '{"path":"a"}'}


@pytest.mark.asyncio
async def test_cancellation_closes_upstream_without_recording(monkeypatch, tmp_path):
    upstream = GatedStream()
    server, stream, requests = setup(monkeypatch, tmp_path, upstream)
    await asyncio.wait_for(anext(stream), 1)
    await stream.aclose()
    assert upstream.closed and len(requests) == 1
    assert server._session_turns.get("session-1", []) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["transport", "missing_done", "invalid_json", "error_event"])
async def test_midstream_error_never_replays_or_records_success(monkeypatch, tmp_path, kind):
    tails = {
        "missing_done": [chunk({"content": "truncated"})],
        "invalid_json": [b"data: invalid\n\n"],
        "error_event": [b'data: {"error":{"message":"secret"}}\n\n'],
    }
    upstream = GatedStream(tail=tails.get(kind), error=kind == "transport")
    upstream.release.set()
    server, stream, requests = setup(monkeypatch, tmp_path, upstream)
    parts = [part async for part in stream]
    assert any('"error"' in part for part in parts)
    assert not any("[DONE]" in part or "secret" in part or "private upstream" in part for part in parts)
    assert server._session_turns.get("session-1", []) == []
    assert len(requests) == 1 and upstream.closed


@pytest.mark.asyncio
async def test_failover_before_stream_starts(monkeypatch, tmp_path):
    upstream = GatedStream()
    upstream.release.set()
    server, stream, requests = setup(monkeypatch, tmp_path, upstream, first_status=503)
    parts = [part async for part in stream]
    assert parts[-1] == "data: [DONE]\n\n"
    assert [r.url.host for r in requests] == ["primary.test", "backup.test"]
    assert len(server._session_turns["session-1"]) == 1


@pytest.mark.asyncio
async def test_stream_deadline_closes_connection(monkeypatch, tmp_path):
    upstream = GatedStream()
    server, stream, requests = setup(monkeypatch, tmp_path, upstream)
    monkeypatch.setattr("skillclaw.api_server._llm_request_timeout_seconds", lambda: 0.05)
    parts = [part async for part in stream]
    assert "504" in parts[-1] and upstream.closed
    assert server._session_turns.get("session-1", []) == []
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_chat_route_uses_live_generator_and_disables_buffering(monkeypatch, tmp_path):
    server = SkillClawAPIServer(
        SkillClawConfig(
            claw_type="none",
            use_skills=False,
            use_prm=False,
            record_enabled=False,
            record_dir=str(tmp_path),
        )
    )
    seen = []

    async def live(body, session_id, turn_type, session_done):
        seen.append((body, session_id, turn_type, session_done))
        yield 'data: {"choices":[]}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(server, "_stream_chat_and_track", live)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Session-Id": "explicit"},
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert response.status_code == 200
    assert response.headers["x-accel-buffering"] == "no"
    assert "[DONE]" in response.text and seen[0][1] == "explicit"
