import httpx
import pytest

from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig


@pytest.mark.asyncio
async def test_forward_to_llm_responses_preserves_codex_native_tools(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "resp_native",
                "object": "response",
                "status": "completed",
                "output": [],
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        llm_api_base="http://upstream.test/v1",
        llm_api_key="upstream-key",
        llm_model_id="upstream-model",
        llm_api_mode="responses",
    )

    body = {
        "model": "skillclaw-model",
        "input": "hi",
        "stream": True,
        "tools": [
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
            {"type": "custom", "name": "js_repl"},
            {"type": "web_search"},
            {"type": "namespace", "name": "mcp__cccc__"},
        ],
    }

    result = await server._forward_to_llm_responses(body)

    assert result["id"] == "resp_native"
    assert captured["url"] == "http://upstream.test/v1/responses"
    assert captured["json"]["model"] == "upstream-model"
    assert captured["json"]["stream"] is False
    assert captured["json"]["tools"] == body["tools"]
    assert captured["headers"] == {"Authorization": "Bearer upstream-key"}


@pytest.mark.asyncio
async def test_responses_endpoint_uses_native_forward_when_enabled():
    server = SkillClawAPIServer(
        SkillClawConfig(
            llm_api_mode="responses",
            llm_api_base="http://upstream.test/v1",
            llm_model_id="upstream-model",
            proxy_api_key="skillclaw",
            record_enabled=False,
        )
    )
    seen = {}

    async def fake_forward(body):
        seen["body"] = body
        return {
            "id": "resp_native",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "upstream-model",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "native ok", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    server._forward_to_llm_responses = fake_forward
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")
    try:
        response = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer skillclaw", "Session_id": "codex-session-1"},
            json={
                "model": "skillclaw-model",
                "input": "hi",
                "stream": False,
                "tools": [{"type": "custom", "name": "js_repl"}],
            },
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert response.json()["id"] == "resp_native"
    assert seen["body"]["tools"] == [{"type": "custom", "name": "js_repl"}]


@pytest.mark.asyncio
async def test_forward_to_llm_responses_stream_preserves_upstream_sse(monkeypatch):
    captured = {}

    class FakeStreamResponse:
        def raise_for_status(self):
            return None

        async def aiter_raw(self):
            yield b'data: {"type":"response.created"}\n\n'
            yield b'data: {"type":"response.completed"}\n\n'
            yield b'data: [DONE]\n\n'

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json, headers):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeStreamContext()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        llm_api_base="http://upstream.test/v1",
        llm_api_key="upstream-key",
        llm_model_id="upstream-model",
        llm_api_mode="responses",
    )

    body = {
        "model": "skillclaw-model",
        "input": "hi",
        "stream": True,
        "tools": [{"type": "custom", "name": "js_repl"}],
    }

    chunks = []
    async for chunk in server._stream_llm_responses(body):
        chunks.append(chunk)

    assert chunks == [
        b'data: {"type":"response.created"}\n\n',
        b'data: {"type":"response.completed"}\n\n',
        b'data: [DONE]\n\n',
    ]
    assert captured["method"] == "POST"
    assert captured["url"] == "http://upstream.test/v1/responses"
    assert captured["json"]["stream"] is True
    assert captured["json"]["model"] == "upstream-model"
    assert captured["json"]["tools"] == body["tools"]


@pytest.mark.asyncio
async def test_responses_endpoint_passthroughs_native_stream():
    server = SkillClawAPIServer(
        SkillClawConfig(
            llm_api_mode="responses",
            llm_api_base="http://upstream.test/v1",
            llm_model_id="upstream-model",
            proxy_api_key="skillclaw",
            record_enabled=False,
        )
    )

    async def fake_stream(body):
        yield b'data: {"type":"response.created","upstream":true}\n\n'
        yield b'data: [DONE]\n\n'

    server._stream_llm_responses = fake_stream
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")
    try:
        response = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer skillclaw", "Session_id": "codex-session-1"},
            json={"model": "skillclaw-model", "input": "hi", "stream": True},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert response.text == 'data: {"type":"response.created","upstream":true}\n\ndata: [DONE]\n\n'


@pytest.mark.asyncio
async def test_responses_chat_bridge_merges_previous_response_history(monkeypatch, tmp_path):
    monkeypatch.setattr(SkillClawAPIServer, "_load_tokenizer", lambda self: None)
    server = SkillClawAPIServer(
        SkillClawConfig(
            proxy_api_key="skillclaw",
            record_enabled=False,
            record_dir=str(tmp_path),
            claw_type="nanoclaw",
        )
    )
    calls = []

    async def fake_handle_request(body, session_id, turn_type, session_done):
        calls.append(body)
        idx = len(calls)
        return {
            "response": {
                "id": f"chatcmpl_{idx}",
                "created": 0,
                "choices": [
                    {"message": {"role": "assistant", "content": f"ok {idx}"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        }

    server._handle_request = fake_handle_request
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")
    try:
        first = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer skillclaw", "Session_id": "codex-session-1"},
            json={"model": "skillclaw-model", "input": "first", "store": True},
        )
        second = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer skillclaw", "Session_id": "codex-session-1"},
            json={
                "model": "skillclaw-model",
                "input": "second",
                "previous_response_id": first.json()["id"],
                "store": True,
            },
        )
    finally:
        await client.aclose()

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok 1"},
        {"role": "user", "content": "second"},
    ]


@pytest.mark.asyncio
async def test_responses_continuation_keeps_new_instructions_first(monkeypatch, tmp_path):
    monkeypatch.setattr(SkillClawAPIServer, "_load_tokenizer", lambda self: None)
    server = SkillClawAPIServer(
        SkillClawConfig(
            proxy_api_key="skillclaw",
            record_enabled=False,
            record_dir=str(tmp_path),
            claw_type="nanoclaw",
        )
    )
    calls = []

    async def fake_handle_request(body, session_id, turn_type, session_done):
        calls.append(body)
        idx = len(calls)
        return {
            "response": {
                "id": f"chatcmpl_{idx}",
                "created": 0,
                "choices": [
                    {"message": {"role": "assistant", "content": f"ok {idx}"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        }

    server._handle_request = fake_handle_request
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")
    try:
        first = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer skillclaw", "Session_id": "codex-session-1"},
            json={"model": "skillclaw-model", "input": "first", "store": True},
        )
        second = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer skillclaw", "Session_id": "codex-session-1"},
            json={
                "model": "skillclaw-model",
                "instructions": "new system instructions",
                "input": "second",
                "previous_response_id": first.json()["id"],
                "store": True,
            },
        )
    finally:
        await client.aclose()

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls[1]["messages"] == [
        {"role": "system", "content": "new system instructions"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok 1"},
        {"role": "user", "content": "second"},
    ]


@pytest.mark.asyncio
async def test_responses_continuation_deduplicates_replayed_output_items(monkeypatch, tmp_path):
    monkeypatch.setattr(SkillClawAPIServer, "_load_tokenizer", lambda self: None)
    server = SkillClawAPIServer(
        SkillClawConfig(
            proxy_api_key="skillclaw",
            record_enabled=False,
            record_dir=str(tmp_path),
            claw_type="nanoclaw",
        )
    )
    calls = []

    async def fake_handle_request(body, session_id, turn_type, session_done):
        calls.append(body)
        idx = len(calls)
        if idx == 1:
            return {
                "response": {
                    "id": "chatcmpl_1",
                    "created": 0,
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "need tool",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "Skill", "arguments": '{"name":"debug"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            }
        return {
            "response": {
                "id": f"chatcmpl_{idx}",
                "created": 0,
                "choices": [
                    {"message": {"role": "assistant", "content": f"ok {idx}"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        }

    server._handle_request = fake_handle_request
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")
    try:
        first = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer skillclaw", "Session_id": "codex-session-1"},
            json={"model": "skillclaw-model", "input": "first", "store": True},
        )
        first_payload = first.json()
        second = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer skillclaw", "Session_id": "codex-session-1"},
            json={
                "model": "skillclaw-model",
                "input": [
                    *first_payload["output"],
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "second"}],
                    },
                ],
                "previous_response_id": first_payload["id"],
                "store": True,
            },
        )
    finally:
        await client.aclose()

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls[1]["messages"] == [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "need tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Skill", "arguments": '{"name":"debug"}'},
                }
            ],
        },
        {"role": "user", "content": "second"},
    ]


def test_prepare_responses_forward_keeps_native_codex_items_out_of_chat_conversion():
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        llm_api_base="http://upstream.test/v1/",
        llm_api_key="upstream-key",
        llm_model_id="upstream-model",
        llm_api_mode="responses",
    )
    native_tools = [
        {"type": "custom", "name": "js_repl", "description": "Run JavaScript"},
        {"type": "web_search", "search_context_size": "medium"},
        {
            "type": "namespace",
            "name": "mcp__cccc__",
            "tools": [{"name": "cccc_context_get", "input_schema": {"type": "object"}}],
        },
    ]
    body = {
        "model": "skillclaw-model",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "tools": native_tools,
        "tool_choice": {"type": "custom", "name": "js_repl"},
        "parallel_tool_calls": True,
        "session_id": "private-session",
        "session_done": False,
        "turn_type": "user",
    }

    url, send_body, headers = server._prepare_responses_forward(body, stream=True)

    assert url == "http://upstream.test/v1/responses"
    assert headers == {"Authorization": "Bearer upstream-key"}
    assert send_body["model"] == "upstream-model"
    assert send_body["stream"] is True
    assert send_body["tools"] is native_tools
    assert send_body["tool_choice"] == {"type": "custom", "name": "js_repl"}
    assert send_body["parallel_tool_calls"] is True
    assert send_body["input"] == body["input"]
    assert "messages" not in send_body
    assert "session_id" not in send_body
    assert "session_done" not in send_body
    assert "turn_type" not in send_body
