from skillclaw.protocols import openai_responses


def test_codex_responses_message_items_convert_developer_to_system():
    body = {
        "model": "skillclaw-model",
        "instructions": "base instructions",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "repo instructions"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "do work"}],
            },
        ],
    }

    converted = openai_responses.to_openai_body(body, "fallback-model")

    assert converted["messages"] == [
        {"role": "system", "content": "base instructions"},
        {"role": "system", "content": "repo instructions"},
        {"role": "user", "content": "do work"},
    ]


def test_codex_responses_drops_non_chat_completion_tools_but_keeps_function_tools():
    body = {
        "model": "skillclaw-model",
        "input": "hi",
        "tools": [
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
            {"type": "custom", "name": "js_repl"},
            {"type": "web_search"},
            {"type": "namespace", "name": "mcp__cccc__"},
        ],
    }

    converted = openai_responses.to_openai_body(body, "fallback-model")

    assert converted["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "",
                "parameters": {"type": "object"},
            },
        }
    ]


async def _collect_response_stream(response_payload):
    events = []
    async for chunk in openai_responses.stream_response(response_payload):
        if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
            events.append(chunk.removeprefix("data: ").strip())
        elif chunk.strip() == "data: [DONE]":
            events.append("[DONE]")
    return events


def test_codex_responses_stream_includes_response_completed_and_done():
    import asyncio
    import json

    response_payload = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "skillclaw-model",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    events = asyncio.run(_collect_response_stream(response_payload))
    parsed = [json.loads(event) for event in events if event != "[DONE]"]

    assert any(event["type"] == "response.completed" for event in parsed)
    assert events[-1] == "[DONE]"


def test_codex_responses_multimodal_image_input_converts_to_openai_chat_content_parts():
    body = {
        "model": "skillclaw-model",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe this image"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        ],
    }

    converted = openai_responses.to_openai_body(body, "fallback-model")

    assert converted["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]


def test_codex_responses_skill_function_call_and_output_convert_to_chat_tool_roundtrip():
    body = {
        "model": "skillclaw-model",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_skill_1",
                "name": "Skill",
                "arguments": {"name": "paper-reviewer"},
            },
            {
                "type": "function_call_output",
                "call_id": "call_skill_1",
                "output": "Loaded paper-reviewer skill instructions",
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "continue"}],
            },
        ],
    }

    converted = openai_responses.to_openai_body(body, "fallback-model")

    assert converted["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_skill_1",
                    "type": "function",
                    "function": {"name": "Skill", "arguments": '{"name": "paper-reviewer"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_skill_1",
            "content": "Loaded paper-reviewer skill instructions",
        },
        {"role": "user", "content": "continue"},
    ]


def test_openai_chat_tool_call_response_converts_to_codex_responses_function_call_item():
    payload = {
        "id": "chatcmpl_tool",
        "created": 1,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_skill_1",
                            "type": "function",
                            "function": {"name": "Skill", "arguments": '{"name":"paper-reviewer"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    converted = openai_responses.from_openai_chat_payload(payload, "skillclaw-model")

    assert converted["output"] == [
        {
            "type": "function_call",
            "id": "fc_skill_1",
            "call_id": "call_skill_1",
            "name": "Skill",
            "arguments": '{"name":"paper-reviewer"}',
            "status": "completed",
        }
    ]


def test_codex_responses_chat_bridge_converts_function_tool_choice_shortcut():
    body = {
        "model": "skillclaw-model",
        "input": "Use the skill",
        "tools": [{"type": "function", "name": "Skill", "parameters": {"type": "object"}}],
        "tool_choice": {"type": "function", "name": "Skill"},
    }

    converted = openai_responses.to_openai_body(body, "fallback-model")

    assert converted["tool_choice"] == {"type": "function", "function": {"name": "Skill"}}


def test_codex_responses_chat_bridge_rejects_native_tool_choice():
    body = {
        "model": "skillclaw-model",
        "input": "Use js",
        "tools": [{"type": "custom", "name": "js_repl"}],
        "tool_choice": {"type": "custom", "name": "js_repl"},
    }

    try:
        openai_responses.to_openai_body(body, "fallback-model")
    except ValueError as exc:
        assert "not supported in chat bridge mode" in str(exc)
    else:
        raise AssertionError("expected native tool_choice to be rejected")
