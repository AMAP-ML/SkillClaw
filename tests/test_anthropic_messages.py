from skillclaw.protocols import anthropic_messages


def test_anthropic_tool_result_blocks_convert_to_openai_tool_messages():
    body = {
        "model": "claude-code-test",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Skill", "input": {"name": "debug"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Skill instructions"}
                ],
            },
        ],
    }

    converted = anthropic_messages.to_openai_body(body)

    assert converted["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "Skill", "arguments": '{"name": "debug"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "Skill instructions"},
    ]


def test_openai_tool_calls_convert_to_anthropic_tool_use_blocks():
    openai_resp = {
        "id": "chatcmpl_1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Skill", "arguments": '{"name":"debug"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }

    converted = anthropic_messages.from_openai_response(openai_resp, "claude-code-test")

    assert converted["stop_reason"] == "tool_use"
    assert converted["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "Skill", "input": {"name": "debug"}}
    ]


async def _collect_stream_events(result, model):
    events = []
    async for chunk in anthropic_messages.stream_from_openai_result(result, model):
        if not chunk.startswith("event: "):
            continue
        header, data_line = chunk.strip().split("\n", 1)
        events.append((header.removeprefix("event: "), data_line.removeprefix("data: ")))
    return events


def test_streaming_openai_tool_calls_emit_anthropic_tool_use_events():
    import asyncio
    import json
    result = {
        "response": {
            "id": "chatcmpl_1",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Skill", "arguments": '{"name":"debug"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    }

    events = asyncio.run(_collect_stream_events(result, "claude-code-test"))
    parsed = [(name, json.loads(data)) for name, data in events]

    assert any(
        name == "content_block_start"
        and payload["content_block"] == {"type": "tool_use", "id": "call_1", "name": "Skill", "input": {}}
        for name, payload in parsed
    )
    assert any(
        name == "content_block_delta"
        and payload["delta"] == {"type": "input_json_delta", "partial_json": '{"name":"debug"}'}
        for name, payload in parsed
    )



def test_streaming_openai_tool_calls_use_tool_use_stop_reason_even_if_finish_reason_is_stop():
    import asyncio
    import json
    result = {
        "response": {
            "id": "chatcmpl_1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Skill", "arguments": '{"name":"debug"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    }

    events = asyncio.run(_collect_stream_events(result, "claude-code-test"))
    parsed = [(name, json.loads(data)) for name, data in events]

    assert any(
        name == "message_delta"
        and payload["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
        for name, payload in parsed
    )

def test_anthropic_system_blocks_preserve_text_and_cache_control():
    body = {
        "model": "claude-code-test",
        "system": [
            {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Use tools carefully."},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }

    converted = anthropic_messages.to_openai_body(body)

    assert converted["messages"][0] == {
        "role": "system",
        "content": "You are Claude Code. Use tools carefully.",
    }


def test_anthropic_tools_and_tool_choice_convert_to_openai_function_schema():
    body = {
        "model": "claude-code-test",
        "messages": [{"role": "user", "content": "Use a skill"}],
        "tools": [
            {
                "name": "Skill",
                "description": "Load a named skill",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "Skill"},
    }

    converted = anthropic_messages.to_openai_body(body)

    assert converted["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": "Load a named skill",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
    ]
    assert converted["tool_choice"] == {"type": "function", "function": {"name": "Skill"}}


def test_openai_response_with_tool_calls_uses_tool_use_stop_reason_even_if_finish_reason_is_stop():
    openai_resp = {
        "id": "chatcmpl_1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Skill", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }

    converted = anthropic_messages.from_openai_response(openai_resp, "claude-code-test")

    assert converted["stop_reason"] == "tool_use"



def test_anthropic_server_web_search_tool_is_not_converted_to_function_tool():
    body = {
        "model": "claude-code-test",
        "messages": [{"role": "user", "content": "search docs"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "Skill", "description": "Load skill", "input_schema": {"type": "object"}},
        ],
    }

    converted = anthropic_messages.to_openai_body(body)

    assert converted["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": "Load skill",
                "parameters": {"type": "object"},
            },
        }
    ]

def test_anthropic_multimodal_image_input_converts_to_openai_chat_content_parts():
    body = {
        "model": "claude-code-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AAAA",
                        },
                    },
                ],
            }
        ],
    }

    converted = anthropic_messages.to_openai_body(body)

    assert converted["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
