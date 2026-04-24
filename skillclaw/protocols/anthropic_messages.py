"""Anthropic Messages compatibility for Claude Code style clients."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from .common import json_dumps_tool_args, json_loads_tool_input

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}


def _flatten_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif "content" in item:
                    parts.append(_flatten_tool_result_content(item.get("content")))
            elif item is not None:
                parts.append(str(item))
        return " ".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _image_block_to_openai_part(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    if source.get("type") == "base64":
        media_type = str(source.get("media_type") or "image/png")
        data = str(source.get("data") or "")
        if data:
            return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    url = source.get("url") or block.get("url") or block.get("image_url")
    if isinstance(url, str) and url:
        return {"type": "image_url", "image_url": {"url": url}}
    return None


def _tools_to_openai_tools(tools: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted
    for item in tools:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        # Anthropic server tools are not client function tools; a chat upstream
        # cannot execute them unless they are handled by a native protocol path.
        if item_type.startswith("web_search") or item_type in {"server_tool_use", "web_search_tool_result"}:
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(item.get("description") or ""),
                    "parameters": item.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _tool_choice_to_openai(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return "required" if tool_choice == "any" else tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        name = str(tool_choice.get("name") or "").strip()
        if name:
            return {"type": "function", "function": {"name": name}}
    return tool_choice


def to_openai_body(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic /v1/messages request body to OpenAI chat format."""
    messages: list[dict[str, Any]] = list(body.get("messages", []))

    system = body.get("system")
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = " ".join(
                blk.get("text", "") for blk in system if isinstance(blk, dict) and blk.get("type") == "text"
            )
        else:
            system_text = str(system)
        messages = [{"role": "system", "content": system_text}] + messages

    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list):
            normalized.append(msg)
            continue

        text_parts: list[str] = []
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for idx, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
                    content_parts.append({"type": "text", "text": text})
            elif block_type == "image":
                image_part = _image_block_to_openai_part(block)
                if image_part:
                    content_parts.append(image_part)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or f"toolu_{idx}"),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or "unknown_tool"),
                            "arguments": json_dumps_tool_args(block.get("input")),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": _flatten_tool_result_content(block.get("content")),
                    }
                )

        text = " ".join(text_parts).strip()
        has_image = any(part.get("type") == "image_url" for part in content_parts)
        openai_content: str | list[dict[str, Any]] = content_parts if has_image else text
        if role == "assistant":
            assistant_msg = {**msg, "content": text}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            normalized.append(assistant_msg)
            continue
        if tool_results:
            normalized.extend(tool_results)
            if text:
                normalized.append({**msg, "content": text})
            continue
        normalized.append({**msg, "content": openai_content})

    openai_body: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": normalized,
        "max_tokens": body.get("max_tokens", 2048),
    }
    tools = _tools_to_openai_tools(body.get("tools"))
    if tools:
        openai_body["tools"] = tools
    if "tool_choice" in body:
        openai_body["tool_choice"] = _tool_choice_to_openai(body.get("tool_choice"))
    for opt in ("temperature", "top_p", "stop_sequences", "stream"):
        if opt in body:
            key = "stop" if opt == "stop_sequences" else opt
            openai_body[key] = body[opt]
    return openai_body


def from_openai_response(openai_resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert an OpenAI chat completion response to Anthropic /v1/messages format."""
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
    content_text = message.get("content") or ""
    raw_tool_calls = message.get("tool_calls")
    tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = "tool_use" if tool_calls else _STOP_REASON_MAP.get(finish_reason, "end_turn")

    content_blocks: list[dict[str, Any]] = []
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})
    for idx, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or f"call_{idx}"),
                "name": str(function.get("name") or "unknown_tool"),
                "input": json_loads_tool_input(function.get("arguments")),
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = openai_resp.get("usage", {})
    return {
        "id": openai_resp.get("id", "msg_skillclaw"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def stream_from_openai_result(result: dict[str, Any], model: str) -> AsyncIterator[str]:
    """Yield Anthropic-format SSE events from an internal OpenAI chat result."""
    payload = result["response"]
    choice = payload.get("choices", [{}])[0]
    anthropic_payload = from_openai_response(payload, model)
    content_blocks = anthropic_payload.get("content", [])
    stop_reason = anthropic_payload.get("stop_reason") or "end_turn"
    usage = payload.get("usage", {})
    msg_id = payload.get("id", "msg_skillclaw")

    def sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": 0},
            },
        },
    )
    yield sse("ping", {"type": "ping"})

    for index, block in enumerate(content_blocks):
        block_type = block.get("type") if isinstance(block, dict) else None
        if block_type == "tool_use":
            input_obj = block.get("input") if isinstance(block.get("input"), dict) else {}
            partial_json = json.dumps(input_obj, ensure_ascii=False, separators=(",", ":"))
            yield sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id", f"call_{index}"),
                        "name": block.get("name", "unknown_tool"),
                        "input": {},
                    },
                },
            )
            if partial_json and partial_json != "{}":
                yield sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": partial_json},
                    },
                )
            yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
            continue

        text = str(block.get("text") or "") if isinstance(block, dict) else ""
        yield sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        )
        if text:
            yield sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        yield sse("content_block_stop", {"type": "content_block_stop", "index": index})

    yield sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage.get("completion_tokens", 0)},
        },
    )
    yield sse("message_stop", {"type": "message_stop"})
