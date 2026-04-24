"""OpenAI Responses compatibility for Codex style clients."""

from __future__ import annotations

import json
import re
import time
from typing import Any, AsyncIterator


def normalize_content_to_text(content: Any) -> str:
    """Flatten Responses-style content blocks to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"input_text", "output_text", "text"}:
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return " ".join(parts)
    return str(content) if content is not None else ""


def content_to_openai_chat_content(content: Any) -> str | list[dict[str, Any]]:
    """Convert Responses message content into chat-completions content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"input_text", "output_text", "text"}:
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
        elif item_type in {"input_image", "image_url"}:
            image_url = item.get("image_url") or item.get("url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if isinstance(image_url, str) and image_url:
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
    if not parts:
        return ""
    if all(part.get("type") == "text" for part in parts):
        return " ".join(str(part.get("text") or "") for part in parts).strip()
    return parts


def tool_choice_to_openai_chat(tool_choice: Any) -> Any:
    """Convert Responses tool_choice to chat-completions tool_choice."""
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        raise ValueError(f"Responses tool_choice {tool_choice!r} is not supported in chat bridge mode")
    if not isinstance(tool_choice, dict):
        return tool_choice

    choice_type = str(tool_choice.get("type") or "").strip()
    if choice_type == "function":
        if isinstance(tool_choice.get("function"), dict):
            return tool_choice
        name = str(tool_choice.get("name") or "").strip()
        if name:
            return {"type": "function", "function": {"name": name}}
    if choice_type in {"auto", "none", "required"}:
        return choice_type
    raise ValueError(f"Responses tool_choice type {choice_type!r} is not supported in chat bridge mode")


def tools_to_openai_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert Responses function-tool schemas to chat-completions tool schemas."""
    converted: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted

    for item in tools:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function":
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(item.get("description") or ""),
                        "parameters": item.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
            continue
        if item.get("function") and item_type in {None, "function"}:
            converted.append(item)
    return converted


def to_openai_body(body: dict[str, Any], default_model: str) -> dict[str, Any]:
    """Convert an OpenAI Responses request body to chat-completions format."""
    raw_input = body.get("input")
    if raw_input is None:
        raise ValueError("input is required")

    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if instructions is not None:
        messages.append({"role": "system", "content": normalize_content_to_text(instructions)})

    def append_tool_call(item: dict[str, Any]) -> None:
        call_id = str(item.get("call_id") or item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()
        arguments = item.get("arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            arguments = str(arguments)
        arguments = arguments.strip() or "{}"
        if not call_id or not name:
            raise ValueError("function_call items require call_id and name")
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        )

    def append_tool_output(item: dict[str, Any]) -> None:
        call_id = str(item.get("call_id") or item.get("tool_call_id") or "").strip()
        if not call_id:
            raise ValueError("function_call_output items require call_id")
        output = item.get("output", "")
        if output is None:
            output = ""
        if not isinstance(output, str):
            output = str(output)
        messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                raise ValueError("input items must be strings or objects")

            item_type = item.get("type")
            if item_type == "function_call":
                append_tool_call(item)
                continue
            if item_type == "function_call_output":
                append_tool_output(item)
                continue
            if item_type == "reasoning":
                continue

            role = str(item.get("role") or "user").strip() or "user"
            if role == "developer":
                role = "system"
            if role == "tool":
                append_tool_output(item)
                continue
            messages.append({"role": role, "content": content_to_openai_chat_content(item.get("content", ""))})
    else:
        raise ValueError("input must be a string or an array")

    if not messages:
        raise ValueError("input must produce at least one message")

    openai_body: dict[str, Any] = {
        "model": body.get("model") or default_model,
        "messages": messages,
    }
    tools = tools_to_openai_tools(body.get("tools"))
    if tools:
        openai_body["tools"] = tools
    if "temperature" in body:
        openai_body["temperature"] = body["temperature"]
    if "top_p" in body:
        openai_body["top_p"] = body["top_p"]
    if "tool_choice" in body:
        openai_body["tool_choice"] = tool_choice_to_openai_chat(body["tool_choice"])
    if "parallel_tool_calls" in body:
        openai_body["parallel_tool_calls"] = body["parallel_tool_calls"]
    if "max_output_tokens" in body:
        openai_body["max_tokens"] = body["max_output_tokens"]
    return openai_body


def function_item_id(call_id: str, index: int) -> str:
    raw = str(call_id or "").strip()
    if raw.startswith("fc_"):
        return raw
    if raw.startswith("call_") and len(raw) > len("call_"):
        return f"fc_{raw[len('call_') :]}"
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", raw)
    if cleaned:
        return f"fc_{cleaned[:48]}"
    return f"fc_{index}"


def from_openai_chat_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert a chat-completions payload to a Responses API payload."""
    choice = payload.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
    content_text = normalize_content_to_text(message.get("content", ""))
    tool_calls = list(message.get("tool_calls") or []) if isinstance(message.get("tool_calls"), list) else []

    output_items: list[dict[str, Any]] = []
    for idx, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function", {}) if isinstance(tool_call.get("function"), dict) else {}
        call_id = str(tool_call.get("id") or tool_call.get("call_id") or f"call_{idx}").strip() or f"call_{idx}"
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            arguments = str(arguments)
        output_items.append(
            {
                "type": "function_call",
                "id": function_item_id(call_id, idx),
                "call_id": call_id,
                "name": str(function.get("name") or ""),
                "arguments": arguments or "{}",
                "status": "completed",
            }
        )

    if content_text or not output_items:
        output_items.append(
            {
                "id": f"msg_{payload.get('id') or 'skillclaw'}_{len(output_items)}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content_text, "annotations": []}],
            }
        )

    usage = payload.get("usage", {})
    response_payload = {
        "id": payload.get("id") or f"resp_skillclaw_{int(time.time() * 1000)}",
        "object": "response",
        "created_at": payload.get("created", int(time.time())),
        "status": "completed",
        "model": model,
        "output": output_items,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
    }
    if content_text:
        response_payload["output_text"] = content_text
    return response_payload


async def stream_response(response_payload: dict[str, Any]) -> AsyncIterator[str]:
    """Yield OpenAI Responses API-compatible SSE events."""
    sequence_number = 0

    def event(payload: dict[str, Any]) -> str:
        nonlocal sequence_number
        payload["sequence_number"] = sequence_number
        sequence_number += 1
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    initial_response = dict(response_payload)
    initial_response["status"] = "in_progress"
    initial_response["output"] = []
    initial_response["usage"] = None
    yield event({"type": "response.created", "response": initial_response})
    yield event({"type": "response.in_progress", "response": initial_response})

    for index, item in enumerate(response_payload.get("output", [])):
        yield event({"type": "response.output_item.added", "output_index": index, "item": item})

        if item.get("type") == "function_call":
            arguments = str(item.get("arguments") or "")
            if arguments:
                yield event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": item.get("id", ""),
                        "output_index": index,
                        "delta": arguments,
                    }
                )
            yield event(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item.get("id", ""),
                    "output_index": index,
                    "arguments": arguments,
                }
            )

        if item.get("type") == "message":
            for content_index, part in enumerate(item.get("content", [])):
                if part.get("type") != "output_text":
                    continue
                item_id = str(item.get("id") or "")
                base_part = {"type": "output_text", "text": "", "annotations": []}
                yield event(
                    {
                        "type": "response.content_part.added",
                        "output_index": index,
                        "content_index": content_index,
                        "item_id": item_id,
                        "part": base_part,
                    }
                )
                text = str(part.get("text") or "")
                if text:
                    yield event(
                        {
                            "type": "response.output_text.delta",
                            "output_index": index,
                            "content_index": content_index,
                            "item_id": item_id,
                            "delta": text,
                            "logprobs": [],
                        }
                    )
                yield event(
                    {
                        "type": "response.output_text.done",
                        "output_index": index,
                        "content_index": content_index,
                        "item_id": item_id,
                        "text": text,
                        "logprobs": [],
                    }
                )
                yield event(
                    {
                        "type": "response.content_part.done",
                        "output_index": index,
                        "content_index": content_index,
                        "item_id": item_id,
                        "part": {"type": "output_text", "text": text, "annotations": []},
                    }
                )
        yield event({"type": "response.output_item.done", "output_index": index, "item": item})

    yield event({"type": "response.completed", "response": response_payload})
    yield "data: [DONE]\n\n"
