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

_CLAUDE_TOOL_NAME_ALIASES = {
    "agent": "Agent",
    "ask_user_question": "AskUserQuestion",
    "ask-user-question": "AskUserQuestion",
    "askuserquestion": "AskUserQuestion",
    "bash": "Bash",
    "cron_create": "CronCreate",
    "cron-create": "CronCreate",
    "croncreate": "CronCreate",
    "cron_delete": "CronDelete",
    "cron-delete": "CronDelete",
    "crondelete": "CronDelete",
    "cron_list": "CronList",
    "cron-list": "CronList",
    "cronlist": "CronList",
    "edit": "Edit",
    "edit_file": "Edit",
    "enter_plan_mode": "EnterPlanMode",
    "enter-plan-mode": "EnterPlanMode",
    "enterplanmode": "EnterPlanMode",
    "enter_worktree": "EnterWorktree",
    "enter-worktree": "EnterWorktree",
    "enterworktree": "EnterWorktree",
    "exit_plan_mode": "ExitPlanMode",
    "exit-plan-mode": "ExitPlanMode",
    "exitplanmode": "ExitPlanMode",
    "exit_worktree": "ExitWorktree",
    "exit-worktree": "ExitWorktree",
    "exitworktree": "ExitWorktree",
    "glob": "Glob",
    "grep": "Grep",
    "list": "LS",
    "ls": "LS",
    "multi_edit": "MultiEdit",
    "multi-edit": "MultiEdit",
    "multiedit": "MultiEdit",
    "notebook_edit": "NotebookEdit",
    "notebook-edit": "NotebookEdit",
    "notebookedit": "NotebookEdit",
    "notebook_read": "NotebookRead",
    "notebook-read": "NotebookRead",
    "notebookread": "NotebookRead",
    "read": "Read",
    "read_file": "Read",
    "read-file": "Read",
    "file_read": "Read",
    "file-read": "Read",
    "readfile": "Read",
    "schedule_wakeup": "ScheduleWakeup",
    "schedule-wakeup": "ScheduleWakeup",
    "schedulewakeup": "ScheduleWakeup",
    "skill": "Skill",
    "task": "Task",
    "task_output": "TaskOutput",
    "task-output": "TaskOutput",
    "taskoutput": "TaskOutput",
    "task_stop": "TaskStop",
    "task-stop": "TaskStop",
    "taskstop": "TaskStop",
    "todo_write": "TodoWrite",
    "todo-write": "TodoWrite",
    "todowrite": "TodoWrite",
    "web_fetch": "WebFetch",
    "web-fetch": "WebFetch",
    "webfetch": "WebFetch",
    "web_search": "WebSearch",
    "web-search": "WebSearch",
    "websearch": "WebSearch",
    "write": "Write",
    "write_file": "Write",
    "write-file": "Write",
    "file_write": "Write",
    "file-write": "Write",
    "writefile": "Write",
}
_FILE_PATH_TOOLS = {"Edit", "MultiEdit", "Read", "Write"}
_PATH_TOOLS = {"Glob", "Grep", "LS"}
_NOTEBOOK_PATH_TOOLS = {"NotebookEdit", "NotebookRead"}


def _normalize_tool_use_name(name: Any, tool_names: set[str] | None = None) -> str:
    raw = str(name or "unknown_tool")
    if tool_names is None:
        return raw
    if raw in tool_names:
        return raw

    lower_tool_names = {tool_name.lower(): tool_name for tool_name in tool_names}
    exact_case_match = lower_tool_names.get(raw.lower())
    if exact_case_match:
        return exact_case_match

    alias = _CLAUDE_TOOL_NAME_ALIASES.get(raw.lower())
    if alias in tool_names:
        return alias
    return raw


def _sanitize_tool_use_input(name: str, value: Any) -> dict[str, Any]:
    """Normalize OpenAI-style tool arguments back to Claude Code schemas."""
    parsed = json_loads_tool_input(value)
    sanitized = dict(parsed)
    if name == "Bash" and "command" not in sanitized and "cmd" in sanitized:
        sanitized["command"] = sanitized.pop("cmd")

    if name in _FILE_PATH_TOOLS and "file_path" not in sanitized:
        for alias in ("path", "file"):
            if alias in sanitized:
                sanitized["file_path"] = sanitized[alias]
                break

    if name in _FILE_PATH_TOOLS and "file_path" in sanitized:
        sanitized.pop("path", None)
        sanitized.pop("file", None)

    if name in _PATH_TOOLS and "path" not in sanitized:
        for alias in ("file_path", "file"):
            if alias in sanitized:
                sanitized["path"] = sanitized[alias]
                break

    if name in _PATH_TOOLS and "path" in sanitized:
        sanitized.pop("file_path", None)
        sanitized.pop("file", None)

    if name in _NOTEBOOK_PATH_TOOLS and "notebook_path" not in sanitized:
        for alias in ("path", "file_path", "file"):
            if alias in sanitized:
                sanitized["notebook_path"] = sanitized[alias]
                break

    if name in _NOTEBOOK_PATH_TOOLS and "notebook_path" in sanitized:
        sanitized.pop("path", None)
        sanitized.pop("file_path", None)
        sanitized.pop("file", None)

    if name == "Edit":
        for src, dst in (("oldString", "old_string"), ("newString", "new_string"), ("replaceAll", "replace_all")):
            if src in sanitized and dst not in sanitized:
                sanitized[dst] = sanitized.pop(src)
    elif name == "MultiEdit" and isinstance(sanitized.get("edits"), list):
        edits = []
        for edit in sanitized["edits"]:
            if not isinstance(edit, dict):
                edits.append(edit)
                continue
            item = dict(edit)
            for src, dst in (("oldString", "old_string"), ("newString", "new_string"), ("replaceAll", "replace_all")):
                if src in item and dst not in item:
                    item[dst] = item.pop(src)
            edits.append(item)
        sanitized["edits"] = edits

    if name == "Read" and sanitized.get("pages") == "":
        sanitized.pop("pages", None)

    return sanitized


def _flatten_openai_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif item is not None:
                parts.append(str(item))
        return " ".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _anthropic_usage_from_openai_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached_tokens = 0
    if isinstance(details, dict):
        cached_tokens = int(details.get("cached_tokens") or 0)
    input_tokens = max(0, prompt_tokens - cached_tokens)
    out = {"input_tokens": input_tokens, "output_tokens": completion_tokens}
    if cached_tokens:
        out["cache_read_input_tokens"] = cached_tokens
    return out


def _tool_result_to_openai_content(block: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if value:
                text_parts.append(value)
            return
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if isinstance(value, dict):
            value_type = value.get("type")
            if value_type in {"text", "input_text", "output_text"}:
                text = value.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
                return
            if value_type == "image":
                image_part = _image_block_to_openai_part(value)
                if image_part:
                    image_parts.append(image_part)
                return
            if "content" in value:
                collect(value.get("content"))
            return
        if value is not None:
            text_parts.append(str(value))

    collect(block.get("content"))
    text = " ".join(part for part in text_parts if part).strip()
    if block.get("is_error") is True:
        text = f"Tool error: {text}" if text else "Tool error"
    elif not text:
        text = "(image result attached)" if image_parts else "(empty)"
    return text, image_parts


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
                tool_text, tool_images = _tool_result_to_openai_content(block)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": tool_text,
                    }
                )
                content_parts.extend(tool_images)

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
            if text or content_parts:
                normalized.append({**msg, "content": openai_content})
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


def from_openai_response(
    openai_resp: dict[str, Any],
    model: str,
    tool_names: set[str] | None = None,
) -> dict[str, Any]:
    """Convert an OpenAI chat completion response to Anthropic /v1/messages format."""
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
    content_text = _flatten_openai_message_content(message.get("content"))
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
        tool_name = _normalize_tool_use_name(function.get("name"), tool_names)
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or f"call_{idx}"),
                "name": tool_name,
                "input": _sanitize_tool_use_input(tool_name, function.get("arguments")),
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = openai_resp.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    return {
        "id": openai_resp.get("id", "msg_skillclaw"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _anthropic_usage_from_openai_usage(usage),
    }


async def stream_from_openai_result(
    result: dict[str, Any],
    model: str,
    tool_names: set[str] | None = None,
) -> AsyncIterator[str]:
    """Yield Anthropic-format SSE events from an internal OpenAI chat result."""
    payload = result["response"]
    anthropic_payload = from_openai_response(payload, model, tool_names)
    content_blocks = anthropic_payload.get("content", [])
    stop_reason = anthropic_payload.get("stop_reason") or "end_turn"
    usage = payload.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    anthropic_usage = _anthropic_usage_from_openai_usage(usage)
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
                "usage": {**anthropic_usage, "output_tokens": 0},
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
            "usage": {"output_tokens": anthropic_usage.get("output_tokens", 0)},
        },
    )
    yield sse("message_stop", {"type": "message_stop"})
