# Adapted from MetaClaw
"""
FastAPI proxy server for SkillClaw.

Intercepts LLM requests from Claw agents, injects skills into system
prompts, forwards to a real LLM API, and optionally collects PRM scores.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from itertools import count
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import SkillClawConfig
from .data_formatter import ConversationSample
from .prm_scorer import PRMScorer
from .skill_manager import SkillManager
from .utils import run_llm
from .protocols import anthropic_messages as anthropic_protocol
from .protocols import openai_responses as responses_protocol

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

_NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type"}


# ------------------------------------------------------------------ #
# Helper utilities                                                     #
# ------------------------------------------------------------------ #


def _flatten_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _normalize_assistant_content_parts(content: list[dict]) -> tuple[str, list[dict]]:
    """Extract plain text and OpenAI-style tool_calls from assistant content parts."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for i, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif item_type == "toolCall":
            name = item.get("name")
            args = item.get("arguments", {})
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args = "{}"
            tc_id = item.get("id") or f"call_{i}"
            tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name or "unknown_tool",
                        "arguments": args,
                    },
                }
            )
    return (" ".join(text_parts).strip(), tool_calls)



_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_HANDLE_RE = re.compile(r"^call_(?:kimi|xml)_\d+$")
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([a-zA-Z0-9_.-]+)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)
_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_ARGS_MAX_CHARS = 4_000
_TOOL_RESULT_CONTENT_MAX_CHARS = 4_000
_SESSION_IDLE_CLOSE_SECONDS = 180
_SESSION_SWEEP_INTERVAL_SECONDS = 15
_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 15
_VALID_TURN_TYPES = {"main", "side"}
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_READ_TOOL_NAMES = {"read", "file_read", "read_file", "readfile"}
_HERMES_SKILL_READ_TOOL_NAMES = {"skill_view"}
_SKILL_WRITE_TOOL_NAMES = {
    "write",
    "file_write",
    "write_file",
    "writefile",
    "create_file",
    "edit",
    "edit_file",
    "replace",
    "replace_in_file",
    "append",
    "append_file",
    "patch",
    "apply_patch",
    "move",
    "rename",
    "mv",
}
_HERMES_SKILL_WRITE_TOOL_NAMES = {"skill_manage"}
_SHELL_TOOL_NAMES = {"shell", "exec", "bash", "terminal"}
_PATCH_PATH_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
_SHELL_SKILL_PATH_RE = re.compile(
    r"([~./A-Za-z0-9_\-][^\n\"'`]*?"
    r"(?:SKILL\.md|references/[^\s\"'`]+|scripts/[^\s\"'`]+|assets/[^\s\"'`]+|history/[^\s\"'`]+))"
)


def _extract_skill_names(items: list[Any] | None) -> set[str]:
    names: set[str] = set()
    for item in items or []:
        if isinstance(item, dict):
            raw = item.get("skill_name") or item.get("name") or item.get("skill")
        else:
            raw = item
        name = str(raw or "").strip()
        if name:
            names.add(name)
    return names


def _extract_modified_skill_names(turns: list[dict] | None) -> set[str]:
    names: set[str] = set()
    for turn in turns or []:
        if isinstance(turn, dict):
            names.update(_extract_skill_names(turn.get("modified_skills")))
    return names


def _llm_request_timeout_seconds() -> float:
    raw = str(os.environ.get("SKILLCLAW_LLM_REQUEST_TIMEOUT_S", "120")).strip()
    try:
        timeout = float(raw)
    except ValueError:
        return 120.0
    return timeout if timeout > 0 else 120.0


def _resolve_turn_type(
    header_turn_type: Optional[str],
    body_turn_type: Any,
    *,
    default: str = "main",
) -> str:
    """Resolve request turn_type safely.

    Defaults to ``main`` to avoid silently dropping record/PRM paths when
    clients include a session id but forget to provide turn_type.
    """
    if default not in _VALID_TURN_TYPES:
        default = "main"
    candidate = header_turn_type if header_turn_type is not None else body_turn_type
    raw = str(candidate or "").strip().lower()
    if not raw:
        return default
    if raw in _VALID_TURN_TYPES:
        return raw
    logger.warning("[SessionDetect] invalid turn_type=%r; fallback=%s", raw, default)
    return default


def _resolve_session_done(
    header_session_done: Optional[str],
    body_session_done: Any,
) -> bool:
    """Resolve session_done from header or body."""
    candidate = header_session_done if header_session_done is not None else body_session_done
    if isinstance(candidate, bool):
        return candidate
    if candidate is None:
        return False
    return str(candidate).strip().lower() in _TRUE_STRINGS


def _normalize_tool_name(raw_name: str, args_raw: str) -> str:
    """
    Normalize tool names from model output.
    Fixes common drift where a call handle (e.g. call_kimi_0) is emitted as
    function name instead of the actual tool name.
    """
    name = (raw_name or "").strip()
    if name.startswith("functions."):
        name = name.split(".", 1)[1]
    if not _TOOL_HANDLE_RE.fullmatch(name):
        return name or "unknown_tool"

    try:
        args_obj = json.loads(args_raw or "{}")
    except Exception:
        args_obj = {}
    if isinstance(args_obj, dict):
        if isinstance(args_obj.get("command"), str) and args_obj.get("command"):
            return "exec"
        if isinstance(args_obj.get("sessionId"), str) and args_obj.get("sessionId"):
            return "process"
    return "unknown_tool"


def _normalize_tool_call_name(raw_name: str) -> str:
    """Strip transport-specific prefixes from a tool name."""
    name = str(raw_name or "").strip()
    if name.startswith("functions."):
        return name.split(".", 1)[1]
    return name


def _deduplicate_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        clean = str(path or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _looks_like_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        return False
    return (
        "/" in text
        or "\\" in text
        or text.startswith("~")
        or text.endswith("SKILL.md")
    )


def _extract_skill_paths_from_patch(raw_text: str) -> list[str]:
    return _deduplicate_paths(
        [
            match.group(1).strip()
            for match in _PATCH_PATH_RE.finditer(str(raw_text or ""))
            if match.group(1).strip()
        ]
    )


def _extract_skill_paths_from_shell(command: str) -> list[str]:
    return _deduplicate_paths(
        [
            match.group(1).strip()
            for match in _SHELL_SKILL_PATH_RE.finditer(str(command or ""))
            if match.group(1).strip()
        ]
    )


def _extract_skill_paths_from_args_dict(args: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in (
        "path",
        "file",
        "file_path",
        "target",
        "destination",
        "dest",
        "to",
        "source",
        "src",
        "old_path",
        "new_path",
    ):
        value = args.get(key)
        if isinstance(value, str) and _looks_like_path(value):
            paths.append(value.strip())

    raw_paths = args.get("paths")
    if isinstance(raw_paths, list):
        for item in raw_paths:
            if isinstance(item, str) and _looks_like_path(item):
                paths.append(item.strip())
    return _deduplicate_paths(paths)


def _extract_skill_paths_from_tool_call(tool_call: dict) -> tuple[str, list[str]]:
    func = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    tool_name = _normalize_tool_call_name(func.get("name") or "")
    args_raw = func.get("arguments", "{}")
    if not isinstance(args_raw, str):
        try:
            args_raw = json.dumps(args_raw, ensure_ascii=False)
        except Exception:
            args_raw = "{}"

    paths: list[str] = []
    args_obj: Any = None
    try:
        args_obj = json.loads(args_raw)
    except Exception:
        args_obj = None

    if isinstance(args_obj, dict):
        paths.extend(_extract_skill_paths_from_args_dict(args_obj))
        if tool_name.lower() in _SHELL_TOOL_NAMES:
            command = str(args_obj.get("command") or args_obj.get("cmd") or "")
            paths.extend(_extract_skill_paths_from_shell(command))

    if tool_name.lower() in {"apply_patch", "patch"}:
        paths.extend(_extract_skill_paths_from_patch(args_raw))
    elif tool_name.lower() in _SHELL_TOOL_NAMES:
        paths.extend(_extract_skill_paths_from_shell(args_raw))

    return tool_name, _deduplicate_paths(paths)


def _extract_hermes_skill_name_from_tool_call(tool_call: dict) -> tuple[str, str, str]:
    """Extract Hermes-native skill name + relative file path from skill calls."""
    func = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    tool_name = _normalize_tool_call_name(func.get("name") or "")
    args_raw = func.get("arguments", "{}")
    if not isinstance(args_raw, str):
        try:
            args_raw = json.dumps(args_raw, ensure_ascii=False)
        except Exception:
            args_raw = "{}"

    try:
        args_obj = json.loads(args_raw)
    except Exception:
        args_obj = {}

    if not isinstance(args_obj, dict):
        return tool_name, "", ""

    rel_path = ""
    for key in ("file_path", "path"):
        value = args_obj.get(key)
        if isinstance(value, str) and value.strip():
            rel_path = value.strip()
            break
    for key in ("skill_name", "name", "skill"):
        value = args_obj.get(key)
        if isinstance(value, str) and value.strip():
            return tool_name, value.strip(), rel_path
    return tool_name, "", rel_path


def _resolve_skill_reference(
    path: str,
    skill_path_map: dict[str, dict[str, str]],
) -> dict[str, str]:
    expanded = os.path.expanduser(str(path or "").strip())
    real_path = os.path.realpath(expanded) if expanded else ""
    skill_info = (
        skill_path_map.get(real_path) or skill_path_map.get(expanded) or skill_path_map.get(str(path or "").strip())
    )
    if skill_info:
        return {
            "skill_id": str(skill_info.get("skill_id", "") or ""),
            "skill_name": str(skill_info.get("skill_name", "") or ""),
            "path": str(path or "").strip(),
        }
    return {
        "skill_id": "",
        "skill_name": "",
        "path": str(path or "").strip(),
    }


def _resolve_skill_reference_by_name(
    skill_name: str,
    skill_path_map: dict[str, dict[str, str]],
    rel_path: str = "",
) -> dict[str, str]:
    clean_name = str(skill_name or "").strip()
    if not clean_name:
        return {"skill_id": "", "skill_name": "", "path": ""}
    normalized_rel = str(rel_path or "").strip().replace("\\", "/").lstrip("./")
    if normalized_rel:
        suffix = f"/{normalized_rel}"
        for path, skill_info in skill_path_map.items():
            if str(skill_info.get("skill_name", "") or "").strip() != clean_name:
                continue
            candidate = str(path or "").replace("\\", "/")
            if candidate.endswith(suffix) or candidate == normalized_rel:
                return {
                    "skill_id": str(skill_info.get("skill_id", "") or ""),
                    "skill_name": clean_name,
                    "path": str(path or ""),
                }
    for path, skill_info in skill_path_map.items():
        if str(skill_info.get("skill_name", "") or "").strip() == clean_name:
            return {
                "skill_id": str(skill_info.get("skill_id", "") or ""),
                "skill_name": clean_name,
                "path": str(path or ""),
            }
    return {"skill_id": "", "skill_name": clean_name, "path": ""}


def _extract_tool_calls_from_text(text: str) -> tuple[str, list[dict]]:
    """
    Parse tool-call tags embedded in assistant text into OpenAI-style tool_calls.
    Supports Kimi markers and Qwen <tool_call> wrappers.
    """
    if not text:
        return "", []

    tool_calls: list[dict] = []

    for i, m in enumerate(_KIMI_TOOL_CALL_RE.finditer(text)):
        raw_name = (m.group(1) or "").strip()
        args_raw = (m.group(2) or "{}").strip()
        tool_name = _normalize_tool_name(raw_name, args_raw)
        try:
            args_obj = json.loads(args_raw)
            args_str = json.dumps(args_obj, ensure_ascii=False)
        except Exception:
            args_str = args_raw if args_raw else "{}"
        tool_calls.append(
            {
                "id": f"call_kimi_{i}",
                "type": "function",
                "function": {"name": tool_name or "unknown_tool", "arguments": args_str},
            }
        )

    for i, m in enumerate(_QWEN_TOOL_CALL_RE.finditer(text), start=len(tool_calls)):
        payload_raw = (m.group(1) or "").strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            continue
        name = (
            payload.get("name") or payload.get("tool_name") or payload.get("function", {}).get("name") or "unknown_tool"
        )
        args = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = "{}"
        name = _normalize_tool_name(str(name), args)
        tool_calls.append(
            {
                "id": f"call_xml_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )

    clean = text
    clean = _THINK_RE.sub("", clean)
    clean = clean.replace("</think>", "")
    # Keep tool call data only in structured field; strip markup from plain text.
    clean = re.sub(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", "", clean, flags=re.DOTALL)
    clean = _QWEN_TOOL_CALL_RE.sub("", clean)
    clean = clean.strip()
    return clean, tool_calls


def _assistant_message_has_tool_calls(message: dict[str, Any]) -> bool:
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list) and raw_tool_calls:
        return True

    raw_content = message.get("content")
    if isinstance(raw_content, list):
        _, part_tool_calls = _normalize_assistant_content_parts(raw_content)
        return bool(part_tool_calls)
    if isinstance(raw_content, str) and raw_content:
        _, text_tool_calls = _extract_tool_calls_from_text(raw_content)
        return bool(text_tool_calls)
    return False


def _restore_missing_reasoning_content(
    messages: list[dict[str, Any]],
    prior_turns: list[dict[str, Any]],
) -> int:
    """Backfill reasoning_content for prior assistant tool-call messages."""
    assistant_tool_indices = [
        idx
        for idx, msg in enumerate(messages)
        if isinstance(msg, dict) and msg.get("role") == "assistant" and _assistant_message_has_tool_calls(msg)
    ]
    prior_tool_turns = [turn for turn in prior_turns if isinstance(turn, dict) and turn.get("tool_calls")]
    if not assistant_tool_indices or not prior_tool_turns:
        return 0

    restored = 0
    for msg_idx, turn in zip(reversed(assistant_tool_indices), reversed(prior_tool_turns)):
        msg = messages[msg_idx]
        if msg.get("reasoning_content"):
            continue
        reasoning = str(turn.get("reasoning_content") or "").strip()
        if not reasoning:
            continue
        messages[msg_idx] = {**msg, "reasoning_content": reasoning}
        restored += 1
    return restored


def _deduplicate_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Deduplicate tool calls while preserving order.

    Priority key is tool-call id. When id is missing, fallback to
    (function.name, function.arguments).
    """
    deduped: list[dict] = []
    seen: set[str] = set()
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_id = str(tc.get("id") or "").strip()
        func = tc.get("function") or {}
        fn_name = str(func.get("name") or "")
        fn_args = str(func.get("arguments") or "")
        key = f"id:{tc_id}" if tc_id else f"fn:{fn_name}|args:{fn_args}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tc)
    return deduped


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Normalize OpenClaw-style messages into chat-template-compatible format."""
    out = []
    for msg in messages:
        m = dict(msg)
        role = m.get("role")

        if role == "developer":
            m["role"] = "system"
            role = "system"

        # OpenClaw tool result message → OpenAI tool message
        if role == "toolResult":
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": _flatten_message_content(m.get("content")),
            }
            tc_id = m.get("toolCallId") or m.get("tool_call_id")
            if tc_id:
                tool_msg["tool_call_id"] = tc_id
            tool_name = m.get("toolName") or m.get("name")
            if tool_name:
                tool_msg["name"] = tool_name
            out.append(tool_msg)
            continue

        # assistant content parts may contain text + toolCall blocks
        raw = m.get("content")
        if role == "assistant" and isinstance(raw, list):
            text, tool_calls = _normalize_assistant_content_parts(raw)
            m["content"] = text
            if tool_calls:
                m["tool_calls"] = tool_calls
        elif not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_message_content(raw)

        out.append(m)
    return out


def _extract_last_user_instruction(messages: list[dict]) -> str:
    """Return the most recent user message text from the current turn context."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _flatten_message_content(msg.get("content"))
            if text:
                return text
    return ""


_ERROR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"exited with code (?!0\b)\d+|exit code (?!0\b)\d+|exit status (?!0\b)\d+",
            re.IGNORECASE,
        ),
        "exit_code",
    ),
    (re.compile(r"Traceback \(most recent call last\)|\.py\", line \d+", re.IGNORECASE), "traceback"),
    (re.compile(r"Permission denied|EACCES|PermissionError", re.IGNORECASE), "permission"),
    (re.compile(r"No such file|FileNotFoundError|ENOENT|not found", re.IGNORECASE), "not_found"),
    (re.compile(r"command not found|not recognized as|is not recognized", re.IGNORECASE), "command_not_found"),
    (re.compile(r"timed?\s*out|TimeoutError|ETIMEDOUT", re.IGNORECASE), "timeout"),
    (re.compile(r"(?:^|\W)(?:Error|Exception):\s", re.MULTILINE), "generic_error"),
]


def _classify_tool_error(content: str) -> tuple[bool, str | None]:
    """Return (has_error, error_type) by matching content against known patterns."""
    for pattern, error_type in _ERROR_PATTERNS:
        if pattern.search(content):
            return True, error_type
    return False, None


def _extract_recent_tool_results(messages: list[dict]) -> list[dict]:
    """Extract tool results from the most recent tool-call round in messages.

    Scans backwards from the end of *messages*, collecting all consecutive
    tool / toolResult messages that appear after the last assistant message.
    Returns a list of summary dicts suitable for skill feedback tracking.
    """
    results: list[dict] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role in ("toolResult", "tool"):
            content = _flatten_message_content(msg.get("content"))
            tool_name = msg.get("toolName") or msg.get("name") or msg.get("tool_name") or "unknown"
            has_error, error_type = _classify_tool_error(content)
            results.append(
                {
                    "tool_name": tool_name,
                    "tool_call_id": (msg.get("toolCallId") or msg.get("tool_call_id") or ""),
                    "content": content[:_TOOL_RESULT_CONTENT_MAX_CHARS],
                    "has_error": has_error,
                    "error_type": error_type,
                }
            )
        elif role == "user":
            continue
        else:
            break
    results.reverse()
    return results


def _extract_recent_tool_result_messages(messages: list[dict]) -> list[dict]:
    """Extract raw tool result messages from the most recent tool round.

    This preserves the original payload shape so cloud sessions can retain a
    complete tool-execution snapshot for future analysis. No truncation or
    error classification is applied here.
    """
    results: list[dict] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role in ("toolResult", "tool"):
            try:
                results.append(json.loads(json.dumps(msg, ensure_ascii=False)))
            except Exception:
                results.append(dict(msg))
        elif role == "user":
            continue
        else:
            break
    results.reverse()
    return results


def _assemble_streaming_chat_completion(
    events: list[dict[str, Any]],
    *,
    fallback_model: str,
) -> dict[str, Any]:
    """Collapse OpenAI-style SSE chat chunks into a single response dict."""
    import time

    builders: dict[int, dict[str, Any]] = {}
    response_id = ""
    response_model = fallback_model
    response_created = int(time.time())
    usage: dict[str, Any] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        response_id = str(event.get("id") or response_id)
        response_model = str(event.get("model") or response_model)
        created = event.get("created")
        if isinstance(created, int):
            response_created = created
        if isinstance(event.get("usage"), dict):
            usage = dict(event["usage"])

        for choice in event.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            index = int(choice.get("index", 0))
            entry = builders.setdefault(
                index,
                {
                    "role": "assistant",
                    "content_parts": [],
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta") or {}
            if isinstance(delta.get("role"), str):
                entry["role"] = delta["role"]

            content = delta.get("content")
            if isinstance(content, str):
                entry["content_parts"].append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        entry["content_parts"].append(item["text"])

            for tc in delta.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                tc_index = int(tc.get("index", 0))
                tool_entry = entry["tool_calls"].setdefault(
                    tc_index,
                    {
                        "id": tc.get("id") or f"call_{tc_index}",
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc.get("id"):
                    tool_entry["id"] = tc["id"]
                if tc.get("type"):
                    tool_entry["type"] = tc["type"]
                fn = tc.get("function") or {}
                if isinstance(fn.get("name"), str):
                    tool_entry["function"]["name"] += fn["name"]
                if isinstance(fn.get("arguments"), str):
                    tool_entry["function"]["arguments"] += fn["arguments"]

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                entry["finish_reason"] = finish_reason

    choices: list[dict[str, Any]] = []
    for index in sorted(builders):
        entry = builders[index]
        message: dict[str, Any] = {
            "role": entry["role"],
            "content": "".join(entry["content_parts"]),
        }
        if entry["tool_calls"]:
            message["tool_calls"] = [entry["tool_calls"][i] for i in sorted(entry["tool_calls"])]
        choices.append(
            {
                "index": index,
                "message": message,
                "finish_reason": entry["finish_reason"] or "stop",
            }
        )

    return {
        "id": response_id or f"chatcmpl-stream-{response_created}",
        "object": "chat.completion",
        "created": response_created,
        "model": response_model,
        "choices": choices
        or [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


async def _collect_sse_chat_events(response) -> list[dict[str, Any]]:
    """Read SSE `data:` lines from a streaming chat completion response."""
    events: list[dict[str, Any]] = []
    async for line in response.aiter_lines():
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _build_tool_summaries(tool_calls: list[dict]) -> list[dict]:
    """Build tool summary dicts from the model's tool_calls.

    Extracts the tool name and key arguments (``command`` for shell-like
    tools, ``path`` for file-based tools) into a compact format for the
    turn record.  ``has_error`` defaults to ``False`` and is merged later
    when actual tool results arrive.
    """
    summaries: list[dict] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = _normalize_tool_call_name(func.get("name", "unknown"))
        args_raw = func.get("arguments", "{}")
        if not isinstance(args_raw, str):
            try:
                args_raw = json.dumps(args_raw, ensure_ascii=False)
            except Exception:
                args_raw = "{}"
        try:
            args = json.loads(args_raw)
        except Exception:
            args = {}
        _, skill_paths = _extract_skill_paths_from_tool_call(tc)

        summary: dict[str, Any] = {
            "tool_name": name,
            "tool_call_id": str(tc.get("id") or ""),
            "arguments": args_raw[:_TOOL_ARGS_MAX_CHARS],
            "has_error": False,
        }

        if name.lower() in _SHELL_TOOL_NAMES:
            cmd = str(args.get("command") or args.get("cmd") or "")
            if cmd:
                summary["command"] = cmd[:_TOOL_ARGS_MAX_CHARS]

        path = str(args.get("path") or args.get("file") or args.get("file_path") or "")
        if path:
            summary["path"] = path
        elif skill_paths:
            summary["path"] = skill_paths[0]

        summaries.append(summary)
    return summaries


def _extract_read_skills_from_tool_calls(
    tool_calls: list[dict],
    skill_path_map: dict[str, dict[str, str]],
) -> list[dict]:
    """Identify which skill bundle files were read from the model's tool_calls.

    Returns a list of ``{"skill_id": ..., "skill_name": ...}`` dicts for
    each ``read`` tool call whose ``path`` argument points inside a skill.
    """
    read_skills: list[dict] = []
    seen_ids: set[str] = set()
    for tc in tool_calls:
        tool_name, skill_paths = _extract_skill_paths_from_tool_call(tc)
        normalized = tool_name.lower()
        if normalized in _HERMES_SKILL_READ_TOOL_NAMES:
            _, skill_name, rel_path = _extract_hermes_skill_name_from_tool_call(tc)
            skill_ref = _resolve_skill_reference_by_name(skill_name, skill_path_map, rel_path)
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("skill_name")
            if dedupe_key and dedupe_key not in seen_ids:
                read_skills.append(skill_ref)
                seen_ids.add(dedupe_key)
            continue
        if normalized not in _READ_TOOL_NAMES:
            continue
        for path in skill_paths:
            skill_ref = _resolve_skill_reference(path, skill_path_map)
            if not skill_ref.get("skill_id") and not skill_ref.get("skill_name"):
                continue
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("path") or skill_ref.get("skill_name")
            if not dedupe_key or dedupe_key in seen_ids:
                continue
            read_skills.append(skill_ref)
            seen_ids.add(dedupe_key)

    return read_skills


def _extract_modified_skills_from_tool_calls(
    tool_calls: list[dict],
    skill_path_map: dict[str, dict[str, str]],
) -> list[dict]:
    """Identify skill bundle files the model attempted to write or update."""
    modified_skills: list[dict] = []
    seen_ids: set[str] = set()
    for tc in tool_calls:
        tool_name, skill_paths = _extract_skill_paths_from_tool_call(tc)
        normalized = tool_name.lower()
        if normalized in _READ_TOOL_NAMES:
            continue
        if normalized in _HERMES_SKILL_WRITE_TOOL_NAMES:
            _, skill_name, rel_path = _extract_hermes_skill_name_from_tool_call(tc)
            skill_ref = _resolve_skill_reference_by_name(skill_name, skill_path_map, rel_path)
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("skill_name")
            if dedupe_key and dedupe_key not in seen_ids:
                modified_skills.append({**skill_ref, "action": normalized})
                seen_ids.add(dedupe_key)
            continue
        if normalized not in _SKILL_WRITE_TOOL_NAMES and normalized not in _SHELL_TOOL_NAMES:
            continue
        for path in skill_paths:
            skill_ref = _resolve_skill_reference(path, skill_path_map)
            if not skill_ref.get("skill_id") and not skill_ref.get("skill_name"):
                continue
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("path") or skill_ref.get("skill_name")
            if not dedupe_key or dedupe_key in seen_ids:
                continue
            modified_skills.append(
                {
                    **skill_ref,
                    "action": "shell" if normalized in _SHELL_TOOL_NAMES else normalized,
                }
            )
            seen_ids.add(dedupe_key)
    return modified_skills


def _merge_tool_error_info(
    turn_record: dict,
    tool_results: list[dict],
    raw_tool_results: list[dict] | None = None,
) -> None:
    """Merge error information from tool results into the turn record.

    Matches tool results to the ``tool_results`` summaries built from tool
    calls (by position).  Updates ``has_error``, ``error_type``, and
    ``content`` on matching entries, then rebuilds ``tool_errors``.
    ``raw_tool_results`` preserves the original tool payloads for cloud upload.
    """
    summaries = turn_record.get("tool_results", [])
    observations: list[dict] = []

    if raw_tool_results is not None:
        raw_snapshot: list[dict] = []
        for item in raw_tool_results:
            if not isinstance(item, dict):
                continue
            try:
                raw_snapshot.append(json.loads(json.dumps(item, ensure_ascii=False)))
            except Exception:
                raw_snapshot.append(dict(item))
        turn_record["tool_results_raw"] = raw_snapshot
    else:
        turn_record.setdefault("tool_results_raw", [])

    for i, result in enumerate(tool_results):
        obs: dict[str, Any] = {
            "tool_name": result.get("tool_name", "unknown"),
            "tool_call_id": result.get("tool_call_id", ""),
            "has_error": bool(result.get("has_error", False)),
        }
        if result.get("error_type"):
            obs["error_type"] = result["error_type"]
        content = result.get("content", "")
        if content:
            obs["content"] = str(content)[:_TOOL_RESULT_CONTENT_MAX_CHARS]
        observations.append(obs)

        if i < len(summaries):
            summaries[i]["has_error"] = bool(result.get("has_error", False))
            summaries[i]["tool_name"] = result.get("tool_name", summaries[i].get("tool_name", "unknown"))
            if result.get("tool_call_id"):
                summaries[i]["tool_call_id"] = result["tool_call_id"]
            if result.get("error_type"):
                summaries[i]["error_type"] = result["error_type"]
            else:
                summaries[i].pop("error_type", None)
            content = result.get("content", "")
            if content:
                summaries[i]["content"] = str(content)[:_TOOL_RESULT_CONTENT_MAX_CHARS]
            else:
                summaries[i].pop("content", None)
        else:
            entry: dict[str, Any] = {
                "tool_name": result.get("tool_name", "unknown"),
                "tool_call_id": result.get("tool_call_id", ""),
                "has_error": bool(result.get("has_error", False)),
            }
            if result.get("error_type"):
                entry["error_type"] = result["error_type"]
            content = result.get("content", "")
            if content:
                entry["content"] = str(content)[:_TOOL_RESULT_CONTENT_MAX_CHARS]
            summaries.append(entry)

    turn_record["tool_observations"] = observations
    turn_record["tool_errors"] = [
        {
            "tool_name": s.get("tool_name", "unknown"),
            **({"tool_call_id": s["tool_call_id"]} if s.get("tool_call_id") else {}),
            **({"error_type": s["error_type"]} if s.get("error_type") else {}),
            **({"content": s["content"]} if s.get("content") else {}),
        }
        for s in summaries
        if s.get("has_error")
    ]


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _rewrite_new_session_bootstrap_prompt(messages: list[dict]) -> tuple[list[dict], int]:
    """Rewrite OpenClaw /new bootstrap user prompt to a safer variant.

    Some upstream providers over-trigger policy filters on the stock bootstrap
    text ("A new session was started via /new or /reset ..."). This keeps
    behavior while avoiding brittle phrasing.
    """
    rewritten = 0
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        if msg.get("role") != "user":
            out.append(msg)
            continue
        text = _flatten_message_content(msg.get("content"))
        lowered = text.lower()
        if "a new session was started via /new or /reset" in lowered:
            out.append(
                {
                    **msg,
                    "content": (
                        "A new chat session just started. "
                        "Greet the user briefly in 1-3 sentences and ask what they want to do."
                    ),
                }
            )
            rewritten += 1
            continue
        out.append(msg)
    return out, rewritten


# ------------------------------------------------------------------ #
# Protocol compatibility wrappers                                      #
# ------------------------------------------------------------------ #


def _anthropic_to_openai_body(body: dict[str, Any]) -> dict[str, Any]:
    return anthropic_protocol.to_openai_body(body)


def _normalize_responses_content(content: Any) -> str:
    return responses_protocol.normalize_content_to_text(content)


def _responses_tools_to_openai_tools(tools: Any) -> list[dict]:
    return responses_protocol.tools_to_openai_tools(tools)


def _responses_to_openai_body(body: dict[str, Any], default_model: str) -> dict[str, Any]:
    try:
        return responses_protocol.to_openai_body(body, default_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _responses_function_item_id(call_id: str, index: int) -> str:
    return responses_protocol.function_item_id(call_id, index)


def _openai_chat_to_responses_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
    return responses_protocol.from_openai_chat_payload(payload, model)


def _openai_to_anthropic_response(openai_resp: dict[str, Any], model: str) -> dict[str, Any]:
    return anthropic_protocol.from_openai_response(openai_resp, model)


# ------------------------------------------------------------------ #
# SkillClawAPIServer                                                    #
# ------------------------------------------------------------------ #


class SkillClawAPIServer:
    """Proxy between client agents and the upstream model with SkillClaw hooks.

    OpenClaw sends ``X-Session-Id`` and ``X-Turn-Type`` headers with every
    request. The proxy injects skills, records conversation artifacts when
    enabled, and can attach PRM scoring when configured. Side tasks
    (``turn_type != "main"``) are forwarded but do not generate the main
    conversation artifact path.

    Parameters
    ----------
    config:
        SkillClawConfig instance.
    skill_manager:
        Optional SkillManager for injecting skills into system prompts.
    prm_scorer:
        Optional PRMScorer. If None, all samples get reward=0.
    """

    def __init__(
        self,
        config: SkillClawConfig,
        sampling_client=None,
        skill_manager: Optional[SkillManager] = None,
        prm_scorer: Optional[PRMScorer] = None,
        last_request_tracker=None,
    ):
        self.config = config
        self._sampling_client = sampling_client
        self.skill_manager = skill_manager
        self.prm_scorer = prm_scorer
        self._last_request_tracker = last_request_tracker
        self._last_request_at = time.time()

        self._served_model = config.served_model_name
        self._expected_api_key = config.proxy_api_key
        os.makedirs(config.record_dir, exist_ok=True)
        # System prompt compression is only used for OpenClaw (whose verbose
        # system prompt benefits from compression).  Non-OpenClaw agents send
        # short/no system prompts, and the compressed OpenClaw text can trigger
        # content filters on strict providers (e.g. Azure).
        self._compress_system_prompt = config.claw_type == "openclaw"
        cache_suffix = f"{config.claw_type}_{config.llm_provider}"
        self._system_prompt_cache_file = os.path.join(config.record_dir, f"system_prompt_cache_{cache_suffix}.json")

        # State machines
        self._index_counter = count(0)
        self._group_counter = count(0)
        self._turn_counts: dict[str, int] = {}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}  # session → {turn → data}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}  # session → {turn → task}
        self._pending_records: dict[str, dict] = {}  # for record logging
        self._session_effective: dict[str, int] = {}  # at-least-one guarantee
        self._session_turns: dict[str, list] = {}
        self._session_last_active: dict[str, float] = {}  # session -> unix_ts
        self._closing_sessions: set[str] = set()  # session ids currently being closed
        self._background_tasks: set[asyncio.Task] = set()  # transient async tasks (upload, submit)
        self._responses_store: dict[str, dict[str, Any]] = {}  # response_id -> stored response/history
        self._session_sweeper_task: Optional[asyncio.Task] = None
        self._session_idle_close_seconds = max(
            0,
            int(getattr(config, "session_idle_close_seconds", _SESSION_IDLE_CLOSE_SECONDS)),
        )
        self._session_sweep_interval_seconds = max(
            1,
            int(getattr(config, "session_sweep_interval_seconds", _SESSION_SWEEP_INTERVAL_SECONDS)),
        )
        self._shutdown_drain_timeout_seconds = max(
            1,
            int(getattr(config, "shutdown_drain_timeout_seconds", _SHUTDOWN_DRAIN_TIMEOUT_SECONDS)),
        )

        # Session boundary detection for non-OpenClaw agents (QwenPaw, IronClaw, etc.)
        # Maps pseudo-session key (e.g. "tui-model") to tracking metadata.
        self._tui_session_meta: dict[str, dict] = {}
        _INACTIVITY_TIMEOUT = 300  # seconds — treat as new session after 5 min idle
        self._tui_inactivity_timeout = _INACTIVITY_TIMEOUT

        # Record files
        self._record_file = ""
        self._prm_record_file = ""
        if config.record_enabled:
            os.makedirs(config.record_dir, exist_ok=True)
            self._record_file = os.path.join(config.record_dir, "conversations.jsonl")
            self._prm_record_file = os.path.join(config.record_dir, "prm_scores.jsonl")
            with open(self._record_file, "w"):
                pass
            with open(self._prm_record_file, "w"):
                pass

        # Tokenizer is used for prompt length accounting/truncation and for
        # optional tokenized conversation sample export.
        self._tokenizer = self._load_tokenizer()
        self.app = self._build_app()

        # Threading lifecycle (set by start())
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._server_stopped_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Tokenizer                                                            #
    # ------------------------------------------------------------------ #

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        except Exception as e:
            logger.warning("[OpenClaw] could not load tokenizer: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # FastAPI app                                                          #
    # ------------------------------------------------------------------ #

    def _build_app(self) -> FastAPI:
        owner = self

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            owner._ready_event.set()
            owner._start_session_idle_sweeper()
            try:
                yield
            finally:
                owner._ready_event.clear()
                await owner._shutdown_cleanup()

        app = FastAPI(title="SkillClaw Proxy", lifespan=lifespan)
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.get("/v1/models")
        async def list_models(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            model_id = owner._served_model
            return JSONResponse(
                content={
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "skillclaw",
                        }
                    ],
                }
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            # Update idle tracker so the scheduler knows the user is active
            owner._mark_request_activity()
            await owner._check_auth(authorization)

            body = await request.json()
            incoming_messages = body.get("messages", [])
            if isinstance(incoming_messages, list):
                rewritten_messages, _ = _rewrite_new_session_bootstrap_prompt(incoming_messages)
                body["messages"] = rewritten_messages
            _raw_sid = x_session_id or body.get("session_id") or ""
            # OpenClaw sends X-Session-Id/X-Turn-Type on every request.
            # Non-OpenClaw agents (QwenPaw, IronClaw, etc.) don't — detect
            # session boundaries heuristically so session upload and state
            # cleanup still work correctly.
            if _raw_sid:
                session_id = _raw_sid
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")
            else:
                msg_count = len(body.get("messages") or [])
                session_id = await owner._resolve_tui_session(
                    body.get("model", "default"),
                    msg_count,
                )
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")
            session_done = _resolve_session_done(x_session_done, body.get("session_done"))
            # Do not infer session_done from bootstrap text — only explicit
            # X-Session-Done or body session_done trigger session close.

            stream = bool(body.get("stream", False))
            result = await owner._handle_request(
                body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
            )
            if stream:
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(content=result["response"])

        @app.post("/v1/responses")
        async def responses(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
            codex_session_id: Optional[str] = Header(default=None, alias="session_id"),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            owner._mark_request_activity()
            await owner._check_auth(authorization)

            body = await request.json()
            if owner._responses_native_enabled():
                if bool(body.get("stream", False)):
                    return StreamingResponse(
                        owner._stream_llm_responses(body),
                        media_type="text/event-stream",
                    )
                response_payload = await owner._forward_to_llm_responses(body)
                return JSONResponse(content=response_payload)

            previous_response_id = str(body.get("previous_response_id") or "").strip()
            store_response = bool(body.get("store", True))
            openai_body = _responses_to_openai_body(body, owner._served_model)
            if previous_response_id:
                stored = owner._responses_store.get(previous_response_id)
                if stored is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"previous_response_id not found: {previous_response_id}",
                    )
                openai_body["messages"] = _merge_previous_response_messages(
                    list(stored.get("messages") or []),
                    list(openai_body.get("messages") or []),
                )
            _raw_sid = x_session_id or codex_session_id or body.get("session_id") or ""
            if _raw_sid:
                session_id = _raw_sid
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")
            else:
                msg_count = len(openai_body.get("messages") or [])
                session_id = await owner._resolve_tui_session(
                    openai_body.get("model", owner._served_model),
                    msg_count,
                )
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")
            session_done = _resolve_session_done(x_session_done, body.get("session_done"))

            result = await owner._handle_request(
                openai_body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
            )
            response_payload = _openai_chat_to_responses_payload(
                result["response"],
                model=openai_body.get("model", owner._served_model),
            )
            assistant_message = (
                result.get("response", {}).get("choices", [{}])[0].get("message", {})
                if isinstance(result.get("response"), dict)
                else {}
            )
            if store_response:
                owner._responses_store[response_payload["id"]] = {
                    "response": response_payload,
                    "messages": [
                        *list(openai_body.get("messages") or []),
                        assistant_message if isinstance(assistant_message, dict) else {},
                    ],
                }
            if bool(body.get("stream", False)):
                return StreamingResponse(
                    owner._stream_responses_response(response_payload),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=response_payload)

        @app.get("/v1/responses/{response_id}")
        async def get_response(
            response_id: str,
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            stored = owner._responses_store.get(response_id)
            if stored is None:
                raise HTTPException(status_code=404, detail="response not found")
            return JSONResponse(content=stored["response"])

        @app.delete("/v1/responses/{response_id}")
        async def delete_response(
            response_id: str,
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            stored = owner._responses_store.pop(response_id, None)
            if stored is None:
                raise HTTPException(status_code=404, detail="response not found")
            return JSONResponse(content={"id": response_id, "object": "response", "deleted": True})

        # ---------------------------------------------------------------- #
        # Anthropic-compatible endpoint — used by NanoClaw (credential proxy
        # forwards container Anthropic SDK calls to ANTHROPIC_BASE_URL).
        # ---------------------------------------------------------------- #

        @app.post("/v1/messages")
        async def anthropic_messages(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
            x_session_id: Optional[str] = Header(default=None),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            owner._mark_request_activity()
            # Accept Anthropic-style x-api-key as well as Bearer token.
            auth_header = authorization or (f"Bearer {x_api_key}" if x_api_key else None)
            await owner._check_auth(auth_header)

            raw_body = await request.json()
            stream = bool(raw_body.get("stream", False))
            openai_body = _anthropic_to_openai_body(raw_body)
            model = raw_body.get("model") or owner._served_model

            incoming_messages = openai_body.get("messages", [])
            if isinstance(incoming_messages, list):
                rewritten_messages, _ = _rewrite_new_session_bootstrap_prompt(incoming_messages)
                openai_body["messages"] = rewritten_messages

            _raw_sid = x_session_id or ""
            if _raw_sid:
                session_id = _raw_sid
                turn_type = _resolve_turn_type(x_turn_type, raw_body.get("turn_type"), default="main")
            else:
                msg_count = len(openai_body.get("messages") or [])
                session_id = await owner._resolve_tui_session(model, msg_count)
                turn_type = _resolve_turn_type(x_turn_type, raw_body.get("turn_type"), default="main")
            session_done = _resolve_session_done(x_session_done, raw_body.get("session_done"))

            result = await owner._handle_request(
                openai_body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
            )
            if stream:
                return StreamingResponse(
                    owner._stream_anthropic_response(result, model),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=_openai_to_anthropic_response(result["response"], model))

        return app

    async def _check_auth(self, authorization: Optional[str]):
        if not self._expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self._expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    def _mark_request_activity(self) -> None:
        self._last_request_at = time.time()
        if self._last_request_tracker is not None:
            try:
                self._last_request_tracker.touch()
            except Exception:
                pass

    def last_request_age_seconds(self) -> Optional[float]:
        last = getattr(self, "_last_request_at", None)
        if last is None:
            return None
        return max(0.0, time.time() - float(last))

    def active_session_count(self) -> int:
        return len(self._collect_active_session_ids())

    def is_idle_for_validation(self, idle_after_seconds: int) -> bool:
        age = self.last_request_age_seconds()
        if age is None:
            return False
        if self.active_session_count() > 0:
            return False
        return age >= max(0, int(idle_after_seconds))

    # ------------------------------------------------------------------ #
    # TUI session boundary detection (QwenPaw / IronClaw / generic clients) #
    # ------------------------------------------------------------------ #

    async def _resolve_tui_session(self, model: str, msg_count: int) -> str:
        """Return a session_id for agents that don't send X-Session-Id.

        Detects new-conversation boundaries by two heuristics:
          1. Message count dropped — the client started a fresh conversation.
          2. Inactivity timeout — the user was idle for >N seconds.

        When a boundary is detected the old session is flushed (session data
        uploaded, state dicts cleaned up) and a new unique id is assigned.
        """
        import uuid

        tui_key = f"tui-{model}"
        now = time.time()
        meta = self._tui_session_meta.get(tui_key)

        if meta is None:
            # First request for this model — start a fresh session.
            sid = f"tui-{model}-{uuid.uuid4().hex[:8]}"
            self._tui_session_meta[tui_key] = {
                "session_id": sid,
                "last_msg_count": msg_count,
                "last_request_time": now,
            }
            logger.info("[SessionDetect] new TUI session %s (first request)", sid)
            return sid

        new_session = False
        if msg_count < meta["last_msg_count"]:
            # Message count dropped → client started a new conversation.
            new_session = True
            logger.info(
                "[SessionDetect] msg count dropped %d → %d — new session",
                meta["last_msg_count"],
                msg_count,
            )
        elif (now - meta["last_request_time"]) > self._tui_inactivity_timeout:
            new_session = True
            idle_sec = int(now - meta["last_request_time"])
            logger.info(
                "[SessionDetect] inactivity %ds > %ds — new session",
                idle_sec,
                self._tui_inactivity_timeout,
            )

        if new_session:
            old_sid = meta["session_id"]
            await self._close_session(old_sid, reason="tui_boundary")
            sid = f"tui-{model}-{uuid.uuid4().hex[:8]}"
            self._tui_session_meta[tui_key] = {
                "session_id": sid,
                "last_msg_count": msg_count,
                "last_request_time": now,
            }
            logger.info("[SessionDetect] new TUI session %s (replacing %s)", sid, old_sid)
            return sid

        # Same session — update tracking.
        meta["last_msg_count"] = msg_count
        meta["last_request_time"] = now
        return meta["session_id"]

    def _touch_session(self, session_id: str) -> None:
        if session_id:
            self._session_last_active[session_id] = time.time()

    def _collect_active_session_ids(self) -> list[str]:
        session_ids = set(self._session_last_active.keys())
        session_ids.update(self._pending_records.keys())
        session_ids.update(self._session_turns.keys())
        session_ids.update(self._pending_turn_data.keys())
        session_ids.update(self._turn_counts.keys())
        session_ids.update(self._session_effective.keys())
        session_ids.update(self._prm_tasks.keys())
        return sorted(s for s in session_ids if s and s not in self._closing_sessions)

    def _collect_idle_session_ids(self, now: Optional[float] = None) -> list[str]:
        if self._session_idle_close_seconds <= 0:
            return []
        if now is None:
            now = time.time()
        threshold = float(self._session_idle_close_seconds)
        return sorted(
            sid
            for sid, ts in self._session_last_active.items()
            if sid and sid not in self._closing_sessions and (now - float(ts)) >= threshold
        )

    def _start_session_idle_sweeper(self) -> None:
        if self._session_idle_close_seconds <= 0:
            logger.info("[SessionDetect] idle sweeper disabled (timeout <= 0)")
            return
        if self._session_sweeper_task is not None and not self._session_sweeper_task.done():
            return
        self._session_sweeper_task = asyncio.create_task(self._session_idle_sweeper_loop())
        self._session_sweeper_task.add_done_callback(self._task_done_cb)
        logger.info(
            "[SessionDetect] idle sweeper started (timeout=%ss interval=%ss)",
            self._session_idle_close_seconds,
            self._session_sweep_interval_seconds,
        )

    async def _session_idle_sweeper_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._session_sweep_interval_seconds)
                stale_ids = self._collect_idle_session_ids()
                for sid in stale_ids:
                    await self._close_session(sid, reason="idle_timeout")
        except asyncio.CancelledError:
            logger.info("[SessionDetect] idle sweeper stopped")
            raise

    async def _await_background_tasks(self, timeout_seconds: float) -> None:
        pending = [t for t in list(self._background_tasks) if not t.done()]
        if not pending:
            return
        done, still_pending = await asyncio.wait(pending, timeout=timeout_seconds)
        if still_pending:
            logger.warning(
                "[OpenClaw] background drain timeout: %d task(s) still running",
                len(still_pending),
            )
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
        else:
            logger.info("[OpenClaw] background drain complete (%d task(s))", len(done))

    async def _drain_active_sessions(self, reason: str) -> None:
        active_ids = self._collect_active_session_ids()
        if not active_ids:
            return
        logger.info("[SessionDetect] draining %d active session(s): reason=%s", len(active_ids), reason)
        for sid in active_ids:
            await self._close_session(sid, reason=reason)

    async def _shutdown_cleanup(self) -> None:
        if self._session_sweeper_task is not None:
            self._session_sweeper_task.cancel()
            await asyncio.gather(self._session_sweeper_task, return_exceptions=True)
            self._session_sweeper_task = None
        await self._drain_active_sessions(reason="server_shutdown")
        await self._await_background_tasks(self._shutdown_drain_timeout_seconds)

    async def _close_session(self, session_id: str, reason: str = "explicit") -> None:
        """Flush a session: submit remaining samples, upload session data, clean up state."""
        if not session_id:
            return
        if session_id in self._closing_sessions:
            return
        self._closing_sessions.add(session_id)
        try:
            self._flush_pending_record(session_id, None)
            pending_snapshot = {
                turn_num: dict(turn_data) for turn_num, turn_data in self._pending_turn_data.get(session_id, {}).items()
            }
            self._maybe_submit_ready_samples(session_id, force_last_prm=True)
            prm_tasks = list(self._prm_tasks.get(session_id, {}).values())
            if prm_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*prm_tasks, return_exceptions=True),
                        timeout=_SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[SessionDetect] PRM drain timed out for session=%s", session_id)
            for turn_num in sorted(pending_snapshot.keys()):
                turn_data = pending_snapshot[turn_num]
                prm_result = turn_data.pop("prm_result", None)
                prm_task = self._prm_tasks.get(session_id, {}).get(turn_num)
                if prm_result is None and prm_task is not None and prm_task.done():
                    try:
                        prm_result = prm_task.result()
                    except (asyncio.CancelledError, Exception):
                        prm_result = None
                await self._submit_turn_sample(
                    turn_num,
                    turn_data,
                    session_id,
                    prm_result,
                )
            eff = self._session_effective.pop(session_id, 0)
            self._turn_counts.pop(session_id, None)
            self._pending_turn_data.pop(session_id, None)
            prm_tasks = self._prm_tasks.pop(session_id, {})
            for task in prm_tasks.values():
                if isinstance(task, asyncio.Task) and not task.done():
                    task.cancel()
            logger.info(
                "[SessionDetect] closed session=%s reason=%s (effective_samples=%d)",
                session_id,
                reason,
                eff,
            )
            if self.skill_manager:
                self.skill_manager._save_stats()
            turns = self._session_turns.pop(session_id, [])
            modified_skill_names = _extract_modified_skill_names(turns)
            if turns and self.config.sharing_enabled:
                self._safe_create_task(self._upload_session_data(session_id, turns))
            if self.config.sharing_enabled:
                self._safe_create_task(self._pull_skills_from_cloud(skip_names=modified_skill_names))
            self._session_last_active.pop(session_id, None)
            for key, meta in list(self._tui_session_meta.items()):
                if isinstance(meta, dict) and meta.get("session_id") == session_id:
                    self._tui_session_meta.pop(key, None)
        finally:
            self._closing_sessions.discard(session_id)

    # ------------------------------------------------------------------ #
    # Record helpers                                                       #
    # ------------------------------------------------------------------ #

    def _flush_pending_record(self, session_id: str, next_state):
        """Write out the buffered record for *session_id* and fire PRM."""
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if next_state:
            ns_role = next_state.get("role", "?")
            ns_content = _flatten_message_content(next_state.get("content"))
            logger.info(
                f"{_GREEN}[OpenClaw] session={session_id} turn={rec['turn']} "
                f"next_state role={ns_role} len={len(ns_content)}: "
                f"{ns_content[:200]}{_RESET}"
            )
            self._fire_prm_scoring(
                session_id,
                rec["turn"],
                rec["response_text"],
                rec.get("instruction_text", ""),
                next_state,
            )
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[OpenClaw] failed to write record: %s", e)

    def _buffer_record(
        self, session_id: str, turn_num: int, messages: list, prompt_text: str, response_text: str, tool_calls: list
    ):
        if not self._record_file:
            return
        instruction_text = _extract_last_user_instruction(messages)
        self._pending_records[session_id] = {
            "session_id": session_id,
            "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "tool_calls": tool_calls or None,
        }

    def _append_prm_record(self, session_id: str, turn_num: int, score: float, votes: list):
        if not self._prm_record_file:
            return
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "turn": turn_num,
                            "score": score,
                            "votes": votes,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except OSError as e:
            logger.warning("[OpenClaw] failed to write PRM record: %s", e)

    def purge_record_files(self):
        """Clear all record JSONL files."""
        for path, label in [
            (self._record_file, "record"),
            (self._prm_record_file, "PRM record"),
        ]:
            if not path:
                continue
            try:
                open(path, "w").close()
                logger.info("[OpenClaw] %s file purged: %s", label, path)
            except OSError as e:
                logger.warning("[OpenClaw] failed to purge %s file: %s", label, e)

    # ------------------------------------------------------------------ #
    # PRM scoring                                                          #
    # ------------------------------------------------------------------ #

    def _fire_prm_scoring(
        self,
        session_id: str,
        turn_num: int,
        response_text: str,
        instruction_text: str,
        next_state,
        submit_ready_samples: bool = True,
    ):
        if not self.prm_scorer or not next_state:
            return
        inst_text = instruction_text or ""
        task = asyncio.create_task(
            self.prm_scorer.evaluate(response_text, inst_text, session_id=session_id, turn_num=turn_num)
        )
        task.add_done_callback(self._task_done_cb)
        if submit_ready_samples:
            task.add_done_callback(lambda _t: self._on_prm_done(session_id, turn_num, _t))
        else:
            task.add_done_callback(lambda _t: self._on_prm_done_without_submit(session_id, turn_num, _t))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        td = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if td is not None:
            td["has_next_state"] = True

    def _apply_prm_result(
        self,
        session_id: str,
        turn_num: int,
        prm_result: Optional[dict],
    ) -> None:
        score = prm_result.get("score", 0.0) if prm_result else 0.0
        turns = self._session_turns.get(session_id, [])
        # turn_num is 1-based; list index is 0-based
        idx = turn_num - 1
        if 0 <= idx < len(turns):
            turns[idx]["prm_score"] = score
            injected = turns[idx].get("injected_skills", [])
            if injected and self.skill_manager:
                self.skill_manager.record_feedback(injected, score)
            read = turns[idx].get("read_skills", [])
            if read and self.skill_manager:
                read_names = [r["skill_name"] for r in read if isinstance(r, dict) and r.get("skill_name")]
                if read_names:
                    self.skill_manager.record_feedback(read_names, score)
        pending_turn = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if isinstance(pending_turn, dict):
            pending_turn["prm_result"] = prm_result

    def _on_prm_done(self, session_id: str, turn_num: int, task: asyncio.Task):
        """Callback after PRM scoring completes — write score back and update skill stats."""
        if task.cancelled():
            return
        try:
            prm_result = task.result()
        except Exception:
            return
        self._apply_prm_result(session_id, turn_num, prm_result)
        if session_id in self._closing_sessions:
            return
        self._maybe_submit_ready_samples(session_id)

    def _on_prm_done_without_submit(self, session_id: str, turn_num: int, task: asyncio.Task):
        """Callback used for close-session PRM tasks; records score only."""
        if task.cancelled():
            return
        try:
            prm_result = task.result()
        except Exception:
            return
        self._apply_prm_result(session_id, turn_num, prm_result)

    # ------------------------------------------------------------------ #
    # Request handling                                                     #
    # ------------------------------------------------------------------ #

    def _read_cached_system_prompt(self) -> str:
        try:
            with open(self._system_prompt_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached = data.get("compressed_system_prompt", "")
            return cached if isinstance(cached, str) else ""
        except Exception:
            return ""

    def _write_cached_system_prompt(self, prompt: str):
        try:
            with open(self._system_prompt_cache_file, "w", encoding="utf-8") as f:
                json.dump({"compressed_system_prompt": prompt}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("[OpenClaw] failed to write system prompt cache: %s", e)

    async def _handle_request(
        self,
        body: dict[str, Any],
        session_id: str,
        turn_type: str,
        session_done: bool,
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        self._touch_session(session_id)
        rewritten = 0
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and msg.get("content", "").startswith("A new chat session just started.")
            ):
                rewritten += 1
        if rewritten:
            logger.info("[OpenClaw] rewrote %d /new bootstrap user prompt(s) for provider safety", rewritten)

        def _prompt_len(msgs):
            try:
                norm_msgs = _normalize_messages_for_template(msgs)
                text = self._tokenizer.apply_chat_template(
                    norm_msgs,
                    tools=body.get("tools"),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                return 0

        # Compress verbose system prompts (OpenClaw only).  Non-OpenClaw
        # agents send short or no system prompts; compressing them wastes an
        # LLM call and the cached OpenClaw prompt can trigger content filters.
        cached_system = ""
        if self._compress_system_prompt:
            cached_system = self._read_cached_system_prompt()
            if not cached_system:
                raw_system = ""
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "system":
                        raw_system = _flatten_message_content(m.get("content"))
                        break
                if raw_system:
                    try:
                        cached_system = await asyncio.to_thread(
                            run_llm,
                            [{"role": "user", "content": raw_system}],
                            self.config,
                        )
                        cached_system = (cached_system or raw_system).strip()
                    except Exception as e:
                        logger.warning(
                            "[OpenClaw] system prompt compression failed: %s — using raw system prompt",
                            e,
                        )
                        cached_system = raw_system.strip()
                    self._write_cached_system_prompt(cached_system)

            if cached_system:
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "system":
                        m["content"] = cached_system

        restored_reasoning = _restore_missing_reasoning_content(
            messages,
            self._session_turns.get(session_id, []),
        )
        if restored_reasoning:
            logger.info(
                "[OpenClaw] restored reasoning_content on %d prior assistant tool-call message(s)",
                restored_reasoning,
            )

        tools = body.get("tools")

        # Inject skills into system message for main turns
        injected_skills: list[str] = []
        if self.skill_manager and turn_type == "main":
            messages, injected_skills = self._inject_skills(messages)
        if self._compress_system_prompt and cached_system:
            logger.info(
                "[OpenClaw] system prompt cached len=%d",
                _prompt_len([{"role": "system", "content": cached_system}]),
            )

        # Truncate to fit within max_context_tokens (keep system + most-recent messages)
        max_prompt = self.config.max_context_tokens - int(body.get("max_tokens") or 2048)
        if max_prompt > 0:
            messages = self._truncate_messages(messages, tools, max_prompt)

        forward_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        if "model" not in forward_body:
            forward_body["model"] = self._served_model
        forward_body["messages"] = messages  # potentially skill-injected

        output = await self._forward_to_llm(forward_body)
        output["model"] = forward_body.get("model") or self._served_model

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        if not isinstance(assistant_msg, dict):
            assistant_msg = {"role": "assistant", "content": _flatten_message_content(assistant_msg)}

        raw_tool_calls = assistant_msg.get("tool_calls") or []
        tool_calls = list(raw_tool_calls) if isinstance(raw_tool_calls, list) else []

        raw_content = assistant_msg.get("content")
        if isinstance(raw_content, list):
            part_text, part_tool_calls = _normalize_assistant_content_parts(raw_content)
            content = part_text
            tool_calls.extend(part_tool_calls)
        else:
            content = _flatten_message_content(raw_content)

        # Upstream models sometimes emit tool calls as text tags instead of
        # structured `message.tool_calls`; parse and normalize both sources.
        clean_content, text_tool_calls = _extract_tool_calls_from_text(content)
        if text_tool_calls:
            content = clean_content
            tool_calls.extend(text_tool_calls)
        tool_calls = _deduplicate_tool_calls(tool_calls)

        assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        else:
            assistant_msg.pop("tool_calls", None)
        choice["message"] = assistant_msg
        if isinstance(output.get("choices"), list) and output["choices"]:
            output["choices"][0] = choice
        else:
            output["choices"] = [choice]

        reasoning = assistant_msg.get("reasoning_content") or ""

        logger.info(f"{_YELLOW}[OpenClaw] [{turn_type}] session={session_id} prompt_msgs={len(messages)}{_RESET}")
        logger.info(
            f"{_RED}[OpenClaw] [{turn_type}] session={session_id} "
            f"thinking={len(reasoning)} chars, response:\n{content}{_RESET}"
        )
        if tool_calls:
            logger.info("[OpenClaw] tool_calls: %s", json.dumps(tool_calls, ensure_ascii=False)[:500])

        if turn_type == "main":
            tool_results = _extract_recent_tool_results(messages)
            prev_turns = self._session_turns.get(session_id, [])
            if tool_results and prev_turns:
                raw_tool_results = _extract_recent_tool_result_messages(messages)
                _merge_tool_error_info(prev_turns[-1], tool_results, raw_tool_results)

            if session_id in self._pending_records and messages:
                self._flush_pending_record(session_id, messages[-1])

            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""

            norm_msgs = _normalize_messages_for_template(messages)
            norm_resp = _normalize_messages_for_template([response_msg])[0]
            full_norm = norm_msgs + [norm_resp]

            skill_path_map = self.skill_manager.get_skill_path_map() if self.skill_manager else {}
            read_skills = _extract_read_skills_from_tool_calls(
                tool_calls,
                skill_path_map,
            )
            modified_skills = _extract_modified_skills_from_tool_calls(
                tool_calls,
                skill_path_map,
            )
            tool_summaries = _build_tool_summaries(tool_calls)
            if read_skills:
                logger.info(
                    "[SkillManager] model read %d skill(s): %s",
                    len(read_skills),
                    ", ".join(r.get("skill_name", "?") for r in read_skills),
                )
            if modified_skills:
                logger.info(
                    "[SkillManager] model modified %d skill(s): %s",
                    len(modified_skills),
                    ", ".join(r.get("skill_name", "?") for r in modified_skills),
                )

            user_instruction = _extract_last_user_instruction(messages)

            if self._tokenizer is None:
                self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
                turn_num = self._turn_counts[session_id]
                prompt_text_simple = "\n".join(
                    f"{m.get('role', '?')}: {_flatten_message_content(m.get('content', ''))}" for m in messages
                )
                response_text_simple = content or (json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "")
                self._buffer_record(
                    session_id,
                    turn_num,
                    messages,
                    prompt_text_simple,
                    response_text_simple,
                    tool_calls,
                )
                self._session_turns.setdefault(session_id, []).append(
                    {
                        "turn_num": turn_num,
                        "prompt_text": user_instruction,
                        "response_text": response_text_simple,
                        "reasoning_content": reasoning or None,
                        "tool_calls": tool_calls,
                        "read_skills": read_skills,
                        "modified_skills": modified_skills,
                        "tool_results": tool_summaries,
                        "tool_results_raw": [],
                        "tool_observations": [],
                        "tool_errors": [],
                        "injected_skills": injected_skills,
                        "prm_score": None,
                    }
                )
                self._pending_turn_data.setdefault(session_id, {})[turn_num] = {
                    "prompt_ids": [],
                    "response_ids": [],
                    "response_logprobs": [],
                    "prompt_text": prompt_text_simple,
                    "response_text": response_text_simple,
                }
                if session_done:
                    await self._close_session(session_id)
                output["session_id"] = session_id
                return {"response": output}

            prompt_text = self._tokenizer.apply_chat_template(
                norm_msgs,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self._tokenizer.apply_chat_template(
                full_norm,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )

            if full_text.startswith(prompt_text):
                response_text = full_text[len(prompt_text) :]
            else:
                logger.warning("[OpenClaw] prompt_text not prefix of full_text, using full_text as response")
                response_text = full_text

            prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = self._tokenizer(response_text, add_special_tokens=False)["input_ids"]

            if not response_ids and not response_text.strip() and not tool_calls:
                logger.info("[OpenClaw] MAIN session=%s → empty response, skipping", session_id)
                output["session_id"] = session_id
                return {"response": output}

            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

            turn_data = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text,
                "response_text": response_text,
            }

            self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
            turn_num = self._turn_counts[session_id]

            logger.info(
                "[OpenClaw] MAIN session=%s turn=%d prompt_tokens=%d response_tokens=%d",
                session_id,
                turn_num,
                len(prompt_ids),
                len(response_ids),
            )
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            self._session_turns.setdefault(session_id, []).append(
                {
                    "turn_num": turn_num,
                    "prompt_text": user_instruction,
                    "response_text": response_text,
                    "reasoning_content": reasoning or None,
                    "tool_calls": tool_calls,
                    "read_skills": read_skills,
                    "modified_skills": modified_skills,
                    "tool_results": tool_summaries,
                    "tool_results_raw": [],
                    "tool_observations": [],
                    "tool_errors": [],
                    "injected_skills": injected_skills,
                    "prm_score": None,
                }
            )
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[OpenClaw] SIDE session=%s → skipped (no training data)", session_id)

        if session_done:
            await self._close_session(session_id)

        output["session_id"] = session_id
        return {"response": output}

    # ------------------------------------------------------------------ #
    # LLM forwarding                                                       #
    # ------------------------------------------------------------------ #

    async def _forward_to_llm(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward to a real LLM API.

        Supports providers:
          - ``"openai"`` (default) — any OpenAI-compatible ``/v1/chat/completions`` endpoint.
          - ``"openrouter"`` — OpenRouter gateway (OpenAI-compatible + routing extensions).
          - ``"bedrock"`` — AWS Bedrock Converse API via :class:`BedrockChatClient`.
        """
        if self.config.llm_provider == "bedrock":
            return await self._forward_to_llm_bedrock(body)
        return await self._forward_to_llm_openai(body)

    def _responses_native_enabled(self) -> bool:
        """Return whether /v1/responses should be forwarded as Responses API."""
        return str(getattr(self.config, "llm_api_mode", "chat") or "chat").lower() == "responses"

    def _prepare_responses_forward(self, body: dict[str, Any], *, stream: bool) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Build URL, body, and headers for native Responses forwarding.

        Native mode intentionally keeps Responses-only tools (custom, web_search,
        namespace, etc.) untouched instead of converting the request to chat.
        """
        api_base = self.config.llm_api_base.rstrip("/")
        if not api_base:
            raise HTTPException(
                status_code=503,
                detail="llm_api_base is not configured. Run 'skillclaw setup' first.",
            )

        send_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        send_body["model"] = self.config.llm_model_id or body.get("model", "")
        send_body["stream"] = stream

        headers: dict[str, str] = {}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        return f"{api_base}/responses", send_body, headers

    async def _forward_to_llm_responses(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward a Codex Responses payload to an upstream Responses API."""
        import httpx

        url, send_body, headers = self._prepare_responses_forward(body, stream=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                    resp = await client.post(
                        url,
                        json=send_body,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as e:
                response_text = e.response.text[:200]
                if attempt < max_retries - 1:
                    wait = min(2**attempt + random.uniform(0, 1), 10)
                    logger.warning(
                        "[OpenClaw] upstream Responses error (attempt %d/%d), retrying in %.1fs: %s %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        e.response.status_code,
                        response_text,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("[OpenClaw] upstream Responses error: %s %s", e.response.status_code, response_text)
                raise HTTPException(status_code=502, detail=f"Upstream Responses error: {e}") from e
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = min(2**attempt + random.uniform(0, 1), 10)
                    logger.warning(
                        "[OpenClaw] Responses forward failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("[OpenClaw] Responses forward failed: %s", e, exc_info=True)
                raise HTTPException(status_code=502, detail=f"Responses forward error: {e}") from e

    async def _stream_llm_responses(self, body: dict[str, Any]):
        """Passthrough upstream Responses SSE without aggregating or rewriting events."""
        import httpx

        url, send_body, headers = self._prepare_responses_forward(body, stream=True)
        try:
            async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                async with client.stream("POST", url, json=send_body, headers=headers) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
        except httpx.HTTPStatusError as e:
            response_text = e.response.text[:200]
            logger.error("[OpenClaw] upstream Responses stream error: %s %s", e.response.status_code, response_text)
            raise HTTPException(status_code=502, detail=f"Upstream Responses stream error: {e}") from e
        except Exception as e:
            logger.error("[OpenClaw] Responses stream failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Responses stream error: {e}") from e

    async def _forward_to_llm_openai(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward to an OpenAI-compatible API."""
        import httpx

        api_base = self.config.llm_api_base.rstrip("/")
        if not api_base:
            raise HTTPException(
                status_code=503,
                detail="llm_api_base is not configured. Run 'skillclaw setup' first.",
            )

        # Strip Tinker-specific fields not supported by standard OpenAI APIs
        send_body = {k: v for k, v in body.items() if k not in {"logprobs", "top_logprobs", "stream_options"}}
        send_body["model"] = self.config.llm_model_id or body.get("model", "")
        send_body["stream"] = False

        headers: dict[str, str] = {}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

        # OpenRouter-specific headers and body extensions
        if self.config.llm_provider == "openrouter":
            if self.config.openrouter_app_name:
                headers["X-Title"] = self.config.openrouter_app_name
            if self.config.openrouter_app_url:
                headers["HTTP-Referer"] = self.config.openrouter_app_url
            # Routing strategy
            route = self.config.openrouter_route
            if route and route != "fallback":
                send_body["provider"] = {"sort": route}
            # Fallback model list
            fallback = self.config.openrouter_fallback_models
            if fallback:
                models = [m.strip() for m in fallback.split(",") if m.strip()]
                if models:
                    send_body["models"] = [send_body.get("model", "")] + models
            # Data collection policy
            if self.config.openrouter_data_policy == "deny":
                send_body.setdefault("provider", {})
                send_body["provider"]["data_collection"] = "deny"

        max_retries = 6
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                    resp = await client.post(
                        f"{api_base}/chat/completions",
                        json=send_body,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as e:
                response_text = e.response.text[:200]
                if e.response.status_code == 400 and "'temperature' is not supported" in e.response.text:
                    logger.info("[OpenClaw] upstream rejects temperature param, retrying without it")
                    send_body.pop("temperature", None)
                    continue
                if e.response.status_code == 400 and "Stream must be set to true" in e.response.text:
                    logger.info("[OpenClaw] upstream requires stream=true, retrying with SSE collection")
                    stream_body = dict(send_body)
                    stream_body["stream"] = True
                    try:
                        async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                            async with client.stream(
                                "POST",
                                f"{api_base}/chat/completions",
                                json=stream_body,
                                headers=headers,
                            ) as stream_resp:
                                stream_resp.raise_for_status()
                                events = await _collect_sse_chat_events(stream_resp)
                        return _assemble_streaming_chat_completion(
                            events,
                            fallback_model=send_body.get("model", ""),
                        )
                    except httpx.HTTPStatusError as stream_error:
                        logger.error(
                            "[OpenClaw] upstream SSE retry error: %s %s",
                            stream_error.response.status_code,
                            stream_error.response.text[:200],
                        )
                        raise HTTPException(
                            status_code=502,
                            detail=f"Upstream LLM SSE retry error: {stream_error}",
                        ) from stream_error
                    except Exception as stream_error:
                        logger.error("[OpenClaw] upstream SSE retry failed: %s", stream_error, exc_info=True)
                        raise HTTPException(
                            status_code=502,
                            detail=f"Upstream LLM SSE retry failed: {stream_error}",
                        ) from stream_error
                # Retryable upstream error — retry if attempts remain
                if attempt < max_retries - 1:
                    wait = min(2**attempt + random.uniform(0, 1), 30)
                    logger.warning(
                        "[OpenClaw] upstream LLM error (attempt %d/%d), retrying in %.1fs: %s %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        e.response.status_code,
                        response_text,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("[OpenClaw] upstream LLM error: %s %s", e.response.status_code, response_text)
                raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e}") from e
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = min(2**attempt + random.uniform(0, 1), 30)
                    logger.warning(
                        "[OpenClaw] LLM forward failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("[OpenClaw] LLM forward failed: %s", e, exc_info=True)
                raise HTTPException(status_code=502, detail=f"LLM forward error: {e}") from e

    async def _forward_to_llm_bedrock(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward to AWS Bedrock via BedrockChatClient."""
        from .bedrock_client import BedrockChatClient

        model_id = self.config.llm_model_id
        if not model_id:
            raise HTTPException(
                status_code=503,
                detail="llm.model_id (Bedrock inference profile) is not configured.",
            )

        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.6)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 8192

        try:
            client = BedrockChatClient(
                model_id=model_id,
                region=self.config.bedrock_region,
            )
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            # Convert BedrockChatClient dataclass response to OpenAI-compatible dict
            choice = resp.choices[0] if resp.choices else None
            return {
                "id": f"chatcmpl-bedrock-{int(time.time())}",
                "object": "chat.completion",
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": choice.message.role if choice else "assistant",
                            "content": choice.message.content if choice else "",
                        },
                        "finish_reason": choice.finish_reason if choice else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            }
        except Exception as e:
            logger.error("[OpenClaw] Bedrock forward failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Bedrock forward error: {e}") from e

    # ------------------------------------------------------------------ #
    # Session data upload (cloud)                                          #
    # ------------------------------------------------------------------ #

    async def _upload_session_data(
        self,
        session_id: str,
        turns: list[dict],
    ) -> None:
        """Upload the complete session turn records to cloud storage.

        Session data and skill data live in *separate* cloud paths so they
        can be consumed independently:
          - sessions: ``{group_id}/sessions/{session_id}.jsonl``
          - skills:   ``{group_id}/skills/{name}/SKILL.md``  (handled by SkillHub)
        """
        try:
            from .skill_hub import SkillHub

            hub = SkillHub.from_config(self.config)
            session_payload = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "user_alias": self.config.sharing_user_alias or os.environ.get("USER", "anonymous"),
                "num_turns": len(turns),
                "turns": turns,
            }

            content = json.dumps(session_payload, ensure_ascii=False)
            oss_key = f"{hub._prefix()}sessions/{session_id}.json"
            hub._bucket.put_object(oss_key, content.encode("utf-8"))
            logger.info(
                "[SkillHub] session uploaded: %s (%d turns, %d bytes)",
                oss_key,
                len(turns),
                len(content),
            )
        except Exception as e:
            logger.warning("[SkillHub] session upload failed: %s", e)

    # ------------------------------------------------------------------ #
    # Skill pull (cloud -> local)                                          #
    # ------------------------------------------------------------------ #

    async def _pull_skills_from_cloud(self, skip_names: Optional[set[str]] = None) -> None:
        """Pull latest skills from cloud storage and reload the skill manager.

        This is a *read-only* operation — local skills are never pushed
        automatically.  Use ``skillclaw skills push`` for explicit uploads.
        """
        try:
            from .skill_hub import SkillHub

            hub = SkillHub.from_config(self.config)
            pull_result = hub.pull_skills(self.config.skills_dir, skip_names=skip_names)
            logger.info(
                "[SkillHub] skill pull: %d downloaded, %d unchanged, %d deleted",
                pull_result["downloaded"],
                pull_result["skipped"],
                pull_result.get("deleted", 0),
            )
            if self.skill_manager and (
                pull_result.get("downloaded", 0) > 0
                or pull_result.get("deleted", 0) > 0
                or pull_result.get("restored_from_backup", False)
            ):
                self.skill_manager.reload()
        except Exception as e:
            logger.warning("[SkillHub] skill pull failed: %s", e)

    # ------------------------------------------------------------------ #
    # Skill injection                                                      #
    # ------------------------------------------------------------------ #

    def _truncate_messages(
        self,
        messages: list[dict],
        tools,
        max_prompt_tokens: int,
    ) -> list[dict]:
        """
        Drop oldest non-system messages until the tokenized prompt fits within
        max_prompt_tokens.  The system message (if any) is always kept.
        At least one user message is always kept even if it alone exceeds the limit.
        """
        if self._tokenizer is None:
            return messages

        def _prompt_len(msgs):
            try:
                norm_msgs = _normalize_messages_for_template(msgs)
                text = self._tokenizer.apply_chat_template(
                    norm_msgs,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                return 0

        if _prompt_len(messages) <= max_prompt_tokens:
            return messages

        # Split into system and non-system messages
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        non_sys = [m for m in messages if m.get("role") != "system"]

        # Greedily keep most-recent messages
        kept = []
        for msg in reversed(non_sys):
            candidate = sys_msgs + list(reversed(kept + [msg]))
            if _prompt_len(candidate) <= max_prompt_tokens:
                kept.append(msg)
            elif not kept:
                kept.append(msg)  # keep at least one user message
                break
            else:
                break

        result = sys_msgs + list(reversed(kept))
        dropped = len(messages) - len(result)
        if dropped > 0:
            logger.warning(
                "[OpenClaw] context truncated: dropped %d oldest messages (%d → %d tokens, limit=%d)",
                dropped,
                _prompt_len(messages),
                _prompt_len(result),
                max_prompt_tokens,
            )
        return result

    def _inject_skills(self, messages: list[dict]) -> tuple[list[dict], list[str]]:
        """Inject an OpenClaw-compatible skill catalog into the system message.

        Lists ALL eligible skills as an XML ``<available_skills>`` catalog
        with ``<name>``, ``<description>``, and ``<location>`` per entry.
        The model is instructed to ``read`` at most one SKILL.md when
        relevant (lazy loading), matching OpenClaw's injection behaviour.

        Returns (modified_messages, listed_skill_names).
        """
        if not self.skill_manager:
            return messages, []

        try:
            self.skill_manager.refresh_if_changed()
        except Exception as e:
            logger.warning("[SkillManager] failed to refresh local skills: %s", e)

        skill_text = self.skill_manager.build_injection_prompt(
            max_chars=getattr(self.config, "max_skills_prompt_chars", 30_000),
        )
        if not skill_text:
            return messages, []

        all_skills = self.skill_manager.get_all_skills()
        skill_names = [s.get("name", "unknown_skill") for s in all_skills if isinstance(s, dict)]
        logger.info(
            "[SkillManager] listing %d skills in catalog: %s",
            len(skill_names),
            ", ".join(skill_names)[:400],
        )

        self.skill_manager.record_injection(skill_names)

        messages = list(messages)
        sys_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        if sys_indices:
            idx = sys_indices[0]
            existing = _flatten_message_content(messages[idx].get("content", ""))
            messages[idx] = {**messages[idx], "content": existing + "\n\n" + skill_text}
        else:
            messages.insert(0, {"role": "system", "content": skill_text})

        return messages, skill_names

    # ------------------------------------------------------------------ #
    # Sample submission                                                    #
    # ------------------------------------------------------------------ #

    def _maybe_submit_ready_samples(
        self,
        session_id: str,
        force_no_prm: bool = False,
        force_last_prm: bool = False,
    ):
        """Submit turns whose PRM and teacher queries are done.

        force_no_prm: also submit turns that have no PRM task yet (used at
        session end for the last turn which will never get a next_state).
        force_last_prm: when closing a session, fire PRM for the latest
        pending turn even if it never received a next_state.
        When force is active, pending teacher tasks are also skipped.
        """
        prm_tasks = self._prm_tasks.setdefault(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            # --- PRM readiness ---
            prm_task = prm_tasks.get(turn_num)
            if not self.config.use_prm or not self.prm_scorer:
                pass  # no PRM → submit immediately
            elif force_last_prm and prm_task is None:
                turn_data = pending.get(turn_num)
                if turn_data is not None:
                    prm_task = asyncio.create_task(
                        self.prm_scorer.evaluate(
                            turn_data.get("response_text", ""),
                            turn_data.get("prompt_text", ""),
                            session_id=session_id,
                            turn_num=turn_num,
                        )
                    )
                    prm_task.add_done_callback(self._task_done_cb)
                    prm_task.add_done_callback(
                        lambda _t, sid=session_id, tnum=turn_num: self._on_prm_done_without_submit(sid, tnum, _t)
                    )
                    prm_tasks[turn_num] = prm_task
                continue
            elif prm_task is not None and not prm_task.done():
                continue  # PRM still running
            elif prm_task is None and not force_no_prm:
                continue  # waiting for next_state to fire PRM

            turn_data = pending.pop(turn_num)
            prm_result = None
            cached_prm_result = turn_data.pop("prm_result", None)
            if cached_prm_result is not None:
                prm_result = cached_prm_result
            if prm_task is not None and prm_task.done():
                try:
                    prm_result = prm_task.result()
                except (asyncio.CancelledError, Exception):
                    pass
                prm_tasks.pop(turn_num, None)

            self._safe_create_task(
                self._submit_turn_sample(
                    turn_num,
                    turn_data,
                    session_id,
                    prm_result,
                )
            )

    async def _submit_ready_samples_inline(
        self,
        session_id: str,
        force_no_prm: bool = False,
    ) -> None:
        """Submit ready samples inline, used when closing a session.

        Unlike ``_maybe_submit_ready_samples``, this awaits the submission
        coroutine directly so the final PRM/sample records are durable before
        session cleanup continues.
        """
        prm_tasks = self._prm_tasks.setdefault(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            prm_task = prm_tasks.get(turn_num)
            if not self.config.use_prm or not self.prm_scorer:
                pass
            elif prm_task is not None and not prm_task.done():
                continue
            elif prm_task is None and not force_no_prm:
                continue

            turn_data = pending.pop(turn_num)
            prm_result = None
            cached_prm_result = turn_data.pop("prm_result", None)
            if cached_prm_result is not None:
                prm_result = cached_prm_result
            if prm_task is not None and prm_task.done():
                try:
                    prm_result = prm_task.result()
                except (asyncio.CancelledError, Exception):
                    pass
                prm_tasks.pop(turn_num, None)

            await self._submit_turn_sample(
                turn_num,
                turn_data,
                session_id,
                prm_result,
            )

    async def _submit_turn_sample(
        self,
        turn_num: int,
        turn_data: dict[str, Any],
        session_id: str,
        prm_result: Optional[dict],
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        has_next_state = turn_data.get("has_next_state", False)
        score = prm_result["score"] if prm_result else 0.0

        exclude = not has_next_state or score == 0.0
        # Guarantee at least one tokenized sample per session is retained when
        # sample export is enabled.
        if exclude and has_next_state and self._session_effective.get(session_id, 0) == 0:
            exclude = False
            logger.info(
                "[OpenClaw] promoting session=%s turn with score=0 → loss_mask=1 (at-least-one guarantee)",
                session_id,
            )

        loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
        _ = ConversationSample(
            session_id=session_id,
            turn_num=turn_num,
            prompt_tokens=prompt_ids,
            response_tokens=response_ids,
            response_logprobs=turn_data["response_logprobs"],
            loss_mask=loss_mask,
            reward=score,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
            skill_generation=self.skill_manager.generation if self.skill_manager else 0,
        )

        if not exclude:
            self._session_effective[session_id] = self._session_effective.get(session_id, 0) + 1

        index = next(self._index_counter)
        next(self._group_counter)

        if prm_result:
            self._append_prm_record(session_id, turn_num, score, prm_result.get("votes", []))

        logger.info(
            "[OpenClaw] submitted sample session=%s turn=%d index=%d score=%.1f exclude=%s "
            "prompt_len=%d response_len=%d",
            session_id,
            turn_num,
            index,
            score,
            exclude,
            len(prompt_ids),
            len(response_ids),
        )

    # ------------------------------------------------------------------ #
    # Streaming                                                            #
    # ------------------------------------------------------------------ #

    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        chunk_base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        final = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_responses_response(self, response_payload: dict[str, Any]):
        """Yield OpenAI Responses API-compatible SSE events."""
        async for chunk in responses_protocol.stream_response(response_payload):
            yield chunk

    async def _stream_anthropic_response(self, result: dict[str, Any], model: str):
        """Yield Anthropic-format SSE events from an internal result dict."""
        async for chunk in anthropic_protocol.stream_from_openai_result(result, model):
            yield chunk


    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._server_stopped_event.clear()
        cfg = uvicorn.Config(
            self.app,
            host=self.config.proxy_host,
            port=self.config.proxy_port,
            log_level="info",
        )
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        threading.Thread(target=self._print_ready_banner, daemon=True).start()

    def _run_server(self):
        try:
            self._server.run()
        finally:
            self._server_stopped_event.set()
            self._ready_event.clear()

    def _print_ready_banner(self):
        if not self._ready_event.wait(timeout=30):
            return
        if self._server_stopped_event.is_set():
            return
        backend = f"LLM ({self.config.llm_model_id or 'upstream'})"
        banner = (
            f"\n{'=' * 70}\n"
            f"  SkillClaw proxy ready\n"
            f"  proxy {self.config.proxy_host}:{self.config.proxy_port} → {backend}\n"
            f"  Claw agent has been configured to use this proxy automatically.\n"
            f"{'=' * 70}\n"
        )
        logger.info(f"{_GREEN}{banner}{_RESET}")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._ready_event.clear()
        self._server_stopped_event.set()

    def wait_until_ready(self, timeout_s: float = 30.0) -> bool:
        return self._ready_event.wait(timeout=timeout_s)

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(t: asyncio.Task):
            self._background_tasks.discard(t)
            self._task_done_cb(t)

        task.add_done_callback(_on_done)
        return task

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[OpenClaw] background task failed: %s", exc, exc_info=exc)
