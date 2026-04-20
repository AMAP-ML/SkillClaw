"""
Shared parsing and formatting helpers used across pipeline stages.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


# ------------------------------------------------------------------ #
#  LLM output parsing                                                  #
# ------------------------------------------------------------------ #

def parse_single_skill(text: str) -> Optional[dict]:
    """Extract a single skill JSON object from LLM output."""
    clean = re.sub(r"```(?:json)?\s*", "", text.strip()).strip().rstrip("`")

    try:
        obj = json.loads(clean)
        if isinstance(obj, dict) and obj.get("name"):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(clean[start : end + 1])
            if isinstance(obj, dict) and obj.get("name"):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ------------------------------------------------------------------ #
#  Tool snippet compaction                                             #
# ------------------------------------------------------------------ #

_TOOL_SNIPPET_MAX_ITEMS = 4
_TOOL_SNIPPET_MAX_CHARS = 240


def _clip_text(value: Any, max_chars: int = _TOOL_SNIPPET_MAX_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def compact_tool_calls(tool_calls: Any, max_items: int = _TOOL_SNIPPET_MAX_ITEMS) -> list[dict]:
    """Compress raw tool call list into concise structured snippets."""
    if not isinstance(tool_calls, list):
        return []
    out: list[dict] = []
    for tc in tool_calls[:max_items]:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        out.append(
            {
                "id": str(tc.get("id") or ""),
                "name": str(func.get("name") or "unknown"),
                "arguments": _clip_text(func.get("arguments") or ""),
            }
        )
    return out


def compact_tool_observations(
    observations: Any,
    max_items: int = _TOOL_SNIPPET_MAX_ITEMS,
) -> list[dict]:
    """Compress tool result/observation list into concise structured signals."""
    if not isinstance(observations, list):
        return []
    out: list[dict] = []
    for item in observations[:max_items]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "tool_name": str(item.get("tool_name") or "unknown"),
                "tool_call_id": str(item.get("tool_call_id") or ""),
                "has_error": bool(item.get("has_error", False)),
                "error_type": str(item.get("error_type") or ""),
                "command": _clip_text(item.get("command") or ""),
                "path": _clip_text(item.get("path") or ""),
                "content": _clip_text(item.get("content") or ""),
            }
        )
    return out


# ------------------------------------------------------------------ #
#  SKILL.md rendering                                                  #
# ------------------------------------------------------------------ #

def build_skill_md(skill: dict) -> str:
    """Render a skill dict into SKILL.md content (with YAML frontmatter)."""
    name = skill.get("name", "unknown")
    description = skill.get("description", "")
    category = skill.get("category", "general")
    content = skill.get("content", "")

    needs_quoting = any(c in description for c in ':{}[],"\'#&*!|>%@`\n')
    if needs_quoting:
        escaped = description.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        desc_line = f'description: "{escaped}"'
    else:
        desc_line = f"description: {description}"

    fm_lines = [f"name: {name}", desc_line, f"category: {category}"]

    extra_fm = skill.get("extra_frontmatter")
    if isinstance(extra_fm, dict):
        import yaml

        for key, value in extra_fm.items():
            if key not in ("name", "description", "category"):
                fm_lines.append(f"{key}: {yaml.dump(value, default_flow_style=True).strip()}")

    return "---\n" + "\n".join(fm_lines) + "\n---\n\n" + content + "\n"


def parse_skill_content(name: str, raw_md: str) -> dict[str, Any]:
    """Minimal parse of a SKILL.md frontmatter + body."""
    result: dict[str, Any] = {
        "name": name,
        "description": "",
        "category": "general",
        "content": "",
        "extra_frontmatter": {},
    }
    if not raw_md.startswith("---"):
        result["content"] = raw_md
        return result

    end_idx = raw_md.find("\n---", 3)
    if end_idx == -1:
        result["content"] = raw_md
        return result

    fm_text = raw_md[3:end_idx].strip()
    body = raw_md[end_idx + 4 :].strip()

    try:
        import yaml

        fm = yaml.safe_load(fm_text) or {}
    except Exception:
        fm = {}
        for key in ("name", "description", "category"):
            match = re.search(rf'^{key}:\s*["\']?(.*?)["\']?\s*$', fm_text, re.MULTILINE)
            if match:
                fm[key] = match.group(1)

    if isinstance(fm, dict):
        result["description"] = str(fm.get("description", ""))
        result["category"] = str(fm.get("category", "general"))
        extra = {k: v for k, v in fm.items() if k not in ("name", "description", "category")}
        if extra:
            result["extra_frontmatter"] = extra
    result["content"] = body
    return result
