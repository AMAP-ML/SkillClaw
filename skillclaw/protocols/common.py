"""Shared helpers for protocol adapters."""

from __future__ import annotations

import json
from typing import Any


def json_dumps_tool_args(value: Any) -> str:
    """Return a JSON object string suitable for OpenAI function arguments."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or "{}"
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False)
    except Exception:
        return "{}"


def json_loads_tool_input(value: Any) -> dict[str, Any]:
    """Parse OpenAI function arguments into a tool input object."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}
