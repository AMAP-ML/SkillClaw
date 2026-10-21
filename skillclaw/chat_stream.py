"""Validate SSE frames while preserving incremental chat deltas."""

from __future__ import annotations

import json

from fastapi import HTTPException


async def chat_events(response):
    data = []
    finished = {}

    def parse():
        payload = "\n".join(data)
        data.clear()
        if payload == "[DONE]":
            if not finished or not all(finished.values()):
                raise ValueError("Incomplete chat stream")
            return None
        event = json.loads(payload)
        if not isinstance(event, dict) or event.get("error"):
            raise ValueError("Upstream stream error")
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise ValueError("Invalid chat stream choices")
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
                raise ValueError("Invalid chat stream delta")
            index = int(choice.get("index", 0))
            finished[index] = finished.get(index, False) or bool(choice.get("finish_reason"))
        return event

    try:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                event = parse()
                if event is None:
                    return
                yield event
        if data:
            event = parse()
            if event is None:
                return
            yield event
        # Even a finish_reason frame can precede a transport failure. Require
        # the protocol terminator before committing this turn to session state.
        raise ValueError("Missing stream terminator")
    except Exception as exc:
        # In particular, do not expose a TransportError to failover after
        # successful response headers: emitted text must never be replayed.
        raise HTTPException(status_code=502, detail="Upstream chat stream failed or ended incompletely") from exc
