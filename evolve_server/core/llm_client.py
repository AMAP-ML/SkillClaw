"""
Async LLM client — thin wrapper around the ``openai`` SDK.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any


def _normalize_temperature(model: str, requested: float) -> float:
    normalized = str(model or "").strip().lower()
    if normalized in {"kimi-k2.5", "ccr/kimi-k2.5"}:
        return 1
    return requested


class AsyncLLMClient:
    """OpenAI-compatible async chat client.

    All calls are dispatched to a background thread so the event loop stays
    free while the synchronous ``openai`` SDK performs the HTTP round-trip.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        max_tokens: int = 100000,
        temperature: float = 0.4,
    ) -> None:
        from openai import OpenAI
        import httpx

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=httpx.Timeout(600.0, connect=30.0),  # 10 min max per request
        )
        self.model = model or os.environ.get("EVOLVE_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send a chat completion request and return the assistant content."""
        requested_temperature = kwargs.pop("temperature", self.temperature)
        merged = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": _normalize_temperature(self.model, requested_temperature),
            **kwargs,
        }

        max_retries = 6
        for attempt in range(max_retries):
            try:
                resp = await asyncio.to_thread(
                    self._client.chat.completions.create, **merged,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                body_text = getattr(getattr(exc, "response", None), "text", "") or ""
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code == 400 and "'temperature' is not supported" in body_text:
                    merged.pop("temperature", None)
                    continue
                if status_code == 400 and "Stream must be set to true" in body_text:
                    return await self._chat_via_stream(merged)
                if attempt < max_retries - 1:
                    import random
                    wait = min(2 ** attempt + random.uniform(0, 1), 30)
                    await asyncio.sleep(wait)
                    continue
                raise

    async def _chat_via_stream(self, body: dict[str, Any]) -> str:
        import json
        import httpx

        headers: dict[str, str] = {}
        api_key = getattr(self._client, "api_key", None) or os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request_body = dict(body)
        request_body["stream"] = True
        base_url = str(getattr(self._client, "base_url", "")).rstrip("/")

        content_parts: list[str] = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json=request_body,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for choice in event.get("choices", []) or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            content_parts.append(text)
        return "".join(content_parts)
