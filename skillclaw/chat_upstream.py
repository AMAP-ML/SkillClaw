"""Bounded, ordered failover for OpenAI-compatible chat requests."""

from __future__ import annotations

import asyncio
import copy
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException

_RETRYABLE = {408, 429, 500, 502, 503, 504}


def validate_fallbacks(value):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 3:
        raise ValueError("llm.fallbacks must be a list with at most three OpenAI-compatible chat endpoints")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each llm.fallbacks entry must be a mapping")
        endpoint = str(item.get("api_base") or "").rstrip("/")
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Fallback api_base must be an HTTP(S) URL without embedded credentials")
        if parsed.query or parsed.fragment or not str(item.get("model_id") or "").strip():
            raise ValueError("Fallback requires model_id and an api_base without a query or fragment")
        provider = item.get("provider", "custom")
        if provider not in {"custom", "openai", "openrouter"} or item.get("api_mode", "chat") != "chat":
            raise ValueError("Fallback endpoints currently support OpenAI-compatible chat APIs only")
        result.append(
            {
                "api_base": endpoint,
                "model_id": str(item["model_id"]),
                "api_key": str(item.get("api_key") or ""),
                "provider": provider,
            }
        )
    return result


def chat_endpoints(config, requested_model):
    if not config.llm_api_base:
        raise HTTPException(status_code=503, detail="llm_api_base is not configured. Run 'skillclaw setup' first.")
    try:
        fallbacks = validate_fallbacks(config.llm_fallbacks)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [
        {
            "api_base": config.llm_api_base.rstrip("/"),
            "model_id": config.llm_model_id or requested_model,
            "api_key": config.llm_api_key,
            "provider": config.llm_provider,
        },
        *fallbacks,
    ]


def prepare_chat_request(config, body, endpoint, *, stream=False):
    payload = copy.deepcopy({k: v for k, v in body.items() if k not in {"logprobs", "top_logprobs", "stream_options"}})
    payload.update(model=endpoint["model_id"], stream=stream)
    headers = {}
    if endpoint["api_key"]:
        headers["Authorization"] = f"Bearer {endpoint['api_key']}"
    if endpoint["provider"] == "openrouter":
        if config.openrouter_app_name:
            headers["X-Title"] = config.openrouter_app_name
        if config.openrouter_app_url:
            headers["HTTP-Referer"] = config.openrouter_app_url
        if config.openrouter_route and config.openrouter_route != "fallback":
            payload["provider"] = {"sort": config.openrouter_route}
        models = [m.strip() for m in config.openrouter_fallback_models.split(",") if m.strip()]
        if models:
            payload["models"] = [endpoint["model_id"], *models]
        if config.openrouter_data_policy == "deny":
            payload.setdefault("provider", {})["data_collection"] = "deny"
    return endpoint["api_base"] + "/chat/completions", payload, headers


def retry_after_seconds(response):
    raw = response.headers.get("retry-after", "")
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            stamp = parsedate_to_datetime(raw)
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            return max(0.0, (stamp - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


async def forward_chat(config, body, *, timeout, collect_stream):
    endpoints = chat_endpoints(config, body.get("model", ""))

    async def run():
        index = 0
        url, payload, headers = prepare_chat_request(config, body, endpoints[index])
        last_status = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            # A shared budget bounds the total attempts, including compatibility
            # retries. Fresh payload/headers per endpoint prevent credential leaks.
            for attempt in range(6):
                try:
                    if payload["stream"]:
                        async with client.stream("POST", url, json=payload, headers=headers) as response:
                            if response.is_error:
                                await response.aread()
                            response.raise_for_status()
                            return await collect_stream(response, payload["model"])
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    response = exc.response
                    last_status = response.status_code
                    if (
                        last_status == 400
                        and "'temperature' is not supported" in response.text
                        and "temperature" in payload
                    ):
                        payload.pop("temperature")
                        continue
                    if last_status == 400 and "Stream must be set to true" in response.text and not payload["stream"]:
                        payload["stream"] = True
                        continue
                    if last_status not in _RETRYABLE:
                        break
                    delay = retry_after_seconds(response)
                    # Quota windows are not useful to retry against the same
                    # account. Alternate endpoints are attempted immediately.
                    quota = last_status == 429 and (
                        (delay is not None and delay > 30)
                        or "usage limit" in response.text.lower()
                        or '"1308"' in response.text
                    )
                    if index + 1 >= len(endpoints) and quota:
                        break
                except httpx.TransportError:
                    last_status = None
                    delay = None
                except ValueError as exc:
                    raise HTTPException(status_code=502, detail="Upstream LLM returned invalid JSON") from exc
                if index + 1 < len(endpoints):
                    index += 1
                    url, payload, headers = prepare_chat_request(config, body, endpoints[index])
                    continue
                if attempt < 5:
                    await asyncio.sleep(delay if delay is not None else min(2**attempt + random.uniform(0, 1), 30))
        suffix = f" (HTTP {last_status})" if last_status is not None else ""
        raise HTTPException(status_code=502, detail="Upstream LLM request failed" + suffix)

    try:
        # Covers network operations and Retry-After/backoff together, rather
        # than resetting the deadline at each endpoint.
        return await asyncio.wait_for(run(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Upstream LLM request exceeded its total time budget") from exc
