import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from skillclaw.chat_upstream import forward_chat, validate_fallbacks
from skillclaw.config import SkillClawConfig
from skillclaw.config_store import ConfigStore


@pytest.fixture
def transport(monkeypatch):
    original_client = httpx.AsyncClient
    seen = []
    outcomes = []

    def handler(request):
        seen.append((str(request.url), dict(request.headers), json.loads(request.content)))
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        status, body, headers = result if len(result) == 3 else (*result, {})
        return httpx.Response(status, json=body, headers=headers)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)
    )
    return seen, outcomes


def config(**kwargs):
    return SkillClawConfig(
        llm_api_base="https://primary.test/v1",
        llm_api_key="primary-secret",
        llm_model_id="primary-model",
        llm_fallbacks=[{"api_base": "https://secondary.test/v1", "model_id": "secondary-model"}],
        **kwargs,
    )


async def call(cfg, **kwargs):
    return await forward_chat(cfg, {"model": "alias", "messages": [], **kwargs}, timeout=1, collect_stream=AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [(429, {"error": "Usage limit reached"}), (503, {}), httpx.ConnectError("down")])
async def test_immediate_failover_with_separate_auth_and_model(transport, failure):
    seen, outcomes = transport
    outcomes.extend([failure, (200, {"model": "secondary-model", "choices": []})])
    cfg = config(llm_provider="openrouter", openrouter_fallback_models="other", openrouter_data_policy="deny")
    result = await call(cfg)
    assert result["model"] == "secondary-model"
    assert len(seen) == 2
    assert seen[0][1]["authorization"] == "Bearer primary-secret"
    assert "authorization" not in seen[1][1]
    assert "x-title" not in seen[1][1]
    assert seen[1][2]["model"] == "secondary-model"
    assert "provider" not in seen[1][2] and "models" not in seen[1][2]
    assert cfg.llm_fallbacks[0].get("api_key") is None


@pytest.mark.asyncio
async def test_success_never_calls_fallback(transport):
    seen, outcomes = transport
    outcomes.append((200, {"model": "primary-model"}))
    assert (await call(config()))["model"] == "primary-model"
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_nonretryable_failures_do_not_replay_elsewhere(transport, status):
    seen, outcomes = transport
    outcomes.append((status, {"error": "private upstream detail"}))
    with pytest.raises(HTTPException) as exc:
        await call(config())
    assert exc.value.status_code == 502
    assert "private" not in exc.value.detail
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_retry_budget_shared_across_endpoints(transport, monkeypatch):
    seen, outcomes = transport
    outcomes.extend([(503, {})] * 6)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    cfg = config()
    cfg.llm_fallbacks.append({"api_base": "https://third.test/v1", "model_id": "third", "api_key": "third-key"})
    with pytest.raises(HTTPException):
        await call(cfg)
    assert len(seen) == 6
    assert [entry[0] for entry in seen[:3]] == [
        f"https://{host}.test/v1/chat/completions" for host in ("primary", "secondary", "third")
    ]
    assert seen[2][1]["authorization"] == "Bearer third-key"


@pytest.mark.asyncio
async def test_final_quota_window_fast_fails(transport):
    seen, outcomes = transport
    outcomes.extend([(429, {}, {"Retry-After": "3600"})] * 2)
    with pytest.raises(HTTPException) as exc:
        await call(config())
    assert exc.value.status_code == 502
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_total_deadline_includes_retry_after_sleep(transport):
    seen, outcomes = transport
    outcomes.extend([(503, {}, {"Retry-After": "10"})] * 2)
    with pytest.raises(HTTPException) as exc:
        await forward_chat(config(), {}, timeout=0.02, collect_stream=AsyncMock())
    assert exc.value.status_code == 504
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_temperature_compatibility_retry_is_bounded(transport):
    seen, outcomes = transport
    outcomes.extend([(400, {"error": "'temperature' is not supported"})] * 2)
    with pytest.raises(HTTPException):
        await call(config(), temperature=0.5)
    assert len(seen) == 2
    assert "temperature" not in seen[1][2]


@pytest.mark.asyncio
async def test_stream_required_compatibility(transport):
    seen, outcomes = transport
    outcomes.extend([(400, {"error": "Stream must be set to true"}), (200, {})])
    collect = AsyncMock(return_value={"model": "primary-model"})
    assert await forward_chat(config(), {}, timeout=1, collect_stream=collect) == {"model": "primary-model"}
    assert seen[1][2]["stream"] is True
    collect.assert_awaited_once()


@pytest.mark.parametrize(
    "value",
    [
        "bad",
        [None],
        [{}],
        [{"api_base": "file:///tmp/key", "model_id": "m"}],
        [{"api_base": "https://user:key@host/v1", "model_id": "m"}],
        [{"api_base": "https://host/v1?api_key=secret", "model_id": "m"}],
        [{"api_base": "https://host/v1", "model_id": "m", "api_mode": "responses"}],
        [{"api_base": "https://host/v1", "model_id": "m"}] * 4,
    ],
)
def test_invalid_fallbacks_fail_before_requests(value):
    with pytest.raises(ValueError):
        validate_fallbacks(value)


def test_config_store_preserves_explicit_fallbacks(tmp_path):
    cs = ConfigStore(tmp_path / "config.yaml")
    entries = [{"api_base": "https://secondary.test/v1", "model_id": "backup", "api_key": "separate"}]
    cs.set("llm.fallbacks", entries)
    cfg = cs.to_skillclaw_config()
    assert cfg.llm_fallbacks[0]["api_key"] == "separate"
    assert cfg.llm_fallbacks[0]["model_id"] == "backup"
