from __future__ import annotations

import types

import pytest

from evolve_server.core.config import EvolveServerConfig
from evolve_server.engines.workflow import EvolveServer


@pytest.mark.anyio
async def test_notify_proxy_reload_posts_callback_with_auth(monkeypatch) -> None:
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(
        skill_reload_mode="callback",
        proxy_reload_url="http://proxy.test/",
        proxy_reload_api_key="secret",
    )
    captured = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers):
            captured["url"] = url
            captured["headers"] = headers
            return types.SimpleNamespace(raise_for_status=lambda: None)

    fake_httpx = types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    await server._notify_proxy_reload()

    assert captured == {
        "timeout": 5.0,
        "url": "http://proxy.test/internal/reload-skills",
        "headers": {"Authorization": "Bearer secret"},
    }


@pytest.mark.anyio
async def test_notify_proxy_reload_retries_on_http_error(monkeypatch) -> None:
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(
        skill_reload_mode="callback",
        proxy_reload_url="http://proxy.test",
        proxy_reload_api_key="secret",
    )
    attempts = {"count": 0}

    class FakeResponse:
        def raise_for_status(self):
            raise RuntimeError("401 Unauthorized")

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers):
            attempts["count"] += 1
            return FakeResponse()

    async def fake_sleep(_delay):
        return None

    fake_httpx = types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)
    monkeypatch.setattr("evolve_server.engines.workflow.asyncio.sleep", fake_sleep)

    await server._notify_proxy_reload()

    assert attempts == {"count": 3}


@pytest.mark.anyio
async def test_notify_proxy_reload_skips_non_callback_modes(monkeypatch) -> None:
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(
        skill_reload_mode="poll",
        proxy_reload_url="http://proxy.test",
        proxy_reload_api_key="secret",
    )
    called = {"http": False}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            called["http"] = True

    fake_httpx = types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    await server._notify_proxy_reload()

    assert called == {"http": False}
