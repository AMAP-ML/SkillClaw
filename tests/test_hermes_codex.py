import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient

from skillclaw import claw_adapter, hermes_codex
from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig


def _oauth_result(*, force_refresh: bool = False):
    token = "fresh-runtime-token" if force_refresh else "runtime-token"
    return (
        "https://chatgpt.com/backend-api/codex",
        token,
        {
            "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
            "originator": "codex_cli_rs",
            "ChatGPT-Account-ID": "acct-test",
        },
    )


def test_resolve_upstream_reuses_hermes_credentials_and_headers(monkeypatch):
    monkeypatch.setattr(hermes_codex, "_select_pool_credential", lambda base_url, token: (base_url, token))
    monkeypatch.setattr(
        hermes_codex,
        "_resolver",
        lambda: (
            lambda **_: {
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "runtime-token",
            },
            lambda token: {"originator": "codex_cli_rs", "token-seen": token},
        ),
    )

    assert hermes_codex.resolve_upstream() == (
        "https://chatgpt.com/backend-api/codex",
        "runtime-token",
        {"originator": "codex_cli_rs", "token-seen": "runtime-token"},
    )


def test_hermes_codex_upstream_uses_runtime_oauth(monkeypatch):
    monkeypatch.setattr("skillclaw.hermes_codex.resolve_upstream", _oauth_result)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(llm_provider="hermes-openai-codex")

    assert server._upstream_auth() == _oauth_result()


def test_hermes_codex_responses_preserve_native_transport(monkeypatch):
    monkeypatch.setattr("skillclaw.hermes_codex.resolve_upstream", _oauth_result)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        llm_provider="hermes-openai-codex",
        llm_model_id="gpt-5.6-sol",
        llm_api_mode="responses",
    )

    url, body, headers = server._prepare_responses_forward(
        {
            "model": "skillclaw-model",
            "max_output_tokens": 4096,
            "_skillclaw_codex_session_id": "session-test",
        },
        stream=False,
    )

    assert url == "https://chatgpt.com/backend-api/codex/responses"
    assert body == {"model": "gpt-5.6-sol", "stream": True, "store": False}
    assert headers == {
        "Authorization": "Bearer runtime-token",
        "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
        "originator": "codex_cli_rs",
        "ChatGPT-Account-ID": "acct-test",
        "session_id": "session-test",
        "x-client-request-id": "session-test",
    }


def test_oauth_proxy_rejects_unauthenticated_non_loopback_bind():
    with pytest.raises(ValueError, match="requires proxy.api_key"):
        SkillClawAPIServer(
            SkillClawConfig(
                llm_provider="hermes-openai-codex",
                proxy_host="0.0.0.0",
                proxy_api_key="",
            )
        )


def test_oauth_proxy_rejects_unauthenticated_loopback_bind():
    with pytest.raises(ValueError, match="requires proxy.api_key"):
        SkillClawAPIServer(
            SkillClawConfig(
                llm_provider="hermes-openai-codex",
                proxy_host="127.0.0.1",
                proxy_api_key="",
            )
        )


def test_oauth_proxy_requires_its_local_bearer():
    server = SkillClawAPIServer(
        SkillClawConfig(
            llm_provider="hermes-openai-codex",
            proxy_host="127.0.0.1",
            proxy_api_key="local-proxy-key",
        )
    )

    with TestClient(server.app) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get(
                "/v1/models",
                headers={"Authorization": "Bearer local-proxy-key"},
            ).status_code
            == 200
        )


def test_streaming_auth_failure_is_returned_before_downstream_200(monkeypatch):
    server = SkillClawAPIServer(
        SkillClawConfig(
            llm_provider="hermes-openai-codex",
            llm_api_mode="responses",
            proxy_host="127.0.0.1",
            proxy_api_key="local-proxy-key",
        )
    )

    async def fail_before_first_chunk(*args, **kwargs):
        if False:
            yield b""
        raise HTTPException(status_code=429, detail="Upstream Responses stream error: 429")

    monkeypatch.setattr(server, "_stream_and_track_responses", fail_before_first_chunk)

    with TestClient(server.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/responses",
            headers={
                "Authorization": "Bearer local-proxy-key",
                "x-session-id": "stream-error-test",
            },
            json={"model": "gpt-5.6-sol", "input": [], "stream": True},
        )

    assert response.status_code == 429
    assert response.json()["detail"] == "Upstream Responses stream error: 429"


def _configure_hermes(monkeypatch, tmp_path: Path, cfg: SkillClawConfig) -> Path:
    hermes_home = tmp_path / ".hermes"
    config_path = hermes_home / "config.yaml"
    hermes_home.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
                "custom_providers": [{"name": "keep-me", "base_url": "http://keep.test/v1"}],
                "providers": {"keep-named": {"api": "http://named.test/v1"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(claw_adapter, "_HERMES_HOME", hermes_home)
    monkeypatch.setattr(claw_adapter, "_HERMES_SKILLS_DIR", hermes_home / "skills")
    monkeypatch.setattr(claw_adapter, "_HERMES_BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(claw_adapter, "_LEGACY_SKILLCLAW_SKILLS_DIR", tmp_path / "legacy")
    monkeypatch.setattr("skillclaw.hermes_codex.resolve_upstream", _oauth_result)
    claw_adapter._configure_hermes(cfg)
    return config_path


def test_configure_hermes_codex_oauth_uses_named_native_transport(monkeypatch, tmp_path: Path):
    cfg = SkillClawConfig(
        claw_type="hermes",
        llm_provider="hermes-openai-codex",
        llm_model_id="gpt-5.6-sol",
        proxy_port=31000,
        proxy_api_key="skillclaw-key",
    )
    config_path = _configure_hermes(monkeypatch, tmp_path, cfg)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["model"] == {
        "provider": "custom:skillclaw-codex",
        "default": "gpt-5.6-sol",
    }
    assert data["providers"]["skillclaw-codex"] == {
        "name": "SkillClaw Codex OAuth proxy",
        "api": "http://127.0.0.1:31000/v1",
        "api_key": "skillclaw-key",
        "default_model": "gpt-5.6-sol",
        "transport": "codex_responses",
    }
    assert data["providers"]["keep-named"] == {"api": "http://named.test/v1"}
    assert {p["name"] for p in data["custom_providers"]} == {"keep-me"}
    assert claw_adapter.inspect_hermes_config(cfg)["proxy_match"] is True


def test_configure_hermes_generic_provider_keeps_generic_proxy_mode(monkeypatch, tmp_path: Path):
    cfg = SkillClawConfig(
        claw_type="hermes",
        llm_provider="openrouter",
        llm_model_id="upstream-model",
        served_model_name="skillclaw-model",
        proxy_port=31000,
        proxy_api_key="skillclaw-key",
    )
    config_path = _configure_hermes(monkeypatch, tmp_path, cfg)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["model"] == {
        "provider": "custom",
        "base_url": "http://127.0.0.1:31000/v1",
        "default": "skillclaw-model",
        "api_key": "skillclaw-key",
        "api_mode": "",
    }
    assert data["providers"] == {"keep-named": {"api": "http://named.test/v1"}}
    assert {p["name"] for p in data["custom_providers"]} == {"keep-me"}
    assert claw_adapter.inspect_hermes_config(cfg)["proxy_match"] is True


def test_doctor_reports_missing_hermes_oauth(monkeypatch, tmp_path: Path):
    cfg = SkillClawConfig(
        claw_type="hermes",
        llm_provider="hermes-openai-codex",
        llm_model_id="gpt-5.6-sol",
        proxy_api_key="skillclaw-key",
    )
    _configure_hermes(monkeypatch, tmp_path, cfg)

    def unavailable():
        raise RuntimeError("missing login")

    monkeypatch.setattr("skillclaw.hermes_codex.resolve_upstream", unavailable)
    report = claw_adapter.inspect_hermes_config(cfg)

    assert report["status"] == "warning"
    assert "Hermes Codex OAuth credentials are unavailable." in cast(list[str], report["issues"])
    assert any("hermes auth add openai-codex" in step for step in cast(list[str], report["next_steps"]))


def test_resolver_discovers_source_from_hermes_executable(monkeypatch, tmp_path: Path):
    source = tmp_path / "hermes-agent"
    executable = source / "venv" / "bin" / "hermes"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (source / "hermes_cli").mkdir()
    (source / "hermes_cli" / "auth.py").touch()
    monkeypatch.setattr(hermes_codex.shutil, "which", lambda _: str(executable))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "state-only-home"))

    assert source in hermes_codex._candidate_sources()


def test_resolver_preserves_legacy_hermes_home_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SKILLCLAW_HERMES_HOME", str(tmp_path / "custom-home"))

    assert hermes_codex._candidate_sources()[0] == tmp_path / "custom-home" / "hermes-agent"


def test_adapter_uses_hermes_home_environment_in_fresh_process(tmp_path: Path):
    hermes_home = tmp_path / "profile-home"
    env = {**os.environ, "HERMES_HOME": str(hermes_home)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from skillclaw.claw_adapter import _HERMES_HOME, _HERMES_SKILLS_DIR; "
                "print(_HERMES_HOME); print(_HERMES_SKILLS_DIR)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == [str(hermes_home), str(hermes_home / "skills")]


def test_recover_upstream_refreshes_matching_pool_entry(monkeypatch):
    refreshed = SimpleNamespace(
        runtime_api_key="fresh-token",
        base_url="https://fresh.test/codex",
        inference_base_url=None,
    )

    class FakePool:
        def try_refresh_matching(self, *, api_key_hint):
            assert api_key_hint == "stale-token"
            return refreshed

        def mark_exhausted_and_rotate(self, **kwargs):
            raise AssertionError("successful refresh must not rotate")

    monkeypatch.setattr(hermes_codex, "_pool_runtime", lambda: (FakePool(), "https://default.test"))
    monkeypatch.setattr(
        hermes_codex,
        "_resolver",
        lambda: (lambda **_: {}, lambda token: {"account-token": token}),
    )

    assert hermes_codex.recover_upstream("stale-token", status_code=401) == (
        "https://fresh.test/codex",
        "fresh-token",
        {"account-token": "fresh-token"},
    )


def test_recover_upstream_never_persists_upstream_error_body(monkeypatch):
    captured = {}

    class FakePool:
        def try_refresh_matching(self, *, api_key_hint):
            return None

        def mark_exhausted_and_rotate(self, **kwargs):
            captured.update(kwargs)
            return None

    monkeypatch.setattr(hermes_codex, "_pool_runtime", lambda: (FakePool(), "https://default.test"))

    assert hermes_codex.recover_upstream("secret-token", status_code=429) is None
    assert captured["error_context"] == {"message": "HTTP 429"}
    assert "secret-token" not in str(captured["error_context"])


def test_resolver_does_not_reuse_auth_store_token_when_pool_is_exhausted(monkeypatch):
    class ExhaustedPool:
        def has_credentials(self):
            return True

        def select(self):
            return None

    monkeypatch.setattr(hermes_codex, "_pool_runtime", lambda: (ExhaustedPool(), "https://default.test"))

    with pytest.raises(RuntimeError, match="no available credential"):
        hermes_codex._select_pool_credential("https://upstream.test", "known-exhausted-token")


def test_doctor_distinguishes_missing_hermes_runtime(monkeypatch, tmp_path: Path):
    cfg = SkillClawConfig(
        claw_type="hermes",
        llm_provider="hermes-openai-codex",
        llm_model_id="gpt-5.6-sol",
        proxy_api_key="skillclaw-key",
    )
    _configure_hermes(monkeypatch, tmp_path, cfg)

    def missing_runtime():
        raise hermes_codex.HermesInstallationError("not installed")

    monkeypatch.setattr("skillclaw.hermes_codex.resolve_upstream", missing_runtime)
    report = claw_adapter.inspect_hermes_config(cfg)

    assert "Hermes runtime modules are unavailable to SkillClaw." in cast(list[str], report["issues"])
    assert not any("hermes auth add" in step for step in cast(list[str], report["next_steps"]))


@pytest.mark.asyncio
async def test_nonstreaming_401_forces_one_oauth_refresh(monkeypatch):
    refreshes = []

    def resolve(*, force_refresh=False):
        refreshes.append(force_refresh)
        token = "fresh-token" if force_refresh else "stale-token"
        return "https://upstream.test", token, {"originator": "codex_cli_rs"}

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "unauthorized" if status_code == 401 else "ok"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return self.text.encode()

        async def aiter_text(self):
            if self.status_code == 200:
                yield f"data: {json.dumps({'type': 'response.completed', 'response': self.json()})}\n\n"

        def raise_for_status(self):
            if self.status_code == 401:
                request = httpx.Request("POST", "https://upstream.test/responses")
                response = httpx.Response(401, request=request, text=self.text)
                raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

        def json(self):
            return {"id": "resp-refreshed"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            status = 401 if headers["Authorization"] == "Bearer stale-token" else 200
            return FakeResponse(status)

        def stream(self, method, url, json, headers):
            status = 401 if headers["Authorization"] == "Bearer stale-token" else 200
            return FakeResponse(status)

    monkeypatch.setattr("skillclaw.hermes_codex.resolve_upstream", resolve)
    monkeypatch.setattr("skillclaw.hermes_codex.recover_upstream", lambda *args, **kwargs: None)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(llm_provider="hermes-openai-codex", llm_api_mode="responses")

    result = await server._forward_to_llm_responses({"model": "gpt-5.6-sol"})

    assert result == {"id": "resp-refreshed"}
    assert refreshes == [False, True]


@pytest.mark.asyncio
async def test_nonstreaming_429_exhausts_pool_until_success(monkeypatch):
    sent_tokens = []
    rotations = []

    class FakeResponse:
        def __init__(self, token):
            self.status_code = 429 if token != "healthy-token" else 200
            self.text = "quota exhausted"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return self.text.encode()

        async def aiter_text(self):
            if self.status_code == 200:
                yield f"data: {json.dumps({'type': 'response.completed', 'response': self.json()})}\n\n"

        def raise_for_status(self):
            if self.status_code == 429:
                request = httpx.Request("POST", "https://upstream.test/responses")
                response = httpx.Response(429, request=request, text=self.text)
                raise httpx.HTTPStatusError("rate limited", request=request, response=response)

        def json(self):
            return {"id": "resp-healthy"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            token = headers["Authorization"].removeprefix("Bearer ")
            sent_tokens.append(token)
            return FakeResponse(token)

        def stream(self, method, url, json, headers):
            token = headers["Authorization"].removeprefix("Bearer ")
            sent_tokens.append(token)
            return FakeResponse(token)

    def rotate(token, *, status_code):
        rotations.append((token, status_code))
        next_token = {"old-token": "next-token", "next-token": "healthy-token"}.get(token)
        return ("https://upstream.test", next_token, {}) if next_token else None

    monkeypatch.setattr(
        "skillclaw.hermes_codex.resolve_upstream",
        lambda **_: ("https://upstream.test", "old-token", {}),
    )
    monkeypatch.setattr("skillclaw.hermes_codex.recover_upstream", rotate)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(llm_provider="hermes-openai-codex", llm_api_mode="responses")

    assert await server._forward_to_llm_responses({"model": "gpt-5.6-sol"}) == {"id": "resp-healthy"}
    assert sent_tokens == ["old-token", "next-token", "healthy-token"]
    assert rotations == [("old-token", 429), ("next-token", 429)]


@pytest.mark.asyncio
async def test_nonstreaming_codex_sse_assembles_output_and_preserves_incomplete_status(monkeypatch):
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "status": "in_progress", "content": []},
        },
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "hello "},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "world"},
        {
            "type": "response.incomplete",
            "response": {"id": "resp-incomplete", "status": "incomplete", "output": []},
        },
    ]

    class FakeResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_text(self):
            for event in events:
                yield f"data: {json.dumps(event)}\n\n"

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json, headers):
            return FakeResponse()

    monkeypatch.setattr(
        "skillclaw.hermes_codex.resolve_upstream",
        lambda **_: ("https://upstream.test", "runtime-token", {}),
    )
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(llm_provider="hermes-openai-codex", llm_api_mode="responses")

    response = await server._forward_to_llm_responses({"model": "gpt-5.6-sol"})

    assert response["status"] == "incomplete"
    assert response["output"] == [
        {
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": "hello world", "annotations": []}],
        }
    ]


@pytest.mark.asyncio
async def test_generic_nonstreaming_429_keeps_existing_transient_retries(monkeypatch):
    attempts = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            attempts.append(url)
            request = httpx.Request("POST", url)
            return httpx.Response(429, request=request, text="rate limited")

    async def no_sleep(_):
        pass

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("skillclaw.api_server.asyncio.sleep", no_sleep)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        llm_provider="openai",
        llm_api_base="https://upstream.test/v1",
        llm_api_key="static-key",
        llm_api_mode="responses",
    )

    with pytest.raises(HTTPException) as exc_info:
        await server._forward_to_llm_responses({"model": "generic-model"})

    assert exc_info.value.status_code == 502
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_generic_streaming_401_keeps_existing_502_mapping(monkeypatch):
    class FakeStreamContext:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json, headers):
            request = httpx.Request("POST", url)
            response = httpx.Response(401, request=request, stream=httpx.ByteStream(b"unauthorized"))
            return FakeStreamContext(response)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        llm_provider="openai",
        llm_api_base="https://upstream.test/v1",
        llm_api_key="static-key",
        llm_api_mode="responses",
    )

    with pytest.raises(HTTPException) as exc_info:
        _ = [chunk async for chunk in server._stream_llm_responses({"model": "generic-model"})]

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_streaming_429_reads_error_and_exhausts_pool_until_success(monkeypatch, caplog):
    sent_tokens = []
    rotations = []

    class FakeStreamContext:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json, headers):
            token = headers["Authorization"].removeprefix("Bearer ")
            sent_tokens.append(token)
            request = httpx.Request("POST", "https://upstream.test/responses")
            if token in {"old-token", "next-token"}:
                body = f"quota exhausted; reflected bearer={token}".encode()
                response = httpx.Response(429, request=request, stream=httpx.ByteStream(body))
            else:
                response = httpx.Response(
                    200,
                    request=request,
                    stream=httpx.ByteStream(b'data: {"type":"response.completed"}\n\n'),
                )
            return FakeStreamContext(response)

    def rotate(token, *, status_code):
        rotations.append((token, status_code))
        next_token = {"old-token": "next-token", "next-token": "healthy-token"}.get(token)
        if next_token is None:
            return None
        return "https://upstream.test", next_token, {"originator": "codex_cli_rs"}

    monkeypatch.setattr(
        "skillclaw.hermes_codex.resolve_upstream",
        lambda **_: ("https://upstream.test", "old-token", {"originator": "codex_cli_rs"}),
    )
    monkeypatch.setattr("skillclaw.hermes_codex.recover_upstream", rotate)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(llm_provider="hermes-openai-codex", llm_api_mode="responses")

    chunks = [chunk async for chunk in server._stream_llm_responses({"model": "gpt-5.6-sol"})]

    assert chunks == [b'data: {"type":"response.completed"}\n\n']
    assert sent_tokens == ["old-token", "next-token", "healthy-token"]
    assert rotations == [("old-token", 429), ("next-token", 429)]
    assert "old-token" not in caplog.text
    assert "next-token" not in caplog.text
