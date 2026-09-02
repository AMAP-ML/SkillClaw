"""Tests for the Codex (ChatGPT-account) OAuth credential bridge."""

from __future__ import annotations

import base64
import json
import time

import pytest

from skillclaw import codex_oauth


def _make_jwt(claims: dict) -> str:
    """Build an unsigned JWT whose payload carries *claims*."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def _token(*, exp_offset: float = 3600.0, account_id: str = "acct-123") -> str:
    return _make_jwt(
        {
            "exp": time.time() + exp_offset,
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
        }
    )


# --------------------------------------------------------------------------- #
# JWT parsing                                                                  #
# --------------------------------------------------------------------------- #


def test_decode_jwt_claims_reads_payload():
    token = _make_jwt({"hello": "world"})
    assert codex_oauth.decode_jwt_claims(token)["hello"] == "world"


@pytest.mark.parametrize("bad", ["", "not-a-jwt", "only.two", None, 12345])
def test_decode_jwt_claims_tolerates_garbage(bad):
    """Malformed tokens must never raise -- they should 401 upstream instead."""
    assert codex_oauth.decode_jwt_claims(bad) == {}


def test_account_id_extracted_from_auth_claim():
    assert codex_oauth.account_id_from_token(_token(account_id="acct-xyz")) == "acct-xyz"


def test_account_id_missing_claim_returns_empty():
    assert codex_oauth.account_id_from_token(_make_jwt({"exp": 1})) == ""


# --------------------------------------------------------------------------- #
# Expiry                                                                       #
# --------------------------------------------------------------------------- #


def test_fresh_token_is_not_expired():
    assert codex_oauth.is_expired(_token(exp_offset=3600)) is False


def test_past_exp_is_expired():
    assert codex_oauth.is_expired(_token(exp_offset=-10)) is True


def test_within_skew_window_is_expired():
    """A token expiring inside the skew window is refreshed proactively."""
    assert codex_oauth.is_expired(_token(exp_offset=30), skew_seconds=120) is True


def test_token_without_exp_is_treated_as_valid():
    """No parseable exp must NOT burn the single-use refresh token."""
    assert codex_oauth.is_expired(_make_jwt({"foo": "bar"})) is False


# --------------------------------------------------------------------------- #
# Base URL guard                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://chatgpt.com/backend-api/codex", True),
        ("https://chatgpt.com/backend-api/codex/", True),
        ("https://chatgpt.com/backend-api/codex/responses", True),
        # Look-alikes and downgrades must all be rejected.
        ("http://chatgpt.com/backend-api/codex", False),
        ("https://chatgpt.com.attacker.test/backend-api/codex", False),
        ("https://evil.test/chatgpt.com/backend-api/codex", False),
        ("https://api.openai.com/v1", False),
        ("", False),
    ],
)
def test_is_codex_base_url(url, expected):
    assert codex_oauth.is_codex_base_url(url) is expected


# --------------------------------------------------------------------------- #
# Token store resolution                                                       #
# --------------------------------------------------------------------------- #


def test_load_tokens_prefers_later_expiry(monkeypatch, tmp_path):
    """A rotation written by another agent should win automatically."""
    stale, fresh = _token(exp_offset=60), _token(exp_offset=9999)

    hermes = tmp_path / "hermes-auth.json"
    hermes.write_text(
        json.dumps({"providers": {"openai-codex": {"tokens": {"access_token": stale, "refresh_token": "r-old"}}}})
    )
    codex = tmp_path / "codex-auth.json"
    codex.write_text(json.dumps({"tokens": {"access_token": fresh, "refresh_token": "r-new"}}))

    monkeypatch.setattr(codex_oauth, "_HERMES_AUTH_PATH", hermes)
    monkeypatch.setattr(codex_oauth, "_codex_auth_path", lambda: codex)

    assert codex_oauth.load_tokens()["access_token"] == fresh


def test_load_tokens_empty_when_no_stores(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_oauth, "_HERMES_AUTH_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(codex_oauth, "_codex_auth_path", lambda: tmp_path / "also-nope.json")
    assert codex_oauth.load_tokens() == {}


def test_get_access_token_raises_when_unauthenticated(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_oauth, "_HERMES_AUTH_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(codex_oauth, "_codex_auth_path", lambda: tmp_path / "also-nope.json")
    with pytest.raises(codex_oauth.CodexAuthError) as exc:
        codex_oauth.get_access_token()
    assert exc.value.relogin_required is True


def test_valid_token_is_returned_without_refresh(monkeypatch, tmp_path):
    """The happy path must not touch the network or spend the refresh token."""
    good = _token(exp_offset=9999)
    hermes = tmp_path / "hermes-auth.json"
    hermes.write_text(
        json.dumps({"providers": {"openai-codex": {"tokens": {"access_token": good, "refresh_token": "r"}}}})
    )
    monkeypatch.setattr(codex_oauth, "_HERMES_AUTH_PATH", hermes)
    monkeypatch.setattr(codex_oauth, "_codex_auth_path", lambda: tmp_path / "absent.json")

    def _boom(*a, **k):
        raise AssertionError("refresh must not be called for a valid token")

    monkeypatch.setattr(codex_oauth, "refresh_tokens", _boom)
    assert codex_oauth.get_access_token() == good


def test_expired_token_triggers_refresh_and_persists(monkeypatch, tmp_path):
    expired, renewed = _token(exp_offset=-100), _token(exp_offset=9999)

    hermes = tmp_path / "hermes-auth.json"
    hermes.write_text(
        json.dumps(
            {
                "providers": {"openai-codex": {"tokens": {"access_token": expired, "refresh_token": "r-old"}}},
                "credential_pool": {"openai-codex": [{"id": "a", "access_token": expired, "refresh_token": "r-old"}]},
            }
        )
    )
    codex = tmp_path / "codex-auth.json"
    codex.write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": expired, "refresh_token": "r-old"}})
    )

    monkeypatch.setattr(codex_oauth, "_HERMES_AUTH_PATH", hermes)
    monkeypatch.setattr(codex_oauth, "_codex_auth_path", lambda: codex)
    monkeypatch.setattr(
        codex_oauth,
        "refresh_tokens",
        lambda rt, **k: {"access_token": renewed, "refresh_token": "r-new"},
    )

    assert codex_oauth.get_access_token() == renewed

    # Both stores advance, so neither Hermes nor the Codex CLI is stranded on
    # a refresh token we just consumed.
    h = json.loads(hermes.read_text())
    assert h["providers"]["openai-codex"]["tokens"]["access_token"] == renewed
    assert h["providers"]["openai-codex"]["tokens"]["refresh_token"] == "r-new"
    assert h["credential_pool"]["openai-codex"][0]["refresh_token"] == "r-new"

    c = json.loads(codex.read_text())
    assert c["tokens"]["access_token"] == renewed
    # Unrelated fields survive the rewrite.
    assert c["auth_mode"] == "chatgpt"


def test_persist_skips_absent_stores(monkeypatch, tmp_path):
    """SkillClaw must not invent a credential file for a tool you don't use."""
    hermes = tmp_path / "hermes-auth.json"
    hermes.write_text(json.dumps({"providers": {"openai-codex": {"tokens": {}}}}))
    absent = tmp_path / "absent.json"

    monkeypatch.setattr(codex_oauth, "_HERMES_AUTH_PATH", hermes)
    monkeypatch.setattr(codex_oauth, "_codex_auth_path", lambda: absent)

    written = codex_oauth.persist_tokens({"access_token": "a", "refresh_token": "r"})
    assert str(hermes) in written
    assert not absent.exists()


def test_independent_pool_entries_are_not_overwritten(monkeypatch, tmp_path):
    """Separate accounts in the pool keep their own credentials."""
    expired, renewed = _token(exp_offset=-100), _token(exp_offset=9999)
    hermes = tmp_path / "hermes-auth.json"
    hermes.write_text(
        json.dumps(
            {
                "providers": {"openai-codex": {"tokens": {"access_token": expired, "refresh_token": "r-old"}}},
                "credential_pool": {
                    "openai-codex": [
                        {"id": "alias", "access_token": expired, "refresh_token": "r-old"},
                        {"id": "other", "access_token": "other-tok", "refresh_token": "other-r"},
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(codex_oauth, "_HERMES_AUTH_PATH", hermes)
    monkeypatch.setattr(codex_oauth, "_codex_auth_path", lambda: tmp_path / "absent.json")

    codex_oauth.persist_tokens({"access_token": renewed, "refresh_token": "r-new"})

    pool = json.loads(hermes.read_text())["credential_pool"]["openai-codex"]
    assert pool[0]["refresh_token"] == "r-new"
    assert pool[1]["refresh_token"] == "other-r"


# --------------------------------------------------------------------------- #
# Request headers                                                              #
# --------------------------------------------------------------------------- #


def test_build_auth_headers_includes_identity(monkeypatch):
    token = _token(account_id="acct-hdr")
    monkeypatch.setattr(codex_oauth, "get_access_token", lambda **k: token)

    headers = codex_oauth.build_auth_headers("https://chatgpt.com/backend-api/codex")
    assert headers["Authorization"] == f"Bearer {token}"
    assert headers["ChatGPT-Account-ID"] == "acct-hdr"
    # OpenAI requires third-party harnesses to identify themselves.
    assert headers["originator"] == "codex_cli_rs"
    assert "SkillClaw" in headers["User-Agent"]


def test_build_auth_headers_omits_account_when_claim_absent(monkeypatch):
    monkeypatch.setattr(codex_oauth, "get_access_token", lambda **k: _make_jwt({"exp": time.time() + 999}))
    headers = codex_oauth.build_auth_headers("https://chatgpt.com/backend-api/codex")
    assert "ChatGPT-Account-ID" not in headers
