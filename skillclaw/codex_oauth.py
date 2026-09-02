"""Codex (ChatGPT-account) OAuth credential bridge for SkillClaw.

SkillClaw's stock forwarding path sends a *static* ``Authorization: Bearer
<api_key>`` taken from ``~/.skillclaw/config.yaml``.  That model does not fit
OpenAI's ``chatgpt.com/backend-api/codex`` endpoint, which authenticates with a
short-lived OAuth access token issued against a ChatGPT (Plus/Pro) account and
additionally requires harness-identity headers.

This module supplies both, reusing the credentials that Hermes / the Codex CLI
already maintain on disk.

Design notes
------------
*Piggyback, don't compete.*  Codex OAuth ``refresh_token``s are **single-use**:
whoever redeems one invalidates every other copy.  Hermes is the active agent on
this machine and refreshes on its own schedule, so SkillClaw deliberately does
NOT refresh on a timer.  It re-reads the token store on every request and uses
whatever Hermes most recently wrote.  Only when the on-disk access token is
genuinely expired (and Hermes has not yet rotated it) does SkillClaw perform a
refresh itself -- and then it writes the result back to *both* stores so Hermes
and the Codex CLI stay in sync instead of being stranded on a consumed token.

Token store precedence mirrors Hermes' own recovery order:
1. ``~/.hermes/auth.json``  -> ``providers.openai-codex.tokens``
2. ``~/.codex/auth.json``   -> ``tokens``          (Codex CLI shared file)

The newest valid access token across both wins.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# OAuth client identity, mirrored from Hermes (hermes_cli/auth.py).
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"

# Refresh this many seconds before the JWT's own ``exp``.  Matches Hermes'
# CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS so both agents consider a token
# "stale" at the same moment rather than fighting over a narrow window.
REFRESH_SKEW_SECONDS = 120

_HERMES_AUTH_PATH = Path.home() / ".hermes" / "auth.json"

# Serialises refreshes within this process.  Cross-process races are handled by
# re-reading the store after acquiring the lock (another agent may have already
# rotated the token while we waited).
_refresh_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Path helpers                                                                 #
# --------------------------------------------------------------------------- #


def _codex_auth_path() -> Path:
    """Return ~/.codex/auth.json, honouring CODEX_HOME like the Codex CLI."""
    codex_home = os.getenv("CODEX_HOME", "").strip()
    if not codex_home:
        codex_home = str(Path.home() / ".codex")
    return Path(codex_home).expanduser() / "auth.json"


# --------------------------------------------------------------------------- #
# JWT inspection                                                               #
# --------------------------------------------------------------------------- #


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Best-effort decode of a JWT payload. Returns {} on any malformed input.

    Never raises: a bad token should surface downstream as a 401 from the API,
    not as a crash inside SkillClaw's request path.
    """
    if not isinstance(token, str) or not token.strip():
        return {}
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def token_expiry(token: str) -> float | None:
    """Return the JWT ``exp`` as a unix timestamp, or None if absent."""
    exp = decode_jwt_claims(token).get("exp")
    if isinstance(exp, (int, float)):
        return float(exp)
    return None


def is_expired(token: str, *, skew_seconds: float = REFRESH_SKEW_SECONDS) -> bool:
    """True when the token is past (exp - skew).

    A token with no parseable ``exp`` is treated as still valid: we would rather
    let the upstream reject it than trigger an unnecessary refresh that consumes
    the single-use refresh token.
    """
    exp = token_expiry(token)
    if exp is None:
        return False
    return time.time() >= (exp - float(skew_seconds))


def account_id_from_token(token: str) -> str:
    """Extract the ChatGPT account id claim used for the account header."""
    claims = decode_jwt_claims(token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        acct = auth_claims.get("chatgpt_account_id")
        if isinstance(acct, str) and acct.strip():
            return acct.strip()
    return ""


# --------------------------------------------------------------------------- #
# Token store reads                                                            #
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("[CodexOAuth] could not read %s: %s", path, e)
        return {}


def _tokens_from_hermes_store() -> dict[str, str]:
    payload = _read_json(_HERMES_AUTH_PATH)
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        return {}
    entry = providers.get("openai-codex")
    if not isinstance(entry, dict):
        return {}
    tokens = entry.get("tokens")
    if not isinstance(tokens, dict):
        return {}
    return {
        "access_token": str(tokens.get("access_token", "") or ""),
        "refresh_token": str(tokens.get("refresh_token", "") or ""),
        "account_id": str(tokens.get("account_id", "") or ""),
    }


def _tokens_from_codex_cli_store() -> dict[str, str]:
    payload = _read_json(_codex_auth_path())
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return {}
    return {
        "access_token": str(tokens.get("access_token", "") or ""),
        "refresh_token": str(tokens.get("refresh_token", "") or ""),
        "account_id": str(tokens.get("account_id", "") or ""),
    }


def load_tokens() -> dict[str, str]:
    """Load the freshest Codex token pair available on this machine.

    Prefers whichever store holds the access token with the later ``exp`` so we
    naturally adopt a rotation performed by Hermes or the Codex CLI.
    """
    candidates = [t for t in (_tokens_from_hermes_store(), _tokens_from_codex_cli_store()) if t.get("access_token")]
    if not candidates:
        return {}
    if len(candidates) == 1:
        return candidates[0]

    def sort_key(tok: dict[str, str]) -> float:
        exp = token_expiry(tok.get("access_token", ""))
        return exp if exp is not None else 0.0

    return max(candidates, key=sort_key)


# --------------------------------------------------------------------------- #
# Token store writes                                                           #
# --------------------------------------------------------------------------- #


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".skillclaw.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    os.replace(tmp, path)


def _persist_to_hermes_store(tokens: dict[str, str], last_refresh: str) -> bool:
    """Update providers.openai-codex in ~/.hermes/auth.json, preserving the rest.

    Also refreshes any credential_pool entries that carried the *previous*
    access/refresh pair, mirroring Hermes' own singleton->pool sync so the pool
    does not keep replaying a token we have just consumed.
    """
    payload = _read_json(_HERMES_AUTH_PATH)
    if not payload:
        return False
    providers = payload.setdefault("providers", {})
    if not isinstance(providers, dict):
        return False
    entry = providers.get("openai-codex")
    if not isinstance(entry, dict):
        entry = {}
    stale = entry.get("tokens") if isinstance(entry.get("tokens"), dict) else {}
    stale_access = str(stale.get("access_token", "") or "")
    stale_refresh = str(stale.get("refresh_token", "") or "")

    merged = dict(stale)
    merged.update(
        {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
        }
    )
    entry["tokens"] = merged
    entry["last_refresh"] = last_refresh
    entry.setdefault("auth_mode", "chatgpt")
    providers["openai-codex"] = entry

    pool = payload.get("credential_pool")
    if isinstance(pool, dict):
        codex_entries = pool.get("openai-codex")
        if isinstance(codex_entries, list):
            for pool_entry in codex_entries:
                if not isinstance(pool_entry, dict):
                    continue
                # Only advance entries that were aliases of the singleton we
                # just rotated; independent accounts keep their own tokens.
                if (
                    str(pool_entry.get("access_token", "") or "") == stale_access
                    or str(pool_entry.get("refresh_token", "") or "") == stale_refresh
                ):
                    pool_entry["access_token"] = tokens["access_token"]
                    pool_entry["refresh_token"] = tokens["refresh_token"]
                    pool_entry["last_refresh"] = last_refresh

    _write_json_atomic(_HERMES_AUTH_PATH, payload)
    return True


def _persist_to_codex_cli_store(tokens: dict[str, str], last_refresh: str) -> bool:
    """Update ~/.codex/auth.json so the Codex CLI does not strand on a dead token."""
    path = _codex_auth_path()
    payload = _read_json(path)
    if not payload:
        return False
    existing = payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {}
    merged = dict(existing)
    merged.update(
        {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
        }
    )
    payload["tokens"] = merged
    payload["last_refresh"] = last_refresh
    _write_json_atomic(path, payload)
    return True


def persist_tokens(tokens: dict[str, str]) -> list[str]:
    """Write a refreshed pair to every store that already tracked Codex auth.

    Returns the list of store paths successfully updated.  Stores that do not
    exist are skipped rather than created -- SkillClaw should never invent a
    credential file for a tool the user does not use.
    """
    from datetime import datetime, timezone

    last_refresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    written: list[str] = []
    for label, path, fn in (
        ("hermes", _HERMES_AUTH_PATH, _persist_to_hermes_store),
        ("codex-cli", _codex_auth_path(), _persist_to_codex_cli_store),
    ):
        try:
            if fn(tokens, last_refresh):
                written.append(str(path))
        except Exception as e:
            logger.warning("[CodexOAuth] failed to persist tokens to %s store (%s): %s", label, path, e)
    return written


# --------------------------------------------------------------------------- #
# Refresh                                                                      #
# --------------------------------------------------------------------------- #


class CodexAuthError(RuntimeError):
    """Raised when a usable Codex OAuth access token cannot be obtained."""

    def __init__(self, message: str, *, code: str = "codex_auth_error", relogin_required: bool = False):
        super().__init__(message)
        self.code = code
        self.relogin_required = relogin_required


def refresh_tokens(refresh_token: str, *, timeout_seconds: float = 20.0) -> dict[str, str]:
    """Redeem a refresh token for a new access token.

    Pure with respect to local state: the caller decides whether to persist.
    """
    import httpx

    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise CodexAuthError(
            "Codex auth is missing refresh_token; run `hermes auth` to re-authenticate.",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )

    with httpx.Client(timeout=httpx.Timeout(max(5.0, float(timeout_seconds)))) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "skillclaw-codex-oauth",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token.strip(),
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code == 429:
        # Quota exhaustion, not a credential problem -- the stored refresh token
        # is still good, so never escalate this to "please re-login".
        raise CodexAuthError(
            "Codex provider quota exhausted (429); credentials are still valid.",
            code="codex_rate_limited",
            relogin_required=False,
        )

    if response.status_code != 200:
        code = "codex_refresh_failed"
        message = f"Codex token refresh failed with status {response.status_code}."
        try:
            err = response.json()
            if isinstance(err, dict):
                err_obj = err.get("error")
                if isinstance(err_obj, dict):
                    code = str(err_obj.get("code") or err_obj.get("type") or code)
                    if err_obj.get("message"):
                        message = f"Codex token refresh failed: {err_obj['message']}"
                elif isinstance(err_obj, str) and err_obj.strip():
                    code = err_obj.strip()
                    desc = err.get("error_description") or err.get("message")
                    if desc:
                        message = f"Codex token refresh failed: {desc}"
        except Exception:
            pass
        relogin = code in {"invalid_grant", "invalid_token", "invalid_request", "refresh_token_reused"} or (
            response.status_code in {401, 403}
        )
        if code == "refresh_token_reused":
            message = (
                "Codex refresh token was already consumed by another client "
                "(Hermes, Codex CLI, or the VS Code extension). "
                "Run `codex` then `hermes auth` to mint a fresh pair."
            )
        raise CodexAuthError(message, code=code, relogin_required=relogin)

    try:
        payload = response.json()
    except Exception as exc:
        raise CodexAuthError(
            "Codex token refresh returned invalid JSON.",
            code="codex_refresh_invalid_json",
            relogin_required=True,
        ) from exc

    access = payload.get("access_token")
    if not isinstance(access, str) or not access.strip():
        raise CodexAuthError(
            "Codex token refresh response was missing access_token.",
            code="codex_refresh_missing_access_token",
            relogin_required=True,
        )

    rotated = payload.get("refresh_token")
    return {
        "access_token": access.strip(),
        "refresh_token": (rotated.strip() if isinstance(rotated, str) and rotated.strip() else refresh_token.strip()),
    }


def get_access_token(*, allow_refresh: bool = True) -> str:
    """Return a currently-valid Codex access token.

    Re-reads the shared token stores on every call so a rotation performed by
    Hermes is picked up immediately and no refresh is spent unnecessarily.
    """
    tokens = load_tokens()
    if not tokens.get("access_token"):
        raise CodexAuthError(
            "No Codex OAuth tokens found. Log in with `hermes auth` (or run `codex`) first.",
            code="codex_auth_missing",
            relogin_required=True,
        )

    if not is_expired(tokens["access_token"]):
        return tokens["access_token"]

    if not allow_refresh:
        raise CodexAuthError(
            "Codex access token is expired and refresh is disabled.",
            code="codex_auth_expired",
            relogin_required=False,
        )

    with _refresh_lock:
        # Another process (usually Hermes itself) may have rotated the token
        # while we waited for the lock -- prefer their work over spending our
        # single-use refresh token.
        latest = load_tokens()
        if latest.get("access_token") and not is_expired(latest["access_token"]):
            return latest["access_token"]

        refreshed = refresh_tokens(latest.get("refresh_token") or tokens.get("refresh_token", ""))
        written = persist_tokens(refreshed)
        logger.info(
            "[CodexOAuth] refreshed Codex access token; synced %d store(s): %s",
            len(written),
            ", ".join(written) or "(none)",
        )
        return refreshed["access_token"]


# --------------------------------------------------------------------------- #
# Request headers                                                              #
# --------------------------------------------------------------------------- #


def is_codex_base_url(base_url: str) -> bool:
    """True only for OpenAI's official Codex endpoint (not look-alike proxies)."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(base_url or "")
        path = parsed.path.rstrip("/")
        return (
            parsed.scheme == "https"
            and parsed.hostname == "chatgpt.com"
            and parsed.port in (None, 443)
            and (path == "/backend-api/codex" or path.startswith("/backend-api/codex/"))
        )
    except (TypeError, ValueError):
        return False


def build_auth_headers(base_url: str, *, allow_refresh: bool = True) -> dict[str, str]:
    """Authorization + harness-identity headers for a Codex OAuth request.

    OpenAI requires third-party harnesses to identify themselves via
    ``originator`` / ``User-Agent``, and to pass the account id from the token's
    ``chatgpt_account_id`` claim.  Without these the endpoint rejects the call.
    """
    access_token = get_access_token(allow_refresh=allow_refresh)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "codex_cli_rs/0.0.0 (SkillClaw)",
        "originator": "codex_cli_rs",
    }
    account_id = account_id_from_token(access_token)
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return headers


def describe() -> dict[str, Any]:
    """Diagnostic snapshot for `skillclaw doctor` / manual troubleshooting."""
    tokens = load_tokens()
    access = tokens.get("access_token", "")
    exp = token_expiry(access)
    return {
        "hermes_store": str(_HERMES_AUTH_PATH),
        "hermes_store_present": _HERMES_AUTH_PATH.is_file(),
        "codex_cli_store": str(_codex_auth_path()),
        "codex_cli_store_present": _codex_auth_path().is_file(),
        "access_token_present": bool(access),
        "access_token_expired": is_expired(access) if access else None,
        "expires_at_unix": exp,
        "expires_in_seconds": (round(exp - time.time()) if exp else None),
        "account_id": account_id_from_token(access) if access else "",
        "refresh_token_present": bool(tokens.get("refresh_token")),
    }
