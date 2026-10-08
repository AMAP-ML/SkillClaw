"""SkillClaw session capture as a native Hermes plugin (no proxy).

The default Hermes integration rewrites ``~/.hermes/config.yaml`` so that all model
traffic goes through the SkillClaw proxy. That cannot work when Hermes talks to a
provider the proxy cannot forward to (subscription OAuth such as ChatGPT/Codex or
Claude, several providers across Hermes profiles, provider-side fallback chains).

This plugin keeps Hermes' own provider routing untouched and only observes finished
turns through Hermes' ``post_llm_call`` hook. Turns of one Hermes session are written
as ONE session record in the format the evolve server already consumes
(``{group_id}/sessions/<id>.json`` with ``num_turns`` + ``turns[]``), including
``read_skills`` / ``modified_skills`` derived from Hermes' ``skill_view`` /
``skill_manage`` tool calls, so sessions are grouped by the skills they used instead
of all landing in the no-skill bucket.

Configuration (environment variables, read at registration time):

- ``SKILLCLAW_CAPTURE_ROOT``  local shared root, same as ``sharing.local_root``
  of the evolve server (required; the plugin is inert without it)
- ``SKILLCLAW_GROUP_ID``      group id (default ``default``)
- ``SKILLCLAW_USER_ALIAS``    alias recorded in sessions (default: Hermes profile name)
- ``SKILLCLAW_CAPTURE_IDLE_SECONDS``  idle time after which an open session is
  published to the queue (default 1800)

What is stored: the user message and final assistant text of each turn (secrets
redacted), the model, tool NAMES and skill names. System prompts, earlier history,
tool arguments and tool results are never stored.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path

MAX_TURNS = 40
MAX_RECORD_BYTES = 1024 * 1024
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,120}")
_SKILL_REF = re.compile(r"[A-Za-z0-9_.:/-]{1,200}")
SKILL_READ_TOOLS = {"skill_view"}
SKILL_WRITE_TOOLS = {"skill_manage"}
INTERNAL_SESSION_PREFIXES = ("skillclaw-evolve-", "skillclaw-validate-")


def scrub(value: object) -> str:
    text = str(value or "")
    try:  # Hermes' own redactor when running inside Hermes
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text, force=True)
    except Exception:
        pass
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{20,}", "[REDACTED]", text)
    return text


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("session record exceeds size bound")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _arguments(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _skill_name(args: dict) -> str:
    for key in ("name", "skill_name", "skill"):
        value = args.get(key)
        if isinstance(value, str) and _SKILL_REF.fullmatch(value.strip()):
            return value.strip().split("/")[-1]
    return ""


def current_turn_tools(conversation_history: list | None) -> tuple[list[str], list[dict], list[dict]]:
    """Tool names and skill references of the current turn (after the last user message)."""
    names: set[str] = set()
    read: dict[str, dict] = {}
    modified: dict[str, dict] = {}
    for msg in reversed(conversation_history or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            break
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = str((call.get("function") or {}).get("name") or "")
            if not _TOOL_NAME.fullmatch(name):
                continue
            names.add(name)
            if name not in SKILL_READ_TOOLS and name not in SKILL_WRITE_TOOLS:
                continue
            args = _arguments(call)
            skill = _skill_name(args)
            if not skill:
                continue
            ref = {"skill_id": "", "skill_name": skill}
            file_path = args.get("file_path")
            if name in SKILL_READ_TOOLS and isinstance(file_path, str) and _SKILL_REF.fullmatch(file_path):
                ref["path"] = file_path
            (read if name in SKILL_READ_TOOLS else modified).setdefault(skill, ref)
    return sorted(names), list(read.values()), list(modified.values())


class SessionCapture:
    """Accumulates turns per Hermes session and publishes idle sessions to the queue."""

    def __init__(
        self, root: str | Path, group_id: str = "default", user_alias: str = "hermes", idle_seconds: int = 1800
    ) -> None:
        self.root = Path(root).expanduser()
        self.group_id = group_id
        self.user_alias = user_alias
        self.idle_seconds = idle_seconds
        self.queue = self.root / group_id / "sessions"
        self.state = self.root / ".hermes-capture" / group_id
        self.open_dir = self.state / "open"
        self.seen_dir = self.state / "seen"

    # -- hook entry points -------------------------------------------------

    def on_turn(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        user_message: str = "",
        assistant_response: str = "",
        conversation_history: list | None = None,
        model: str = "",
        platform: str = "",
        **_: object,
    ) -> None:
        if not session_id or not turn_id or not assistant_response:
            return
        if str(session_id).startswith(INTERNAL_SESSION_PREFIXES):
            return
        tool_names, read_skills, modified_skills = current_turn_tools(conversation_history)
        turn = {
            "prompt_text": scrub(user_message)[:24000],
            "response_text": scrub(assistant_response)[:48000],
            "model": model,
            "platform": platform,
            "tool_names": tool_names,
            "read_skills": read_skills,
            "modified_skills": modified_skills,
            "turn_hash": hashlib.sha256(str(turn_id).encode()).hexdigest(),
        }
        session_hash = hashlib.sha256(str(session_id).encode()).hexdigest()
        marker = self.seen_dir / hashlib.sha256(f"{session_id}\0{turn_id}".encode()).hexdigest()
        open_path = self.open_dir / f"{self.user_alias}-{session_hash}.json"
        with self._lock():
            if marker.exists():  # duplicate hook delivery
                return
            record = self._load(open_path) or {
                "session_id": f"{self.user_alias}-{session_hash}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "user_alias": self.user_alias,
                "num_turns": 0,
                "turns": [],
            }
            record["turns"].append(turn)
            record["num_turns"] = len(record["turns"])
            try:
                _atomic_json(open_path, record)
            except ValueError:  # too large: close the episode, start a new one with this turn
                record["turns"].pop()
                record["num_turns"] = len(record["turns"])
                if record["turns"]:
                    _atomic_json(open_path, record)
                    self._publish(open_path)
                record.update(turns=[turn], num_turns=1)
                _atomic_json(open_path, record)
            self.seen_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.close(os.open(marker, os.O_CREAT | os.O_WRONLY, 0o600))
            if record["num_turns"] >= MAX_TURNS:
                self._publish(open_path)
            self._flush_idle_locked(exclude=open_path)

    def on_session_closed(self, *, session_id: str = "", old_session_id: str = "", **_: object) -> None:
        """Gateway/CLI reset or shutdown: publish the finished session right away."""
        sid = old_session_id or session_id
        if not sid:
            return
        path = self.open_dir / f"{self.user_alias}-{hashlib.sha256(str(sid).encode()).hexdigest()}.json"
        with self._lock():
            if path.exists():
                self._publish(path)

    def flush_idle(self, idle_seconds: int | None = None) -> list[str]:
        with self._lock():
            return self._flush_idle_locked(idle_seconds=idle_seconds)

    # -- internals -----------------------------------------------------------

    def _lock(self):
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = open(self.state / ".lock", "a")

        class _Held:
            def __enter__(self_inner):
                fcntl.flock(handle, fcntl.LOCK_EX)
                return handle

            def __exit__(self_inner, *exc):
                handle.close()

        return _Held()

    @staticmethod
    def _load(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _publish(self, path: Path) -> str | None:
        record = self._load(path)
        if not record or not record.get("turns"):
            path.unlink(missing_ok=True)
            return None
        first = str(record["turns"][0].get("turn_hash", ""))[:12] or str(int(time.time()))
        key = f"{record['session_id']}-{first}"  # unique per episode, never overwritten
        record["session_id"] = key
        _atomic_json(self.queue / f"{key}.json", record)
        path.unlink(missing_ok=True)
        return key

    def _flush_idle_locked(self, idle_seconds: int | None = None, exclude: Path | None = None) -> list[str]:
        idle = self.idle_seconds if idle_seconds is None else idle_seconds
        now = time.time()
        published = []
        if not self.open_dir.exists():
            return published
        for path in sorted(self.open_dir.glob("*.json")):
            if path == exclude or now - path.stat().st_mtime < idle:
                continue
            key = self._publish(path)
            if key:
                published.append(key)
        return published


def _profile_name(home: Path) -> str:
    return home.name if home.parent.name == "profiles" else "default"


def register(ctx) -> None:
    root = os.environ.get("SKILLCLAW_CAPTURE_ROOT", "").strip()
    if not root:
        return
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:
        home = Path.home() / ".hermes"
    capture = SessionCapture(
        root,
        group_id=os.environ.get("SKILLCLAW_GROUP_ID", "default").strip() or "default",
        user_alias=os.environ.get("SKILLCLAW_USER_ALIAS", "").strip() or _profile_name(home),
        idle_seconds=int(os.environ.get("SKILLCLAW_CAPTURE_IDLE_SECONDS", "1800") or 1800),
    )
    ctx.register_hook("post_llm_call", capture.on_turn)
    ctx.register_hook("on_session_finalize", capture.on_session_closed)
