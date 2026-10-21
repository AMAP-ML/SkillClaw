"""Opt-in, local Hermes turn capture; provider routing is never changed.

Only finished-turn text, model/platform, tool names and skill references are stored.
System prompts, history and tool arguments/results are not persisted. Redaction is
best-effort, NOT anonymization. See ../README.md for lifecycle and privacy limits.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path

MAX_TURNS = 40
MAX_RECORD_BYTES = 1024 * 1024
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,120}")
_SKILL_REF = re.compile(r"[A-Za-z0-9_.:/-]{1,200}")
_COMPONENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}")
_HASH = re.compile(r"[a-f0-9]{64}")
_WRITE_ACTIONS = {"create", "edit", "patch", "delete", "write_file", "remove_file"}
INTERNAL_SESSION_PREFIXES = ("skillclaw-evolve-", "skillclaw-validate-")
logger = logging.getLogger(__name__)


def scrub(value: object) -> str:
    text = str(value or "")
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text, force=True)
    except Exception:
        pass
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{20,}", "[REDACTED]", text)
    return text


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir(path: Path) -> None:
    if not path.exists():
        _mkdir(path.parent)
        path.mkdir(mode=0o700, exist_ok=True)
        _sync_dir(path.parent)


def _encoded(value: dict) -> bytes:
    data = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("session record exceeds size bound")
    return data


def _atomic_json(path: Path, value: dict) -> None:
    data = _encoded(value)
    _mkdir(path.parent)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
        _sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _arguments(call: dict) -> dict:
    function = call.get("function")
    raw = function.get("arguments") if isinstance(function, dict) else None
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _skill_name(args: dict) -> str:
    value = args.get("name")
    if isinstance(value, str) and _SKILL_REF.fullmatch(value):
        return value.split("/")[-1]
    return ""


def current_turn_tools(conversation_history: list | None) -> tuple[list[str], list[dict], list[dict]]:
    """Attribute attempted calls, not successful results, after the last user message.

    skill_view has a single name/file_path; parallel calls are separate tool_calls.
    skill_manage advertises operations[]; older transcripts use flat action/name.
    """
    names: set[str] = set()
    read: dict[str, dict] = {}
    modified: dict[str, dict] = {}
    for msg in reversed(conversation_history or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            break
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                continue
            name = call["function"].get("name")
            if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
                continue
            names.add(name)
            if name not in {"skill_view", "skill_manage"}:
                continue
            args = _arguments(call)
            operations = args.get("operations", [args]) if name == "skill_manage" else [args]
            if not isinstance(operations, list):
                continue
            for op in operations:
                if not isinstance(op, dict):
                    continue
                action = op.get("action")
                if name == "skill_manage" and (not isinstance(action, str) or action not in _WRITE_ACTIONS):
                    continue
                skill = _skill_name(op)
                if not skill:
                    continue
                ref = {"skill_id": "", "skill_name": skill}
                file_path = op.get("file_path")
                if name == "skill_view" and isinstance(file_path, str) and _SKILL_REF.fullmatch(file_path):
                    ref["path"] = file_path
                (read if name == "skill_view" else modified).setdefault(skill, ref)
    return sorted(names), list(read.values()), list(modified.values())


def _home() -> Path:
    try:
        from hermes_constants import get_hermes_home
    except ImportError:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser().resolve()
    return Path(get_hermes_home()).expanduser().resolve()


class SessionCapture:
    """Profile-local accumulation with durable receipts and atomic queue handoff."""

    def __init__(
        self,
        root: str | Path,
        group_id: str = "default",
        user_alias: str = "hermes",
        idle_seconds: float = 1800,
        profile_id: str | None = None,
    ) -> None:
        if not _COMPONENT.fullmatch(group_id) or not _COMPONENT.fullmatch(user_alias):
            raise ValueError("group_id and user_alias must be safe path components (1-80 characters)")
        if not math.isfinite(idle_seconds) or idle_seconds <= 0:
            raise ValueError("idle_seconds must be finite and positive")
        self.root = Path(root).expanduser().resolve()
        self.group_id = group_id
        self.user_alias = user_alias
        self.idle_seconds = idle_seconds
        identity = f"{profile_id if profile_id is not None else _home()}\0{user_alias}"
        self.namespace = hashlib.sha256(identity.encode()).hexdigest()
        self.queue = self.root / group_id / "sessions"
        self.state = self.root / ".hermes-capture" / group_id / self.namespace
        self.open_dir = self.state / "open"
        self.seen_dir = self.state / "seen"
        self._stopped = threading.Event()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="skillclaw-capture-idle", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._worker is not None:
            self._worker.join()

    def _run(self) -> None:
        while not self._stopped.wait(min(30.0, self.idle_seconds)):
            try:
                self.flush_idle()
            except (OSError, ValueError):
                logger.warning("SkillClaw idle capture failed; retained state for retry")

    def _session_hash(self, session_id: str) -> str:
        return hashlib.sha256(f"{self.namespace}\0{session_id}".encode()).hexdigest()

    def _remember(self, session_hash: str, turn_hash: str) -> None:
        marker = self.seen_dir / f"{session_hash}-{turn_hash}"
        if not marker.exists():
            _atomic_json(marker, {})

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
        if self._stopped.is_set() or not session_id or not turn_id or not assistant_response:
            return
        if str(session_id).startswith(INTERNAL_SESSION_PREFIXES):
            return
        tool_names, read_skills, modified_skills = current_turn_tools(conversation_history)
        turn = {
            "prompt_text": scrub(user_message)[:24000],
            "response_text": scrub(assistant_response)[:48000],
            "model": scrub(model)[:200],
            "platform": scrub(platform)[:120],
            "tool_names": tool_names,
            "read_skills": read_skills,
            "modified_skills": modified_skills,
            "turn_hash": hashlib.sha256(str(turn_id).encode()).hexdigest(),
        }
        session_hash = self._session_hash(str(session_id))
        marker = self.seen_dir / f"{session_hash}-{turn['turn_hash']}"
        open_path = self.open_dir / f"{session_hash}.json"
        fresh = {
            "session_id": f"{self.user_alias}-{session_hash}-{turn['turn_hash']}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_alias": self.user_alias,
            "num_turns": 1,
            "turns": [turn],
        }
        _encoded(fresh)  # Reject an oversized single turn before publishing or changing anything.
        with self._lock():
            if self._stopped.is_set() or marker.exists():
                return
            record = self._load(open_path)
            if record and any(t["turn_hash"] == turn["turn_hash"] for t in record["turns"]):
                self._remember(session_hash, turn["turn_hash"])
                return  # Recovered a crash after record commit but before receipt commit.
            if record:
                candidate = {**record, "turns": [*record["turns"], turn], "num_turns": record["num_turns"] + 1}
                try:
                    _encoded(candidate)
                    fits = record["num_turns"] < MAX_TURNS
                except ValueError:
                    fits = False
                if fits:
                    fresh = candidate
                else:
                    self._publish(open_path)
            _atomic_json(open_path, fresh)
            self._remember(session_hash, turn["turn_hash"])
            if fresh["num_turns"] >= MAX_TURNS:
                self._publish(open_path)

    def on_session_closed(self, *, session_id: str = "", old_session_id: str = "", **_: object) -> None:
        sid = old_session_id or session_id
        if self._stopped.is_set() or not sid:
            return
        with self._lock():
            if not self._stopped.is_set():
                self._publish(self.open_dir / f"{self._session_hash(str(sid))}.json")

    def flush_idle(self, idle_seconds: float | None = None) -> list[str]:
        with self._lock():
            if self._stopped.is_set():
                return []
            return self._flush_idle_locked(idle_seconds)

    def _lock(self):
        _mkdir(self.state)
        handle = os.fdopen(os.open(self.state / ".lock", os.O_CREAT | os.O_RDWR, 0o600), "a")

        class _Held:
            def __enter__(self_inner):
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                except BaseException:
                    handle.close()
                    raise
                return handle

            def __exit__(self_inner, *exc):
                handle.close()

        return _Held()

    @staticmethod
    def _load(path: Path) -> dict | None:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        record = None
        try:
            record = json.loads(raw)
            valid = (
                len(raw) <= MAX_RECORD_BYTES
                and isinstance(record, dict)
                and isinstance(record.get("session_id"), str)
                and re.fullmatch(r"[A-Za-z0-9_.-]{1,220}", record["session_id"])
                and isinstance(record.get("turns"), list)
                and bool(record["turns"])
                and record.get("num_turns") == len(record["turns"])
                and all(isinstance(t, dict) and _HASH.fullmatch(str(t.get("turn_hash", ""))) for t in record["turns"])
            )
        except (ValueError, UnicodeError):
            valid = False
        if not valid:
            raise ValueError("invalid capture state; preserved for manual recovery")
        return record

    def _publish(self, path: Path) -> str | None:
        record = self._load(path)
        if record is None:
            return None
        # Receipts must be durable BEFORE handoff. A consumer can remove the queue
        # file immediately; recovery must not depend on its continued presence.
        for turn in record["turns"]:
            self._remember(path.stem, turn["turn_hash"])
        _mkdir(self.queue)
        target = self.queue / f"{record['session_id']}.json"
        if target.exists():
            raise FileExistsError("capture queue collision; preserved both records")
        # A single rename, not copy + unlink: no two-copy process-crash window.
        os.replace(path, target)
        _sync_dir(self.queue)
        _sync_dir(self.open_dir)
        return record["session_id"]

    def _flush_idle_locked(self, idle_seconds: float | None = None) -> list[str]:
        idle = self.idle_seconds if idle_seconds is None else idle_seconds
        now = time.time()
        published = []
        for path in sorted(self.open_dir.glob("*.json")):
            try:
                if now - path.stat().st_mtime < idle:
                    continue
                key = self._publish(path)
                if key:
                    published.append(key)
            except (OSError, ValueError):
                logger.warning("SkillClaw could not publish a record; retained state for manual recovery or retry")
        return published


def _profile_name(home: Path) -> str:
    return home.name if home.parent.name == "profiles" else "default"


def register(ctx) -> None:
    root = os.environ.get("SKILLCLAW_CAPTURE_ROOT", "").strip()
    if not root:
        return
    home = _home()
    capture = SessionCapture(
        root,
        group_id=os.environ.get("SKILLCLAW_GROUP_ID", "default").strip() or "default",
        user_alias=os.environ.get("SKILLCLAW_USER_ALIAS", "").strip() or _profile_name(home),
        idle_seconds=float(os.environ.get("SKILLCLAW_CAPTURE_IDLE_SECONDS", "1800") or 1800),
        profile_id=str(home),
    )
    ctx.register_hook("post_llm_call", capture.on_turn)
    ctx.register_hook("on_session_finalize", capture.on_session_closed)
    ctx.register_hook("on_session_reset", capture.on_session_closed)
    ctx.on_unload(capture.stop)
    capture.start()
