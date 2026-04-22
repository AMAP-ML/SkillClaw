"""
Snapshot builder for the SkillClaw dashboard.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evolve_server.core.skill_registry import SkillIDRegistry
from evolve_server.core.utils import build_skill_md
from evolve_server.storage.oss_helpers import fetch_version_bundle, load_version_bundle_record
from skillclaw.skill_bundle import bundle_entrypoint_text, read_skill_bundle_with_meta

from .config import SkillClawConfig
from .skill_hub import SkillHub
from .validation_store import ValidationStore

logger = logging.getLogger(__name__)

_CORE_FRONTMATTER_KEYS = {"name", "description", "metadata", "category"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_skill_id(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def _parse_iso8601(raw: str) -> datetime | None:
    value = str(raw or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _latest_timestamp(*values: str) -> str:
    latest: tuple[datetime, str] | None = None
    for value in values:
        parsed = _parse_iso8601(value)
        if parsed is None:
            continue
        if latest is None or parsed > latest[0]:
            latest = (parsed, value)
    return latest[1] if latest else ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def _bundle_record(
    *,
    tree_sha256: str = "",
    files: Any = None,
    format_name: str = "bundle_v1",
    entrypoint: str = "SKILL.md",
) -> dict[str, Any]:
    normalized_files = [dict(item) for item in files if isinstance(item, dict)] if isinstance(files, list) else []
    return {
        "format": str(format_name or "bundle_v1"),
        "entrypoint": str(entrypoint or "SKILL.md"),
        "tree_sha256": str(tree_sha256 or ""),
        "files": normalized_files,
    }


def _truncate(text: str, limit: int = 180) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _trim_message(text: str, limit: int = 6000) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _normalize_timestamp(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""

    parsed = _parse_iso8601(value)
    if parsed is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    if parsed is None:
        return value
    return parsed.isoformat()


def _extract_skill_names(items: Any) -> list[str]:
    names: set[str] = set()
    if not isinstance(items, list):
        return []
    for item in items:
        if isinstance(item, dict):
            raw = item.get("skill_name") or item.get("name") or item.get("skill")
        else:
            raw = item
        name = str(raw or "").strip()
        if name:
            names.add(name)
    return sorted(names)


def _extract_message_text(message: Any) -> str:
    if isinstance(message, str):
        return message.strip()
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = str(item.get("text", "") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def _clean_transcript_text(text: str) -> str:
    value = str(text or "").strip()
    wrapped = re.fullmatch(r"<user_query>\s*(.*?)\s*</user_query>", value, flags=re.DOTALL)
    if wrapped:
        value = wrapped.group(1).strip()
    return _trim_message(value)


def _guess_category(skills_dir: Path, skill_path: Path) -> str:
    try:
        rel_parts = skill_path.resolve().relative_to(skills_dir.resolve()).parts
    except Exception:
        return "general"
    if len(rel_parts) >= 3:
        return str(rel_parts[0] or "general")
    return "general"


def _parse_skill_document(
    raw: str,
    *,
    fallback_name: str = "",
    fallback_category: str = "general",
) -> dict[str, Any]:
    body = raw.strip()
    fm: dict[str, Any] = {}
    if raw.startswith("---"):
        end_idx = raw.find("\n---", 3)
        if end_idx != -1:
            try:
                parsed = yaml.safe_load(raw[3:end_idx].strip()) or {}
                if isinstance(parsed, dict):
                    fm = parsed
            except yaml.YAMLError:
                fm = {}
            body = raw[end_idx + 4 :].strip()

    metadata = fm.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    skillclaw_meta = metadata.get("skillclaw")
    if not isinstance(skillclaw_meta, dict):
        skillclaw_meta = {}

    category = (
        str(skillclaw_meta.get("category") or fm.get("category") or fallback_category or "general").strip() or "general"
    )
    name = str(fm.get("name") or fallback_name or "").strip()
    description = str(fm.get("description") or "").strip()
    extra_frontmatter = {k: v for k, v in fm.items() if k not in _CORE_FRONTMATTER_KEYS}

    return {
        "name": name,
        "description": description,
        "category": category,
        "metadata": metadata,
        "extra_frontmatter": extra_frontmatter,
        "content": body,
        "skill_md": raw,
    }


def _load_local_skills(config: SkillClawConfig, warnings: list[str]) -> dict[str, dict[str, Any]]:
    skills_dir = Path(config.skills_dir).expanduser()
    if not skills_dir.is_dir():
        return {}

    stats = _read_json(skills_dir / "skill_stats.json", {})
    if not isinstance(stats, dict):
        stats = {}

    skills: dict[str, dict[str, Any]] = {}
    for skill_path in sorted(skills_dir.rglob("SKILL.md")):
        bundle_files, bundle_records, local_tree_sha = read_skill_bundle_with_meta(skill_path.parent)
        try:
            raw = bundle_entrypoint_text(bundle_files)
        except Exception:
            raw = _read_text(skill_path)
        if not raw:
            warnings.append(f"failed to read local skill file: {skill_path}")
            continue
        parsed = _parse_skill_document(
            raw,
            fallback_name=skill_path.parent.name,
            fallback_category=_guess_category(skills_dir, skill_path),
        )
        name = str(parsed.get("name") or skill_path.parent.name).strip()
        if not name:
            continue
        stat = stats.get(name)
        if not isinstance(stat, dict):
            stat = {}
        mtime = ""
        try:
            mtime = datetime.fromtimestamp(skill_path.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            pass
        local_sha = _hash_text(raw)
        local_bundle_record = _bundle_record(
            tree_sha256=local_tree_sha,
            files=bundle_records,
        )

        skills[name] = {
            "name": name,
            "skill_id": _stable_skill_id(name),
            "description": str(parsed.get("description") or ""),
            "category": str(parsed.get("category") or "general"),
            "metadata": parsed.get("metadata") or {},
            "extra_frontmatter": parsed.get("extra_frontmatter") or {},
            "content": str(parsed.get("content") or ""),
            "skill_md": str(parsed.get("skill_md") or ""),
            "source": "local",
            "has_local": True,
            "has_remote": False,
            "local_path": str(skill_path),
            "uploaded_at": "",
            "uploaded_by": "",
            "updated_at": mtime,
            "local_updated_at": mtime,
            "remote_updated_at": "",
            "current_version": 1,
            "current_sha": local_sha,
            "local_sha": local_sha,
            "remote_sha": "",
            "current_tree_sha": local_tree_sha,
            "local_tree_sha": local_tree_sha,
            "remote_tree_sha": "",
            "local_inject_count": int(stat.get("inject_count", 0) or 0),
            "observed_injection_count": 0,
            "read_count": 0,
            "modified_count": 0,
            "session_count": 0,
            "effectiveness": float(stat.get("effectiveness", 0.0) or 0.0),
            "positive_count": int(stat.get("positive_count", 0) or 0),
            "negative_count": int(stat.get("negative_count", 0) or 0),
            "neutral_count": int(stat.get("neutral_count", 0) or 0),
            "last_injected_at": str(stat.get("last_injected_at", "") or ""),
            "stats": stat,
            "manifest": {},
            "registry": {},
            "versions": [],
            "local_bundle_record": local_bundle_record,
            "remote_bundle_record": {},
        }

    return skills


def _skillclaw_state_dir(config: SkillClawConfig) -> Path:
    return Path(config.skills_dir).expanduser().parent / "state"


def _find_transcript_path(session_id: str, transcript_paths: list[str]) -> Path | None:
    candidates: list[Path] = []
    for raw_path in transcript_paths:
        path = Path(str(raw_path)).expanduser()
        if path.stem == session_id or session_id in path.as_posix():
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.stem != session_id, len(str(item))))
    return candidates[0]


def _parse_cursor_transcript_turns(transcript_path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current_turn: dict[str, Any] | None = None

    try:
        with transcript_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                role = str(record.get("role", "") or "").strip().lower()
                if role not in {"user", "assistant"}:
                    continue

                text = _clean_transcript_text(_extract_message_text(record.get("message")))
                if not text:
                    continue

                if role == "user":
                    if current_turn is not None:
                        turns.append(current_turn)
                    current_turn = {
                        "turn_num": len(turns) + 1,
                        "prompt_text": text,
                        "response_text": "",
                        "reasoning_content": None,
                        "tool_calls": [],
                        "read_skills": [],
                        "modified_skills": [],
                        "tool_results": [],
                        "tool_results_raw": [],
                        "tool_observations": [],
                        "tool_errors": [],
                        "injected_skills": [],
                        "prm_score": None,
                    }
                    continue

                if current_turn is None:
                    continue
                if current_turn["response_text"]:
                    current_turn["response_text"] += "\n\n" + text
                else:
                    current_turn["response_text"] = text
    except OSError as exc:
        warnings.append(f"failed to read local transcript '{transcript_path}': {exc}")
        return []

    if current_turn is not None:
        turns.append(current_turn)

    return turns


def _record_dir_candidates(config: SkillClawConfig) -> list[Path]:
    raw = str(getattr(config, "record_dir", "") or "").strip()
    if not raw:
        return []

    record_dir = Path(raw).expanduser()
    if record_dir.is_absolute():
        return [record_dir]

    candidates: list[Path] = []
    seen: set[str] = set()
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidate = (parent / record_dir).resolve()
        key = str(candidate)
        if key in seen:
            continue
        candidates.append(candidate)
        seen.add(key)
    return candidates


def _resolve_record_dir(config: SkillClawConfig) -> Path | None:
    candidates = _record_dir_candidates(config)
    for candidate in candidates:
        if candidate.is_dir() or (candidate / "conversations.jsonl").exists():
            return candidate
    return candidates[0] if candidates else None


def _extract_record_instruction(record: dict[str, Any]) -> str:
    instruction = _clean_transcript_text(str(record.get("instruction_text", "") or ""))
    if instruction:
        return instruction

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "").strip().lower()
            if role != "user":
                continue
            text = _clean_transcript_text(_extract_message_text(message))
            if text:
                return text

    return _clean_transcript_text(str(record.get("prompt_text", "") or ""))


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _load_record_prm_scores(record_dir: Path, warnings: list[str]) -> dict[tuple[str, int], float]:
    prm_scores_path = record_dir / "prm_scores.jsonl"
    if not prm_scores_path.is_file():
        return {}

    scores: dict[tuple[str, int], float] = {}
    try:
        with prm_scores_path.open(encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    warnings.append(f"failed to parse PRM record '{prm_scores_path}' line {line_no}")
                    continue
                if not isinstance(payload, dict):
                    continue
                session_id = str(payload.get("session_id", "") or "").strip()
                if not session_id:
                    continue
                try:
                    turn_num = int(payload.get("turn", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if turn_num <= 0:
                    continue
                score = payload.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    scores[(session_id, turn_num)] = float(score)
    except OSError as exc:
        warnings.append(f"failed to read PRM records '{prm_scores_path}': {exc}")

    return scores


def _load_record_sessions(config: SkillClawConfig, warnings: list[str]) -> list[dict[str, Any]]:
    record_dir = _resolve_record_dir(config)
    if record_dir is None:
        return []

    conversations_path = record_dir / "conversations.jsonl"
    if not conversations_path.is_file():
        return []

    prm_scores = _load_record_prm_scores(record_dir, warnings)
    grouped: dict[str, dict[str, Any]] = {}
    line_counter = 0

    try:
        with conversations_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line_counter += 1
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    warnings.append(f"failed to parse conversation record '{conversations_path}' line {line_counter}")
                    continue
                if not isinstance(payload, dict):
                    continue

                session_id = str(payload.get("session_id", "") or "").strip()
                if not session_id:
                    continue

                timestamp = _normalize_timestamp(str(payload.get("timestamp", "") or ""))
                try:
                    turn_num = int(payload.get("turn", 0) or 0)
                except (TypeError, ValueError):
                    turn_num = 0

                group = grouped.setdefault(
                    session_id,
                    {
                        "session_id": session_id,
                        "timestamp": "",
                        "turns": {},
                        "line_index": {},
                        "record_path": str(conversations_path),
                    },
                )
                group["timestamp"] = _latest_timestamp(group["timestamp"], timestamp) or timestamp

                if turn_num <= 0:
                    turn_num = max(group["turns"].keys(), default=0) + 1

                turn_payload = {
                    "turn_num": turn_num,
                    "prompt_text": _extract_record_instruction(payload),
                    "response_text": _trim_message(str(payload.get("response_text", "") or "")),
                    "reasoning_content": None,
                    "tool_calls": _normalize_tool_calls(payload.get("tool_calls")),
                    "read_skills": [],
                    "modified_skills": [],
                    "tool_results": [],
                    "tool_results_raw": [],
                    "tool_observations": [],
                    "tool_errors": [],
                    "injected_skills": [],
                    "prm_score": prm_scores.get((session_id, turn_num)),
                }

                existing_line = group["line_index"].get(turn_num, -1)
                if line_counter >= existing_line:
                    group["turns"][turn_num] = turn_payload
                    group["line_index"][turn_num] = line_counter
    except OSError as exc:
        warnings.append(f"failed to read local conversations '{conversations_path}': {exc}")
        return []

    sessions: list[dict[str, Any]] = []
    for session_id, group in grouped.items():
        turns = [group["turns"][turn_num] for turn_num in sorted(group["turns"])]
        sessions.append(
            {
                "session_id": session_id,
                "timestamp": str(group.get("timestamp", "") or ""),
                "user_alias": "local",
                "num_turns": len(turns),
                "turns": turns,
                "source": "local",
                "outcome": "",
                "outcome_reasons": [],
                "outcome_reason_count": 0,
                "active_skills": [],
                "transcript_path": "",
                "trajectory_path": "",
                "record_path": str(group.get("record_path", "") or ""),
            }
        )

    return sessions


def _merge_session_turns(
    base_turns: list[dict[str, Any]],
    overlay_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_turn: dict[int, dict[str, Any]] = {}
    for turn in base_turns:
        if not isinstance(turn, dict):
            continue
        try:
            turn_num = int(turn.get("turn_num", 0) or 0)
        except (TypeError, ValueError):
            continue
        if turn_num <= 0:
            continue
        by_turn[turn_num] = dict(turn)

    list_fields = {
        "tool_calls",
        "read_skills",
        "modified_skills",
        "tool_results",
        "tool_results_raw",
        "tool_observations",
        "tool_errors",
        "injected_skills",
    }
    nullable_fields = {"prm_score", "reasoning_content"}

    for turn in overlay_turns:
        if not isinstance(turn, dict):
            continue
        try:
            turn_num = int(turn.get("turn_num", 0) or 0)
        except (TypeError, ValueError):
            continue
        if turn_num <= 0:
            continue

        current = by_turn.get(turn_num)
        if current is None:
            by_turn[turn_num] = dict(turn)
            continue

        merged = dict(current)
        for key, value in turn.items():
            if key == "turn_num":
                continue
            if key in list_fields:
                if isinstance(value, list) and value:
                    merged[key] = value
                else:
                    merged.setdefault(key, current.get(key, []))
                continue
            if key in nullable_fields:
                if value is not None:
                    merged[key] = value
                continue
            if str(value or "").strip():
                merged[key] = value
        by_turn[turn_num] = merged

    return [by_turn[turn_num] for turn_num in sorted(by_turn)]


def _merge_local_sessions(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged_turns = _merge_session_turns(
        list(base.get("turns") or []),
        list(overlay.get("turns") or []),
    )
    merged["turns"] = merged_turns
    merged["timestamp"] = _latest_timestamp(
        str(base.get("timestamp", "") or ""),
        str(overlay.get("timestamp", "") or ""),
    ) or str(overlay.get("timestamp", "") or base.get("timestamp", "") or "")
    merged["user_alias"] = str(overlay.get("user_alias", "") or base.get("user_alias", "") or "local")
    merged["num_turns"] = max(
        int(base.get("num_turns", 0) or 0),
        int(overlay.get("num_turns", 0) or 0),
        len(merged_turns),
    )
    merged["source"] = "local"
    merged["outcome"] = str(overlay.get("outcome", "") or base.get("outcome", "") or "")

    outcome_reasons = [
        str(item or "").strip()
        for item in [*(base.get("outcome_reasons") or []), *(overlay.get("outcome_reasons") or [])]
        if str(item or "").strip()
    ]
    deduped_outcome_reasons = list(dict.fromkeys(outcome_reasons))
    merged["outcome_reasons"] = deduped_outcome_reasons[:20]
    merged["outcome_reason_count"] = len(deduped_outcome_reasons)
    merged["active_skills"] = sorted(
        {
            str(item or "").strip()
            for item in [*(base.get("active_skills") or []), *(overlay.get("active_skills") or [])]
            if str(item or "").strip()
        }
    )
    merged["transcript_path"] = str(base.get("transcript_path", "") or overlay.get("transcript_path", "") or "")
    merged["trajectory_path"] = str(base.get("trajectory_path", "") or overlay.get("trajectory_path", "") or "")
    merged["record_path"] = str(overlay.get("record_path", "") or base.get("record_path", "") or "")
    return merged


def _load_state_sessions(config: SkillClawConfig, warnings: list[str]) -> list[dict[str, Any]]:
    state_dir = _skillclaw_state_dir(config)
    conv_offsets_path = state_dir / "conv_offsets.json"
    trajectories_dir = state_dir / "trajectories"

    conv_offsets = _read_json(conv_offsets_path, {})
    transcript_paths = list(conv_offsets.keys()) if isinstance(conv_offsets, dict) else []

    trajectories: dict[str, dict[str, Any]] = {}
    if trajectories_dir.is_dir():
        for trajectory_path in sorted(trajectories_dir.glob("*.json")):
            payload = _read_json(trajectory_path, {})
            if not isinstance(payload, dict):
                continue
            session_id = str(payload.get("conversation_id", "") or trajectory_path.stem).strip()
            if not session_id:
                continue
            payload["_trajectory_path"] = str(trajectory_path)
            trajectories[session_id] = payload

    session_ids = set(trajectories.keys())
    for raw_path in transcript_paths:
        session_id = Path(str(raw_path)).stem.strip()
        if session_id:
            session_ids.add(session_id)

    sessions: list[dict[str, Any]] = []
    for session_id in sorted(session_ids):
        trajectory = trajectories.get(session_id, {})
        transcript_path = _find_transcript_path(session_id, transcript_paths)
        turns = _parse_cursor_transcript_turns(transcript_path, warnings) if transcript_path else []
        active_skills = sorted(
            {str(item or "").strip() for item in (trajectory.get("active_skills") or []) if str(item or "").strip()}
        )
        if turns and active_skills:
            turns[0]["injected_skills"] = active_skills
        elif active_skills:
            turns = [
                {
                    "turn_num": 1,
                    "prompt_text": "(local trajectory imported without transcript)",
                    "response_text": "",
                    "reasoning_content": None,
                    "tool_calls": [],
                    "read_skills": [],
                    "modified_skills": [],
                    "tool_results": [],
                    "tool_results_raw": [],
                    "tool_observations": [],
                    "tool_errors": [],
                    "injected_skills": active_skills,
                    "prm_score": None,
                }
            ]

        timestamp = str(trajectory.get("end_time") or trajectory.get("start_time") or "")
        if not timestamp and transcript_path is not None:
            try:
                timestamp = datetime.fromtimestamp(transcript_path.stat().st_mtime, tz=timezone.utc).isoformat()
            except OSError:
                timestamp = ""

        outcome_reasons = [
            str(item or "").strip() for item in (trajectory.get("outcome_reasons") or []) if str(item or "").strip()
        ]
        sessions.append(
            {
                "session_id": session_id,
                "timestamp": timestamp,
                "user_alias": "local",
                "num_turns": len(turns),
                "turns": turns,
                "source": "local",
                "outcome": str(trajectory.get("outcome", "") or ""),
                "outcome_reasons": outcome_reasons[:20],
                "outcome_reason_count": len(outcome_reasons),
                "active_skills": active_skills,
                "transcript_path": str(transcript_path) if transcript_path is not None else "",
                "trajectory_path": str(trajectory.get("_trajectory_path", "") or ""),
            }
        )

    sessions.sort(
        key=lambda item: (
            str(item.get("timestamp", "") or ""),
            str(item.get("session_id", "") or ""),
        ),
        reverse=True,
    )
    return sessions


def _load_local_sessions(config: SkillClawConfig, warnings: list[str]) -> list[dict[str, Any]]:
    sessions_by_id: dict[str, dict[str, Any]] = {}

    for session in _load_state_sessions(config, warnings):
        session_id = str(session.get("session_id", "") or "")
        if session_id:
            sessions_by_id[session_id] = session

    for session in _load_record_sessions(config, warnings):
        session_id = str(session.get("session_id", "") or "")
        if not session_id:
            continue
        existing = sessions_by_id.get(session_id)
        if existing is None:
            sessions_by_id[session_id] = session
        else:
            sessions_by_id[session_id] = _merge_local_sessions(existing, session)

    sessions = list(sessions_by_id.values())
    sessions.sort(
        key=lambda item: (
            str(item.get("timestamp", "") or ""),
            str(item.get("session_id", "") or ""),
        ),
        reverse=True,
    )
    return sessions


def _load_shared_skills(
    config: SkillClawConfig,
    warnings: list[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not config.sharing_enabled or not config.dashboard_include_shared:
        return {}, [], [], {}

    try:
        hub = SkillHub.from_config(config)
    except Exception as exc:
        warnings.append(f"failed to initialize shared storage: {exc}")
        return {}, [], [], {}

    try:
        manifest = hub._load_remote_manifest()
    except Exception as exc:
        warnings.append(f"failed to load shared manifest: {exc}")
        manifest = {}

    registry = SkillIDRegistry()
    try:
        registry.load_from_oss(hub._bucket, hub._prefix())
    except Exception as exc:
        warnings.append(f"failed to load shared registry: {exc}")

    registry_entries = registry.all_entries()
    skills: dict[str, dict[str, Any]] = {}
    candidate_docs_by_skill: dict[str, dict[str, str]] = defaultdict(dict)
    validation_jobs: list[dict[str, Any]] = []

    try:
        validation_store = ValidationStore.from_config(config)
        for job in validation_store.list_jobs():
            if not isinstance(job, dict):
                continue
            job_id = str(job.get("job_id", "") or "")
            results = validation_store.list_results(job_id) if job_id else []
            decision = validation_store.load_decision(job_id) if job_id else None
            accepted_count = sum(1 for item in results if isinstance(item, dict) and item.get("accepted") is True)
            rejected_count = sum(1 for item in results if isinstance(item, dict) and item.get("accepted") is not True)
            score_values = [
                float(item["score"])
                for item in results
                if isinstance(item, dict)
                and isinstance(item.get("score"), (int, float))
                and not isinstance(item.get("score"), bool)
            ]
            mean_score = round(sum(score_values) / len(score_values), 3) if score_values else None
            decision_status = ""
            if isinstance(decision, dict):
                decision_status = str(decision.get("status", "") or "")
            if decision_status:
                status = decision_status
            elif not results:
                status = "pending"
            else:
                status = "review"

            candidate_skill = job.get("candidate_skill")
            candidate_name = ""
            if isinstance(candidate_skill, dict):
                candidate_name = str(candidate_skill.get("name", "") or "")
                if candidate_name:
                    try:
                        candidate_md = build_skill_md(candidate_skill)
                    except Exception:
                        candidate_md = ""
                    if candidate_md:
                        candidate_docs_by_skill[candidate_name][_hash_text(candidate_md)] = candidate_md

            validation_jobs.append(
                {
                    "job_id": job_id,
                    "created_at": str(job.get("created_at", "") or ""),
                    "skill_name": str(job.get("candidate_skill_name", "") or candidate_name),
                    "proposed_action": str(job.get("proposed_action", "") or ""),
                    "status": status,
                    "result_count": len(results),
                    "accepted_count": accepted_count,
                    "rejected_count": rejected_count,
                    "mean_score": mean_score,
                    "job": job,
                    "results": results,
                    "decision": decision or {},
                }
            )
    except Exception as exc:
        warnings.append(f"failed to load validation jobs: {exc}")

    for name, record in manifest.items():
        raw = ""
        try:
            raw = hub._bucket.get_object(hub._skill_key(name)).read().decode("utf-8")
        except Exception as exc:
            warnings.append(f"failed to fetch shared skill '{name}': {exc}")
        parsed = _parse_skill_document(
            raw,
            fallback_name=name,
            fallback_category=str(record.get("category", "general") or "general"),
        )
        registry_entry = registry_entries.get(name)
        if not isinstance(registry_entry, dict):
            registry_entry = {}
        history = registry_entry.get("history")
        if not isinstance(history, list):
            history = []
        enriched_history: list[dict[str, Any]] = []
        current_sha = str(registry_entry.get("content_sha") or record.get("sha256") or (_hash_text(raw) if raw else ""))
        current_version = int(registry_entry.get("version", 0) or record.get("version", 0) or 0)
        if current_version <= 0 and current_sha:
            current_version = 1
        remote_bundle_record = _bundle_record(
            tree_sha256=str(record.get("tree_sha256") or registry_entry.get("tree_sha256") or ""),
            files=record.get("files") or registry_entry.get("files") or [],
            format_name=str(record.get("format") or registry_entry.get("format") or "bundle_v1"),
            entrypoint=str(record.get("entrypoint") or registry_entry.get("entrypoint") or "SKILL.md"),
        )
        current_tree_sha = str(remote_bundle_record.get("tree_sha256") or current_sha)
        history_latest = ""
        for item in history:
            if isinstance(item, dict):
                version_entry = dict(item)
                version_num = int(version_entry.get("version", 0) or 0)
                version_bundle_record = _bundle_record(
                    tree_sha256=str(version_entry.get("tree_sha256", "") or ""),
                    files=version_entry.get("files") or [],
                    format_name=str(version_entry.get("format") or "bundle_v1"),
                    entrypoint=str(version_entry.get("entrypoint") or "SKILL.md"),
                )
                if (not version_bundle_record["files"] or not version_bundle_record["tree_sha256"]) and version_num > 0:
                    persisted_bundle_record = load_version_bundle_record(hub._bucket, hub._prefix(), name, version_num)
                    if isinstance(persisted_bundle_record, dict):
                        version_bundle_record = _bundle_record(
                            tree_sha256=str(
                                persisted_bundle_record.get("tree_sha256")
                                or version_bundle_record.get("tree_sha256")
                                or ""
                            ),
                            files=persisted_bundle_record.get("files") or version_bundle_record.get("files") or [],
                            format_name=str(
                                persisted_bundle_record.get("format")
                                or version_bundle_record.get("format")
                                or "bundle_v1"
                            ),
                            entrypoint=str(
                                persisted_bundle_record.get("entrypoint")
                                or version_bundle_record.get("entrypoint")
                                or "SKILL.md"
                            ),
                        )
                content_sha = str(version_entry.get("content_sha", "") or "")
                snapshot_md = ""
                if content_sha:
                    snapshot_md = candidate_docs_by_skill.get(name, {}).get(content_sha, "")
                if not snapshot_md and content_sha and raw and content_sha == current_sha:
                    snapshot_md = raw
                if not snapshot_md and version_num > 0 and version_bundle_record.get("files"):
                    try:
                        snapshot_bundle = fetch_version_bundle(
                            hub._bucket,
                            hub._prefix(),
                            name,
                            version_num,
                            version_bundle_record,
                        )
                    except Exception:
                        snapshot_bundle = {}
                    if snapshot_bundle:
                        try:
                            snapshot_md = bundle_entrypoint_text(snapshot_bundle)
                        except Exception:
                            snapshot_md = ""
                if snapshot_md:
                    parsed_snapshot = _parse_skill_document(
                        snapshot_md,
                        fallback_name=name,
                        fallback_category=str(record.get("category", "general") or "general"),
                    )
                    version_entry["skill_md"] = snapshot_md
                    version_entry["content"] = str(parsed_snapshot.get("content") or "")
                if version_bundle_record.get("files"):
                    version_entry["bundle_record"] = version_bundle_record
                    version_entry["tree_sha256"] = str(version_bundle_record.get("tree_sha256") or "")
                    version_entry["format"] = str(version_bundle_record.get("format") or "bundle_v1")
                    version_entry["entrypoint"] = str(version_bundle_record.get("entrypoint") or "SKILL.md")
                    version_entry["files"] = list(version_bundle_record.get("files") or [])
                enriched_history.append(version_entry)
                history_latest = _latest_timestamp(history_latest, str(version_entry.get("timestamp", "") or ""))

        skills[name] = {
            "name": name,
            "skill_id": str(registry_entry.get("skill_id") or _stable_skill_id(name)),
            "description": str(parsed.get("description") or record.get("description") or ""),
            "category": str(parsed.get("category") or record.get("category") or "general"),
            "metadata": parsed.get("metadata") or {},
            "extra_frontmatter": parsed.get("extra_frontmatter") or {},
            "content": str(parsed.get("content") or ""),
            "skill_md": str(parsed.get("skill_md") or ""),
            "source": "shared",
            "has_local": False,
            "has_remote": True,
            "local_path": "",
            "uploaded_at": str(record.get("uploaded_at", "") or ""),
            "uploaded_by": str(record.get("uploaded_by", "") or ""),
            "updated_at": _latest_timestamp(
                str(record.get("uploaded_at", "") or ""),
                history_latest,
            ),
            "local_updated_at": "",
            "remote_updated_at": _latest_timestamp(
                str(record.get("uploaded_at", "") or ""),
                history_latest,
            ),
            "current_version": current_version,
            "current_sha": current_sha,
            "local_sha": "",
            "remote_sha": current_sha,
            "current_tree_sha": current_tree_sha,
            "local_tree_sha": "",
            "remote_tree_sha": current_tree_sha,
            "local_inject_count": 0,
            "observed_injection_count": 0,
            "read_count": 0,
            "modified_count": 0,
            "session_count": 0,
            "effectiveness": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "last_injected_at": "",
            "stats": {},
            "manifest": record,
            "registry": registry_entry,
            "versions": enriched_history,
            "local_bundle_record": {},
            "remote_bundle_record": remote_bundle_record,
        }

    sessions: list[dict[str, Any]] = []
    try:
        prefix = f"{hub._prefix()}sessions/"
        for obj in hub._bucket.iter_objects(prefix=prefix):
            if not str(obj.key).endswith(".json"):
                continue
            try:
                payload = json.loads(hub._bucket.get_object(obj.key).read().decode("utf-8"))
                if isinstance(payload, dict):
                    sessions.append(payload)
            except Exception as exc:
                warnings.append(f"failed to parse session object '{obj.key}': {exc}")
    except Exception as exc:
        warnings.append(f"failed to list shared sessions: {exc}")

    sessions.sort(
        key=lambda item: (
            str(item.get("timestamp", "") or ""),
            str(item.get("session_id", "") or ""),
        ),
        reverse=True,
    )
    validation_jobs.sort(
        key=lambda item: (
            str(item.get("created_at", "") or ""),
            str(item.get("job_id", "") or ""),
        ),
        reverse=True,
    )
    return skills, sessions, validation_jobs, registry_entries


def build_dashboard_snapshot(config: SkillClawConfig) -> dict[str, Any]:
    warnings: list[str] = []
    local_skills = _load_local_skills(config, warnings)
    local_sessions = _load_local_sessions(config, warnings)
    shared_skills, shared_sessions, validation_jobs, registry_entries = _load_shared_skills(config, warnings)

    skills_by_name: dict[str, dict[str, Any]] = {name: dict(skill) for name, skill in local_skills.items()}

    for name, shared_skill in shared_skills.items():
        current = skills_by_name.get(name)
        if current is None:
            skills_by_name[name] = dict(shared_skill)
            continue

        current["source"] = "both"
        current["has_remote"] = True
        current["skill_id"] = str(shared_skill.get("skill_id") or current.get("skill_id") or _stable_skill_id(name))
        current["uploaded_at"] = str(shared_skill.get("uploaded_at", "") or current.get("uploaded_at", ""))
        current["uploaded_by"] = str(shared_skill.get("uploaded_by", "") or current.get("uploaded_by", ""))
        current["updated_at"] = _latest_timestamp(
            str(current.get("updated_at", "") or ""),
            str(shared_skill.get("updated_at", "") or ""),
        )
        current["local_updated_at"] = str(current.get("local_updated_at", "") or current.get("updated_at", "") or "")
        current["local_sha"] = str(current.get("local_sha", "") or current.get("current_sha", "") or "")
        current["remote_updated_at"] = str(
            shared_skill.get("remote_updated_at", "") or shared_skill.get("updated_at", "") or ""
        )
        current["remote_sha"] = str(shared_skill.get("remote_sha", "") or shared_skill.get("current_sha", "") or "")
        current["local_tree_sha"] = str(current.get("local_tree_sha", "") or current.get("current_tree_sha", "") or "")
        current["remote_tree_sha"] = str(
            shared_skill.get("remote_tree_sha", "") or shared_skill.get("current_tree_sha", "") or ""
        )
        current["current_version"] = int(
            shared_skill.get("current_version", 0) or current.get("current_version", 0) or 0
        )
        current["current_sha"] = str(shared_skill.get("current_sha", "") or current.get("current_sha", ""))
        current["current_tree_sha"] = str(
            shared_skill.get("current_tree_sha", "") or current.get("current_tree_sha", "") or current["current_sha"]
        )
        current["manifest"] = shared_skill.get("manifest") or {}
        current["registry"] = shared_skill.get("registry") or {}
        current["versions"] = list(shared_skill.get("versions") or [])
        current["remote_skill_md"] = shared_skill.get("skill_md", "")
        current["remote_content"] = shared_skill.get("content", "")
        current["remote_bundle_record"] = shared_skill.get("remote_bundle_record") or {}
        current["local_bundle_record"] = current.get("local_bundle_record") or {}
        if not current.get("description"):
            current["description"] = str(shared_skill.get("description", "") or "")
        if not current.get("metadata"):
            current["metadata"] = shared_skill.get("metadata") or {}
        if not current.get("extra_frontmatter"):
            current["extra_frontmatter"] = shared_skill.get("extra_frontmatter") or {}

    usage_by_name: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "observed_injection_count": 0,
            "read_count": 0,
            "modified_count": 0,
            "session_ids": set(),
        }
    )
    link_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    session_summaries: list[dict[str, Any]] = []

    sessions_by_id: dict[str, dict[str, Any]] = {}
    for session in local_sessions + shared_sessions:
        session_id = str(session.get("session_id", "") or "")
        if not session_id:
            continue
        existing = sessions_by_id.get(session_id)
        if existing is None:
            sessions_by_id[session_id] = session
            continue
        if str(session.get("source", "") or "") == "shared":
            sessions_by_id[session_id] = session

    for session in sessions_by_id.values():
        session_id = str(session.get("session_id", "") or "")
        if not session_id:
            continue
        turns = session.get("turns")
        if not isinstance(turns, list):
            turns = []
        prompt_preview = ""
        response_preview = ""
        prm_scores: list[float] = []
        session_skill_names: set[str] = set()
        injected_names_all: set[str] = set()
        read_names_all: set[str] = set()
        modified_names_all: set[str] = set()

        for turn in turns:
            if not isinstance(turn, dict):
                continue
            if not prompt_preview:
                prompt_preview = _truncate(str(turn.get("prompt_text", "") or ""))
            if not response_preview:
                response_preview = _truncate(str(turn.get("response_text", "") or ""))

            prm_score = turn.get("prm_score")
            if isinstance(prm_score, (int, float)) and not isinstance(prm_score, bool):
                prm_scores.append(float(prm_score))

            injected_names = _extract_skill_names(turn.get("injected_skills"))
            read_names = _extract_skill_names(turn.get("read_skills"))
            modified_names = _extract_skill_names(turn.get("modified_skills"))

            injected_names_all.update(injected_names)
            read_names_all.update(read_names)
            modified_names_all.update(modified_names)
            session_skill_names.update(injected_names)
            session_skill_names.update(read_names)
            session_skill_names.update(modified_names)

            for name in injected_names:
                usage = usage_by_name[name]
                usage["observed_injection_count"] += 1
                usage["session_ids"].add(session_id)
                link_counts[(session_id, name, "injected")] += 1
            for name in read_names:
                usage = usage_by_name[name]
                usage["read_count"] += 1
                usage["session_ids"].add(session_id)
                link_counts[(session_id, name, "read")] += 1
            for name in modified_names:
                usage = usage_by_name[name]
                usage["modified_count"] += 1
                usage["session_ids"].add(session_id)
                link_counts[(session_id, name, "modified")] += 1

        avg_prm_score = round(sum(prm_scores) / len(prm_scores), 3) if prm_scores else None
        session_summaries.append(
            {
                "session_id": session_id,
                "timestamp": str(session.get("timestamp", "") or ""),
                "user_alias": str(session.get("user_alias", "") or ""),
                "num_turns": int(session.get("num_turns", len(turns)) or len(turns)),
                "avg_prm_score": avg_prm_score,
                "source": str(session.get("source", "") or ""),
                "outcome": str(session.get("outcome", "") or ""),
                "outcome_reasons": list(session.get("outcome_reasons") or []),
                "outcome_reason_count": int(session.get("outcome_reason_count", 0) or 0),
                "skill_names": sorted(session_skill_names),
                "injected_skills": sorted(injected_names_all),
                "read_skills": sorted(read_names_all),
                "modified_skills": sorted(modified_names_all),
                "prompt_preview": prompt_preview,
                "response_preview": response_preview,
                "turns": turns,
            }
        )

    for name, usage in usage_by_name.items():
        skill = skills_by_name.get(name)
        if skill is None:
            registry_entry = registry_entries.get(name)
            if not isinstance(registry_entry, dict):
                registry_entry = {}
            skill = {
                "name": name,
                "skill_id": str(registry_entry.get("skill_id") or _stable_skill_id(name)),
                "description": "",
                "category": "general",
                "metadata": {},
                "extra_frontmatter": {},
                "content": "",
                "skill_md": "",
                "source": "observed",
                "has_local": False,
                "has_remote": False,
                "local_path": "",
                "uploaded_at": "",
                "uploaded_by": "",
                "updated_at": "",
                "local_updated_at": "",
                "remote_updated_at": "",
                "current_version": int(registry_entry.get("version", 0) or 0),
                "current_sha": str(registry_entry.get("content_sha", "") or ""),
                "local_sha": "",
                "remote_sha": str(registry_entry.get("content_sha", "") or ""),
                "current_tree_sha": str(registry_entry.get("tree_sha256", "") or ""),
                "local_tree_sha": "",
                "remote_tree_sha": str(registry_entry.get("tree_sha256", "") or ""),
                "local_inject_count": 0,
                "observed_injection_count": 0,
                "read_count": 0,
                "modified_count": 0,
                "session_count": 0,
                "effectiveness": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "last_injected_at": "",
                "stats": {},
                "manifest": {},
                "registry": registry_entry,
                "versions": (
                    list(registry_entry.get("history") or []) if isinstance(registry_entry.get("history"), list) else []
                ),
                "local_bundle_record": {},
                "remote_bundle_record": _bundle_record(
                    tree_sha256=str(registry_entry.get("tree_sha256", "") or ""),
                    files=registry_entry.get("files") or [],
                    format_name=str(registry_entry.get("format") or "bundle_v1"),
                    entrypoint=str(registry_entry.get("entrypoint") or "SKILL.md"),
                ),
            }
            skills_by_name[name] = skill

        skill["observed_injection_count"] = int(usage["observed_injection_count"])
        skill["read_count"] = int(usage["read_count"])
        skill["modified_count"] = int(usage["modified_count"])
        skill["session_count"] = len(usage["session_ids"])

    normalized_skills: list[dict[str, Any]] = []
    for name in sorted(skills_by_name):
        skill = skills_by_name[name]
        versions = [item for item in (skill.get("versions") or []) if isinstance(item, dict)]
        if not versions and skill.get("current_sha"):
            versions = [
                {
                    "version": int(skill.get("current_version", 0) or 1),
                    "content_sha": str(skill.get("current_sha", "") or ""),
                    "tree_sha256": str(skill.get("current_tree_sha", "") or ""),
                    "timestamp": str(
                        skill.get("updated_at") or skill.get("uploaded_at") or skill.get("last_injected_at") or ""
                    ),
                    "action": "snapshot",
                    "skill_md": str(skill.get("remote_skill_md") or skill.get("skill_md") or ""),
                    "content": str(skill.get("remote_content") or skill.get("content") or ""),
                    "bundle_record": skill.get("remote_bundle_record") or skill.get("local_bundle_record") or {},
                }
            ]
        skill["versions"] = versions
        if not skill.get("updated_at"):
            skill["updated_at"] = _latest_timestamp(
                str(skill.get("uploaded_at", "") or ""),
                str(skill.get("last_injected_at", "") or ""),
            )
        normalized_skills.append(skill)

    normalized_skills.sort(
        key=lambda item: (
            -int(item.get("session_count", 0) or 0),
            -int(item.get("observed_injection_count", 0) or 0),
            -int(item.get("local_inject_count", 0) or 0),
            str(item.get("name", "") or ""),
        )
    )
    session_summaries.sort(
        key=lambda item: (
            str(item.get("timestamp", "") or ""),
            str(item.get("session_id", "") or ""),
        ),
        reverse=True,
    )

    skill_id_by_name = {
        str(skill.get("name", "") or ""): str(skill.get("skill_id", "") or "")
        for skill in normalized_skills
        if str(skill.get("name", "") or "")
    }
    session_skill_links = [
        {
            "session_id": session_id,
            "skill_id": skill_id_by_name.get(skill_name, _stable_skill_id(skill_name)),
            "skill_name": skill_name,
            "relation": relation,
            "count": count,
        }
        for (session_id, skill_name, relation), count in sorted(link_counts.items())
    ]

    return {
        "generated_at": _utc_now_iso(),
        "meta": {
            "warnings": warnings,
            "sharing_enabled": bool(config.sharing_enabled),
            "dashboard_include_shared": bool(config.dashboard_include_shared),
            "sharing_backend": str(config.sharing_backend or ""),
            "sharing_group_id": str(config.sharing_group_id or "default"),
            "sharing_local_root": str(config.sharing_local_root or ""),
            "sharing_user_alias": str(config.sharing_user_alias or ""),
            "skills_dir": str(config.skills_dir or ""),
            "dashboard_db_path": str(config.dashboard_db_path or ""),
            "dashboard_evolve_server_url": str(config.dashboard_evolve_server_url or ""),
        },
        "skills": normalized_skills,
        "sessions": session_summaries,
        "session_skill_links": session_skill_links,
        "validation_jobs": validation_jobs,
    }
