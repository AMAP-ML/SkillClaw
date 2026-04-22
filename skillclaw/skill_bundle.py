"""Helpers for reading, hashing, and writing multi-file skill bundles."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

_BUNDLE_ENTRYPOINT = "SKILL.md"
_IGNORED_NAMES = {".DS_Store"}
_IGNORED_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


class SkillBundleError(ValueError):
    """Raised when a bundle is malformed or a bundle path is unsafe."""


def _coerce_bytes(data: bytes | bytearray | str) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    raise TypeError(f"Unsupported bundle payload type: {type(data).__name__}")


def normalize_bundle_rel_path(rel_path: str) -> str:
    value = str(rel_path or "").strip().replace("\\", "/")
    if not value:
        raise SkillBundleError("Bundle path must not be empty")
    parts = PurePosixPath(value).parts
    if not parts:
        raise SkillBundleError("Bundle path must not be empty")
    if any(part in {"", ".", ".."} for part in parts):
        raise SkillBundleError(f"Unsafe bundle path: {rel_path!r}")
    return "/".join(parts)


def is_ignored_bundle_rel_path(rel_path: str) -> bool:
    parts = PurePosixPath(normalize_bundle_rel_path(rel_path)).parts
    if any(part in _IGNORED_DIR_NAMES for part in parts[:-1]):
        return True
    leaf = parts[-1]
    if leaf in _IGNORED_NAMES:
        return True
    return any(leaf.endswith(suffix) for suffix in _IGNORED_SUFFIXES)


def read_skill_bundle(skill_dir: str | os.PathLike[str]) -> dict[str, bytes]:
    root = Path(skill_dir)
    if not root.is_dir():
        return {}

    bundle: dict[str, bytes] = {}
    for rel_path in list_skill_bundle_paths(root):
        path = root / Path(rel_path)
        bundle[rel_path] = path.read_bytes()
    return bundle


def list_skill_bundle_paths(skill_dir: str | os.PathLike[str]) -> list[str]:
    root = Path(skill_dir)
    if not root.is_dir():
        return []

    paths: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        if is_ignored_bundle_rel_path(rel_path):
            continue
        paths.append(rel_path)
    return paths


def coerce_skill_bundle(bundle_files: Mapping[str, bytes | bytearray | str]) -> dict[str, bytes]:
    bundle: dict[str, bytes] = {}
    for raw_rel_path, raw_data in bundle_files.items():
        rel_path = normalize_bundle_rel_path(raw_rel_path)
        if is_ignored_bundle_rel_path(rel_path):
            continue
        bundle[rel_path] = _coerce_bytes(raw_data)
    return bundle


def bundle_file_records(bundle_files: Mapping[str, bytes | bytearray | str]) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for rel_path, raw_data in sorted(coerce_skill_bundle(bundle_files).items()):
        data = _coerce_bytes(raw_data)
        records.append({
            "path": rel_path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        })
    return records


def bundle_tree_sha256(bundle_files: Mapping[str, bytes | bytearray | str]) -> str:
    digest = hashlib.sha256()
    for record in bundle_file_records(bundle_files):
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_skill_bundle_with_meta(
    skill_dir: str | os.PathLike[str],
) -> tuple[dict[str, bytes], list[dict[str, int | str]], str]:
    bundle = read_skill_bundle(skill_dir)
    records = bundle_file_records(bundle)
    tree_sha = bundle_tree_sha256(bundle)
    return bundle, records, tree_sha


def bundle_entrypoint_bytes(
    bundle_files: Mapping[str, bytes | bytearray | str],
    entrypoint: str = _BUNDLE_ENTRYPOINT,
) -> bytes:
    bundle = coerce_skill_bundle(bundle_files)
    key = normalize_bundle_rel_path(entrypoint)
    if key not in bundle:
        raise SkillBundleError(f"Skill bundle is missing required entrypoint {key}")
    return bundle[key]


def bundle_entrypoint_text(
    bundle_files: Mapping[str, bytes | bytearray | str],
    entrypoint: str = _BUNDLE_ENTRYPOINT,
) -> str:
    return bundle_entrypoint_bytes(bundle_files, entrypoint).decode("utf-8")


def write_skill_bundle(
    skill_dir: str | os.PathLike[str],
    bundle_files: Mapping[str, bytes | bytearray | str],
    *,
    clean: bool = False,
) -> None:
    root = Path(skill_dir)
    if clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for rel_path, data in sorted(coerce_skill_bundle(bundle_files).items()):
        path = root / Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def bundle_has_only_entrypoint(
    bundle_files: Mapping[str, bytes | bytearray | str],
    entrypoint: str = _BUNDLE_ENTRYPOINT,
) -> bool:
    bundle = coerce_skill_bundle(bundle_files)
    return set(bundle.keys()) == {normalize_bundle_rel_path(entrypoint)}


def bundle_paths(bundle_files: Mapping[str, bytes | bytearray | str] | Iterable[str]) -> list[str]:
    if isinstance(bundle_files, Mapping):
        paths = bundle_files.keys()
    else:
        paths = bundle_files
    out: list[str] = []
    for rel_path in paths:
        clean = normalize_bundle_rel_path(str(rel_path))
        if is_ignored_bundle_rel_path(clean):
            continue
        out.append(clean)
    return sorted(set(out))
