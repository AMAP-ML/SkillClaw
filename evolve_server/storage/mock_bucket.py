"""
Local filesystem bucket backend that mimics the ``oss2.Bucket`` API.

Historically this module only exposed ``MockBucket`` for ``--mock`` mode.
It now also serves as the generic local backend for evolve-server runs
that should read/write sessions + skills from a normal directory tree.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "mock"


class _LocalObject:
    """Returned by :pymethod:`LocalBucket.get_object`."""

    def __init__(self, data: bytes, key: str) -> None:
        self._data = data
        self.key = key

    def read(self) -> bytes:
        return self._data


class _LocalObjectInfo:
    """Lightweight stand-in used by :class:`LocalObjectIterator`."""

    def __init__(self, key: str) -> None:
        self.key = key


class LocalObjectIterator:
    """Iterate over files in the local backend that start with *prefix*.

    Mimics ``oss2.ObjectIterator(bucket, prefix=...)``.
    """

    def __init__(self, bucket: "LocalBucket", prefix: str = "") -> None:
        self._bucket = bucket
        self._prefix = prefix

    def __iter__(self) -> Iterator[_LocalObjectInfo]:
        root = self._bucket._root
        for dirpath, _dirs, filenames in os.walk(root):
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                if rel.startswith(self._prefix):
                    yield _LocalObjectInfo(rel)


class LocalBucket:
    """Drop-in replacement for ``oss2.Bucket`` backed by a local directory.

    Parameters
    ----------
    root:
        Local directory that acts as the "bucket root".  Defaults to
        ``evolve_server/mock/``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = str(root or _DEFAULT_LOCAL_DIR)
        logger.info("[LocalBucket] using local root: %s", self._root)

    def get_object(self, key: str) -> _LocalObject:
        path = os.path.join(self._root, key)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LocalBucket: key not found: {key}")
        with open(path, "rb") as f:
            data = f.read()
        return _LocalObject(data, key)

    def put_object(self, key: str, data: bytes | str | io.IOBase) -> None:
        path = os.path.join(self._root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            raw = data
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raw = data.read()
        with open(path, "wb") as f:
            f.write(raw)
        logger.info("[LocalBucket] wrote %d bytes → %s", len(raw), path)

    def delete_object(self, key: str) -> None:
        path = os.path.join(self._root, key)
        if os.path.isfile(path):
            os.remove(path)
            logger.info("[LocalBucket] deleted %s", path)


# Backward-compatible aliases used by the original mock-only code path.
MockBucket = LocalBucket
MockObjectIterator = LocalObjectIterator
