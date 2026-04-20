"""
Shared object storage backends for SkillClaw.

Supports three deployment modes:

- ``local``: local filesystem directory tree
- ``s3``: AWS S3 and S3-compatible APIs such as MinIO / R2 / B2
- ``oss``: Alibaba Cloud OSS
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterator


class ObjectInfo:
    """Lightweight object listing entry with a single ``key`` field."""

    def __init__(self, key: str) -> None:
        self.key = key


class _BytesObject:
    """Simple in-memory object body that exposes ``read()``."""

    def __init__(self, data: bytes, key: str) -> None:
        self._data = data
        self.key = key

    def read(self) -> bytes:
        return self._data


def _read_bytes(data: bytes | str | io.IOBase) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    return data.read()


def normalize_backend(backend: str | None, *, endpoint: str = "", local_root: str = "") -> str:
    """Map user-facing aliases into the concrete backend names we support."""
    value = str(backend or "").strip().lower().replace("_", "-")
    aliases = {
        "filesystem": "local",
        "fs": "local",
        "s3-compatible": "s3",
        "minio": "s3",
        "r2": "s3",
        "b2": "s3",
    }
    if value in aliases:
        return aliases[value]
    if value:
        return value
    if local_root:
        return "local"
    if endpoint and "aliyuncs.com" in endpoint:
        return "oss"
    return ""


def is_not_found_error(exc: Exception) -> bool:
    """Best-effort check for backends that signal missing objects differently."""
    if isinstance(exc, FileNotFoundError):
        return True
    name = type(exc).__name__
    return "NoSuchKey" in name or "NotFound" in name or "NoSuchBucket" in name


class LocalObjectStore:
    """Filesystem-backed object store rooted at a local directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = str(Path(root).expanduser())

    def get_object(self, key: str) -> _BytesObject:
        path = os.path.join(self._root, key)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LocalObjectStore: key not found: {key}")
        with open(path, "rb") as f:
            return _BytesObject(f.read(), key)

    def put_object(self, key: str, data: bytes | str | io.IOBase) -> None:
        path = os.path.join(self._root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(_read_bytes(data))

    def delete_object(self, key: str) -> None:
        path = os.path.join(self._root, key)
        if os.path.isfile(path):
            os.remove(path)

    def iter_objects(self, prefix: str = "") -> Iterator[ObjectInfo]:
        root = Path(self._root)
        if not root.exists():
            return iter(())
        keys: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel.startswith(prefix):
                keys.append(rel)
        return iter(ObjectInfo(key) for key in sorted(keys))


class S3ObjectStore:
    """S3-compatible backend built on top of ``boto3``."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "",
        session_token: str = "",
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "S3-compatible skill sharing requires the 'boto3' package. "
                "Install it with: pip install skillclaw[sharing]"
            ) from exc

        session_kwargs = {}
        if access_key_id:
            session_kwargs["aws_access_key_id"] = access_key_id
        if secret_access_key:
            session_kwargs["aws_secret_access_key"] = secret_access_key
        if session_token:
            session_kwargs["aws_session_token"] = session_token
        if region:
            session_kwargs["region_name"] = region
        session = boto3.session.Session(**session_kwargs)

        client_kwargs = {}
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        if region:
            client_kwargs["region_name"] = region

        self._client = session.client("s3", **client_kwargs)
        self._bucket = bucket

    def get_object(self, key: str):
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"]

    def put_object(self, key: str, data: bytes | str | io.IOBase) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=_read_bytes(data))

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def iter_objects(self, prefix: str = "") -> Iterator[ObjectInfo]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents") or []:
                key = item.get("Key")
                if key:
                    yield ObjectInfo(str(key))


class OSSObjectStore:
    """Alibaba Cloud OSS backend built on top of ``oss2``."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "",
        session_token: str = "",
    ) -> None:
        del region, session_token
        try:
            import oss2
        except ImportError as exc:
            raise ImportError(
                "Alibaba Cloud OSS skill sharing requires the 'oss2' package. "
                "Install it with: pip install skillclaw[sharing]"
            ) from exc

        auth = oss2.Auth(access_key_id, secret_access_key)
        self._bucket = oss2.Bucket(auth, endpoint, bucket)

    def get_object(self, key: str):
        return self._bucket.get_object(key)

    def put_object(self, key: str, data: bytes | str | io.IOBase) -> None:
        if isinstance(data, (bytes, bytearray, str)):
            self._bucket.put_object(key, _read_bytes(data))
            return
        self._bucket.put_object(key, data)

    def delete_object(self, key: str) -> None:
        self._bucket.delete_object(key)

    def iter_objects(self, prefix: str = "") -> Iterator[ObjectInfo]:
        import oss2

        for obj in oss2.ObjectIterator(self._bucket, prefix=prefix):
            yield ObjectInfo(obj.key)


def build_object_store(
    *,
    backend: str | None,
    endpoint: str = "",
    bucket: str = "",
    access_key_id: str = "",
    secret_access_key: str = "",
    region: str = "",
    session_token: str = "",
    local_root: str = "",
):
    """Create the configured object storage backend."""
    resolved = normalize_backend(backend, endpoint=endpoint, local_root=local_root)
    if not resolved and (bucket or endpoint):
        resolved = "oss" if endpoint and "aliyuncs.com" in endpoint else "s3"
    if resolved == "local":
        if not local_root:
            raise ValueError("Local storage backend requires local_root.")
        return LocalObjectStore(local_root)
    if resolved == "s3":
        if not bucket:
            raise ValueError("S3 storage backend requires a bucket.")
        return S3ObjectStore(
            endpoint=endpoint,
            bucket=bucket,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
            session_token=session_token,
        )
    if resolved == "oss":
        if not endpoint or not bucket:
            raise ValueError("OSS storage backend requires endpoint and bucket.")
        return OSSObjectStore(
            endpoint=endpoint,
            bucket=bucket,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )
    raise ValueError(f"Unsupported storage backend: {backend!r}")
