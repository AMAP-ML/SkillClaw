from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

try:
    import httpx  # noqa: F401
except ModuleNotFoundError:
    httpx_stub = types.ModuleType("httpx")
    httpx_stub.BaseTransport = object
    httpx_stub.Client = object
    httpx_stub.Response = object
    sys.modules["httpx"] = httpx_stub

from evolve_server.core.config import EvolveServerConfig  # noqa: E402
from evolve_server.engines.workflow import EvolveServer  # noqa: E402
from skillclaw.nacos_skill_hub import NacosSkillHub, _bundle_to_nacos_zip  # noqa: E402

SKILL_MD = """---
name: demo-skill
description: Demo skill
---

# Demo Skill
"""

UPDATED_SKILL_MD = """---
name: demo-skill
description: Demo skill
---

# Demo Skill

Updated content.
"""


class FakeNacosClient:
    def __init__(self, items: list[dict], downloads: dict[tuple[str, str], bytes]) -> None:
        self.items = items
        self.downloads = downloads
        self.uploads: list[dict] = []
        self.submits: list[tuple[str, str]] = []
        self.download_calls: list[dict] = []

    def list_skills(self) -> list[dict]:
        return self.items

    def get_skill(self, name: str) -> dict:
        for item in self.items:
            if item.get("name") == name:
                return item
        return {}

    def download_skill_zip(self, name: str, *, version: str | None = None, label: str = "latest", admin: bool = False):
        self.download_calls.append({"name": name, "version": version, "label": label, "admin": admin})
        return self.downloads[(name, str(version or ""))]

    def upload_skill_zip(self, *, zip_bytes: bytes, filename: str, overwrite: bool, target_version: str | None) -> str:
        self.uploads.append(
            {
                "zip_bytes": zip_bytes,
                "filename": filename,
                "overwrite": overwrite,
                "target_version": target_version,
            }
        )
        return "demo-skill"

    def submit(self, name: str, version: str) -> str:
        self.submits.append((name, version))
        return version


def _write_skill(root: Path, body: str = SKILL_MD) -> None:
    skill_dir = root / "demo-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def _zip(body: str) -> bytes:
    return _bundle_to_nacos_zip("demo-skill", {"SKILL.md": body.encode("utf-8")})


def test_push_skips_when_reviewing_version_matches_local_bundle(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir)
    client = FakeNacosClient(
        [{"name": "demo-skill", "reviewingVersion": "0.0.1", "labels": {}}],
        {("demo-skill", "0.0.1"): _zip(SKILL_MD)},
    )

    result = NacosSkillHub(client=client).push_skills(str(skills_dir))

    assert result["skipped"] == 1
    assert result["uploaded"] == 0
    assert client.uploads == []
    assert client.download_calls == [{"name": "demo-skill", "version": "0.0.1", "label": "latest", "admin": True}]


def test_push_fails_when_reviewing_version_has_different_content(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, UPDATED_SKILL_MD)
    client = FakeNacosClient(
        [{"name": "demo-skill", "reviewingVersion": "0.0.1", "labels": {}}],
        {("demo-skill", "0.0.1"): _zip(SKILL_MD)},
    )

    with pytest.raises(RuntimeError, match="already has reviewing version 0.0.1"):
        NacosSkillHub(client=client).push_skills(str(skills_dir))

    assert client.uploads == []
    assert client.submits == []


def test_push_fails_when_reviewed_version_has_different_content(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, UPDATED_SKILL_MD)
    client = FakeNacosClient(
        [
            {
                "name": "demo-skill",
                "labels": {},
                "versions": [{"version": "0.0.3", "status": "reviewed"}],
            }
        ],
        {("demo-skill", "0.0.3"): _zip(SKILL_MD)},
    )

    with pytest.raises(RuntimeError, match="already has reviewing version 0.0.3"):
        NacosSkillHub(client=client).push_skills(str(skills_dir))

    assert client.download_calls == [{"name": "demo-skill", "version": "0.0.3", "label": "latest", "admin": True}]
    assert client.uploads == []
    assert client.submits == []


def test_push_overwrites_existing_editing_version(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, UPDATED_SKILL_MD)
    client = FakeNacosClient(
        [{"name": "demo-skill", "editingVersion": "0.0.1", "labels": {}}],
        {("demo-skill", "0.0.1"): _zip(SKILL_MD)},
    )

    result = NacosSkillHub(client=client).push_skills(str(skills_dir))

    assert result["uploaded"] == 1
    assert result["submitted"] == 1
    assert client.uploads[0]["target_version"] == "0.0.1"
    assert client.uploads[0]["overwrite"] is True
    assert client.submits == [("demo-skill", "0.0.1")]


def test_evolve_upload_skips_existing_editing_version_with_different_content() -> None:
    client = FakeNacosClient(
        [{"name": "demo-skill", "editingVersion": "0.0.1", "labels": {}}],
        {("demo-skill", "0.0.1"): _zip(SKILL_MD)},
    )
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(skill_storage_backend="nacos")
    server._nacos_skill_client = client
    server._load_remote_skill_record = lambda name: client.get_skill(name)

    status = server._upload_skill(
        {"name": "demo-skill", "description": "Demo skill", "content": "Updated content."},
        "improve",
    )

    assert status == "skipped_existing_editing"
    assert client.uploads == []
    assert client.submits == []


def test_evolve_upload_skips_existing_reviewed_version_with_different_content() -> None:
    client = FakeNacosClient(
        [
            {
                "name": "demo-skill",
                "labels": {},
                "versions": [{"version": "0.0.3", "status": "reviewed"}],
            }
        ],
        {("demo-skill", "0.0.3"): _zip(SKILL_MD)},
    )
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(skill_storage_backend="nacos")
    server._nacos_skill_client = client
    server._load_remote_skill_record = lambda name: client.get_skill(name)

    status = server._upload_skill(
        {"name": "demo-skill", "description": "Demo skill", "content": "Updated content."},
        "improve",
    )

    assert status == "skipped_existing_reviewing"
    assert client.download_calls == [{"name": "demo-skill", "version": "0.0.3", "label": "latest", "admin": True}]
    assert client.uploads == []
    assert client.submits == []


def test_evolve_upload_creates_version_when_nacos_skill_has_no_versions() -> None:
    client = FakeNacosClient(
        [{"name": "demo-skill", "labels": {}, "versions": [], "editingVersion": None, "reviewingVersion": None}],
        {},
    )
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(skill_storage_backend="nacos")
    server._nacos_skill_client = client
    server._load_remote_skill_record = lambda name: client.get_skill(name)

    status = server._upload_skill(
        {"name": "demo-skill", "description": "Demo skill", "content": "New content."},
        "create_skill",
    )

    assert status == "uploaded"
    assert client.uploads[0]["target_version"] == "0.0.1"
    assert client.submits == [("demo-skill", "0.0.1")]


def test_evolve_fetch_returns_none_when_nacos_skill_has_no_published_label() -> None:
    client = FakeNacosClient(
        [{"name": "demo-skill", "labels": {}, "editingVersion": None, "reviewingVersion": None}],
        {},
    )
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(skill_storage_backend="nacos")
    server._nacos_skill_client = client
    server._load_remote_skill_record = lambda name: client.get_skill(name)

    content = server._fetch_skill("demo-skill")

    assert content is None
    assert client.download_calls == []


def test_evolve_fetch_downloads_nacos_published_label_version() -> None:
    client = FakeNacosClient(
        [{"name": "demo-skill", "labels": {"latest": "0.0.3"}, "editingVersion": None, "reviewingVersion": None}],
        {("demo-skill", "0.0.3"): _zip(SKILL_MD)},
    )
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(skill_storage_backend="nacos", nacos_label="latest")
    server._nacos_skill_client = client
    server._load_remote_skill_record = lambda name: client.get_skill(name)

    content = server._fetch_skill("demo-skill")

    assert content == SKILL_MD
    assert client.download_calls == [{"name": "demo-skill", "version": "0.0.3", "label": "latest", "admin": False}]


def test_evolve_fetch_downloads_reviewed_version_as_reviewing_working_version() -> None:
    client = FakeNacosClient(
        [
            {
                "name": "demo-skill",
                "labels": {},
                "editingVersion": None,
                "reviewingVersion": None,
                "versions": [
                    {"version": "0.0.3", "status": "reviewed"},
                    {"version": "0.0.2", "status": "published"},
                ],
            }
        ],
        {("demo-skill", "0.0.3"): _zip(SKILL_MD)},
    )
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(skill_storage_backend="nacos", nacos_label="latest")
    server._nacos_skill_client = client
    server._load_remote_skill_record = lambda name: client.get_skill(name)

    content = server._fetch_skill("demo-skill")

    assert content == SKILL_MD
    assert client.download_calls == [{"name": "demo-skill", "version": "0.0.3", "label": "latest", "admin": True}]


def test_evolve_fetch_downloads_largest_published_version_when_latest_label_missing() -> None:
    client = FakeNacosClient(
        [
            {
                "name": "demo-skill",
                "labels": {},
                "editingVersion": None,
                "reviewingVersion": None,
                "versions": [
                    {"version": "0.0.2", "status": "published"},
                    {"version": "0.0.4", "status": "published"},
                ],
            }
        ],
        {("demo-skill", "0.0.4"): _zip(SKILL_MD)},
    )
    server = EvolveServer.__new__(EvolveServer)
    server.config = EvolveServerConfig(skill_storage_backend="nacos", nacos_label="latest")
    server._nacos_skill_client = client
    server._load_remote_skill_record = lambda name: client.get_skill(name)

    content = server._fetch_skill("demo-skill")

    assert content == SKILL_MD
    assert client.download_calls == [{"name": "demo-skill", "version": "0.0.4", "label": "latest", "admin": False}]
