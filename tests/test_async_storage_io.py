from __future__ import annotations

import asyncio

import pytest

from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig
from skillclaw.skill_hub import SkillHub


@pytest.mark.anyio
async def test_session_upload_runs_object_store_write_in_worker_thread(monkeypatch, tmp_path) -> None:
    calls = []

    class Bucket:
        def put_object(self, key, data):
            calls.append(("put", key, data))

    class Hub:
        _bucket = Bucket()

        @staticmethod
        def _prefix():
            return "group/"

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(("to_thread", func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(SkillHub, "object_storage_from_config", classmethod(lambda cls, config: Hub()))
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(sharing_enabled=True, sharing_group_id="group")

    assert await server._upload_session_data("session-1", [{"turn_num": 1}]) is True
    assert calls[0][0] == "to_thread"
    assert calls[1][0] == "put"


@pytest.mark.anyio
async def test_skill_pull_runs_storage_sync_in_worker_thread(monkeypatch, tmp_path) -> None:
    calls = []

    class Hub:
        def pull_skills(self, skills_dir, *, skip_names=None):
            calls.append(("pull", skills_dir, skip_names))
            return {
                "downloaded": 0,
                "skipped": 1,
                "failed": 0,
                "deleted": 0,
                "total_remote": 1,
                "restored_from_backup": False,
            }

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(("to_thread", func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(SkillHub, "from_config", classmethod(lambda cls, config: Hub()))
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(sharing_enabled=True, skills_dir=str(tmp_path / "skills"))
    server.skill_manager = None

    await server._pull_skills_from_cloud(skip_names={"local-only"})
    assert calls[0][0] == "to_thread"
    assert calls[1] == ("pull", str(tmp_path / "skills"), {"local-only"})
