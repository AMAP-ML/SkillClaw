from unittest.mock import AsyncMock, Mock

import pytest

from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig
from skillclaw.config_store import ConfigStore
from skillclaw.skill_hub import SkillHub


@pytest.mark.anyio
@pytest.mark.parametrize("enabled,upload", [(True, False), (False, True)])
async def test_disabled_upload_never_opens_storage_or_triggers_evolution(monkeypatch, enabled, upload):
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(sharing_enabled=enabled, sharing_upload_sessions=upload)
    storage = Mock(side_effect=AssertionError("must not open session storage"))
    monkeypatch.setattr(SkillHub, "object_storage_from_config", storage)
    server._trigger_evolve = AsyncMock()
    assert await server._upload_session_data("private", [{"prompt_text": "private"}]) is False
    await server._upload_session_snapshot_and_trigger("private", [{"prompt_text": "private"}])
    storage.assert_not_called()
    server._trigger_evolve.assert_not_awaited()


def test_disabled_upload_never_schedules_snapshot():
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(
        sharing_enabled=True, sharing_upload_sessions=False, sharing_session_upload_interval=1
    )
    server._safe_create_task = Mock()
    server._maybe_upload_session_snapshot("private", 1)
    server._safe_create_task.assert_not_called()


@pytest.mark.anyio
async def test_skill_pull_still_works_with_session_upload_disabled(monkeypatch, tmp_path):
    server = object.__new__(SkillClawAPIServer)
    server.config = SkillClawConfig(sharing_enabled=True, sharing_upload_sessions=False, skills_dir=str(tmp_path))
    server.skill_manager = None
    hub = Mock()
    hub.pull_skills.return_value = {"downloaded": 0, "skipped": 0, "failed": 0, "deleted": 0, "total_remote": 0}
    monkeypatch.setattr(SkillHub, "from_config", lambda config: hub)
    await server._pull_skills_from_cloud()
    hub.pull_skills.assert_called_once()


@pytest.mark.parametrize("setting,expected", [("", True), ("  upload_sessions: false\n", False)])
def test_config_preserves_legacy_behavior_and_explicit_opt_out(tmp_path, setting, expected):
    path = tmp_path / "config.yaml"
    path.write_text("sharing:\n  enabled: true\n" + setting)
    assert ConfigStore(path).to_skillclaw_config().sharing_upload_sessions is expected
