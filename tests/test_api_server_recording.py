from __future__ import annotations

from pathlib import Path

from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig


def test_api_server_does_not_create_record_dir_when_recording_disabled(tmp_path: Path) -> None:
    record_dir = tmp_path / "records"
    cfg = SkillClawConfig(record_enabled=False, record_dir=str(record_dir))

    SkillClawAPIServer(cfg)

    assert not record_dir.exists()
