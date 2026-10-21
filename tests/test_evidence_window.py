import json
from datetime import datetime, timedelta, timezone

import pytest

from evolve_server.core.config import EvolveServerConfig
from evolve_server.engines.common import EvolveEngineMixin, _fresh_session_evidence
from evolve_server.storage.mock_bucket import LocalBucket
from skillclaw.config import SkillClawConfig


@pytest.mark.parametrize("seconds,expected", [(0, True), (60, True), (61, False), (-1, False)])
def test_age_window_boundary(seconds, expected):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stamp = (now - timedelta(seconds=seconds)).isoformat()
    assert _fresh_session_evidence({"timestamp": stamp}, now, 60) is expected


@pytest.mark.parametrize("stamp", [None, "", "invalid", "2026-01-01T00:00:00", 123])
def test_undated_or_ambiguous_evidence_is_excluded(stamp):
    assert not _fresh_session_evidence({"timestamp": stamp}, datetime.now(timezone.utc), 60)


def test_timezone_offsets_are_normalized():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _fresh_session_evidence({"timestamp": "2026-01-01T08:00:00+08:00"}, now, 60)


@pytest.mark.asyncio
@pytest.mark.parametrize("window,expected", [(0, 3), (60, 1)])
async def test_drain_excludes_old_evidence_without_consuming_or_deleting_it(tmp_path, window, expected):
    engine = EvolveEngineMixin()
    engine.config = EvolveServerConfig(storage_backend="local", evidence_max_age_seconds=window)
    engine._mock = True
    engine._bucket = LocalBucket(root=str(tmp_path))
    engine._prefix = "group/"
    timestamps = [datetime.now(timezone.utc).isoformat(), "2000-01-01T00:00:00Z", None]
    for i, stamp in enumerate(timestamps):
        engine._bucket.put_object(f"group/sessions/{i}.json", json.dumps({"timestamp": stamp, "turns": []}).encode())
    sessions, keys = await engine._drain_sessions()
    assert len(sessions) == len(keys) == expected
    assert len(list(tmp_path.rglob("*.json"))) == 3
    if window:
        assert keys == ["group/sessions/0.json"]


def test_both_config_entry_points_and_invalid_values(monkeypatch):
    monkeypatch.setenv("EVOLVE_EVIDENCE_MAX_AGE_SECONDS", "604800")
    assert EvolveServerConfig.from_env().evidence_max_age_seconds == 604800
    assert EvolveServerConfig.from_skillclaw_config(SkillClawConfig()).evidence_max_age_seconds == 604800
    with pytest.raises(ValueError, match="non-negative"):
        EvolveServerConfig(evidence_max_age_seconds=-1)
