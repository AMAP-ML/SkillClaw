import os
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner

from skillclaw import cli, runtime_state
from skillclaw.config_store import ConfigStore
from skillclaw.launcher import SkillClawLauncher


def test_instance_defaults_and_config_override(monkeypatch, tmp_path):
    root = tmp_path / "instance"
    monkeypatch.setenv("SKILLCLAW_HOME", str(root))
    monkeypatch.delenv("SKILLCLAW_CONFIG_FILE", raising=False)
    store = ConfigStore()
    assert store.config_file == root / "config.yaml"
    cfg = store.to_skillclaw_config()
    assert cfg.claw_type == "none"
    assert cfg.skills_dir == str(root / "skills")
    assert cfg.record_dir == str(root / "records")
    assert cfg.dashboard_db_path == str(root / "dashboard.db")
    assert cli._default_daemon_log_path() == root / "skillclaw.log"
    other = tmp_path / "explicit.yaml"
    monkeypatch.setenv("SKILLCLAW_CONFIG_FILE", str(other))
    assert ConfigStore().config_file == other
    assert ConfigStore(tmp_path / "arg.yaml").config_file == tmp_path / "arg.yaml"
    assert runtime_state.pid_file_path() == root / "skillclaw.pid"


def test_explicit_data_paths_remain_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLCLAW_HOME", str(tmp_path / "instance"))
    store = ConfigStore(tmp_path / "config.yaml")
    store.save(
        {
            "skills": {"dir": "/explicit/skills"},
            "record": {"dir": "/explicit/records"},
            "dashboard": {"db_path": "/explicit/dashboard.db"},
        }
    )
    cfg = store.to_skillclaw_config()
    assert cfg.skills_dir == "/explicit/skills"
    assert cfg.record_dir == "/explicit/records"
    assert cfg.dashboard_db_path == "/explicit/dashboard.db"


def test_two_instances_have_independent_locks_and_pid_cleanup(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("SKILLCLAW_HOME", str(a))
    launcher_a = SkillClawLauncher(ConfigStore())
    with runtime_state.daemon_start_lock():
        launcher_a._write_pid()
        assert runtime_state.read_pid() == os.getpid()
        monkeypatch.setenv("SKILLCLAW_HOME", str(b))
        launcher_b = SkillClawLauncher(ConfigStore())
        with runtime_state.daemon_start_lock():
            launcher_b._write_pid()
            assert (a / "daemon_start.lock").exists()
            assert (b / "daemon_start.lock").exists()
        launcher_a.stop()
        assert not (a / "skillclaw.pid").exists()
        assert (b / "skillclaw.pid").exists()
        launcher_b.stop()
    assert not list(tmp_path.rglob("*.lock"))


def test_stop_only_signals_selected_instance_and_config_show_reports_real_path(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, pid in [(a, 12345), (b, 23456)]:
        root.mkdir()
        (root / "skillclaw.pid").write_text(str(pid))
    monkeypatch.setenv("SKILLCLAW_HOME", str(a))
    monkeypatch.delenv("SKILLCLAW_CONFIG_FILE", raising=False)
    kill = Mock()
    monkeypatch.setattr(os, "kill", kill)
    runner = CliRunner()
    assert runner.invoke(cli.skillclaw, ["stop"]).exit_code == 0
    assert kill.call_args.args[0] == 12345
    assert (b / "skillclaw.pid").read_text() == "23456"
    store = ConfigStore()
    store.save({"proxy": {"port": 31001}})
    result = runner.invoke(cli.skillclaw, ["config", "show"])
    assert result.exit_code == 0 and str(a / "config.yaml") in result.output


def test_no_instance_home_preserves_default_state_root(monkeypatch):
    monkeypatch.delenv("SKILLCLAW_HOME", raising=False)
    assert runtime_state.pid_file_path() == Path.home() / ".skillclaw" / "skillclaw.pid"
