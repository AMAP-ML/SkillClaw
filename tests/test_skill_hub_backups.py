import os
import shutil
from pathlib import Path

import pytest

from skillclaw.skill_hub import SkillHub


@pytest.fixture
def mirror(tmp_path):
    hub = SkillHub(
        backend="local",
        endpoint="",
        bucket="",
        access_key_id="",
        secret_access_key="",
        local_root=str(tmp_path / "shared"),
        group_id="test",
        user_alias="tester",
    )
    source = tmp_path / "source" / "demo"
    source.mkdir(parents=True)
    text = "---\nname: demo\ndescription: Example\n---\nRemote content\n"
    (source / "SKILL.md").write_text(text)
    hub.push_skills(str(source.parent))
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "local.txt").write_text("preserve me")
    backups = tmp_path / ".skillclaw_backups"
    backups.mkdir()
    for i in range(5):
        (backups / f"skills_20000101T00000{i}Z").mkdir()
    (backups / "other_20000101T000000Z").mkdir()
    return hub, skills, backups, text


def _dangling_link(skills):
    link = skills / "_staging" / "broken"
    link.parent.mkdir()
    try:
        link.symlink_to("../../missing-target")
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    return link


def test_mirror_backup_preserves_dangling_symlink(mirror):
    hub, skills, backups, text = mirror
    link = _dangling_link(skills)
    result = hub.pull_skills(str(skills))
    assert result["downloaded"] == 1
    assert (skills / "demo" / "SKILL.md").read_text() == text
    saved_link = Path(result["backup_dir"]) / link.relative_to(skills)
    assert saved_link.is_symlink()
    assert os.readlink(saved_link) == "../../missing-target"
    assert len(list(backups.glob("skills_*"))) == 3
    assert (backups / "other_20000101T000000Z").is_dir()


def test_repeated_backup_failure_removes_partial_copies(mirror, monkeypatch):
    hub, skills, backups, _ = mirror

    def fail_copy(src, dst, **kwargs):
        Path(dst).mkdir()
        (Path(dst) / "partial").write_text("incomplete")
        raise OSError("simulated disk full")

    monkeypatch.setattr(shutil, "copytree", fail_copy)
    for _ in range(5):
        result = hub.pull_skills(str(skills))
        assert result["downloaded"] == 0
        assert result["backup_dir"] == ""
        assert (skills / "local.txt").read_text() == "preserve me"
        assert not (skills / "demo").exists()
        assert len(list(backups.glob("skills_*"))) == 3
        assert not list(backups.rglob("partial"))


@pytest.mark.parametrize("restore_fails", [False, True])
def test_mirror_failure_prunes_and_preserves_recovery_backup(mirror, monkeypatch, restore_fails):
    hub, skills, backups, _ = mirror
    link = _dangling_link(skills)

    def fail_download(*args, **kwargs):
        raise OSError("simulated upstream failure")

    monkeypatch.setattr(hub, "_download_skill_bundle", fail_download)
    copytree = shutil.copytree

    def copy_or_fail_restore(src, dst, *args, **kwargs):
        if restore_fails and Path(dst) == skills:
            raise OSError("simulated restore failure")
        return copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", copy_or_fail_restore)
    result = hub.pull_skills(str(skills))
    assert result["restored_from_backup"] is (not restore_fails)
    assert result["downloaded"] == 0
    saved = Path(result["backup_dir"])
    assert (saved / "local.txt").read_text() == "preserve me"
    assert (saved / link.relative_to(skills)).is_symlink()
    assert len(list(backups.glob("skills_*"))) == 3
    assert not list(skills.parent.glob(".skillclaw_pull_stage_*"))
    if not restore_fails:
        assert (skills / "local.txt").read_text() == "preserve me"
        assert link.is_symlink()
        assert os.readlink(link) == "../../missing-target"


def test_staging_creation_failure_still_prunes_backups(mirror, monkeypatch):
    hub, skills, backups, _ = mirror
    makedirs = os.makedirs
    original_inode = (skills / "local.txt").stat().st_ino

    def fail_staging(path, *args, **kwargs):
        if Path(path).name.startswith(".skillclaw_pull_stage_"):
            raise OSError("simulated staging directory failure")
        return makedirs(path, *args, **kwargs)

    monkeypatch.setattr(os, "makedirs", fail_staging)
    for _ in range(5):
        result = hub.pull_skills(str(skills))
        assert result["downloaded"] == 0
        assert result["restored_from_backup"] is False
        assert (skills / "local.txt").read_text() == "preserve me"
        assert (skills / "local.txt").stat().st_ino == original_inode
        assert len(list(backups.glob("skills_*"))) == 3
        assert Path(result["backup_dir"]).is_dir()
