from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillclaw.object_store import LocalObjectStore
from skillclaw.skill_bundle import SkillBundleError, normalize_bundle_rel_path, write_skill_bundle
from skillclaw.skill_hub import SkillHub, _skill_dir_for_root


@pytest.mark.parametrize(
    "unsafe_path",
    ["/tmp/escaped", "C:/tmp/escaped", "C:\\tmp\\escaped", "//server/share/escaped"],
)
def test_bundle_paths_must_be_relative(unsafe_path: str) -> None:
    with pytest.raises(SkillBundleError):
        normalize_bundle_rel_path(unsafe_path)


@pytest.mark.parametrize("unsafe_name", ["../outside", "nested/skill", "nested\\skill", "C:/outside"])
def test_remote_skill_names_cannot_select_another_directory(tmp_path: Path, unsafe_name: str) -> None:
    with pytest.raises(SkillBundleError):
        _skill_dir_for_root(str(tmp_path / "skills"), unsafe_name)


def test_remote_skill_categories_must_be_single_segments(tmp_path: Path) -> None:
    with pytest.raises(SkillBundleError):
        _skill_dir_for_root(str(tmp_path / "skills"), "safe-skill", "../outside")


def test_bundle_write_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "skill"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SkillBundleError):
        write_skill_bundle(
            root,
            {
                "SKILL.md": b"---\nname: safe\n---\n",
                "linked/escaped.txt": b"escaped",
            },
        )

    assert not (outside / "escaped.txt").exists()


def test_pull_rejects_manifest_name_that_escapes_skills_root(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    skills_dir = tmp_path / "victim" / "skills"
    store = LocalObjectStore(str(store_root))
    store.put_object(
        "default/manifest.jsonl",
        (json.dumps({"name": "../outside/pwned", "category": "general"}) + "\n").encode(),
    )
    hub = SkillHub(
        backend="local",
        endpoint="",
        bucket="",
        access_key_id="",
        secret_access_key="",
        local_root=str(store_root),
        group_id="default",
    )

    with pytest.raises(SkillBundleError):
        hub.pull_skills(str(skills_dir), mirror=False)

    assert not (tmp_path / "victim" / "outside" / "pwned" / "SKILL.md").exists()
