from __future__ import annotations

import json
from pathlib import Path

from evolve_server.engines.agent_workspace import AgentWorkspace
from skillclaw.skill_hub import SkillHub

SKILL_MD = """---
name: demo-skill
description: Demo bundle skill
category: general
---

# Demo Skill

Use the bundled resources.
"""


def _skill_md(name: str) -> str:
    return f"""---
name: {name}
description: Demo bundle skill
category: general
---

# Demo Skill

Use the bundled resources.
"""


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_skill_hub_push_pull_roundtrips_bundle(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_bytes(skills_dir / "demo-skill" / "SKILL.md", SKILL_MD.encode("utf-8"))
    _write_bytes(skills_dir / "demo-skill" / "references" / "guide.md", b"hello bundle\n")
    _write_bytes(skills_dir / "demo-skill" / "scripts" / "tool.py", b"print('ok')\n")
    _write_bytes(skills_dir / "demo-skill" / "assets" / "icon.bin", b"\x00\x01\x02\x03")

    bucket_root = tmp_path / "bucket"
    hub = SkillHub(
        backend="local",
        endpoint="",
        bucket="",
        access_key_id="",
        secret_access_key="",
        local_root=str(bucket_root),
        group_id="team-a",
        user_alias="tester",
    )

    push_result = hub.push_skills(str(skills_dir))

    assert push_result["uploaded"] == 1
    manifest = hub._load_remote_manifest()
    rec = manifest["demo-skill"]
    assert rec["format"] == "bundle_v1"
    assert rec["entrypoint"] == "SKILL.md"
    assert rec["tree_sha256"]
    assert {item["path"] for item in rec["files"]} == {
        "SKILL.md",
        "references/guide.md",
        "scripts/tool.py",
        "assets/icon.bin",
    }

    restored_dir = tmp_path / "restored-skills"
    pull_result = hub.pull_skills(str(restored_dir))

    assert pull_result["downloaded"] == 1
    assert (restored_dir / "demo-skill" / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD
    assert (restored_dir / "demo-skill" / "references" / "guide.md").read_bytes() == b"hello bundle\n"
    assert (restored_dir / "demo-skill" / "scripts" / "tool.py").read_bytes() == b"print('ok')\n"
    assert (restored_dir / "demo-skill" / "assets" / "icon.bin").read_bytes() == b"\x00\x01\x02\x03"


def test_skill_hub_push_pull_roundtrips_single_file_skill(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    solo_md = _skill_md("solo-skill")
    _write_bytes(skills_dir / "solo-skill" / "SKILL.md", solo_md.encode("utf-8"))

    bucket_root = tmp_path / "bucket"
    hub = SkillHub(
        backend="local",
        endpoint="",
        bucket="",
        access_key_id="",
        secret_access_key="",
        local_root=str(bucket_root),
        group_id="team-a",
        user_alias="tester",
    )

    push_result = hub.push_skills(str(skills_dir))

    assert push_result["uploaded"] == 1
    manifest = hub._load_remote_manifest()
    rec = manifest["solo-skill"]
    assert rec["format"] == "bundle_v1"
    assert rec["entrypoint"] == "SKILL.md"
    assert rec["files"] == [{
        "path": "SKILL.md",
        "sha256": rec["sha256"],
        "size": len(solo_md.encode("utf-8")),
    }]

    restored_dir = tmp_path / "restored-skills"
    pull_result = hub.pull_skills(str(restored_dir))

    assert pull_result["downloaded"] == 1
    assert (restored_dir / "solo-skill" / "SKILL.md").read_text(encoding="utf-8") == solo_md
    restored_files = sorted(
        p.relative_to(restored_dir / "solo-skill").as_posix()
        for p in (restored_dir / "solo-skill").rglob("*")
        if p.is_file()
    )
    assert restored_files == ["SKILL.md"]


def test_skill_hub_persists_bundle_version_snapshots(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "demo-skill"
    _write_bytes(skill_dir / "SKILL.md", SKILL_MD.encode("utf-8"))
    _write_bytes(skill_dir / "references" / "guide.md", b"v1 guide\n")

    bucket_root = tmp_path / "bucket"
    hub = SkillHub(
        backend="local",
        endpoint="",
        bucket="",
        access_key_id="",
        secret_access_key="",
        local_root=str(bucket_root),
        group_id="team-a",
        user_alias="tester",
    )

    first_push = hub.push_skills(str(skills_dir))
    assert first_push["uploaded"] == 1

    _write_bytes(skill_dir / "SKILL.md", _skill_md("demo-skill").encode("utf-8"))
    _write_bytes(skill_dir / "references" / "guide.md", b"v2 guide\n")
    second_push = hub.push_skills(str(skills_dir))
    assert second_push["uploaded"] == 1

    registry_path = bucket_root / "team-a" / "evolve_skill_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry["demo-skill"]
    assert entry["version"] == 2
    assert [item["version"] for item in entry["history"]] == [1, 2]
    assert all(item.get("tree_sha256") for item in entry["history"])
    assert all(
        any(file_item["path"] == "references/guide.md" for file_item in item.get("files", []))
        for item in entry["history"]
    )

    v1_root = bucket_root / "team-a" / "skills" / "demo-skill" / "versions" / "v1"
    v2_root = bucket_root / "team-a" / "skills" / "demo-skill" / "versions" / "v2"
    assert (v1_root / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD
    assert (v1_root / "files" / "references" / "guide.md").read_bytes() == b"v1 guide\n"
    assert (v2_root / "SKILL.md").read_text(encoding="utf-8") == _skill_md("demo-skill")
    assert (v2_root / "files" / "references" / "guide.md").read_bytes() == b"v2 guide\n"


def test_skill_hub_roundtrips_extra_unstructured_files_and_attributes_them(tmp_path: Path) -> None:
    import json

    from skillclaw.api_server import _extract_read_skills_from_tool_calls
    from skillclaw.skill_manager import SkillManager

    skills_dir = tmp_path / "skills"
    extra_md = _skill_md("extra-skill")
    _write_bytes(skills_dir / "extra-skill" / "SKILL.md", extra_md.encode("utf-8"))
    _write_bytes(skills_dir / "extra-skill" / "notes" / "checklist.txt", b"bundle extras stay intact\n")
    _write_bytes(skills_dir / "extra-skill" / "workspace" / "payload.bin", b"\x10\x20\x30\x40")

    bucket_root = tmp_path / "bucket"
    hub = SkillHub(
        backend="local",
        endpoint="",
        bucket="",
        access_key_id="",
        secret_access_key="",
        local_root=str(bucket_root),
        group_id="team-a",
        user_alias="tester",
    )

    push_result = hub.push_skills(str(skills_dir))
    assert push_result["uploaded"] == 1

    manifest = hub._load_remote_manifest()
    rec = manifest["extra-skill"]
    assert {item["path"] for item in rec["files"]} == {
        "SKILL.md",
        "notes/checklist.txt",
        "workspace/payload.bin",
    }

    restored_dir = tmp_path / "restored-skills"
    pull_result = hub.pull_skills(str(restored_dir))
    assert pull_result["downloaded"] == 1
    assert (restored_dir / "extra-skill" / "notes" / "checklist.txt").read_bytes() == b"bundle extras stay intact\n"
    assert (restored_dir / "extra-skill" / "workspace" / "payload.bin").read_bytes() == b"\x10\x20\x30\x40"

    manager = SkillManager(str(restored_dir))
    skill_path_map = manager.get_skill_path_map()
    extra_path = str((restored_dir / "extra-skill" / "notes" / "checklist.txt").resolve())

    read_calls = [
        {
            "id": "call_skill_view_extra",
            "function": {
                "name": "skill_view",
                "arguments": json.dumps({"name": "extra-skill", "file_path": "notes/checklist.txt"}),
            },
        }
    ]

    read_skills = _extract_read_skills_from_tool_calls(read_calls, skill_path_map)

    assert read_skills == [{
        "skill_id": manager.get_all_skills()[0]["id"],
        "skill_name": "extra-skill",
        "path": extra_path,
    }]


def test_agent_workspace_detects_nested_bundle_changes(tmp_path: Path) -> None:
    workspace = AgentWorkspace(tmp_path / "workspace")
    workspace.prepare(
        sessions=[],
        existing_skills={
            "demo-skill": {
                "SKILL.md": SKILL_MD.encode("utf-8"),
                "references/guide.md": b"first\n",
            }
        },
        manifest={"demo-skill": {"name": "demo-skill"}},
        agents_md="# evolve",
        skill_registry_info=None,
    )

    before = workspace.snapshot_skills()
    (workspace.skills_dir / "demo-skill" / "references" / "guide.md").write_text(
        "second\n", encoding="utf-8"
    )

    changes = workspace.collect_changes(before)

    assert len(changes) == 1
    change = changes[0]
    assert change["name"] == "demo-skill"
    assert change["action"] == "improve"
    assert change["raw_md"] == SKILL_MD
    assert change["bundle_files"]["SKILL.md"] == SKILL_MD.encode("utf-8")
    assert change["bundle_files"]["references/guide.md"] == b"second\n"
    assert change["tree_sha256"]


def test_skill_path_map_and_tool_attribution_include_bundle_files(tmp_path: Path) -> None:
    import json

    from skillclaw.api_server import (
        _extract_modified_skills_from_tool_calls,
        _extract_read_skills_from_tool_calls,
    )
    from skillclaw.skill_manager import SkillManager

    skills_dir = tmp_path / "skills"
    _write_bytes(skills_dir / "demo-skill" / "SKILL.md", SKILL_MD.encode("utf-8"))
    _write_bytes(skills_dir / "demo-skill" / "references" / "guide.md", b"look here\n")
    _write_bytes(skills_dir / "demo-skill" / "scripts" / "tool.py", b"print('ok')\n")

    manager = SkillManager(str(skills_dir))
    skill_path_map = manager.get_skill_path_map()

    reference_path = str((skills_dir / "demo-skill" / "references" / "guide.md").resolve())
    script_path = str((skills_dir / "demo-skill" / "scripts" / "tool.py").resolve())
    assert reference_path in skill_path_map
    assert script_path in skill_path_map

    read_calls = [
        {
            "id": "call_read_1",
            "function": {"name": "read", "arguments": json.dumps({"path": reference_path})},
        }
    ]
    write_calls = [
        {
            "id": "call_edit_1",
            "function": {"name": "edit_file", "arguments": json.dumps({"path": script_path})},
        }
    ]

    read_skills = _extract_read_skills_from_tool_calls(read_calls, skill_path_map)
    modified_skills = _extract_modified_skills_from_tool_calls(write_calls, skill_path_map)

    assert read_skills == [{
        "skill_id": manager.get_all_skills()[0]["id"],
        "skill_name": "demo-skill",
        "path": reference_path,
    }]
    assert modified_skills == [{
        "skill_id": manager.get_all_skills()[0]["id"],
        "skill_name": "demo-skill",
        "path": script_path,
        "action": "edit_file",
    }]


def test_hermes_skill_tool_attribution_uses_bundle_child_paths(tmp_path: Path) -> None:
    import json

    from skillclaw.api_server import (
        _extract_modified_skills_from_tool_calls,
        _extract_read_skills_from_tool_calls,
    )
    from skillclaw.skill_manager import SkillManager

    skills_dir = tmp_path / "skills"
    _write_bytes(skills_dir / "demo-skill" / "SKILL.md", SKILL_MD.encode("utf-8"))
    _write_bytes(skills_dir / "demo-skill" / "references" / "guide.md", b"look here\n")
    _write_bytes(skills_dir / "demo-skill" / "scripts" / "tool.py", b"print('ok')\n")

    manager = SkillManager(str(skills_dir))
    skill_path_map = manager.get_skill_path_map()
    skill_id = manager.get_all_skills()[0]["id"]

    reference_path = str((skills_dir / "demo-skill" / "references" / "guide.md").resolve())
    script_path = str((skills_dir / "demo-skill" / "scripts" / "tool.py").resolve())

    read_calls = [
        {
            "id": "call_skill_view_1",
            "function": {
                "name": "skill_view",
                "arguments": json.dumps({"name": "demo-skill", "file_path": "references/guide.md"}),
            },
        }
    ]
    write_calls = [
        {
            "id": "call_skill_manage_1",
            "function": {
                "name": "skill_manage",
                "arguments": json.dumps(
                    {
                        "action": "write_file",
                        "name": "demo-skill",
                        "file_path": "scripts/tool.py",
                    }
                ),
            },
        }
    ]

    read_skills = _extract_read_skills_from_tool_calls(read_calls, skill_path_map)
    modified_skills = _extract_modified_skills_from_tool_calls(write_calls, skill_path_map)

    assert read_skills == [{
        "skill_id": skill_id,
        "skill_name": "demo-skill",
        "path": reference_path,
    }]
    assert modified_skills == [{
        "skill_id": skill_id,
        "skill_name": "demo-skill",
        "path": script_path,
        "action": "skill_manage",
    }]
