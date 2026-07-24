"""Companion-file awareness in the injected skill catalog.

Agents reliably read a skill's SKILL.md but skip its companion files
(references/, scripts/, assets/, extra root docs) — the parts where setup
commands and load-bearing details live. The injected ``<available_skills>``
catalog should surface each skill's companion files as structured data so
the agent reads the *whole* skill, for any skill layout.
"""
from __future__ import annotations

from pathlib import Path

from skillclaw.skill_manager import SkillManager

SKILL_MD = """---
name: {name}
description: {desc}
---

# Body
"""


def _make_skill(
    root: Path,
    name: str,
    *,
    refs: list[str] | None = None,
    scripts: list[str] | None = None,
    root_docs: list[str] | None = None,
) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL_MD.format(name=name, desc=f"Demo {name}"), encoding="utf-8")
    for rel in refs or []:
        p = d / "references" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# ref", encoding="utf-8")
    for rel in scripts or []:
        p = d / "scripts" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/sh\n", encoding="utf-8")
    for rel in root_docs or []:
        (d / rel).write_text("# doc", encoding="utf-8")


def test_catalog_lists_companion_files(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(
        skills_dir,
        "demo-skill",
        refs=["lifecycle.md", "gates.md"],
        scripts=["run.sh"],
        root_docs=["setup.md"],
    )
    mgr = SkillManager(skills_dir=str(skills_dir))
    prompt = mgr.format_skills_for_prompt(mgr.get_all_skills())
    assert "<companion_files>" in prompt
    assert "references/lifecycle.md" in prompt
    assert "references/gates.md" in prompt
    assert "scripts/run.sh" in prompt
    assert "setup.md" in prompt


def test_skill_without_companions_has_no_element(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "bare-skill")
    mgr = SkillManager(skills_dir=str(skills_dir))
    prompt = mgr.format_skills_for_prompt(mgr.get_all_skills())
    assert "<companion_files>" not in prompt


def test_companion_files_capped_and_sorted(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "fat-skill", refs=[f"r{i:02d}.md" for i in range(20)])
    mgr = SkillManager(skills_dir=str(skills_dir))
    skill = mgr.get_all_skills()[0]
    companions = skill.get("companion_files", [])
    assert 0 < len(companions) <= 12
    assert companions == sorted(companions)


def test_compact_format_omits_companions(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "demo-skill", refs=["lifecycle.md"])
    mgr = SkillManager(skills_dir=str(skills_dir))
    compact = mgr.format_skills_compact(mgr.get_all_skills())
    assert "<companion_files>" not in compact


def test_toggle_off_disables_detection(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "demo-skill", refs=["lifecycle.md"], root_docs=["setup.md"])
    mgr = SkillManager(skills_dir=str(skills_dir), include_companion_files=False)
    skill = mgr.get_all_skills()[0]
    assert "companion_files" not in skill
    prompt = mgr.format_skills_for_prompt(mgr.get_all_skills())
    assert "<companion_files>" not in prompt


def test_injection_instruction_mentions_companion_files(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "demo-skill", refs=["lifecycle.md"])
    mgr = SkillManager(skills_dir=str(skills_dir))
    prompt = mgr.build_injection_prompt()
    assert "companion_files" in prompt


def test_toggle_off_prompt_is_byte_identical_to_pre_feature(tmp_path: Path) -> None:
    """With the toggle off, neither the catalog nor the instruction may
    mention companion files — output must match pre-feature behavior."""
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "demo-skill", refs=["lifecycle.md"], root_docs=["setup.md"])
    mgr = SkillManager(skills_dir=str(skills_dir), include_companion_files=False)
    prompt = mgr.build_injection_prompt()
    assert "companion_files" not in prompt


def test_compact_fallback_drops_instruction_bullet(tmp_path: Path) -> None:
    """Oversized catalog -> compact format, which lists no companions; the
    instruction bullet must not tell the agent to read files never listed."""
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "demo-skill", refs=["lifecycle.md"])
    mgr = SkillManager(skills_dir=str(skills_dir))
    prompt = mgr.build_injection_prompt(max_chars=10)
    assert "companion_files" not in prompt


def test_companion_only_change_triggers_refresh(tmp_path: Path) -> None:
    """Adding a references/ file without touching SKILL.md must invalidate
    the cache so refresh_if_changed reloads the catalog."""
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "demo-skill", refs=["lifecycle.md"])
    mgr = SkillManager(skills_dir=str(skills_dir))
    assert mgr.refresh_if_changed() is False
    new_ref = skills_dir / "demo-skill" / "references" / "new.md"
    new_ref.write_text("# new", encoding="utf-8")
    assert mgr.refresh_if_changed() is True
    skill = mgr.get_all_skills()[0]
    assert "references/new.md" in skill.get("companion_files", [])


def test_ignored_noise_excluded(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "demo-skill", scripts=["run.sh"])
    cache = skills_dir / "demo-skill" / "scripts" / "__pycache__"
    cache.mkdir()
    (cache / "run.cpython-311.pyc").write_bytes(b"\x00")
    mgr = SkillManager(skills_dir=str(skills_dir))
    skill = mgr.get_all_skills()[0]
    companions = skill.get("companion_files", [])
    assert companions == ["scripts/run.sh"]


def test_config_mapping_round_trip(tmp_path: Path) -> None:
    """skills.companion_files in config.yaml reaches SkillClawConfig."""
    import yaml

    from skillclaw.config_store import ConfigStore

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"skills": {"enabled": True, "companion_files": False}}))
    assert ConfigStore(cfg).to_skillclaw_config().skills_include_companion_files is False
    cfg.write_text(yaml.safe_dump({"skills": {"enabled": True}}))
    assert ConfigStore(cfg).to_skillclaw_config().skills_include_companion_files is True


def test_paths_are_xml_escaped(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    d = skills_dir / "demo-skill"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL_MD.format(name="demo-skill", desc="Demo"), encoding="utf-8")
    (d / "references" / "a&b.md").write_text("# ref", encoding="utf-8")
    mgr = SkillManager(skills_dir=str(skills_dir))
    prompt = mgr.format_skills_for_prompt(mgr.get_all_skills())
    assert "a&amp;b.md" in prompt
    assert "a&b.md" not in prompt
