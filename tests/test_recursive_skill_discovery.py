"""
Regression tests for recursive SKILL.md discovery.

Covers:
  - ~/.hermes/skills/category/skill/SKILL.md
  - ~/.hermes/profiles/lightning-node/skills/category/skill/SKILL.md
  - depth-3 skills
  - flat skills
  - unrelated directories excluded
  - symlink duplicates not double-counted
  - real-world: all 91 production skills detected
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from skillclaw.skill_manager import SkillManager


def _write_skill(skill_dir: str, name: str, description: str, category: str = "general") -> str:
    """Write a minimal SKILL.md and return its path."""
    skill_path = os.path.join(skill_dir, name)
    os.makedirs(skill_path, exist_ok=True)
    md_path = os.path.join(skill_path, "SKILL.md")
    with open(md_path, "w") as f:
        f.write(f"---\nname: {name}\ndescription: \"{description}\"\n")
        if category != "general":
            f.write(f"category: {category}\n")
        f.write("---\n\n# {name}\n")
    return md_path


class TestRecursiveSkillDiscovery:
    """Prove that _skill_md_paths finds all skills regardless of directory layout."""

    def test_hermes_default_skills_root(self):
        """~/.hermes/skills/category/skill/SKILL.md is found."""
        with tempfile.TemporaryDirectory() as td:
            hermes_skills = os.path.join(td, ".hermes", "skills")
            _write_skill(hermes_skills, "my-skill", "A test skill", category="coding")
            sm = SkillManager(hermes_skills)
            names = {s["name"] for s in sm.skills["all_skills"]}
            assert "my-skill" in names

    def test_hermes_profile_skills_root(self):
        """~/.hermes/profiles/lightning-node/skills/category/skill/SKILL.md is found."""
        with tempfile.TemporaryDirectory() as td:
            profile_skills = os.path.join(td, ".hermes", "profiles", "lightning-node", "skills")
            _write_skill(profile_skills, "lightning-ops", "Lightning node operations", category="infrastructure")
            sm = SkillManager(profile_skills)
            names = {s["name"] for s in sm.skills["all_skills"]}
            assert "lightning-ops" in names
            # Verify _is_hermes_skill_root recognises profile dirs
            assert sm._is_hermes_skill_root()

    def test_depth_3_skills_found(self):
        """depth-3 skills (category/subcategory/skill/SKILL.md) are found."""
        with tempfile.TemporaryDirectory() as td:
            skills_dir = os.path.join(td, "skills")
            deep_path = os.path.join(skills_dir, "mlops", "evaluation", "my-eval-skill")
            os.makedirs(deep_path, exist_ok=True)
            with open(os.path.join(deep_path, "SKILL.md"), "w") as f:
                f.write("---\nname: my-eval-skill\ndescription: \"Eval skill at depth 3\"\n---\n\n# Eval Skill\n")
            sm = SkillManager(skills_dir)
            names = {s["name"] for s in sm.skills["all_skills"]}
            assert "my-eval-skill" in names

    def test_flat_skills_remain_discoverable(self):
        """Flat skills (skills_dir/skill/SKILL.md) are still found."""
        with tempfile.TemporaryDirectory() as td:
            skills_dir = os.path.join(td, "skills")
            _write_skill(skills_dir, "flat-skill", "A flat skill")
            sm = SkillManager(skills_dir)
            names = {s["name"] for s in sm.skills["all_skills"]}
            assert "flat-skill" in names

    def test_unrelated_dirs_excluded(self):
        """Unrelated directories outside the configured skill root are excluded."""
        with tempfile.TemporaryDirectory() as td:
            skills_dir = os.path.join(td, "skills")
            unrelated = os.path.join(td, "not-skills", "intruder")
            os.makedirs(unrelated, exist_ok=True)
            with open(os.path.join(unrelated, "SKILL.md"), "w") as f:
                f.write("---\nname: intruder\ndescription: \"Should not be found\"\n---\n\n# Intruder\n")
            _write_skill(skills_dir, "real-skill", "A real skill")
            sm = SkillManager(skills_dir)
            names = {s["name"] for s in sm.skills["all_skills"]}
            assert "real-skill" in names
            assert "intruder" not in names

    def test_symlink_duplicates_not_double_counted(self):
        """Symlink duplicates are not returned twice (realpath dedup)."""
        with tempfile.TemporaryDirectory() as td:
            skills_dir = os.path.join(td, "skills")
            os.makedirs(skills_dir, exist_ok=True)
            real_skill_dir = os.path.join(td, "real-skills", "linked-skill")
            os.makedirs(real_skill_dir, exist_ok=True)
            with open(os.path.join(real_skill_dir, "SKILL.md"), "w") as f:
                f.write("---\nname: linked-skill\ndescription: \"Linked skill\"\n---\n\n# Linked\n")
            # Create symlink inside skills_dir
            os.symlink(real_skill_dir, os.path.join(skills_dir, "linked-skill"))
            sm = SkillManager(skills_dir)
            linked = [s for s in sm.skills["all_skills"] if s["name"] == "linked-skill"]
            assert len(linked) == 1, f"Expected 1, got {len(linked)}"

    def test_real_world_91_skills_detected(self):
        """All 91 current skills are detected in the user's production profile."""
        skills_dir = os.path.expanduser("~/.hermes/profiles/lightning-node/skills")
        if not os.path.isdir(skills_dir):
            pytest.skip("Production skills directory not available")
        sm = SkillManager(skills_dir)
        count = len(sm.skills["all_skills"])
        assert count >= 91, f"Expected at least 91 skills, found {count}"
        # Verify key Lightning skills are present
        names = {s["name"] for s in sm.skills["all_skills"]}
        expected = {
            "lightning-node-ops",
            "core-lightning-ops",
            "lndg-remote-db-ops",
            "loopd-cost-reconciliation",
            "intervention-ledger-reconciliation",
            "hermes-agent",
            "obsidian",
            "github-workflow",
        }
        missing = expected - names
        assert not missing, f"Missing critical skills: {missing}"

    def test_is_hermes_skill_root_profile(self):
        """_is_hermes_skill_root returns True for profile skills directories."""
        with tempfile.TemporaryDirectory() as td:
            profile_skills = os.path.join(td, ".hermes", "profiles", "my-profile", "skills")
            os.makedirs(profile_skills, exist_ok=True)
            sm = SkillManager(profile_skills)
            assert sm._is_hermes_skill_root()

    def test_is_hermes_skill_root_not_hermes(self):
        """_is_hermes_skill_root returns False for non-Hermes directories."""
        with tempfile.TemporaryDirectory() as td:
            skills_dir = os.path.join(td, "random-skills")
            os.makedirs(skills_dir, exist_ok=True)
            sm = SkillManager(skills_dir)
            assert not sm._is_hermes_skill_root()
