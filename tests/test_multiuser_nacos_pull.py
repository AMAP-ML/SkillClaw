"""Tests for multi-user Nacos skill pull scenarios.

Simulates a second user on a different machine starting a fresh proxy and
pulling skills that were published to Nacos by the first user's session.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from skillclaw.nacos_skill_hub import (
    NacosSkillClient,
    NacosSkillHub,
    _bundle_to_nacos_zip,
)
from skillclaw.skill_manager import SkillManager

WEEKLY_REPORT_SKILL_MD = """\
---
name: corp-weekly-report
description: "Generate weekly report in corp format with sections, JIRA links, and feishu sync."
---

# Corp Weekly Report

Write weekly reports following the company format:
- Title: [Name] W{week} Weekly Report
- Sections: Completed / In Progress / Next Week / Risks
- Each item: [Code] Description | JIRA link | Completion %
"""


def _json(data):
    return httpx.Response(200, json={"code": 0, "data": data})


def _build_nacos_handler(skill_name: str, skill_md: str, version: str = "0.0.1"):
    """Build a mock Nacos handler that serves a single published skill."""
    zip_bytes = _bundle_to_nacos_zip(skill_name, {"SKILL.md": skill_md.encode("utf-8")})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/admin/ai/skills/list":
            return _json(
                {
                    "totalCount": 1,
                    "pageItems": [
                        {
                            "name": skill_name,
                            "description": "Corp weekly report format skill",
                            "labels": {"latest": version},
                        }
                    ],
                }
            )
        if request.url.path == "/v3/admin/ai/skills":
            return _json(
                {
                    "name": skill_name,
                    "labels": {"latest": version},
                    "versions": [
                        {"version": version, "status": "published"},
                    ],
                }
            )
        if request.url.path == "/v3/client/ai/skills":
            assert request.url.params["name"] == skill_name
            return httpx.Response(200, content=zip_bytes)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return handler


class TestMultiUserNacosPull:
    """Simulate a second user pulling published skills from Nacos."""

    def test_fresh_directory_pulls_all_published_skills(self, tmp_path: Path) -> None:
        """Second user with an empty skills dir can pull all published skills."""
        handler = _build_nacos_handler("corp-weekly-report", WEEKLY_REPORT_SKILL_MD)
        client = NacosSkillClient(
            server="http://nacos.test",
            namespace_id="weekly-demo-claude",
            transport=httpx.MockTransport(handler),
        )
        hub = NacosSkillHub(client=client, label="latest")
        skills_dir = tmp_path / "user-b-skills"

        result = hub.pull_skills(str(skills_dir))

        assert result["downloaded"] == 1
        assert result["total_remote"] == 1
        skill_md_path = skills_dir / "corp-weekly-report" / "SKILL.md"
        assert skill_md_path.exists()
        content = skill_md_path.read_text(encoding="utf-8")
        assert "corp-weekly-report" in content
        assert "Weekly Report" in content

    def test_pulled_skills_loadable_by_skill_manager(self, tmp_path: Path) -> None:
        """Skills pulled from Nacos can be loaded by SkillManager and injected."""
        handler = _build_nacos_handler("corp-weekly-report", WEEKLY_REPORT_SKILL_MD)
        client = NacosSkillClient(
            server="http://nacos.test",
            namespace_id="weekly-demo-claude",
            transport=httpx.MockTransport(handler),
        )
        hub = NacosSkillHub(client=client, label="latest")
        skills_dir = tmp_path / "skills"

        hub.pull_skills(str(skills_dir))

        mgr = SkillManager(skills_dir=str(skills_dir))
        all_skills = mgr.get_all_skills()
        assert len(all_skills) == 1
        assert all_skills[0]["name"] == "corp-weekly-report"

        prompt = mgr.build_injection_prompt()
        assert "corp-weekly-report" in prompt
        assert "available_skills" in prompt

    def test_second_user_idempotent_pull(self, tmp_path: Path) -> None:
        """Pulling twice is idempotent — second pull skips already-present skills."""
        handler = _build_nacos_handler("corp-weekly-report", WEEKLY_REPORT_SKILL_MD)
        client = NacosSkillClient(
            server="http://nacos.test",
            namespace_id="weekly-demo-claude",
            transport=httpx.MockTransport(handler),
        )
        hub = NacosSkillHub(client=client, label="latest")
        skills_dir = tmp_path / "skills"

        result1 = hub.pull_skills(str(skills_dir))
        assert result1["downloaded"] == 1

        result2 = hub.pull_skills(str(skills_dir))
        assert result2["downloaded"] == 0
        assert result2["skipped"] == 1

    def test_skill_manager_reload_picks_up_pulled_skills(self, tmp_path: Path) -> None:
        """SkillManager starts empty, then picks up skills after pull + reload."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        mgr = SkillManager(skills_dir=str(skills_dir))
        assert len(mgr.get_all_skills()) == 0

        handler = _build_nacos_handler("corp-weekly-report", WEEKLY_REPORT_SKILL_MD)
        client = NacosSkillClient(
            server="http://nacos.test",
            namespace_id="weekly-demo-claude",
            transport=httpx.MockTransport(handler),
        )
        hub = NacosSkillHub(client=client, label="latest")
        hub.pull_skills(str(skills_dir))

        mgr.reload()
        assert len(mgr.get_all_skills()) == 1
        assert mgr.get_all_skills()[0]["name"] == "corp-weekly-report"

    def test_multiple_skills_from_different_users(self, tmp_path: Path) -> None:
        """Multiple skills published by different users are all pulled."""
        skill_a_md = "---\nname: skill-from-user-a\ndescription: Skill A\n---\n\n# Skill A\n"
        skill_b_md = "---\nname: skill-from-user-b\ndescription: Skill B\n---\n\n# Skill B\n"
        zip_a = _bundle_to_nacos_zip("skill-from-user-a", {"SKILL.md": skill_a_md.encode()})
        zip_b = _bundle_to_nacos_zip("skill-from-user-b", {"SKILL.md": skill_b_md.encode()})

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v3/admin/ai/skills/list":
                return _json(
                    {
                        "totalCount": 2,
                        "pageItems": [
                            {"name": "skill-from-user-a", "labels": {"latest": "0.0.1"}},
                            {"name": "skill-from-user-b", "labels": {"latest": "0.0.2"}},
                        ],
                    }
                )
            if request.url.path == "/v3/client/ai/skills":
                name = request.url.params["name"]
                if name == "skill-from-user-a":
                    return httpx.Response(200, content=zip_a)
                if name == "skill-from-user-b":
                    return httpx.Response(200, content=zip_b)
            raise AssertionError(f"unexpected: {request.method} {request.url}")

        client = NacosSkillClient(
            server="http://nacos.test",
            namespace_id="team-ns",
            transport=httpx.MockTransport(handler),
        )
        hub = NacosSkillHub(client=client, label="latest")
        skills_dir = tmp_path / "skills"

        result = hub.pull_skills(str(skills_dir))

        assert result["downloaded"] == 2
        assert (skills_dir / "skill-from-user-a" / "SKILL.md").exists()
        assert (skills_dir / "skill-from-user-b" / "SKILL.md").exists()

        mgr = SkillManager(skills_dir=str(skills_dir))
        names = {s["name"] for s in mgr.get_all_skills()}
        assert names == {"skill-from-user-a", "skill-from-user-b"}


class TestPollLoopImmediateFirstPull:
    """Test that the skill reload poll loop pulls immediately on first iteration."""

    @pytest.mark.asyncio
    async def test_poll_loop_first_iteration_no_delay(self) -> None:
        """The first poll iteration should pull immediately without waiting."""
        import asyncio

        pull_times: list[float] = []
        import time

        original_time = time.monotonic()

        async def mock_pull(skip_names=None):
            pull_times.append(time.monotonic() - original_time)

        server = SimpleNamespace(
            config=SimpleNamespace(
                sharing_enabled=True,
                sharing_skill_reload_mode="poll",
                sharing_skill_reload_interval_seconds=30,
            ),
            skill_manager=None,
            _skill_reload_interval_seconds=30,
            _skill_reload_task=None,
            _pull_skills_from_cloud=mock_pull,
        )

        from skillclaw.api_server import SkillClawAPIServer

        loop = SkillClawAPIServer._skill_reload_poll_loop

        task = asyncio.create_task(loop(server))
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(pull_times) >= 1, "first pull should have happened immediately"
        assert pull_times[0] < 2.0, f"first pull took {pull_times[0]:.1f}s — should be immediate, not delayed"
