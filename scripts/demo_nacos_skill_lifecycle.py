#!/usr/bin/env python
"""Demo SkillClaw's Nacos-backed skill lifecycle without a real Nacos server.

The demo uses httpx.MockTransport to show the exact lifecycle SkillClaw now
drives against Nacos:

1. Upload local skill zip as a target version.
2. Submit the version for review.
3. In this mock, Nacos audit is disabled, so submit auto-publishes latest.
4. Pull latest back into a local skills directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillclaw.nacos_skill_hub import NacosSkillClient, NacosSkillHub, _bundle_to_nacos_zip


SKILL_MD = """---
name: demo-skill
description: Demo Nacos lifecycle skill
---

# Demo Skill

Use this skill to demonstrate Nacos-backed lifecycle management.
"""


def _json(data):
    return httpx.Response(200, json={"code": 0, "data": data})


def main() -> None:
    calls: list[str] = []
    published_zip = _bundle_to_nacos_zip(
        "demo-skill",
        {
            "SKILL.md": SKILL_MD.encode("utf-8"),
            "references/demo.md": b"demo reference\n",
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/v3/admin/ai/skills/list":
            return _json({"totalCount": 0, "pageItems": []})
        if request.url.path == "/v3/admin/ai/skills/upload":
            return _json("demo-skill")
        if request.url.path == "/v3/admin/ai/skills/submit":
            return _json("v1")
        if request.url.path == "/v3/client/ai/skills":
            return httpx.Response(200, content=published_zip)
        raise RuntimeError(f"unexpected request: {request.method} {request.url}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skills_dir = root / "skills"
        skill_dir = skills_dir / "demo-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "demo.md").write_text("demo reference\n", encoding="utf-8")

        client = NacosSkillClient(
            server="http://nacos-demo.local",
            namespace_id="public",
            transport=httpx.MockTransport(handler),
        )
        hub = NacosSkillHub(client=client)

        push = hub.push_skills(str(skills_dir))
        pulled_dir = root / "pulled"

        def list_after_publish(request: httpx.Request) -> httpx.Response:
            calls.append(f"{request.method} {request.url.path}")
            if request.url.path == "/v3/admin/ai/skills/list":
                return _json(
                    {
                        "totalCount": 1,
                        "pageItems": [{"name": "demo-skill", "labels": {"latest": "v1"}}],
                    }
                )
            if request.url.path == "/v3/client/ai/skills":
                return httpx.Response(200, content=published_zip)
            raise RuntimeError(f"unexpected request: {request.method} {request.url}")

        client._client = httpx.Client(transport=httpx.MockTransport(list_after_publish))
        pull = hub.pull_skills(str(pulled_dir))

        print("SkillClaw Nacos lifecycle demo")
        print("mock mode: Nacos audit disabled, submit auto-publishes latest")
        print(f"push: uploaded={push['uploaded']} submitted={push['submitted']}")
        print(f"pull: downloaded={pull['downloaded']} total_remote={pull['total_remote']}")
        print(f"pulled skill: {pulled_dir / 'demo-skill' / 'SKILL.md'}")
        print("nacos calls:")
        for call in calls:
            print(f"  {call}")


if __name__ == "__main__":
    main()
