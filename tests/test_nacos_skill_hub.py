from __future__ import annotations

from pathlib import Path

import httpx

from skillclaw.nacos_skill_hub import NacosSkillClient, NacosSkillHub, _bundle_to_nacos_zip


SKILL_MD = """---
name: demo-skill
description: Demo skill
---

# Demo Skill
"""


def _write_skill(root: Path, body: str = SKILL_MD) -> None:
    skill_dir = root / "demo-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def _json(data):
    return httpx.Response(200, json={"code": 0, "data": data})


def test_nacos_push_uploads_then_submits_without_publish(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir)
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/v3/admin/ai/skills/list":
            return _json({"totalCount": 0, "pageItems": []})
        if request.method == "POST" and request.url.path == "/v3/admin/ai/skills/upload":
            assert request.url.params.get("targetVersion") is None
            assert b'name="targetVersion"' in request.content
            assert b"v1" in request.content
            return _json("demo-skill")
        if request.method == "POST" and request.url.path == "/v3/admin/ai/skills/submit":
            assert request.content == b"namespaceId=public&skillName=demo-skill&version=v1"
            return _json("v1")
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = NacosSkillClient(
        server="http://nacos.test",
        namespace_id="public",
        transport=httpx.MockTransport(handler),
    )
    hub = NacosSkillHub(client=client)

    result = hub.push_skills(str(skills_dir))

    assert result["uploaded"] == 1
    assert result["submitted"] == 1
    assert result["published"] == 0
    assert ("POST", "/v3/admin/ai/skills/publish") not in calls


def test_nacos_publish_is_explicit_after_review() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v3/admin/ai/skills/publish":
            assert b"updateLatestLabel=true" in request.content
            return _json("ok")
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = NacosSkillClient(
        server="http://nacos.test",
        namespace_id="public",
        transport=httpx.MockTransport(handler),
    )
    hub = NacosSkillHub(client=client)

    result = hub.publish_skill("demo-skill", "v1")

    assert result["published"] == 1
    assert result["updated_latest_label"] is True
    assert ("POST", "/v3/admin/ai/skills/publish") in calls


def test_nacos_pull_downloads_latest_zip(tmp_path: Path) -> None:
    zip_bytes = _bundle_to_nacos_zip(
        "demo-skill",
        {
            "SKILL.md": SKILL_MD.encode("utf-8"),
            "references/guide.md": b"hello\n",
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/admin/ai/skills/list":
            return _json(
                {
                    "totalCount": 1,
                    "pageItems": [
                        {
                            "name": "demo-skill",
                            "description": "Demo skill",
                            "labels": {"latest": "v3"},
                        }
                    ],
                }
            )
        if request.url.path == "/v3/client/ai/skills":
            assert request.url.params["name"] == "demo-skill"
            assert request.url.params["version"] == "v3"
            return httpx.Response(200, content=zip_bytes)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = NacosSkillClient(
        server="http://nacos.test",
        namespace_id="public",
        transport=httpx.MockTransport(handler),
    )
    hub = NacosSkillHub(client=client)
    restored = tmp_path / "restored"

    result = hub.pull_skills(str(restored))

    assert result["downloaded"] == 1
    assert (restored / "demo-skill" / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD
    assert (restored / "demo-skill" / "references" / "guide.md").read_bytes() == b"hello\n"


def test_nacos_pull_skips_failed_download_and_continues(tmp_path: Path) -> None:
    zip_bytes = _bundle_to_nacos_zip("demo-skill", {"SKILL.md": SKILL_MD.encode("utf-8")})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/admin/ai/skills/list":
            return _json(
                {
                    "totalCount": 2,
                    "pageItems": [
                        {
                            "name": "broken-skill",
                            "description": "Missing latest label",
                            "labels": {},
                        },
                        {
                            "name": "demo-skill",
                            "description": "Demo skill",
                            "labels": {"latest": "v1"},
                        },
                    ],
                }
            )
        if request.url.path == "/v3/client/ai/skills":
            if request.url.params["name"] == "broken-skill":
                return httpx.Response(404)
            assert request.url.params["name"] == "demo-skill"
            return httpx.Response(200, content=zip_bytes)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = NacosSkillClient(
        server="http://nacos.test",
        namespace_id="public",
        transport=httpx.MockTransport(handler),
    )
    hub = NacosSkillHub(client=client)
    restored = tmp_path / "restored"

    result = hub.pull_skills(str(restored))

    assert result["downloaded"] == 1
    assert result["failed"] == 1
    assert result["failed_names"] == ["broken-skill"]
    assert (restored / "demo-skill" / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD
    assert not (restored / "broken-skill").exists()
