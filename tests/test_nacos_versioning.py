from __future__ import annotations

from skillclaw.nacos_versions import _next_version


def test_next_version_defaults_to_semver_for_new_skill() -> None:
    assert _next_version({}) == "0.0.1"


def test_next_version_increments_existing_semver_patch() -> None:
    summary = {"labels": {"latest": "0.0.3"}}
    detail = {"versions": [{"version": "0.0.2"}, {"version": "0.0.4"}]}

    assert _next_version(summary, detail) == "0.0.5"


def test_next_version_keeps_existing_v_format() -> None:
    summary = {"reviewingVersion": "v1"}

    assert _next_version(summary) == "v2"


def test_next_version_uses_current_version_format_when_formats_are_mixed() -> None:
    summary = {"labels": {"latest": "0.0.3"}, "reviewingVersion": "v9"}

    assert _next_version(summary) == "0.0.4"
