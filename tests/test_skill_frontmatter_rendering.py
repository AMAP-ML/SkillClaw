import copy

import pytest
import yaml

from evolve_server.core.utils import build_skill_md, parse_skill_content


@pytest.mark.parametrize("value", ["plain", 12, True, None, "yes", "a: b", "line one\nline two", ["x", 2]])
def test_extra_scalar_roundtrip(value):
    extra = {"custom": value, "following": {"nested": [1, True]}}
    skill = {"name": "example", "description": "Example", "content": "Body", "extra_frontmatter": extra}
    rendered = build_skill_md(skill)
    metadata = yaml.safe_load(rendered.split("---\n", 2)[1])
    assert metadata["custom"] == value
    assert type(metadata["custom"]) is type(value)
    assert parse_skill_content("example", rendered)["extra_frontmatter"] == extra


def test_body_preserved_verbatim():
    body = " \n---\n# Body\n\nText\n "
    assert build_skill_md({"content": body}).split("\n---\n", 1)[1] == "\n" + body + "\n"


def test_nested_reserved_and_quoted_fields_roundtrip():
    extra = {
        "name": "wrong",
        "description": "wrong",
        "category": "wrong",
        "metadata": {"nested": {"list": [True, None, "yes", {"key": "value"}]}},
        "key: with colon": "value\nnext",
    }
    skill = {
        "name": "yes",
        "description": "null",
        "category": "on",
        "content": "# Body\n\nText",
        "extra_frontmatter": extra,
    }
    before = copy.deepcopy(skill)
    rendered = build_skill_md(skill)
    metadata = yaml.safe_load(rendered.split("---\n", 2)[1])
    assert metadata == {
        "name": "yes",
        "description": "null",
        "category": "on",
        **{k: v for k, v in extra.items() if k not in {"name", "description", "category"}},
    }
    parsed = parse_skill_content("yes", rendered)
    assert parsed["extra_frontmatter"] == {
        k: v for k, v in extra.items() if k not in {"name", "description", "category"}
    }
    assert parsed["content"] == skill["content"]
    assert skill == before


def test_defaults_and_non_mapping_extras():
    for extra in (None, [], "ignored"):
        rendered = build_skill_md({"extra_frontmatter": extra})
        assert yaml.safe_load(rendered.split("---\n", 2)[1]) == {
            "name": "unknown",
            "description": "",
            "category": "general",
        }
