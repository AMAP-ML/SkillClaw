from evolve_server.core.constants import NO_SKILL_KEY
from evolve_server.pipeline.aggregation import aggregate_sessions_by_skill
from evolve_server.pipeline.summarizer import _extract_session_metadata


def test_injected_skill_catalog_does_not_count_as_skill_reference():
    session = {
        "session_id": "s1",
        "turns": [
            {
                "prompt_text": "task",
                "response_text": "answer",
                "injected_skills": ["api-helper", "debug-helper"],
            }
        ],
    }

    _extract_session_metadata(session)
    grouped = aggregate_sessions_by_skill([session])

    assert session["_skills_referenced"] == set()
    assert grouped == {NO_SKILL_KEY: [session]}


def test_read_or_modified_skills_count_as_skill_references():
    session = {
        "session_id": "s1",
        "turns": [
            {
                "read_skills": [{"skill_name": "api-helper"}],
                "modified_skills": [{"skill_name": "debug-helper"}],
                "injected_skills": ["catalog-only"],
            }
        ],
    }

    _extract_session_metadata(session)
    grouped = aggregate_sessions_by_skill([session])

    assert session["_skills_referenced"] == {"api-helper", "debug-helper"}
    assert set(grouped) == {"api-helper", "debug-helper"}
