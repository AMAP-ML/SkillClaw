from skillclaw.api_server import SkillClawAPIServer, _estimate_openai_body_input_tokens


def _truncate(messages: list[dict], max_prompt_tokens: int) -> list[dict]:
    server = object.__new__(SkillClawAPIServer)
    return server._truncate_messages(messages, tools=None, max_prompt_tokens=max_prompt_tokens)


def test_truncation_stops_when_system_message_alone_exceeds_budget() -> None:
    system = {"role": "system", "content": "x" * 100_000}
    older = {"role": "user", "content": "old"}
    newest = {"role": "user", "content": "new"}

    assert _truncate([system, older, newest], max_prompt_tokens=1) == [system, newest]


def test_truncation_drops_only_messages_needed_to_fit() -> None:
    system = {"role": "system", "content": "system"}
    older = {"role": "user", "content": "x" * 20_000}
    newest = {"role": "user", "content": "keep me"}
    limit = _estimate_openai_body_input_tokens({"messages": [system, newest], "tools": None})

    assert _truncate([system, older, newest], max_prompt_tokens=limit) == [system, newest]
