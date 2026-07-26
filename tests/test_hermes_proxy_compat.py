import pytest

from skillclaw.api_server import SkillClawAPIServer, _normalize_openai_chat_token_fields
from skillclaw.config import SkillClawConfig


@pytest.mark.asyncio
async def test_hermes_reasoning_effort_is_not_forwarded_upstream(tmp_path):
    server = SkillClawAPIServer(
        SkillClawConfig(
            proxy_api_key="skillclaw",
            record_enabled=False,
            record_dir=str(tmp_path),
            use_skills=False,
            llm_model_id="gpt-4o",
            llm_api_base="https://api.openai.com/v1",
        )
    )
    seen = {}

    async def fake_forward(body):
        seen["body"] = dict(body)
        return {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    server._forward_to_llm = fake_forward
    result = await server._handle_request(
        {
            "model": "skillclaw-model",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
            "session_id": "hermes-smoke",
            "turn_type": "main",
        },
        session_id="hermes-smoke",
        turn_type="main",
        session_done=False,
    )

    assert result["response"]["choices"][0]["message"]["content"] == "OK"
    assert "reasoning_effort" not in seen["body"]
    assert "session_id" not in seen["body"]
    assert "turn_type" not in seen["body"]


@pytest.mark.asyncio
async def test_excessive_hermes_max_tokens_is_capped_before_upstream(tmp_path):
    server = SkillClawAPIServer(
        SkillClawConfig(
            proxy_api_key="skillclaw",
            record_enabled=False,
            record_dir=str(tmp_path),
            use_skills=False,
            llm_model_id="gpt-4o",
            llm_api_base="https://api.openai.com/v1",
        )
    )
    seen = {}

    async def fake_forward(body):
        seen["body"] = dict(body)
        return {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    server._forward_to_llm = fake_forward
    await server._handle_request(
        {
            "model": "skillclaw-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 65536,
        },
        session_id="hermes-smoke",
        turn_type="main",
        session_done=False,
    )

    assert seen["body"]["max_tokens"] == 8192


def test_gpt5_chat_requests_use_max_completion_tokens():
    body = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 20,
    }

    _normalize_openai_chat_token_fields(body)

    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 20
