"""
Unit tests for embedding API client.

Run with: pytest tests/test_embedding_api.py
"""

import json
from typing import List

import httpx
import numpy as np
import pytest

from skillclaw.embedding_api_client import EmbeddingAPIClient


def _make_transport(response_json: dict, status_code: int = 200, capture: List[httpx.Request] | None = None):
    """Build a MockTransport that returns a fixed JSON response."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        return httpx.Response(status_code, json=response_json)

    return httpx.MockTransport(_handler)


def _make_client(api_key=None, *, transport, **kwargs) -> EmbeddingAPIClient:
    client = EmbeddingAPIClient(
        api_url="https://api.example.com",
        model="test-model",
        api_key=api_key,
        **kwargs,
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    client._client = httpx.Client(transport=transport, headers=headers)
    return client


class TestEmbeddingAPIClient:

    def test_encode_basic(self):
        transport = _make_transport({
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ]
        })
        client = _make_client("test-key", transport=transport)

        embeddings = client.encode(["Hello world", "Test text"], normalize_embeddings=False)

        assert embeddings.shape == (2, 3)
        assert np.allclose(embeddings[0], [0.1, 0.2, 0.3])
        assert np.allclose(embeddings[1], [0.4, 0.5, 0.6])

    def test_encode_with_normalization(self):
        transport = _make_transport({
            "data": [{"embedding": [3.0, 4.0], "index": 0}]
        })
        client = _make_client(transport=transport)

        embeddings = client.encode(["test"], normalize_embeddings=True)

        assert np.allclose(embeddings[0], [0.6, 0.8], atol=1e-6)
        assert np.allclose(np.linalg.norm(embeddings[0]), 1.0)

    def test_encode_empty_input(self):
        client = EmbeddingAPIClient(api_url="https://api.example.com", model="test-model")
        embeddings = client.encode([])
        assert embeddings.shape == (0, 0)

    def test_encode_with_authorization(self):
        captured: List[httpx.Request] = []
        transport = _make_transport(
            {"data": [{"embedding": [0.1, 0.2], "index": 0}]},
            capture=captured,
        )
        client = _make_client("secret-key-123", transport=transport)

        client.encode(["test"])

        assert len(captured) == 1
        assert captured[0].headers["Authorization"] == "Bearer secret-key-123"

    def test_encode_without_api_key(self):
        captured: List[httpx.Request] = []
        transport = _make_transport(
            {"data": [{"embedding": [0.1, 0.2], "index": 0}]},
            capture=captured,
        )
        client = _make_client(api_key=None, transport=transport)

        client.encode(["test"])

        assert len(captured) == 1
        assert "authorization" not in {k.lower() for k in captured[0].headers.keys()}

    def test_encode_out_of_order_responses(self):
        transport = _make_transport({
            "data": [
                {"embedding": [0.3, 0.3], "index": 2},
                {"embedding": [0.1, 0.1], "index": 0},
                {"embedding": [0.2, 0.2], "index": 1},
            ]
        })
        client = _make_client(transport=transport)

        embeddings = client.encode(["a", "b", "c"], normalize_embeddings=False)

        assert np.allclose(embeddings[0], [0.1, 0.1])
        assert np.allclose(embeddings[1], [0.2, 0.2])
        assert np.allclose(embeddings[2], [0.3, 0.3])

    def test_api_error_handling_4xx(self):
        transport = _make_transport({"error": "Invalid API key"}, status_code=401)
        client = _make_client("invalid-key", transport=transport)

        with pytest.raises(httpx.HTTPStatusError):
            client.encode(["test"])

    def test_api_error_handling_5xx_retries(self):
        call_count = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(500, json={"error": "Internal Server Error"})

        client = _make_client(transport=httpx.MockTransport(_handler), max_retries=2)

        with pytest.raises(httpx.HTTPStatusError):
            client.encode(["test"])

        assert call_count == 2

    def test_invalid_response_format(self):
        transport = _make_transport({"invalid": "response"})
        client = _make_client(transport=transport)

        with pytest.raises(ValueError, match="Unexpected API response format"):
            client.encode(["test"])

    def test_large_batch(self):
        num_texts = 100
        embedding_dim = 384

        rng = np.random.default_rng(42)
        embeddings_data = [
            {"embedding": rng.random(embedding_dim).tolist(), "index": i}
            for i in range(num_texts)
        ]

        transport = _make_transport({"data": embeddings_data})
        client = _make_client(transport=transport)

        texts = [f"text_{i}" for i in range(num_texts)]
        embeddings = client.encode(texts)

        assert embeddings.shape == (num_texts, embedding_dim)
        assert embeddings.dtype == np.float32

    def test_batch_chunking(self):
        call_count = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            n = len(body["input"])
            data = [{"embedding": [0.1, 0.2], "index": i} for i in range(n)]
            return httpx.Response(200, json={"data": data})

        client = _make_client(transport=httpx.MockTransport(_handler), batch_size=3)
        embeddings = client.encode([f"text_{i}" for i in range(7)], normalize_embeddings=False)

        assert call_count == 3  # ceil(7/3) = 3 API calls
        assert embeddings.shape == (7, 2)


class TestEmbeddingAPIIntegration:

    @pytest.mark.skipif(True, reason="Requires actual API key")
    def test_skill_manager_with_api(self):
        import tempfile
        from pathlib import Path

        from skillclaw.skill_manager import SkillManager

        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)

            for skill_name in ["test-skill-1", "test-skill-2"]:
                skill_path = skills_dir / skill_name
                skill_path.mkdir()
                (skill_path / "SKILL.md").write_text(
                    f"---\nname: {skill_name}\ndescription: Test skill for {skill_name}\n"
                    f"---\n\n# {skill_name}\n\nTest content\n"
                )

            skill_manager = SkillManager(
                skills_dir=str(skills_dir),
                retrieval_mode="embedding",
                embedding_type="api",
                embedding_api_url="https://api.jina.ai/v1",
                embedding_api_model="jina-embeddings-v5-text-small",
                embedding_api_key="your-api-key",
            )

            results = skill_manager.retrieve("test query", top_k=2)
            assert len(results) <= 2
