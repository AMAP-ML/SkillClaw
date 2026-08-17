"""
Unit tests for embedding API client.

Run with: pytest tests/test_embedding_api.py
"""

import json
import numpy as np
import pytest
import responses
from unittest.mock import MagicMock, patch

from skillclaw.embedding_api_client import EmbeddingAPIClient


class TestEmbeddingAPIClient:
    """Test cases for EmbeddingAPIClient."""

    @responses.activate
    def test_encode_basic(self):
        """Test basic encoding with mock API."""
        # Mock API response in OpenAI format
        responses.add(
            responses.POST,
            "https://api.example.com/embeddings",
            json={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3], "index": 0},
                    {"embedding": [0.4, 0.5, 0.6], "index": 1},
                ]
            },
            status=200,
        )

        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
            api_key="test-key",
        )

        texts = ["Hello world", "Test text"]
        embeddings = client.encode(texts, normalize_embeddings=False)

        assert embeddings.shape == (2, 3)
        assert np.allclose(embeddings[0], [0.1, 0.2, 0.3])
        assert np.allclose(embeddings[1], [0.4, 0.5, 0.6])

    @responses.activate
    def test_encode_with_normalization(self):
        """Test L2 normalization of embeddings."""
        responses.add(
            responses.POST,
            "https://api.example.com/embeddings",
            json={
                "data": [
                    {"embedding": [3.0, 4.0], "index": 0},  # magnitude = 5
                ]
            },
            status=200,
        )

        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
        )

        embeddings = client.encode(["test"], normalize_embeddings=True)

        # After normalization: [3/5, 4/5] = [0.6, 0.8]
        assert np.allclose(embeddings[0], [0.6, 0.8], atol=1e-6)
        # Check L2 norm is 1
        assert np.allclose(np.linalg.norm(embeddings[0]), 1.0)

    @responses.activate
    def test_encode_empty_input(self):
        """Test encoding with empty input."""
        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
        )

        embeddings = client.encode([])

        assert embeddings.shape == (0, 0)

    @responses.activate
    def test_encode_with_authorization(self):
        """Test that API key is correctly included in request."""
        responses.add(
            responses.POST,
            "https://api.example.com/embeddings",
            json={"data": [{"embedding": [0.1, 0.2], "index": 0}]},
            status=200,
        )

        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
            api_key="secret-key-123",
        )

        embeddings = client.encode(["test"])

        # Check that request includes Authorization header
        assert len(responses.calls) == 1
        assert responses.calls[0].request.headers["Authorization"] == "Bearer secret-key-123"

    @responses.activate
    def test_encode_without_api_key(self):
        """Test encoding without API key (for local services)."""
        responses.add(
            responses.POST,
            "https://api.example.com/embeddings",
            json={"data": [{"embedding": [0.1, 0.2], "index": 0}]},
            status=200,
        )

        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
            api_key=None,
        )

        embeddings = client.encode(["test"])

        # Check that request does not include Authorization header
        assert len(responses.calls) == 1
        assert "Authorization" not in responses.calls[0].request.headers

    @responses.activate
    def test_encode_out_of_order_responses(self):
        """Test that embeddings are correctly sorted by index."""
        responses.add(
            responses.POST,
            "https://api.example.com/embeddings",
            json={
                "data": [
                    {"embedding": [0.3, 0.3], "index": 2},
                    {"embedding": [0.1, 0.1], "index": 0},
                    {"embedding": [0.2, 0.2], "index": 1},
                ]
            },
            status=200,
        )

        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
        )

        embeddings = client.encode(["a", "b", "c"], normalize_embeddings=False)

        # Should be sorted by index
        assert np.allclose(embeddings[0], [0.1, 0.1])
        assert np.allclose(embeddings[1], [0.2, 0.2])
        assert np.allclose(embeddings[2], [0.3, 0.3])

    @responses.activate
    def test_api_error_handling(self):
        """Test error handling for API failures."""
        responses.add(
            responses.POST,
            "https://api.example.com/embeddings",
            json={"error": "Invalid API key"},
            status=401,
        )

        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
            api_key="invalid-key",
        )

        with pytest.raises(Exception):  # Should raise RequestException
            client.encode(["test"])

    @responses.activate
    def test_invalid_response_format(self):
        """Test error handling for invalid response format."""
        responses.add(
            responses.POST,
            "https://api.example.com/embeddings",
            json={"invalid": "response"},
            status=200,
        )

        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
        )

        with pytest.raises(ValueError, match="Unexpected API response format"):
            client.encode(["test"])

    @responses.activate
    def test_large_batch(self):
        """Test encoding a large batch of texts."""
        num_texts = 100
        embedding_dim = 384

        # Generate mock embeddings
        embeddings_data = [
            {"embedding": list(np.random.rand(embedding_dim).astype(float)), "index": i}
            for i in range(num_texts)
        ]

        responses.add(
            responses.POST,
            "https://api.example.com/embeddings",
            json={"data": embeddings_data},
            status=200,
        )

        client = EmbeddingAPIClient(
            api_url="https://api.example.com",
            model="test-model",
        )

        texts = [f"text_{i}" for i in range(num_texts)]
        embeddings = client.encode(texts)

        assert embeddings.shape == (num_texts, embedding_dim)
        assert embeddings.dtype == np.float32


class TestEmbeddingAPIIntegration:
    """Integration tests with SkillManager."""

    @pytest.mark.skipif(True, reason="Requires actual API key")
    def test_skill_manager_with_api(self):
        """Test SkillManager integration with embedding API.
        
        This test is skipped by default as it requires a real API key.
        Set embedding API credentials and remove @skipif to run.
        """
        from skillclaw.skill_manager import SkillManager
        from pathlib import Path
        import tempfile

        # Create temporary skill directory
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)

            # Create sample skills
            for skill_name in ["test-skill-1", "test-skill-2"]:
                skill_path = skills_dir / skill_name
                skill_path.mkdir()
                (skill_path / "SKILL.md").write_text(
                    f"""---
name: {skill_name}
description: Test skill for {skill_name}
---

# {skill_name}

Test content
"""
                )

            # Initialize with API
            skill_manager = SkillManager(
                skills_dir=str(skills_dir),
                retrieval_mode="embedding",
                embedding_type="api",
                embedding_api_url="https://api.jina.ai/v1",
                embedding_api_model="jina-embeddings-v5-text-small",
                embedding_api_key="your-api-key",
            )

            # Test retrieval
            results = skill_manager.retrieve("test query", top_k=2)
            assert len(results) <= 2
