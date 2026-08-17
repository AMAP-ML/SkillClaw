"""
Embedding API client supporting OpenAI-compatible APIs.

Supports any embedding service with OpenAI API format, including:
- OpenAI (https://api.openai.com/v1/embeddings)
- Jina (https://api.jina.ai/v1/embeddings)
- Azure OpenAI
- LocalAI
- Ollama (with OpenAI-compatible server)
"""

import logging
from typing import List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class EmbeddingAPIClient:
    """Client for OpenAI-compatible embedding APIs."""

    def __init__(
        self,
        api_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize embedding API client.

        Args:
            api_url: Base URL of the embedding API (e.g., "https://api.openai.com/v1")
            model: Model name to use for embeddings
            api_key: API key for authentication (optional for local services)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._session = None

    @property
    def session(self):
        """Lazy-load requests session."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings using the API.

        Args:
            texts: List of text strings to encode
            normalize_embeddings: Whether to normalize embeddings (L2)
            show_progress_bar: Whether to show progress bar (ignored for API)
            convert_to_numpy: Whether to return numpy array

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if show_progress_bar:
            logger.warning("show_progress_bar parameter is not supported for embedding API client and will be ignored")

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings = self._call_api(texts)

        if normalize_embeddings:
            # L2 normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms

        if convert_to_numpy:
            return embeddings.astype(np.float32)
        return embeddings

    def _call_api(self, texts: List[str]) -> np.ndarray:
        """Call the embedding API and return embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            numpy array of embeddings
        """
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = self.session.post(
                f"{self.api_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API request failed: {e}")
            raise

        data = response.json()

        # Extract embeddings from response
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        if "data" not in data:
            raise ValueError(f"Unexpected API response format: {data}")

        embeddings_list = sorted(data["data"], key=lambda x: x.get("index", 0))
        embeddings = np.array(
            [item["embedding"] for item in embeddings_list],
            dtype=np.float32,
        )

        logger.debug(f"Retrieved {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        return embeddings

    def __del__(self):
        """Close session when client is destroyed."""
        if self._session is not None:
            self._session.close()
