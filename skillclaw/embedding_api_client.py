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
import time
from typing import List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 256
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0


class EmbeddingAPIClient:
    """Client for OpenAI-compatible embedding APIs."""

    def __init__(
        self,
        api_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_retries: int = _MAX_RETRIES,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.Client(timeout=self.timeout, headers=headers)
        return self._client

    def close(self):
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if len(texts) <= self.batch_size:
            embeddings = self._call_api(texts)
        else:
            chunks = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                chunks.append(self._call_api(batch))
            embeddings = np.concatenate(chunks, axis=0)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        if convert_to_numpy:
            return embeddings.astype(np.float32)
        return embeddings

    def _call_api(self, texts: List[str]) -> np.ndarray:
        payload = {"model": self.model, "input": texts}

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.api_url}/embeddings",
                    json=payload,
                )
                response.raise_for_status()
                return self._parse_response(response.json())
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                last_exc = e
            except httpx.TransportError as e:
                last_exc = e

            if attempt < self.max_retries - 1:
                delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Embedding API request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.max_retries, delay, last_exc,
                )
                time.sleep(delay)

        logger.error("Embedding API request failed after %d attempts: %s", self.max_retries, last_exc)
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _parse_response(data: dict) -> np.ndarray:
        if "data" not in data:
            raise ValueError(f"Unexpected API response format: {data}")

        embeddings_list = sorted(data["data"], key=lambda x: x.get("index", 0))
        return np.array(
            [item["embedding"] for item in embeddings_list],
            dtype=np.float32,
        )
