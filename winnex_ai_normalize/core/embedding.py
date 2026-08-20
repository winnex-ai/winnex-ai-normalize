"""
winnex-ai-normalize — embedding providers (with failover).

Extracted from the Maestro's `EmbeddingService` (winnex-madhava-maestro):
connects to an OpenAI-compatible `/v1/embeddings` endpoint (e.g.
Qwen3-Embedding-0.6B), with a bounded LRU cache and L2 normalization.

NO fallback to fake vectors: if the configured providers are unavailable,
the error is explicit (a normalizer must not fabricate embeddings).

Provider failover: the providers are tried in `config.provider_order`
(lowest priority first); if one fails, the next is tried. If ALL fail,
a RuntimeError is raised with the collected errors.
"""
import logging
import os
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger("winnex-ai-normalize.embedding")


class EmbeddingProvider:
    """A single embedding provider (OpenAI-compatible /v1/embeddings)."""

    def __init__(self, config, http_client_factory=None):
        self.name = config.name
        self.model = config.model
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.api_key_env = config.api_key_env
        self.dim = config.dim
        self._http_factory = http_client_factory

    def _post(self, path: str, payload: dict) -> dict:
        import httpx
        headers = {}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}{path}", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts → (n, d) float32 L2-normalized."""
        result = self._post("/embeddings", {"model": self.model, "input": texts})
        data = sorted(result.get("data", []), key=lambda x: x.get("index", 0))
        if not data:
            raise RuntimeError(f"provider {self.name}: empty embeddings response")
        vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-12)   # L2-normalize (cosine contract)
        if not np.isfinite(vecs).all():
            raise RuntimeError(f"provider {self.name}: embeddings contain NaN/inf")
        self.dim = vecs.shape[1]
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def check_available(self) -> dict:
        try:
            r = self._post("/embeddings", {"model": self.model, "input": ["health-check"]})
            return {"available": bool(r.get("data")), "provider": self.name,
                    "model": self.model, "base_url": self.base_url}
        except Exception as e:
            return {"available": False, "provider": self.name, "error": str(e)}


class EmbeddingService:
    """Embedding service with provider failover + bounded cache.

    Tries the providers in `config.provider_order`; the first that succeeds
    serves the batch. If ALL fail, raises RuntimeError with the collected
    errors (NO silent fallback to fake vectors).
    """

    def __init__(self, config=None, provider_factory=EmbeddingProvider):
        from .config import load_config
        self.config = config or load_config()
        self._providers = {
            p.name: provider_factory(p) for p in self.config.providers
        }
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _cache_put(self, text: str, vec: np.ndarray) -> None:
        with self._lock:
            if len(self._cache) >= max(self.config.cache_max, 1):
                for old in list(self._cache)[: max(0, len(self._cache) - self.config.cache_max + 1)]:
                    self._cache.pop(old, None)
            self._cache[text] = vec

    def _embed_missing(self, texts: List[str]) -> np.ndarray:
        """Embed via the first available provider (failover)."""
        errors = []
        for name in self.config.provider_order:
            p = self._providers.get(name)
            if not p or not p.__dict__.get("_available", True):
                continue
            try:
                return p.embed(texts)
            except Exception as e:
                errors.append(f"{name}: {str(e)[:120]}")
                logger.warning(f"provider {name} failed: {e}")
        raise RuntimeError(
            "EmbeddingService: ALL providers unavailable — refusing to "
            f"fabricate embeddings. Errors: {'; '.join(errors) or 'none configured'}")

    def embed_texts(self, texts: List[str], dim: Optional[int] = None) -> np.ndarray:
        """Embed a list of texts → (n, d) float32, cache repeated texts.

        Args:
            texts: list of strings.
            dim: expected dimension (validated).
        Returns:
            (n, d) float32 L2-normalized embeddings.
        """
        if not texts:
            return np.zeros((0, dim or self.config.default_dim), dtype=np.float32)
        with self._lock:
            missing_idx = [i for i, t in enumerate(texts) if t not in self._cache]
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            result = self._embed_missing(missing_texts)
            if dim and result.shape[1] != dim:
                raise ValueError(
                    f"provider returned dim {result.shape[1]}, expected {dim}")
            for i, v in zip(missing_idx, result):
                self._cache_put(texts[i], v)
        with self._lock:
            vecs = np.stack([self._cache[t] for t in texts]).astype(np.float32)
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text → (d,) float32."""
        return self.embed_texts([text])[0]

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def check_available(self) -> dict:
        for name in self.config.provider_order:
            p = self._providers.get(name)
            if p:
                r = p.check_available()
                if r.get("available"):
                    return r
                logger.warning(f"provider {name} unavailable: {r.get('error')}")
        return {"available": False,
                "error": "no provider available",
                "tried": self.config.provider_order}


# Singleton for reuse
_service = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
