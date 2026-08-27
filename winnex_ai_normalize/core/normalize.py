"""
winnex-ai-normalize — embedding normalization (the plug that feeds Madhava).

The Madhava engine is AGNOSTIC: it consumes float32 vectors. This module
normalizes ANY input (text via a provider, or raw vectors) into a valid
float32 corpus ready for `winnex_madhava.build_engine`:

  - validates dimension, dtype, NaN/inf (fail loudly — no silent fallback),
  - L2-normalizes when the cosine contract requires it,
  - quantizes float32 → uint8 for the engine's native corpus
    (the exact scale logic from the Maestro's `quantize_corpus`).

License: Business Source License 1.1 (BSL 1.1)
"""
import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("winnex-ai-normalize")


# ---------------------------------------------------------------------------
# Validation (fail loudly — no fake embeddings)
# ---------------------------------------------------------------------------
def validate_embeddings(
    vectors: np.ndarray,
    dim: Optional[int] = None,
    require_unit_norm: bool = False,
) -> np.ndarray:
    """Validate a (n, d) embedding matrix. Raises on invalid input.

    Args:
        vectors: (n, d) float32 (or float64) embeddings.
        dim: expected dimensionality (raises if mismatch).
        require_unit_norm: if True, each row must be unit-norm (cosine).
    Returns:
        the array as float32, contiguous.
    Raises:
        ValueError: on wrong dim, NaN/inf, or empty.
    """
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2D (n, d), got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("embeddings must not be empty")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(
            f"dimension mismatch: expected {dim}, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError("embeddings contain NaN or inf — refusing to proceed")
    if require_unit_norm:
        norms = np.linalg.norm(arr, axis=1)
        bad = np.where((norms < 1e-6) | (np.abs(norms - 1.0) > 1e-2))[0]
        if len(bad):
            raise ValueError(
                f"{len(bad)} rows are not unit-norm (cosine contract) — "
                "call normalize_l2() first or fix the provider")
    return arr


# ---------------------------------------------------------------------------
# L2 normalization (cosine contract)
# ---------------------------------------------------------------------------
def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit norm (the cosine contract).

    Raises on a zero-norm row (cannot be normalized meaningfully).
    """
    arr = validate_embeddings(vectors)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("a row has zero norm — cannot L2-normalize")
    return arr / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Quantization float32 → uint8 (the engine's native corpus)
# ---------------------------------------------------------------------------
def quantize_corpus(embeddings: np.ndarray) -> np.ndarray:
    """Quantize float32 normalized embeddings to the engine's uint8 corpus.

    The native C++ engine requires a uint8 corpus and interprets each byte as
    an integer coordinate in [0, 255] (see `load_raw` in winnex-madhava: it
    casts uint8 → float32 without re-scaling, then L2-normalizes). Passing raw
    float32 normalized embeddings straight into `build_engine` silently
    truncates them (`astype(np.uint8)` maps everything in [-1, 1] to 0) — a
    zero corpus whose cosine scores are all 0.0.

    This is the exact scale logic from the Maestro's `quantize_corpus`:
      - values in [-1, 1] (embedding domain)   → map to [0, 255] via (x+1)*127.5
      - values in [0, 1] (post-activation)     → scale to [0, 255] via x*255
      - values already in [0, 255]             → left as-is (round+clip)

    Args:
        embeddings: (n, d) float32 normalized embeddings (cosine).
    Returns:
        (n, d) uint8 corpus ready for `build_engine`.
    """
    arr = validate_embeddings(embeddings)
    if arr.size and arr.min() < 0.0:
        # Embedding-domain values (normalized vectors): [-1, 1] -> [0, 255].
        u8 = np.round((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    else:
        # [0, 1] normalized domain (Qwen post-activation): scale to [0, 255].
        u8 = np.round(arr * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(u8, dtype=np.uint8)


# ---------------------------------------------------------------------------
# The normalizer facade (text/vectors → ready-for-madhava)
# ---------------------------------------------------------------------------
class EmbeddingNormalizer:
    """Normalizes text or raw vectors into Madhava-ready float32/uint8.

    The normalizer is ALSO the quality guardian: every corpus that passes
    through it is audited (dataset integrity + embedding-set drift), and the
    audit FLAGS + routes the engine configuration. This is the ingest gate
    that protects end-to-end recall from third-party embedding drift and
    corrupted datasets (e.g. the BIGANN base whose order differs from the GT).
    """

    def __init__(self, config=None, embedding_service=None):
        from .config import load_config
        self.config = config or load_config()
        self._embedding_service = embedding_service

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            from .embedding import EmbeddingService
            # Use THIS normalizer's config (not the env singleton) so a caller
            # can point at a specific provider/service.
            self._embedding_service = EmbeddingService(config=self.config)
        return self._embedding_service

    def vectorize_texts(self, texts, dim=None) -> np.ndarray:
        """Embed texts via the configured provider, normalized to float32.

        Raises if no provider is available (NO fallback to fake vectors).
        """
        dim = dim or self.config.default_dim
        return self.embedding_service.embed_texts(texts, dim=dim)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text → (d,) float32 (query path)."""
        return self.embedding_service.embed_one(text)

    def normalize_vectors(
        self,
        vectors: Union[np.ndarray, list],
        dim: Optional[int] = None,
        require_unit_norm: bool = False,
    ) -> np.ndarray:
        """Validate + normalize raw vectors to float32 (unit-norm optional)."""
        arr = np.asarray(vectors, dtype=np.float32)
        return validate_embeddings(arr, dim=dim, require_unit_norm=require_unit_norm)

    def to_corpus(
        self,
        vectors: Union[np.ndarray, list],
        dim: Optional[int] = None,
    ) -> np.ndarray:
        """Quantize float32 → uint8 (for BIGANN-style RAW BYTE corpora).

        NOTE: this is NOT the cosine path. For float32 embeddings
        (OpenAI/Qwen3) use `build_engine()` which goes through the engine's
        float32 manifold (build_float32) and preserves cosine exactly.
        """
        arr = self.normalize_vectors(vectors, dim=dim)
        return quantize_corpus(arr)

    def build_engine(self, vectors, dim=None, k=10, **engine_kwargs):
        """Build a Madhava engine over float32 embeddings (the CORRECT path).

        Uses `winnex_madhava.build_engine` with the float32 corpus (the
        engine's build_float32 manifold preserves cosine). The madhava
        engine stays agnostic — it receives ready float32 vectors.

        The corpus is AUDITED first (quality flags): corrupted datasets
        (NaN, degenerate, alignment) FAIL loudly; isotropic/redundant corpora
        route to a safer k1_fraction; dim ≥ 384 forces early_exit=False (the
        P0 recall bug). Pass `allow_unsafe=True` to skip the gate.

        Args:
            vectors: (n, d) float32 embeddings (or text via vectorize_texts).
            dim: embedding dimension.
            k: top-k.
            **engine_kwargs: passed to winnex_madhava.build_engine.
        Returns:
            a winnex_madhava engine (or (engine, QualityReport) when
            return_report=True).
        """
        from .quality import build_quality_engine
        arr = self.normalize_vectors(vectors, dim=dim)
        return build_quality_engine(
            np.ascontiguousarray(arr, dtype=np.float32),
            dim=dim or arr.shape[1],
            k=k,
            provider=getattr(self.embedding_service, "current_provider", None),
            engine_kwargs=dict(metric="cosine", normalize_input=True, **engine_kwargs),
        )

    def audit_corpus(self, vectors, dim=None, reference=None) -> "QualityReport":
        """Validate a corpus/embedding set and return a QualityReport.

        This is the explicit quality gate: inspect a dataset BEFORE indexing
        it (the BIGANN-class problem — corrupted order, NaN, degenerate,
        isotropic, redundant, drifted-vs-reference) and get the FLAGS + the
        suggested engine configuration.

        Args:
            vectors: (n, d) embeddings (float32/float64) or uint8 raw bytes.
            dim: expected dimension (mismatch → FAIL flag).
            reference: an embedding batch to compare against (drift /
                alignment — the "multiple embedding sets" case).
        Returns:
            QualityReport (flags, metrics, suggested config).
        """
        from .quality import audit_corpus
        return audit_corpus(
            vectors,
            dim=dim,
            reference=reference,
            provider=getattr(self.embedding_service, "current_provider", None),
        )

    def check_available(self) -> dict:
        """Health check: is the embedding provider reachable?"""
        return self.embedding_service.check_available()
