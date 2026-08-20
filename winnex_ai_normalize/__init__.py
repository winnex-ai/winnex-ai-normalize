"""winnex-ai-normalize — the input-normalization plug for the Madhava engine.

The Madhava engine is AGNOSTIC: it consumes float32 vectors. This package
normalizes ANY input (text via an embedding provider, or raw vectors) into a
valid float32/uint8 corpus ready for `winnex_madhava.build_engine`, with
provider failover and NO silent fallback to fake vectors.

Components:
    - core.config:      JSON-driven provider/config (model, dim, failover order)
    - core.embedding:   EmbeddingService with provider failover + cache
    - core.normalize:   validate / L2-normalize / quantize → Madhava corpus
    - api.server:       OpenAI-compatible /v1/embeddings endpoint

Consumed by: winnex-madhava (direct), winnex-tracer, the Liferay bridges,
and the Maestro — any tool that needs to feed vectors to Madhava.

Business Source License 1.1 (BSL 1.1) | pay@winnex.ai
"""
from .core.config import NormalizeConfig, ProviderConfig, load_config
from .core.embedding import EmbeddingProvider, EmbeddingService, get_embedding_service
from .core.normalize import (
    EmbeddingNormalizer,
    validate_embeddings,
    normalize_l2,
    quantize_corpus,
)

__version__ = "1.0.0"

__all__ = [
    "NormalizeConfig",
    "ProviderConfig",
    "load_config",
    "EmbeddingProvider",
    "EmbeddingService",
    "get_embedding_service",
    "EmbeddingNormalizer",
    "validate_embeddings",
    "normalize_l2",
    "quantize_corpus",
    "__version__",
]
