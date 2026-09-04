"""winnex-ai-normalize — the input-normalization plug for the Madhava engine.

The Madhava engine is AGNOSTIC: it consumes float32 vectors. This package
normalizes ANY input (text via an embedding provider, or raw vectors) into a
valid float32/uint8 corpus ready for `winnex_madhava.build_engine`, with
provider failover and NO silent fallback to fake vectors.

Components:
    - core.config:      JSON-driven provider/config (model, dim, failover order)
    - core.embedding:   EmbeddingService with provider failover + cache + drift tracking
    - core.normalize:   validate / L2-normalize / quantize → Madhava corpus
    - core.quality:     quality FLAGS — the MOTOR's own Cauchy-Schwarz proof,
                        launched on seed queries and captured as the excluded
                        seed set (the flag response) + engine-config routing
    - api.server:       OpenAI-compatible /v1/embeddings + /v1/quality/validate

Consumed by: winnex-madhava (direct), winnex-tracer, the Liferay bridges,
and the Maestro — any tool that needs to feed vectors to Madhava.

The quality gate protects end-to-end recall from the stages OUTSIDE the motor:
third-party embedding quality, dataset integrity (e.g. the corrupted BIGANN
base), and the prefilter/basis routing. The VALIDATION IS THE ENGINE'S OWN:
Cauchy-Schwarz UB < threshold ⟹ the document is mathematically proven not in
the top-K. The validator launches that proof on seed queries and captures the
excluded set — the captured set IS the flag response. FLAGS: pass / warn /
fail.

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
from .core.quality import (
    QualityConfig,
    QualityGateError,
    QualityReport,
    QualityValidator,
    Flag,
    EmbeddingFingerprint,
    audit_corpus,
    build_quality_engine,
    check_embedding_drift,
    load_dataset_preset,
)

__version__ = "1.4.0"

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
    # quality flags
    "QualityConfig",
    "QualityGateError",
    "QualityReport",
    "QualityValidator",
    "Flag",
    "EmbeddingFingerprint",
    "audit_corpus",
    "build_quality_engine",
    "check_embedding_drift",
    "__version__",
]
