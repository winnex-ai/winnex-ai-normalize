"""
winnex-ai-normalize — configuration (JSON-driven).

The single source of truth for the normalization providers, the embedding
model and the provider priority/failover order. Mirrors the Maestro's
`ai_models_config` (JSON-driven, no hardcoded values).

Config sources, in order of precedence:
  1. Constructor / env vars (WINNEX_AI_NORMALIZE_*)
  2. A JSON file path (WINNEX_AI_NORMALIZE_CONFIG)
  3. Sensible defaults.

License: Business Source License 1.1 (BSL 1.1)
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProviderConfig:
    """A normalization provider (embedding source)."""
    name: str                      # "openai" | "qwen3" | "nano" | "xfactor" | "direct"
    type: str = "openai_compat"    # how to call it: openai_compat | local | direct
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""          # env var holding the API key (never hardcoded)
    dim: int = 0                   # 0 = auto-detect
    timeout: float = 20.0
    priority: int = 10             # lower = tried first (failover order)
    enabled: bool = True


@dataclass
class NormalizeConfig:
    """Top-level normalization configuration."""
    default_dim: int = 1024             # Qwen3-Embedding-0.6B
    default_provider: str = "qwen3"     # the first provider to try
    normalize_l2: bool = True           # L2-normalize embeddings (cosine contract)
    validate_nan: bool = True           # fail loudly on NaN/inf
    validate_range: bool = False        # warn (not fail) on out-of-range
    cache_max: int = 512                # bounded embedding cache
    providers: List[ProviderConfig] = field(default_factory=list)
    provider_order: List[str] = field(default_factory=list)  # failover order

    @classmethod
    def from_env(cls) -> "NormalizeConfig":
        """Load from WINNEX_AI_NORMALIZE_* env vars / JSON file."""
        cfg = cls()
        cfg.default_dim = int(os.environ.get("WINNEX_AI_NORMALIZE_DIM", cfg.default_dim))
        cfg.default_provider = os.environ.get(
            "WINNEX_AI_NORMALIZE_PROVIDER", cfg.default_provider)
        # A config JSON may define the providers + failover order.
        json_path = os.environ.get("WINNEX_AI_NORMALIZE_CONFIG", "")
        if json_path and os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            cfg.default_dim = int(data.get("default_dim", cfg.default_dim))
            cfg.default_provider = data.get("default_provider", cfg.default_provider)
            cfg.normalize_l2 = bool(data.get("normalize_l2", cfg.normalize_l2))
            cfg.cache_max = int(data.get("cache_max", cfg.cache_max))
            for p in data.get("providers", []):
                cfg.providers.append(ProviderConfig(**p))
            cfg.provider_order = list(data.get("provider_order", []))
        # Default providers if none configured.
        if not cfg.providers:
            cfg.providers = [
                ProviderConfig(
                    name="qwen3", type="openai_compat",
                    model=os.environ.get("EMBEDDING_MODEL", "/workspace/models/Qwen3-Embedding-0.6B"),
                    base_url=os.environ.get("EMBEDDING_URL", "http://winnex-embedding:8102"),
                    dim=cfg.default_dim, priority=1),
                ProviderConfig(
                    name="openai", type="openai_compat",
                    model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    api_key_env="OPENAI_API_KEY", priority=2),
                ProviderConfig(
                    name="direct", type="direct",
                    dim=cfg.default_dim, priority=99),
            ]
        if not cfg.provider_order:
            cfg.provider_order = [p.name for p in sorted(
                cfg.providers, key=lambda p: p.priority)]
        return cfg


def load_config() -> NormalizeConfig:
    """Convenience loader (cached)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = NormalizeConfig.from_env()
    return _CONFIG


_CONFIG: Optional[NormalizeConfig] = None
