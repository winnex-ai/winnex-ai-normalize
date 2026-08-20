# winnex-ai-normalize

**The input-normalization plug for the Madhava engine.**

The Madhava engine is **agnostic** — it consumes float32 vectors. This
package normalizes **any** input (text via an embedding provider, or raw
vectors) into a valid float32/uint8 corpus ready for
`winnex_madhava.build_engine`, with **provider failover** and **no silent
fallback to fake vectors**.

```
text / raw vectors
      │
      ▼
winnex-ai-normalize
  ├── providers:  OpenAI / Qwen3 / nano / xfactor / direct (failover order)
  ├── validate:   dim + NaN + norm (fail loudly)
  ├── normalize:  L2 unit-norm (cosine contract)
  └── quantize:   float32 → uint8 (the engine's native corpus)
      │
      ▼
winnex-madhava (agnostic) → top-K + Cauchy-Schwarz proof
```

## Install

```bash
pip install winnex-ai-normalize               # core (numpy + httpx)
pip install winnex-ai-normalize[api]          # + OpenAI-compatible /v1/embeddings
pip install winnex-ai-normalize[all]          # + winnex-madhava
```

## Quick start

```python
from winnex_ai_normalize import EmbeddingNormalizer

norm = EmbeddingNormalizer()                  # reads env / JSON config

# 1. Embed text via the configured provider (OpenAI / Qwen3 / ...)
vecs = norm.vectorize_texts(["hypertension treatment", "diabetes care"])
# vecs: (2, d) float32 L2-normalized

# 2. Or normalize raw vectors (validated, no NaN, unit-norm)
vecs = norm.normalize_vectors(my_embeddings, dim=1024)

# 3. Feed the Madhava engine (agnostic)
u8 = norm.to_corpus(vecs, dim=1024)           # uint8 corpus for build_engine
import winnex_madhava as wm
engine = wm.build_engine(u8, dim=1024, metric="cosine", k=10)
```

## OpenAI-compatible API

```bash
uvicorn winnex_ai_normalize.api.server:app --port 8102
curl -X POST http://localhost:8102/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-Embedding-0.6B", "input": ["text one", "text two"]}'
# → {"data": [{"embedding": [...], "index": 0}, ...], ...}
```

## Provider failover (no fake fallback)

Providers are tried in `provider_order` (JSON/env config). If **all**
fail, a `RuntimeError` is raised with the collected errors — a normalizer
must never fabricate embeddings.

```json
{
  "default_dim": 1024,
  "default_provider": "qwen3",
  "providers": [
    {"name": "qwen3", "base_url": "http://winnex-embedding:8102",
     "model": "/workspace/models/Qwen3-Embedding-0.6B", "priority": 1},
    {"name": "openai", "base_url": "https://api.openai.com/v1",
     "model": "text-embedding-3-small", "api_key_env": "OPENAI_API_KEY", "priority": 2}
  ],
  "provider_order": ["qwen3", "openai"]
}
```

## Consumed by

- **winnex-madhava** (direct — the engine stays agnostic)
- **winnex-tracer** (audit + commitment)
- **Liferay bridges** (tracer-gov-liferay, tracer-med-liferay)
- **Maestro** (winnex-ai-server / winnex-ai-engine) and any external tool

## License

Business Source License 1.1 (BSL 1.1) | pay@winnex.ai |
Winnex Brasil Soluções Empresariais LTDA (CNPJ 58.364.637/0001-47)
