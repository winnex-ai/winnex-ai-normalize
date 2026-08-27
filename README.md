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

## Quality Flags — validating the input, not the search

End-to-end recall is decided **before** the engine, in the quality of the data
that enters it: third-party embedding quality, dataset integrity, and the
prefilter heuristic. A corrupted dataset (e.g. the BIGANN base whose vector
order differs from the ground truth) silently distorts recall while the engine
keeps reporting `bound_violations == 0`.

The quality gate validates the input using **the engine's own Cauchy-Schwarz
proof** — no reimplemented math:

```
Cauchy-Schwarz (upper bound):
    UB(v,q) = ⟨Pv,Pq⟩ + e(v)·e(q)
    UB(v,q) < threshold(K)  ⟹  v is mathematically IMPOSSIBLE to be in the top-K

The validator launches that proof on seed queries and CAPTURES the excluded
set (audit_ids / audit_threshold / pruned_by_bound / pruned_by_prefilter).
The captured set IS the flag response.
```

### Quick start

```python
from winnex_ai_normalize.core.quality import audit_corpus, build_quality_engine
import numpy as np

embeddings = np.random.randn(2000, 384).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# 1. Audit the corpus → flags + the engine's proof coverage + a suggested config.
report = audit_corpus(embeddings, dim=384, n_seed_queries=8)
print(report.summary())            # e.g. quality[PASS] ... proof=0.95 ...
print(report.to_dict()["flags"])   # each flag: code / severity / message
print(report.excluded_seed_set[:3])  # docs the engine PROVED outside top-K

# 2. Build with the routed config (or let the gate raise on FAIL flags).
engine, rep = build_quality_engine(embeddings, dim=384, return_report=True)
res = engine.search(embeddings[0].astype(np.float32))
print(res.indices, res.bound_violations)   # 0 violations — the guarantee
```

### Flags (dataset-agnostic)

| Flag | What it validates | Severity |
|---|---|---|
| `dataset.foldable` | **engine proof coverage** (fraction of the corpus the bound proves outside top-K) | pass/warn |
| `dataset.nan` | NaN/inf (silently corrupts scores, still 0 violations) | **fail** |
| `dataset.degenerate` | zero variance | **fail** |
| `embedding.resolution` | top-1 vs top-K cosine gap (third-party embedding quality) | warn |
| `embedding.provider_drift` | space drift between batches | warn/fail |
| `embedding.cross_provider` | provider switch = incomparable spaces | warn |
| `embedding.dim_shift` | dimension changed between batches | **fail** |
| `embedding.anisotropy` | embedding collapse (known LLM artifact) | warn |
| `integrity.corpus_alignment` | corpus vs reference misaligned (the BIGANN class) | **fail** |

### The router — protecting recall from unstructured data

The validator **decides the engine config from the proof the engine produced**:

| Engine proof coverage | Routed config |
|---|---|
| ≥ 50% | `basis=pca_corpus`, `k1_fraction=0.05` (foldable, bound restored) |
| 20–50% | `basis=random`, `k1_fraction=0.10` (moderate) |
| < 20% | probe with `pca_corpus`; if it proves ≥ 50% → pca, else `random`, `k1=0.20` |
| any | `early_exit=False` always (the P0 fix: early-exit breaks recall at dim ≥ 384) |

### Validated on real data

| Dataset | Engine proof coverage | Routed config |
|---|---|---|
| BIGANN `base.u8bin` (60K×128, L2) | **99.3%** proven outside top-10 | `pca_corpus`, k1=0.05 |
| arXiv OpenAI (d=1536, random) | **0.0%** (Fold Limit: loose bound) | `random`, k1=0.20 |
| arXiv OpenAI (d=1536, PCA probe) | **80.2%** | `pca_corpus`, k1=0.05 |
| Qwen2.5-0.5B quantized (d=896) | 0.0% (14 texts, neighbors in top-10) | `random`, k1=0.20 |

### REST endpoint

```bash
curl -X POST http://localhost:8102/v1/quality/validate \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[...]], "dim": 384, "provider": "qwen3", "dtype": "float32"}'
# → {"verdict": "pass", "flags": [...], "suggested_config": {...}}
```

FAIL flags return HTTP 422 with the full report. Use `allow_unsafe=true` only
when you intend to index anyway.

## Consumed by

- **winnex-madhava** (direct — the engine stays agnostic)
- **winnex-tracer** (audit + commitment)
- **Liferay bridges** (tracer-gov-liferay, tracer-med-liferay)
- **Maestro** (winnex-ai-server / winnex-ai-engine) and any external tool

## License

Business Source License 1.1 (BSL 1.1) | pay@winnex.ai |
Winnex Brasil Soluções Empresariais LTDA (CNPJ 58.364.637/0001-47)
