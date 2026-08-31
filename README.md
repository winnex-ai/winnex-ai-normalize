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

### The bottleneck they fix

The recall bottleneck is **not** the search — it is decided **before** the
engine, in the quality of the data that enters it:

```
R_end2end ≈ R_embedding_semantic × R_prefilter(heuristic) × R_bound(tightness) × R_postfilter(exact)
                  ▲                     ▲                        ▲                      ▲
             provider quality      k1_fraction             basis (random vs PCA)   exact on survivors
```

A bad corpus silently distorts recall **while the engine keeps reporting
`bound_violations == 0`** — the proof stays sound on garbage. Real examples:

- **BIGANN corrupted base** (vector order differs from the ground truth):
  the engine returns "correct" top-K for data that means the wrong thing.
  Recall vs the official GT dropped to 0.006; the engine was never at fault.
- **Third-party embedding drift / provider failover** that switches vector
  space mid-corpus: similarities become incomparable, 0 violations still.
- **NaN / zero-variance / dim-shift**: the engine silently produces garbage
  scores with a clean certificate.

The Quality Flags exist to catch these failure classes **at ingest** — to
identify the critic and resolve it **before** the data reaches the Madhava
motor. That is the correction for the bottleneck: the validation moves to the
point where the recall is actually decided.

### How the flags validate (the engine's own proof)

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

### How to use it — the guard at ingest

The flags sit **in front of** `build_engine` and decide whether the data may
reach the motor at all — and if it may, with what configuration:

```python
from winnex_ai_normalize.core.quality import build_quality_engine

# GATE: raises QualityGateError if a FAIL flag fires (bad data never indexes).
engine, report = build_quality_engine(embeddings, dim=384, return_report=True)
# report.verdict == "fail"  → the corpus is blocked (see report.flags).
# report.verdict == "pass"  → the engine is built with the ROUTED config
#                            (basis/k1 chosen from the engine's own proof).
```

This is the correction for the bottleneck: instead of discovering that the
data was bad **after** indexing (when recall already silently degraded), the
quality gate makes the failure **loud and early** — at the point where the
recall is decided, not where it is measured.

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

### Per-dataset presets (config externalizada — motor/normalize agnósticos)

O motor e o normalize são **agnósticos**: a config por dataset vive em presets
JSON (`configs/dataset_<name>.json`), não no código. O operador escolhe o preset
por dataset; o motor apenas aplica os knobs.

```python
from winnex_ai_normalize.core.quality import build_quality_engine, QualityConfig

# 1. Via QualityConfig.from_dataset (carrega o preset JSON com deep-merge)
cfg = QualityConfig.from_dataset("arxiv")        # → pca_corpus, pca_iterations=30
eng, rep = build_quality_engine(X, dim=1536, return_report=True, cfg=cfg)

# 2. Direto no build_quality_engine
eng, rep = build_quality_engine(X, dim=300, return_report=True, dataset="word2vec")
#   → random, k1=0.20 (onde pca_corpus degrada recall — medido 0.677→0.096)
```

Presets incluídos:

| Preset | Quando usar | Config (engine) | Config (quality) |
|---|---|---|---|
| `default` | agnóstico — o roteador decide pela prova | vazio (router) | `probe_pca=true`, `pca_iterations=30` |
| `arxiv` | manifold forte (d=1536, top-1 = 77% var) | `pca_corpus`, `stage1=128`, `pca_iterations=30` | `probe_pca=true` |
| `sift` | manifold moderado (d=128) | `pca_corpus`, `stage1=64` | `probe_pca=true` |
| `word2vec` | manifold fraco — pca DEGRADA recall | `random`, `k1=0.20` | `probe_pca=false` |
| `isotropic` | sem manifold (ruído) — probe desperdiçado | `random`, `k1=0.20` | `probe_pca=false` |

**Por que presets por dataset:** o `pca_corpus` (recomendado para manifold forte)
pode DEGRADAR o recall em dados sem manifold (Word2Vec 0.677→0.096, ProtBERT
1.0→0.134, medidos). E o probe PCA custa ~21-24s em d=1536 mesmo quando não
ajuda — o preset `isotropic` desliga o probe (24.7s→0.06s medido). Externalizar
a config por dataset deixa o operador decidir por dataset, sem tocar no código.

### Validated — the benchmark

`kaggle/bench_quality_flags/benchmark_normalize.py` (Kaggle kernel
`winnex-quality-flags-normalize-1-1-0`) installs this package + winnex-madhava
from PyPI **in isolation**, then asks the 6 binary questions the flags exist
for. Result on Kaggle (v10, winnex-ai-normalize **1.1.0**, exit 0):

| # | Test | Expected | Result |
|---|---|---|---|
| 1 | `valid_manifold` — healthy structured corpus | routes `pca_corpus`, k1=0.05, no FAIL | ✅ PASS |
| 2 | `valid_isotropic` — healthy isotropic corpus | routes `random`, k1=0.20, no FAIL | ✅ PASS |
| 3 | `nan` — NaN in the corpus | `dataset.nan` (FAIL) | ✅ PASS |
| 4 | `degenerate` — zero variance | `dataset.degenerate` (FAIL) | ✅ PASS |
| 5 | `dim_shift` — expected dim ≠ real (BIGANN/offset class) | `embedding.dim_shift` (FAIL) | ✅ PASS |
| 6 | `drift` — batch in a divergent space (`reference=`) | `embedding.provider_drift` (FAIL) | ✅ PASS |

Additional measured corpora (from the earlier full-data benchmark on the same
kernel, real Kaggle datasets + real HuggingFace models): BIGANN proof 99.3%,
arXiv d=1536 random 0% → PCA probe 80.2%, GloVe d=100 proof 94.5% with
recall 1.0, Hacker News OpenAI d=1536 pca-bound 83.4% recall 0.994, MNIST
d=784 proof 100% recall 0.862, and 6 news×model corpora routing `pca` with
recall ≥ 0.99.

### Validity conditions & scope

- The Cauchy-Schwarz proof is **soundness of the pruning**: it guarantees that
  nothing relevant was discarded by the bound. It is **not** a claim of
  `recall = 1.0` end-to-end in the pipeline. Recall also depends on the
  prefilter heuristic (`k1_fraction`) and on the embedding quality — factors
  the proof does not control.
- The bound is valid when the embedding space has an **inner product** and the
  norms are computable (the Cauchy-Schwarz hypotheses). For uint8 raw-byte
  corpora the engine uses L2 on raw bytes — the proof still runs, but the
  [-1,1] embedding domain is lost unless you feed float32 (see
  `quantize_corpus` caveat above).
- The flags catch the **failure classes at ingest** (NaN, degenerate,
  dim-shift, drift, cross-provider, alignment). They do not repair embeddings
  or raise semantic recall — they make the problem **loud and early** so bad
  data never silently distorts a benchmark or a production index.

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
