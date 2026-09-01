# Changelog

All notable changes to `winnex-ai-normalize` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] — 2026-09-01

### Added: `nan_policy` — política de NaN/inf como knob do config (agnóstico)

**Motivação (causa raiz da degradação do Word2Vec Kaggle):** o `pca_corpus`
AMPLIFICA corrupção de dados. Um único NaN/inf no corpus corrompe a matriz de
covariância do PCA (`C = A.T@A/N` → NaN), os autovetores LAPACK ficam NaN, e a
base `set_basis(P1)` projeta em direções NaN → o bound fica "certeiro"
(e(v)≈0, pruned alto) mas o recall despenca. **Medido:** 1 linha NaN em d=300 →
recall pca 0.042 vs random 1.0; com 1 NaN o recall pca foi 0.0000 no teste real.

A política é um **knob do preset JSON** (`quality.nan_policy`), mantendo motor e
normalize agnósticos — a decisão de segurança vive no config, não no código.

- **`QualityConfig.nan_policy`** (novo campo): `"block_pca"` (default) |
  `"block_build"` | `"ignore"`.
  - `block_pca` → corpus com NaN/inf roteia **sempre** `random`/k1=0.20, nunca
    `pca_corpus`; o probe PCA é pulado (early-return, sem mutar o validator).
  - `block_build` → NaN presente é FAIL estrito (bloqueia via QualityGateError
    a menos que `allow_unsafe`).
  - `ignore` → proteção desligada (para medir a degradação / debug).
- **`build_quality_engine`:** quando `nan_policy='block_pca'` e o corpus tem NaN,
  um `basis` forçado via `engine_kwargs` (preset ou chamador) NÃO pode contornar
  a proteção — `basis` é reescrito para `random`.
- **`QualityReport.to_dict`:** expõe `nan_fraction` (telemetria da fração de
  NaN/inf).
- **Presets JSON:** todos (`default`, `arxiv`, `sift`, `isotropic`, `word2vec`)
  declaram `quality.nan_policy="block_pca"`.

**Validação:** 38/38 testes passam (5 novos). Reproduzido na prática: NaN + pca
forçado → engine `RANDOM`, recall 1.0; pca raw sobre NaN → recall 0.0.
