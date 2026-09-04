# Changelog

All notable changes to `winnex-ai-normalize` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] — 2026-09-04

### Added: automatic agnostic `scan_int8` — validated by real recall, not heuristics

> Entrou no commit `0bc0d2a` (2026-09-03) mas ficou sem bump/changelog; incluído
> no 1.3.0 para que a versão publicada documente a feature completa.

O motor winnex-madhava 1.9.11 adicionou `scan_int8` (~1.4-3× mais rápido no scan
do bound). A segurança NÃO depende do basis — depende da dimensão/distribuição:
o erro de quantização int8 reordena o pool quando o gap entre candidatos é menor
(medido: basis random em d≥384 degrada recall 1.0→0.5; d=128 é seguro; pca_corpus
d=1536 é seguro pois concentra a energia). Nenhuma regra de basis/dim captura
isso de forma confiável.

**Solução agnóstica:** o `build_quality_engine` TENTA scan_int8 e VALIDA por
recall real em seed queries (search vs search_exact do MESMO engine). Se o
recall cai abaixo de 0.95, rebuild sem scan_int8 (float32 exato). O sistema
decide por medição, válido para qualquer basis/dimensão. O preset JSON ou o
chamador podem forçar explicitamente via `engine_kwargs['scan_int8']`.
`cfg_match`: o probe do QualityValidator nunca usa scan_int8 — quando o build
final quer scan_int8, o probe não serve e um engine novo é buildado. Validado no
arXiv real: d=384/1536 random → scan_int8 desligado (recall 1.0); d=384/1536
pca_corpus → scan_int8 mantido (recall 1.0).

### Added: recall validation — the router is validated by REAL recall, not bound coverage alone

**Motivação (o colapso silencioso, `ANALISE_COLAPSO_GARANTIA_WITNESS_20260903.md`):**
o roteador decidia a config SÓ pela cobertura da prova (`pruned_by_bound / N`).
Uma config pode ter `bound_violations == 0` e ainda ser `pool_only` com
recall < 1.0 — o top-K retornado é o melhor DENTRO do pool pós-filtro, não o
top-K GLOBAL. **Medido:** manifold fraco word2vec-like (d=300, ncomp≈d), random
k1=0.05 → recall 0.75 vs o próprio `search_exact` do motor, com viol=0 e
20/20 queries `pool_only`. O motor (1.9.10) já expõe `recall_guarantee`
(`pool_only`/`exact_global`), mas o normalize não o lia nem validava o recall.

**Mudanças:**

1. **`QualityValidator._run` agora valida recall REAL por seed query:** compara
   `search()` vs `search_exact()` do MESMO motor (o ceiling global exato; custo
   ~1.1-1.3× o `search` nas seed queries — N=20k d=384: 6.5ms vs 5.7ms). Não
   reimplementa nada — usa a operação exata do motor.
2. **`QualityReport` expõe a telemetria de escopo:** `seed_recall_vs_exact`
   (recall@K médio), `pool_only_frac` (fração de queries cujo top-K NÃO é
   global) e `recall_guarantee_counts` (`{exact_global, pool_only}`). Na rota
   PCA validada, o report reflete o recall da rota ESCOLHIDA (não o da random
   descartada).
3. **Probe PCA validado por recall, não por bound:** se o PCA PROVA ≥50% mas o
   recall real cai abaixo do floor, o PCA está capturando corrupção/ruído
   (e(v)≈0 é manifold falso) → o roteador REVERTE para random/k1 alto em vez de
   sugerir pca_corpus. `_flag_recall_shortfall` emite a flag.
4. **Nova flag `dataset.recall_not_guaranteed`** (`F_RECALL`): emitida quando a
   rota sugerida é pool_only com recall < floor. Severidade: **WARN** quando o
   recall baixo é limitação física do manifold (não bloqueia — o preset
   word2vec já convive com isso); **FAIL** quando uma rota pca escolhida DEGRADA
   o recall (o roteador não deve sugerir base que perde recall).
5. **`QualityConfig.recall_floor`** (novo knob agnóstico, default 0.95), viaja no
   preset JSON (`quality.recall_floor` no `dataset_default.json`).
6. **`build_quality_engine(return_report=True)`** expõe `seed_recall_vs_exact_final`
   e `recall_guarantee_final` do engine EFETIVAMENTE retornado (após reversão de
   basis / toggles de scan_int8) e emite WARN se a config final tiver recall <
   floor — o operador nunca recebe um motor com recall<1.0 e viol=0 sem ver.

**Validação:** 5 novos testes (`test_*_recall*`, `test_weak_manifold*`,
`test_strong_manifold_recall_holds_pca`, `test_recall_floor_configurable`). O
manifold fraco agora expõe recall 0.75 + flag em vez de PASS silencioso; o
manifold forte continua pca_corpus com recall validado 1.0 (sem regressão).

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
