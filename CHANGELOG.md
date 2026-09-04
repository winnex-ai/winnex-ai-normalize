# Changelog

All notable changes to `winnex-ai-normalize` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] — 2026-09-04

### Changed: routing policy moved from CODE to CONFIG (`route_rules`) — the router is now agnostic

**Motivação (diretriz):** *"o sistema deve ser agnóstico; a responsabilidade é do
config, jamais ajustar para um dataset no código."* O router do `QualityValidator`
decidia basis/k1/stage1 por **ifs fixos no código** (`bound_frac >= 0.50 → pca/k1=0.05`,
`0.20–0.50 → random/k1=0.10`, `< 0.20 → probe ...`). Trocar política exigia editar
código — o oposto de agnóstico. A validação de recall do 1.3.0 adicionou MAIS
regras fixas (reversão de pca por recall). Esta versão move a **decisão** para o
config e mantém o código como **medidor + aplicador**.

**O que NÃO mudou (é sinal legítimo do código):** o validator MEDE e EXPÕE —
`bound_fraction`, `bound_fraction_pca`, `seed_recall_vs_exact`,
`pool_only_frac`, `recall_guarantee`, `prefilter_fraction`, flags de integridade.
Isto é agnóstico: vale para qualquer dado.

**Mudanças:**

1. **`QualityConfig.route_rules`** (nova): a TABELA de decisão que mapeia sinais
   medidos → rota do motor (basis / k1_fraction / stage1_dim). Vive no preset
   JSON (`quality.route_rules` no `dataset_default.json`). Formato:
   `[{"when": {metric: ">= 0.50"}, "route": {basis, k1_fraction}}, ..., {"fallback": {...}}]`.
   A primeira regra cujo `when` casa vence; o fallback cobre o resto. Suporta
   `>=, >, <=, <, ==` sobre as chaves de `report.metrics`. A tabela default
   **reproduz o comportamento histórico exato** → callers existentes não mudam.
2. **`QualityValidator` vira medidor + aplicador**: mede os sinais (probes random
   + PCA) e aplica `_match_route(report.metrics, route_rules, recall_floor)`. Os
   `if bound_frac...` de decisão foram REMOVIDOS (0 restantes). O código não
   decide rota — aplica a política do config.
3. **`recall_floor` deixa de reverter no código**: o 1.3.0 revertia pca→random no
   código quando recall < floor. Agora a flag `dataset.recall_not_guaranteed` é
   um SINAL (WARN). Reverter/evitar pca por recall baixo é uma regra explícita
   na `route_rules` do config (se o operador quiser), não código.
4. **`nan_blocked_route`** (nova, no config): quando `nan_policy=block_pca`, a
   rota de segurança aplicada (random/k1=0.20) vem do config, não de um valor
   fixo no código.
5. **Knobs de probe externalizados**: `stage1_probe_random` (era `min(64,d)`),
   `stage1_probe_pca` (era `min(192,d)`), `probe_pca_dim_gate` (era `d>64`).
6. **Report ↔ engine consistente**: o `build_quality_engine` sincroniza o report
   com a config final aplicada (basis/k1/stage1/quant) — corrige o bug onde o
   report dizia stage1=64 mas o engine usava 128 (preset), e o caso onde o report
   dizia pca_corpus mas o engine reutilizado era random.

**Validação:** 48/48 testes passam (4 novos: route_rules default reproduz
comportamento; route_rules muda a rota via config; `_match_route` condições;
recall_shortfall é sinal WARN, não decisão). Benchmark Kaggle (dry-run) PASS com
a tabela default.

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
