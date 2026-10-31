# Changelog

All notable changes to `winnex-ai-normalize` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.6] — 2026-10-30

### Added: `certifiable` — o roteador garante a config, mas não a prova

O roteador respondia *"qual config dá mais cobertura"*, não a pergunta do
chamador: *"o motor que vou construir consegue **provar** o descarte?"*.

Medido, com o roteador no comando:

| corpus | roteia | avisa? | `has_fail` | cobertura real |
|---|---|---|---:|---:|
| isotrópico d=384 | `random s1=64` | sim | **False** | **0,00%** |
| isotrópico d=1536 | `random s1=64` | sim | **False** | **0,00%** |
| baixo posto r=16 | `pca s1=192` | não | False | 99,96% |
| baixo posto r=200 | `pca s1=192` | não | False | 99,96% |

Nas duas primeiras linhas o engine devolve **recall 1.000** com **0% de
cobertura de exclusão**. Os quatro relatórios parecem auditoria bem-sucedida;
só dois são certificáveis, e o chamador não tinha como pedir a diferença —
nem campo no `QualityReport`, nem knob no `QualityConfig`.

Campos novos:

- `QualityReport.certifiable: bool` — a config roteada pode provar exclusão?
- `QualityReport.certification_action: str` — por que não, e o que fazer
  (vazio quando pode). Nomeia as saídas concretas: `audit_exhaustive=True`,
  `basis='pca_corpus'`, ou aceitar um selo `pool_only`.
- Ambos expostos em `to_dict()` e no `summary()` (`certifiable=NO`).

O sinal já existia — `bound_fraction_pca` é a própria sonda do bound. Derivar
o booleano dele transforma um WARN que o chamador talvez não leia num campo
que ele tem de tratar. O limiar usa `_FOLD_BOUND_FRAC` (0,50), o mesmo que a
tabela de rotas já usa para "corpus fortemente dobrável", então o booleano e o
roteamento concordam **por construção**, em vez de virarem duas heurísticas
divergentes.

### Added: `QualityConfig.require_certifiable`

Quando `True`, `build_quality_engine` levanta `_CertificationRequired` se a
config roteada não é certificável. **Default `False`** — o top-K exato é
devolvido de qualquer forma e a maioria dos callers não assina selo.

É distinto de `fail_on_recall`:

| knob | guarda | pergunta |
|---|---|---|
| `fail_on_recall` | a resposta | o top-K devolvido é o global? |
| `require_certifiable` | o selo | o selo prova algo sobre o descarte? |

Um config pode ter `recall = 1.0` e `cobertura = 0.0` — exatamente o caso do
corpus isotrópico, que nada na stack detectava antes.

### Tests

6 testes novos em `tests/test_certifiable_2026_10_30.py`: corpus dobrável →
`True`; isotrópico → `False` **com** ação; `to_dict` expõe; o knob bloqueia o
não-certificável e permite o dobrável; e a distinção entre os dois knobs.
Suíte: **61 passed** (55 anteriores + 6 novos).

## [1.4.5] — 2026-10-29

### Fixed: `stage1_dim` do preset não era clampado em `d`

`dataset_arxiv.json` fixa `stage1_dim = 128`. Os dois probes do validador já
faziam `min(probe, d)`, mas a atribuição que o preset JSON controla
(`report.stage1_dim = int(route_stage1)`) não fazia.

O que isso causava: num corpus de `d < 128`, o preset passava `stage1_dim=128`
adiante. Desde a validação do Madhava (2026-10-29, `stage1_dim > dim` recusado
com `invalid_argument`), o preset passaria a levantar. Antes dela, o motor
aceitava e produzia um bound INVÁLIDO — medido num corpus de d=64:
`UB/exato = 2,008` e `bound_violations > 0`, com o certificado atestando um
descarte que o motor não provou.

Correção: `stage1_dim = min(route_stage1, d)`, com o valor original registrado
em `report.metrics["stage1_dim_clamped_from"]` para não sumir.

O clamp aqui não é degradação silenciosa: `stage1_dim > d` não tem significado
matemático — não existem mais de `d` vetores ortonormais em ℝ^d. Truncar para
`d` é a maior projeção que ainda respeita o bound, e é o que um preset
"stage1 = 128" quer dizer num corpus menor.

Tests: 2 novos (clamp aplicado com o valor original registrado, e o controle
onde o preset cabe em d e nada muda). Suite: 48 passed.

## [1.4.4] — 2026-10-29

### Fixed: `fail_on_recall` — o gate não podia bloquear por recall

O gate tratava dois sinais de qualidade de forma assimétrica:

  F_RESOLUTION (qualidade do embedding)  escala para FAIL via `fail_on_resolution`
  F_RECALL     (recall abaixo do piso)   era WARN hardcoded

O docstring de `_flag_recall_shortfall` dizia que a política de bloqueio
"pertence ao config (route_rules / recall_floor / fail thresholds)" — mas o knob
que a materializaria nunca existiu. `QualityValidator.__init__` aceitava
`fail_on_resolution` e nunca `fail_on_recall`, e `QualityConfig` não tinha o
campo. O gate não podia bloquear por recall, por construção.

Por que isso importa: o certificado do Madhava atesta que nada foi descartado
indevidamente DENTRO do pool. Com recall em 0,50 e o gate deixando passar, o
operador recebe `bound_violations = 0` sobre um top-K que não é o global, e
nenhum sinal bloqueia.

Adicionado, espelhando `fail_on_resolution` de ponta a ponta:

  QualityConfig.fail_on_recall       (default False, compatível com a frente)
  QualityConfig.from_dict            lê do preset JSON do dataset
  QualityValidator.__init__          aceita o parâmetro
  _flag_recall_shortfall             escala para FAIL quando abaixo do piso

Só perda MATERIAL bloqueia: recall abaixo do piso. O caso pool_only com
recall < 1,0 mas >= piso continua WARN mesmo com o knob ligado — bloquear ali
seria bloquear o default do próprio motor.

Verificado nos dois sentidos com piso impossível (1,01):
  fail_on_recall=True  -> QualityGateError, FAIL: dataset.recall_not_guaranteed
  fail_on_recall=False -> build succeeds (controle)

### Fixed: versão dessincronizada

`pyproject.toml` declarava 1.4.3 e `winnex_ai_normalize.__version__` dizia
1.4.2 desde a 1.4.3 — o mesmo defeito de duas fontes de verdade corrigido no
winnex-tracer. Ambos agora em 1.4.4.

Tests: 4 novos em tests/test_quality.py (knob existe, propaga do config,
escala para FAIL, e o controle não bloqueia). Suite: 53 passed.

## [1.4.3] — 2026-09-07

### Fixed: certificado vem do `search_audited`, não do `search`

O caminho de captura do certificado usava `search()`, que não popula
`audit_ids` (é o caminho rápido, sem coleta por documento). O resultado era um
certificado sempre vazio assinado como se fosse válido. Agora usa
`search_audited`, que materializa o registro por documento excluído.

## [1.4.2] — 2026-09-06

### Changed: cleanup — removed dead config knobs

Removed `fold_bound_frac` / `fold_bound_frac_low` / `resolution_gap_warn` from
`QualityConfig` and the default preset JSON. These fields were declared and
loaded but never read — the validator used the module constants directly. The
validator now references the module constants explicitly. No behavior change
(all 49 tests pass).

## [1.4.1] — 2026-09-04

### Fixed: `build_quality_engine` crash when the route picks `pca_corpus` without a PCA probe

The `F_FOLDABLE` flag message formatted `{pca_proved:.0%}`. When the route
table picks `pca_corpus` via its FIRST rule (`random bound_fraction >= 0.50`),
the PCA probe never runs and `pca_proved` is `None` — so the format raised
`TypeError: unsupported format string passed to NoneType.__format__`. This
crashed `build_quality_engine` on ANY strong manifold (measured: d=64/100/128)
with or without `return_report=True`, blocking the production normalize→madhava
flow. The message now treats a missing probe as `nan`, matching the existing
`pca_recall` guard.

Regression test: `test_foldable_flag_pca_without_probe_no_crash` (strong
low-dim manifold → pca route without probe → no crash, flag well-formed).
49/49 tests pass.

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
