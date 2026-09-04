"""
winnex-ai-normalize — quality flags tests (the MOTOR's own validation).

The quality flag is produced by the Madhava engine's OWN Cauchy-Schwarz proof:
for each seed query, the engine emits the per-document upper bound and the set
of documents it PROVES to be outside the top-K (UB < threshold). The validator
LAUNCHES that native validation and CAPTURES the excluded seed set — the
captured set IS the flag response.

This validates dataset integrity (NaN, degenerate), embedding-set drift
(provider switch, dimension shift, space drift) and the engine-config router
(basis / k1_fraction / early_exit) WITHOUT reimplementing the math.

Run:  python -m pytest winnex_ai_normalize/tests/ -v
"""
import os

import numpy as np
import pytest

from winnex_ai_normalize.core.quality import (
    QualityConfig,
    QualityGateError,
    QualityValidator,
    audit_corpus,
    build_quality_engine,
    check_embedding_drift,
    EmbeddingFingerprint,
    FAIL,
    WARN,
    F_NAN,
    F_DEGENERATE,
    F_FOLDABLE,
    F_DRIFT,
    F_CROSS_PROVIDER,
    F_DIM_SHIFT,
    F_COLLAPSE,
    F_RECALL,
)


def _embeddings(n=600, d=128, seed=0):
    """Realistic embedding-like data: low-rank manifold + moderate noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 12)) @ rng.standard_normal((12, d))
    X = X + 0.15 * rng.standard_normal((n, d))
    X = X.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _isotropic(n=400, d=128, seed=4):
    """Isotropic (no-manifold) unit vectors — the hard case for the bound."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# ---------------------------------------------------------------------------
# The flag IS the motor's proof coverage
# ---------------------------------------------------------------------------
def test_proof_coverage_is_the_flag():
    """A manifold corpus → the engine PROVES most of the corpus is out of
    top-K → the flag response is a high proof coverage → pca_corpus route."""
    X = _embeddings()
    rep = audit_corpus(X, n_seed_queries=6)
    # the Cauchy-Schwarz proof fired and was captured
    assert rep.metrics.get("proof_ratio", 0) > 0.5
    assert rep.metrics.get("bound_fraction", 0) >= 0.5
    # the excluded seed set is non-empty (docs PROVEN outside top-K)
    assert len(rep.excluded_seed_set) > 0
    assert rep.excluded_seed_set[0]["upper_bound"] is not None
    # foldable → pca_corpus
    assert rep.basis == "pca_corpus"
    assert rep.k1_fraction <= 0.10
    assert any(f.code == F_FOLDABLE for f in rep.flags)


def test_high_dim_loose_bound_routes_to_larger_k1():
    """The genuine non-foldable case: a corpus where the bound CANNOT prove
    pruning (loose bound → ~0% proof coverage). This is the arXiv d=1536
    with basis=random (measured: 0% proof, 95% prefilter). We emulate it with
    an isotropic sphere in HIGH dimension, where the random projection is too
    loose to prove exclusion → the PREFILTER is the real recall gate →
    k1_fraction raised, basis=random."""
    X = _isotropic(n=300, d=512)
    rep = audit_corpus(X, n_seed_queries=6, probe_pca=False)
    assert rep.basis == "random"
    assert rep.k1_fraction >= 0.15
    assert rep.metrics.get("bound_fraction", 1.0) < 0.5
    assert any(f.code == F_FOLDABLE for f in rep.flags)


def test_excluded_seed_set_is_deterministic():
    X = _embeddings()
    r1 = audit_corpus(X, n_seed_queries=6)
    r2 = audit_corpus(X, n_seed_queries=6)
    ids1 = [d["doc_id"] for d in r1.excluded_seed_set]
    ids2 = [d["doc_id"] for d in r2.excluded_seed_set]
    assert ids1 == ids2  # same seed → same captured set


# ---------------------------------------------------------------------------
# Dataset integrity flags
# ---------------------------------------------------------------------------
def test_nan_fails():
    X = _embeddings()
    X[3, 5] = np.nan
    rep = audit_corpus(X)
    assert rep.has_fail
    codes = {f.code for f in rep.flags}
    assert F_NAN in codes


def test_degenerate_fails():
    X = np.full((50, 128), 3.0, dtype=np.float32)
    rep = audit_corpus(X)
    codes = {f.code for f in rep.flags}
    assert F_DEGENERATE in codes


def test_uint8_traps_flagged():
    X = np.arange(128 * 200, dtype=np.uint8).reshape(200, 128)
    rep = audit_corpus(X)
    assert rep.metric == "l2"
    codes = {f.code for f in rep.flags}
    assert F_COLLAPSE in codes


def test_empty_fails():
    rep = audit_corpus(np.zeros((0, 128), dtype=np.float32))
    assert rep.has_fail


def test_dimension_mismatch_fails():
    X = _embeddings(d=128)
    rep = audit_corpus(X, dim=64)
    assert rep.has_fail
    assert any(f.code == F_DIM_SHIFT for f in rep.flags)


# ---------------------------------------------------------------------------
# Embedding-set drift
# ---------------------------------------------------------------------------
def test_cross_provider_flagged():
    prev = EmbeddingFingerprint(provider="qwen3", dim=128, centroid=np.zeros(128))
    new = EmbeddingFingerprint(provider="openai", dim=128, centroid=np.zeros(128))
    flags = check_embedding_drift(prev, new)
    assert any(f.code == F_CROSS_PROVIDER for f in flags)


def test_dim_shift_fails():
    prev = EmbeddingFingerprint(provider="qwen3", dim=128, centroid=np.zeros(128))
    new = EmbeddingFingerprint(provider="qwen3", dim=256, centroid=np.zeros(256))
    flags = check_embedding_drift(prev, new)
    assert any(f.code == F_DIM_SHIFT and f.severity == FAIL for f in flags)


def test_space_drift_fails():
    prev = EmbeddingFingerprint(provider="qwen3", dim=128,
                                centroid=np.ones(128) / np.sqrt(128))
    new = EmbeddingFingerprint(provider="qwen3", dim=128,
                               centroid=-np.ones(128) / np.sqrt(128))
    flags = check_embedding_drift(prev, new)
    assert any(f.code == F_DRIFT and f.severity == FAIL for f in flags)


def test_same_batch_no_drift():
    c = np.ones(128) / np.sqrt(128)
    prev = EmbeddingFingerprint(provider="qwen3", dim=128, centroid=c.copy())
    new = EmbeddingFingerprint(provider="qwen3", dim=128, centroid=c.copy() * 0.9999)
    flags = check_embedding_drift(prev, new)
    assert not any(f.severity == FAIL for f in flags)


# ---------------------------------------------------------------------------
# Gate + router (build_quality_engine)
# ---------------------------------------------------------------------------
def test_gate_blocks_unsafe():
    X = _embeddings()
    X[0, 0] = np.nan
    with pytest.raises(QualityGateError):
        build_quality_engine(X, dim=128)


def test_gate_allow_unsafe():
    X = _embeddings()
    X[0, 0] = np.nan
    eng = build_quality_engine(X, dim=128, allow_unsafe=True)
    assert eng is not None
    r = eng.search(X[1].astype(np.float32))
    assert r.bound_violations == 0


def test_build_returns_report():
    X = _embeddings(d=128)
    eng, rep = build_quality_engine(X, dim=128, return_report=True)
    assert rep.n == len(X)
    assert rep.dim == 128
    # router forces early_exit False (the P0 fix) and preserves cosine
    assert rep.early_exit is False
    assert rep.metric == "cosine"
    # recall is preserved by the routed config
    q = X[0].astype(np.float32)
    re = eng.search_exact(q)
    r = eng.search(q)
    rec = sum(1 for j in r.indices if j in re.indices) / 10
    assert rec >= 0.9
    assert r.bound_violations == 0


def test_real_arxiv_audit():
    """The engine's own Cauchy-Schwarz validation on the REAL arXiv embeddings
    (d=1536).

    With the RANDOM basis the bound is loose at d=1536 (measured: proves 0%,
    prefilter cuts 95%) → the router conservatively raises k1_fraction. With
    the PCA PROBE the engine proves 80%+ (the foldable case) → pca_corpus.
    """
    path = "/home/wnnx_user/zenodo/arxiv_100k.npy"
    if not os.path.exists(path):
        pytest.skip("arxiv_100k.npy not present — skipping real-data test")
    a = np.load(path, mmap_mode="r")
    X = np.ascontiguousarray(a[:4000]).astype(np.float32)

    # Without the probe: the random basis proves nothing at d=1536
    # (the documented Fold Limit) → k1 raised, excluded set empty.
    rep = audit_corpus(X, dim=1536, n_seed_queries=4, probe_pca=False)
    assert rep.metrics.get("bound_fraction", 1.0) < 0.1
    assert rep.k1_fraction >= 0.15
    assert rep.basis == "random"

    # With the probe: the engine's OWN pca_corpus build proves 80%+ → the
    # corpus is foldable under a tight basis → pca_corpus, k1 small.
    rep2 = audit_corpus(X, dim=1536, n_seed_queries=4, probe_pca=True)
    assert rep2.metrics.get("bound_fraction_pca", 0) >= 0.5
    assert rep2.basis == "pca_corpus"
    assert rep2.k1_fraction <= 0.10
    # the captured exclusions are mathematically sound: UB < threshold
    for rec in rep2.excluded_seed_set[:20]:
        assert rec["upper_bound"] <= rec["threshold"] + 1e-4


# ---------------------------------------------------------------------------
# Phase 2 / GAIA: embedding-quality floor (query resolution + golden set)
# ---------------------------------------------------------------------------

def _blurry_corpus(n=2000, d=128, seed=1):
    """A 'blurry photo' corpus: all vectors nearly identical (low semantic
    separation), so top-1 vs top-K exact cosine gap is tiny."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X = 0.999 * X[0] + 0.001 * rng.standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_query_resolution_exposed():
    """Phase 2a: the per-query top-1 vs top-K resolution is exposed on the
    report (the 'blurry photo' is now measurable per query)."""
    X = _blurry_corpus()
    rep = audit_corpus(X, dim=128, k=10)
    assert len(rep.query_resolution) > 0
    for rec in rep.query_resolution:
        assert "seed_query" in rec and "gap" in rec
    # blurry corpus -> low resolution (below the 0.10 WARN threshold)
    assert rep.metrics["top1_topk_gap"] < 0.10


def test_fail_on_resolution_default_off():
    """Phase 2b: fail_on_resolution defaults to False — the current behavior
    is preserved (WARN, not FAIL) on a blurry corpus."""
    X = _blurry_corpus()
    rep = audit_corpus(X, dim=128, k=10)
    res = [f for f in rep.flags if f.code == "embedding.resolution"]
    assert res and res[0].severity == WARN
    assert not rep.has_fail
    # build_quality_engine does NOT raise by default
    eng = build_quality_engine(X, dim=128, k=10)
    assert eng is not None


def test_fail_on_resolution_enabled_blocks():
    """Phase 2b: with fail_on_resolution=True the same corpus escalates to
    FAIL and build_quality_engine raises QualityGateError."""
    X = _blurry_corpus()
    cfg = QualityConfig(fail_on_resolution=True)
    rep = audit_corpus(X, dim=128, k=10, cfg=cfg)
    res = [f for f in rep.flags if f.code == "embedding.resolution"]
    assert res and res[0].severity == FAIL
    assert rep.has_fail
    with pytest.raises(QualityGateError):
        build_quality_engine(X, dim=128, k=10, cfg=cfg)


def test_golden_model_card_synthetic():
    """Phase 2c: the golden-set evaluator produces a RetrievalModelCard and
    the contract check distinguishes a weak embedder from a strong one."""
    from winnex_ai_normalize.core.golden import eval_provider, check_contract

    class WeakEmbedder:
        name = "test-weak"
        def embed(self, texts):
            v = np.zeros((len(texts), 64), dtype=np.float32)
            for i, t in enumerate(texts):
                for ch in t:
                    v[i, ord(ch) % 64] += 1.0
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    class StrongEmbedder:
        name = "test-strong"
        def embed(self, texts):
            import collections
            v = np.zeros((len(texts), 256), dtype=np.float32)
            for i, t in enumerate(texts):
                toks = collections.Counter(t[j:j+3] for j in range(max(1, len(t)-2)))
                for tok, cnt in toks.items():
                    v[i, abs(hash(tok)) % 256] = cnt
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    weak = eval_provider(WeakEmbedder(), "test-weak", domain="legal", k=3)
    strong = eval_provider(StrongEmbedder(), "test-strong", domain="legal", k=3)
    assert weak.bound_violations == 0      # the proof still holds on both
    assert strong.bound_violations == 0
    assert strong.semantic_recall >= weak.semantic_recall
    # the weak embedder fails the default contract floor, the strong passes
    assert not check_contract(weak)
    assert check_contract(strong)


# ---------------------------------------------------------------------------
# Dataset presets (config externalizada por dataset — motor/normalize agnósticos)
# ---------------------------------------------------------------------------

def test_load_dataset_preset_default_agnostic():
    """O preset default NÃO força basis/k1/stage — o roteador decide. Apenas
    os fixes universais (pca_iterations=30, early_exit=False) são default."""
    from winnex_ai_normalize.core.quality import load_dataset_preset
    d = load_dataset_preset("default")
    assert d["engine"].get("basis") is None          # agnóstico
    assert d["engine"].get("k1_fraction") is None    # agnóstico
    assert d["engine"].get("stage1_dim") is None     # agnóstico
    assert d["engine"].get("pca_iterations") == 30   # fix universal
    assert d["engine"].get("early_exit") is False    # fix universal


def test_quality_config_from_dataset_arxiv():
    """O preset 'arxiv' (manifold forte) roteia pca_corpus + pca_iterations=30."""
    cfg = QualityConfig.from_dataset("arxiv")
    assert cfg.engine_kwargs.get("basis") == "pca_corpus"
    assert cfg.engine_kwargs.get("pca_iterations") == 30
    assert cfg.engine_kwargs.get("stage1_dim") == 128


def test_quality_config_from_dataset_isotropic_no_probe():
    """O preset 'isotropic' (sem manifold) desliga o probe PCA — economiza o
    build ~21-24s em d=1536 quando o PCA não ajuda."""
    cfg = QualityConfig.from_dataset("isotropic")
    assert cfg.probe_pca is False
    assert cfg.engine_kwargs.get("basis") == "random"
    assert cfg.engine_kwargs.get("k1_fraction") == 0.20


def test_quality_config_from_dataset_unknown_falls_back_to_default():
    """Dataset desconhecido → preset default agnóstico (sem crash)."""
    cfg = QualityConfig.from_dataset("nao_existe")
    assert cfg.probe_pca is True
    # default é agnóstico: sem basis/k1 forçados
    assert cfg.engine_kwargs.get("basis") is None


def test_build_quality_engine_honors_forced_basis():
    """BUG FIX (2026-08-31): o cfg_match antigo tinha erro de precedência
    (o ternário `rdim == dim if dim is not None else True and ...` era
    interpretado como `(rdim==dim) if (dim is not None) else ...`, ignorando
    as verificações seguintes) — o basis forçado via engine_kwargs era
    SILENCIOSAMENTE IGNORADO e o motor do probe (random) era reutilizado.

    Com a correção, forçar basis='pca_corpus' deve aplicar a base PCA REAL
    (verificável pelo pruned_by_bound alto, pois config().basis não reflete
    set_basis no build_engine float32+pca)."""
    X = _embeddings(n=3000, d=384, seed=0)
    eng, rep = build_quality_engine(
        X, dim=384, k=10, return_report=True,
        engine_kwargs=dict(basis="pca_corpus", metric="cosine", normalize_input=True))
    q = X[0].astype(np.float32)
    r = eng.search(q)
    # A base PCA REAL foi aplicada → pruned_by_bound alto (>50%)
    pb_frac = r.pruned_by_bound / len(X)
    assert pb_frac > 0.5, f"basis pca_corpus não aplicado (pb={pb_frac:.2f})"
    # recall preservado e 0 violações
    ex = set(eng.search_exact(q).indices)
    rec = sum(1 for i in r.indices if i in ex) / 10
    assert rec >= 0.9
    assert r.bound_violations == 0


def test_build_quality_engine_dataset_preset_applied():
    """build_quality_engine(dataset='arxiv') aplica o preset (pca_corpus,
    pca_iterations=30, stage1=128) sem o chamador precisar forçar kwargs."""
    X = _embeddings(n=3000, d=384, seed=0)
    eng, rep = build_quality_engine(X, dim=384, k=10, return_report=True, dataset="arxiv")
    assert eng.config().pca_iterations == 30
    assert eng.config().stage1_dim == 128
    q = X[0].astype(np.float32)
    r = eng.search(q)
    assert r.bound_violations == 0


# ---------------------------------------------------------------------------
# nan_policy (knob do config): NaN nunca roteia pca_corpus (causa raiz Word2Vec)
# ---------------------------------------------------------------------------
def _embeddings_with_nan(n=600, d=128, seed=0, n_nan=1):
    X = _embeddings(n=n, d=d, seed=seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(n, n_nan, replace=False)
    X[idx, 0] = np.nan
    return X


def test_nan_policy_block_pca_routes_random():
    """nan_policy='block_pca' (default): corpus com NaN → roteador NUNCA
    escolhe pca_corpus; força random + k1 alto, e desliga o probe PCA."""
    X = _embeddings_with_nan()
    rep = audit_corpus(X, dim=128, n_seed_queries=4)
    # dataset.nan é FAIL e o roteador não sugere pca
    assert rep.has_fail
    assert any(f.code == F_NAN for f in rep.flags)
    assert rep.basis == "random"
    assert rep.k1_fraction >= 0.15
    # nan_fraction exposto na telemetria
    assert rep.metrics.get("nan_fraction", 0) > 0


def test_nan_policy_block_pca_overrides_forced_pca():
    """Mesmo quando o chamador FORÇA basis='pca_corpus' com NaN presente,
    a política 'block_pca' impede o contorno — o motor buildado é random."""
    X = _embeddings_with_nan()
    eng, rep = build_quality_engine(
        X, dim=128, k=10, return_report=True, allow_unsafe=True,
        engine_kwargs=dict(basis="pca_corpus", metric="cosine", normalize_input=True))
    # config().basis é RANDOM (sem PCA aplicado via set_basis)
    assert "RANDOM" in str(eng.config().basis)
    # o report não sugere pca
    assert rep.basis == "random"
    assert rep.k1_fraction >= 0.15


def test_nan_policy_block_build_blocks():
    """nan_policy='block_build': NaN presente → QualityGateError (FAIL estrito),
    mesmo com o preset forçando pca_corpus."""
    X = _embeddings_with_nan()
    cfg = QualityConfig(nan_policy="block_build")
    with pytest.raises(QualityGateError):
        build_quality_engine(X, dim=128, k=10, cfg=cfg)
    # allow_unsafe=True prossegue
    eng, rep = build_quality_engine(X, dim=128, k=10, cfg=cfg, allow_unsafe=True,
                                    return_report=True)
    assert eng is not None
    assert rep.has_fail


def test_nan_policy_ignore_does_not_route_random():
    """nan_policy='ignore': a proteção é desligada — o roteador pode sugerir
    pca_corpus mesmo com NaN (para medir a degradação / debug)."""
    X = _embeddings_with_nan()
    cfg = QualityConfig(nan_policy="ignore")
    rep = audit_corpus(X, dim=128, n_seed_queries=4, cfg=cfg)
    # continua FAIL por dataset.nan, mas o basis pode ser pca (sem override)
    assert rep.has_fail
    assert any(f.code == F_NAN for f in rep.flags)


def test_nan_policy_from_dataset_preset():
    """O knob nan_policy viaja no preset JSON (config agnóstica)."""
    cfg = QualityConfig.from_dataset("default")
    assert cfg.nan_policy == "block_pca"
    cfg = QualityConfig.from_dataset("word2vec")
    assert cfg.nan_policy == "block_pca"
    # preset desconhecido → default (block_pca)
    cfg = QualityConfig.from_dataset("nao_existe")
    assert cfg.nan_policy == "block_pca"


# ---------------------------------------------------------------------------
# RECALL VALIDATION (2026-09-04): validate the router by REAL recall (search vs
# the motor's own search_exact), not by bound coverage alone. Expose
# recall_guarantee (pool_only | exact_global) on the report. This is the honest
# fix for the silent collapse: a config with viol=0 can be pool_only with
# recall < 1.0 (measured: word2vec-like weak manifold → 0.75 with pool_only).
# ---------------------------------------------------------------------------

def _weak_manifold(n=941, d=300, ncomp=200, seed=0):
    """word2vec-like weak manifold (ncomp ≈ d): the bound on a random basis is
    loose (proves ~0%), the prefilter is the real recall gate, and the returned
    top-K is pool_only — the documented silent-collapse regime."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, ncomp)) @ rng.standard_normal((ncomp, d))
    X = X.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def test_report_exposes_seed_recall_and_guarantee():
    """The report carries the honest scope of the suggested config: the mean
    seed recall (search vs the motor's own search_exact) and the pool_only /
    exact_global counts. Before this fix the router reported only bound
    coverage — recall<1.0 with viol=0 was invisible."""
    X = _weak_manifold()
    rep = audit_corpus(X, dim=300, n_seed_queries=8)
    m = rep.metrics
    # the telemetry exists
    assert "seed_recall_vs_exact" in m
    assert "pool_only_frac" in m
    assert "recall_guarantee_counts" in m
    # the counts sum to the number of seed queries
    c = m["recall_guarantee_counts"]
    assert c["exact_global"] + c["pool_only"] == rep.metrics["seed_queries"]
    # the values are real numbers (not nan)
    assert m["seed_recall_vs_exact"] == m["seed_recall_vs_exact"]  # not nan


def test_weak_manifold_recall_exposed_pool_only():
    """On a weak manifold the recall of the random baseline is measurably < 1.0
    and pool_only — the silent collapse is now VISIBLE, not hidden behind
    viol=0. The router should either pick the PCA route that restores recall
    (reporting its recall) or flag the random route."""
    X = _weak_manifold()
    rep = audit_corpus(X, dim=300, n_seed_queries=10)
    m = rep.metrics
    # either the pca route was chosen (recall 1.0, validated) or the random
    # route was flagged. In BOTH cases pool_only_frac must be reported.
    assert "pool_only_frac" in m
    if rep.basis == "random":
        # a random route on this manifold is pool_only with recall < floor
        assert rep.k1_fraction >= 0.15
        rflag = [f for f in rep.flags if f.code == F_RECALL]
        assert rflag, "random route on a weak manifold must be flagged"
    else:
        # pca route: its recall must be high (validated, not assumed)
        assert rep.basis == "pca_corpus"
        assert m.get("seed_recall_vs_exact", 0) >= 0.9


def test_recall_flag_on_pool_only_route():
    """When the router keeps a pool_only random route with recall < floor, the
    flag dataset.recall_not_guaranteed is emitted (WARN — weak manifold is a
    physical limit, not corruption)."""
    X = _weak_manifold()
    rep = audit_corpus(X, dim=300, n_seed_queries=10, probe_pca=False)
    assert rep.basis == "random"
    rflags = [f for f in rep.flags if f.code == F_RECALL]
    assert rflags, "a pool_only random route with recall < 1.0 must be flagged"
    assert rflags[0].severity == WARN  # not FAIL: physical limit, not corruption


def test_strong_manifold_recall_holds_pca():
    """Regression: a strong manifold still routes pca_corpus and the validated
    recall is 1.0 — the recall gate did NOT break good routing."""
    X = _embeddings(n=3000, d=128, seed=0)  # low-rank + small noise = strong
    rep = audit_corpus(X, dim=128, n_seed_queries=6)
    assert rep.basis == "pca_corpus"
    assert rep.metrics.get("seed_recall_vs_exact", 0) >= 0.95
    # no FAIL recall flag on a strong manifold
    assert not any(f.code == F_RECALL and f.severity == FAIL for f in rep.flags)


def test_recall_floor_configurable_from_preset():
    """recall_floor is an agnostic knob that travels in the preset JSON."""
    from winnex_ai_normalize.core.quality import load_dataset_preset
    cfg = QualityConfig.from_dataset("default")
    assert cfg.recall_floor == 0.95
    cfg2 = QualityConfig()  # default
    assert cfg2.recall_floor == 0.95
    # a preset override is honored
    assert QualityConfig(recall_floor=0.9).recall_floor == 0.9


def test_recall_shortfall_is_a_signal_not_a_decision():
    """(2026-09-04, agnostic redesign) `dataset.recall_not_guaranteed` is a
    SIGNAL: when the applied route is pool_only with recall below the floor, the
    code emits a WARN and does NOT reverse/block. Reversing ("never pca when
    recall < floor") is a route_rules policy of the CONFIG, not code. We
    unit-test the flag helper: any route with low recall → WARN (never FAIL from
    the code alone)."""
    from winnex_ai_normalize.core.quality import QualityValidator, QualityReport
    # low recall on a pca route → WARN (the code does not decide to fail/reverse)
    v = QualityValidator()
    rep = QualityReport(n=100, dim=128)
    v._flag_recall_shortfall(rep, 0.40, 1.0, ["pool_only"] * 8, 0.95,
                             route="pca_corpus")
    flags = [f for f in rep.flags if f.code == F_RECALL]
    assert flags and flags[0].severity == WARN
    assert not rep.has_fail  # signal only — no block from the code


def test_random_route_with_low_recall_emits_warn_not_fail():
    """A weak manifold that stays on the random route is a PHYSICAL limit (the
    prefilter is the recall gate), not corruption — so the flag is WARN, and the
    build is not blocked by default (the preset word2vec already lives here)."""
    from winnex_ai_normalize.core.quality import QualityValidator, QualityReport
    v = QualityValidator()
    rep = QualityReport(n=100, dim=300)
    v._flag_recall_shortfall(rep, 0.53, 1.0, ["pool_only"] * 6, 0.95,
                             route="random")
    warns = [f for f in rep.flags if f.code == F_RECALL]
    assert warns and warns[0].severity == WARN
    assert not rep.has_fail


def test_route_rules_are_config_not_code():
    """(2026-09-04) The routing decision is the CONFIG's: changing route_rules
    changes the route. Prove it by forcing a corpus that would naturally route
    pca_corpus to instead route random via a custom route table."""
    from winnex_ai_normalize.core.quality import QualityConfig, audit_corpus
    X = _embeddings(n=1500, d=128, seed=0)  # strong manifold → pca by default
    # default table → pca_corpus (bound_frac high)
    rep_default = audit_corpus(X, dim=128, n_seed_queries=6)
    assert rep_default.basis == "pca_corpus"
    # custom table: force random with a low k1 (a policy the operator chose)
    cfg = QualityConfig(route_rules=[
        {"when": {"bound_fraction": ">= 0.0"},
         "route": {"basis": "random", "k1_fraction": 0.05}},
        {"fallback": {"basis": "random", "k1_fraction": 0.20}},
    ])
    rep_custom = audit_corpus(X, dim=128, n_seed_queries=6, cfg=cfg)
    assert rep_custom.basis == "random"
    assert rep_custom.k1_fraction == 0.05


def test_route_rules_default_reproduces_historical_behavior():
    """(2026-09-04) The DEFAULT route table reproduces the historical router
    behavior exactly, so existing callers see no change: strong manifold → pca,
    isotropic high-dim → random/k1 high."""
    X = _embeddings(n=1500, d=128, seed=0)   # strong → pca
    rep = audit_corpus(X, dim=128, n_seed_queries=6)
    assert rep.basis == "pca_corpus"
    assert rep.k1_fraction <= 0.10

    X2 = _isotropic(n=300, d=512, seed=4)     # no manifold → random/k1 high
    rep2 = audit_corpus(X2, dim=512, n_seed_queries=6, probe_pca=False)
    assert rep2.basis == "random"
    assert rep2.k1_fraction >= 0.15


def test_match_route_conditions():
    """(2026-09-04) The route-table interpreter evaluates conditions correctly."""
    from winnex_ai_normalize.core.quality import _match_route
    rules = [
        {"when": {"bound_fraction": ">= 0.50", "seed_recall_vs_exact": ">= 0.95"},
         "route": {"basis": "pca_corpus", "k1_fraction": 0.05}},
        {"fallback": {"basis": "random", "k1_fraction": 0.20}},
    ]
    # both conditions match → pca
    assert _match_route({"bound_fraction": 0.6, "seed_recall_vs_exact": 1.0},
                        rules, 0.95)["basis"] == "pca_corpus"
    # recall below floor → falls to fallback (random)
    assert _match_route({"bound_fraction": 0.6, "seed_recall_vs_exact": 0.80},
                        rules, 0.95)["basis"] == "random"
    # missing metric → fallback
    assert _match_route({"bound_fraction": 0.6}, rules, 0.95)["basis"] == "random"


def test_foldable_flag_pca_without_probe_no_crash():
    """REGRESSION (2026-09-04, 1.4.1): when the route table picks pca_corpus
    via the FIRST rule (random bound_fraction >= 0.50), the PCA probe never
    runs and `pca_proved` is None. The F_FOLDABLE flag message formatted
    `{pca_proved:.0%}` → `TypeError: unsupported format string passed to
    NoneType.__format__`, crashing build_quality_engine on ANY strong manifold
    (measured: d=64/100/128). The fix treats pca_proved None as nan in the
    message. This test builds a strong low-dim manifold (the trigger) and
    asserts no crash + the pca route is applied."""
    rng = np.random.RandomState(0)
    N, d, ncomp = 20000, 64, 8
    comp = rng.randn(ncomp, d).astype(np.float32)
    X = (rng.randn(N, ncomp).astype(np.float32) @ comp).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # strong manifold at d=64: random basis proves >= 50% → pca route via rule 1,
    # the PCA probe is skipped (d <= probe_pca_dim_gate) → pca_proved is None.
    eng, rep = build_quality_engine(X, dim=d, k=10, return_report=True)
    assert rep.basis == "pca_corpus"
    # the F_FOLDABLE flag must be present and well-formed (no crash)
    assert any(f.code == F_FOLDABLE for f in rep.flags)
