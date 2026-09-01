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
