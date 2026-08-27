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
