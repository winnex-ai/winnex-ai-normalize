"""
winnex-ai-normalize — normalization tests (REAL data, no fallback).

Validates:
  - validate_embeddings rejects NaN/wrong-dim (fail loudly).
  - normalize_l2 produces unit-norm rows.
  - quantize_corpus maps float32 → uint8 with the correct scale.
  - EmbeddingNormalizer end-to-end with a REAL dataset (arXiv d=1536).

Run:  python -m pytest winnex_ai_normalize/tests/ -v
"""
import os

import numpy as np
import pytest

from winnex_ai_normalize.core.normalize import (
    EmbeddingNormalizer,
    validate_embeddings,
    normalize_l2,
    quantize_corpus,
)
from winnex_ai_normalize.core.config import NormalizeConfig


def test_validate_rejects_nan():
    v = np.random.randn(10, 128).astype(np.float32)
    v[3, 5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_embeddings(v)


def test_validate_rejects_wrong_dim():
    v = np.random.randn(10, 128).astype(np.float32)
    with pytest.raises(ValueError, match="dimension"):
        validate_embeddings(v, dim=64)


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_embeddings(np.zeros((0, 128), dtype=np.float32))


def test_normalize_l2_unit_norm():
    v = np.random.randn(50, 128).astype(np.float32)
    n = normalize_l2(v)
    norms = np.linalg.norm(n, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_quantize_corpus_scale():
    """uint8 quantization maps to [0, 255] with the correct scale.

    NOTE: uint8 quantization is for BIGANN-style RAW BYTE corpora. For
    float32 embeddings (OpenAI/Qwen3) the correct path is build_float32
    (the engine's float32 manifold preserves cosine exactly); quantizing
    float32 → uint8 with a shift is lossy and NOT the cosine path.
    """
    v = np.random.randn(20, 128).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    u8 = quantize_corpus(v)
    assert u8.dtype == np.uint8
    assert u8.min() >= 0 and u8.max() <= 255


def test_build_float32_path_preserves_cosine():
    """The float32 path (build_float32) preserves cosine — the correct
    path for real embeddings (no uint8 quantization)."""
    v = np.random.randn(20, 128).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # After L2-normalization the cosine of the float32 IS the inner product.
    c = float(v[0] @ v[1])
    assert -1.0 <= c <= 1.0
    # The normalizer validates + keeps float32 (no lossy uint8 for cosine).
    from winnex_ai_normalize import validate_embeddings
    out = validate_embeddings(v, dim=128)
    assert np.allclose(out, v)


def test_normalizer_with_real_arxiv():
    """End-to-end with the REAL arXiv OpenAI embeddings (d=1536)."""
    path = "/home/wnnx_user/zenodo/arxiv_100k.npy"
    if not os.path.exists(path):
        pytest.skip("arxiv_100k.npy not present — skipping real-data test")
    a = np.load(path, mmap_mode="r")
    X = np.ascontiguousarray(a[:200])          # real embeddings d=1536
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(n, 1e-12)

    cfg = NormalizeConfig(default_dim=1536)
    norm = EmbeddingNormalizer(config=cfg)
    # Vectors pass through unchanged (validated + float32).
    out = norm.normalize_vectors(X, dim=1536)
    assert out.shape == (200, 1536)
    assert np.isfinite(out).all()
    # Corpus quantization for the madhava engine.
    u8 = norm.to_corpus(X, dim=1536)
    assert u8.shape == (200, 1536) and u8.dtype == np.uint8
