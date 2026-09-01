"""
winnex-ai-normalize — quality flags: the Madhava engine's own validation.

The recall end-to-end depends on stages OUTSIDE the motor: embedding quality
(third-party providers, multiple embedding sets), dataset integrity (e.g. the
corrupted BIGANN base whose order differs from the ground truth), and the
prefilter heuristic. This module makes winnex-normalize the GUARDIAN at the
ingest point — but the VALIDATION IS THE MOTOR'S OWN:

    Cauchy-Schwarz, upper bound:
        UB(v,q) = ⟨Pv,Pq⟩ + e(v)·e(q)
        UB(v,q) < threshold(K)  ⟹  v is mathematically PROVEN not in the top-K.

The engine already emits this per-document proof in every search() (the audit
hook: audit_ids / audit_threshold / pruned_by_bound / pruned_by_prefilter).
This module LAUNCHES that native validation on a set of SEED QUERIES and
CAPTURES the excluded seed set — the captured set IS the flag response:

    - high pruned_by_bound / N  → the bound proves most of the corpus is out
      of top-K → the corpus is FOLDABLE → route basis=pca_corpus (tightens
      the bound further), keep k1 small.
    - low pruned_by_bound, high pruned_by_prefilter → the bound is loose
      (high dimension / no manifold) → the PREFILTER is the real recall gate
      → raise k1_fraction, keep basis=random.
    - if the random basis proves little, an optional PCA probe (the engine's
      OWN pca_corpus build) decides whether the corpus is foldable under a
      tight basis or genuinely isotropic.

This validates dataset AND embedding quality without reimplementing the math:
the proof is the engine's, not a heuristic approximation.

License: Business Source License 1.1 (BSL 1.1)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

logger = logging.getLogger("winnex-ai-normalize.quality")

# Presets JSON por dataset (motor e normalize AGNÓSTICOS — a config por dataset
# vive no arquivo, não no código). Mesmo padrão do winnex_pipeline/configs/.
# Os presets ficam em <pacote-raiz>/configs/ (winnex-ai-normalize/configs/),
# ao lado do subpacote winnex_ai_normalize/ — subimos 2 níveis a partir de
# winnex_ai_normalize/core/quality.py.
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../winnex_ai_normalize/core
_PKG_DIR = os.path.dirname(_CORE_DIR)                            # .../winnex_ai_normalize
_ROOT_DIR = os.path.dirname(_PKG_DIR)                            # .../winnex-ai-normalize
_CONFIGS_DIR = os.path.join(_ROOT_DIR, "configs")
# Overridável via env (permite apontar para um diretório de presets customizado).
_ENV_CONFIGS_DIR = os.environ.get("WINNEX_AI_NORMALIZE_CONFIGS_DIR", _CONFIGS_DIR)


def _deep_merge(base, override):
    """Recursive dict merge. override values win (mesmo padrão do pipeline)."""
    result = dict(base)
    for k, v in (override or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_dataset_preset(dataset: Optional[str] = None) -> dict:
    """Load a per-dataset preset JSON, deep-merged over the agnostic default.

    - dataset=None or 'default' → the agnostic preset (router decides).
    - dataset='arxiv' → configs/dataset_arxiv.json (manifold-strong: pca_corpus).
    - dataset='isotropic' → configs/dataset_isotropic.json (no-manifold:
      probe_pca=false, k1=0.20 — economiza o probe PCA de ~21-24s em d=1536).

    O motor permanece agnóstico: os knobs (basis, pca_iterations, k1_fraction,
    probe_pca, ...) vêm do preset; o motor apenas os aplica. A config por
    dataset é externalizada no arquivo, não hardcoded.
    """
    default = load_dataset_preset_file("dataset_default")
    if not dataset or dataset in ("default", "default.json", "dataset_default"):
        return default
    # Aceita 'arxiv' → dataset_arxiv.json; 'dataset_arxiv.json' → direto;
    # 'dataset_arxiv' → dataset_arxiv.json.
    name = dataset
    if not name.endswith(".json"):
        name = name if name.startswith("dataset_") else f"dataset_{name}"
        name = f"{name}.json"
    preset = load_dataset_preset_file(name)
    return _deep_merge(default, preset)


def load_dataset_preset_file(name: str) -> dict:
    """Read one preset JSON from the configs dir (empty dict if missing)."""
    fname = name if name.endswith(".json") else f"{name}.json"
    path = os.path.join(_ENV_CONFIGS_DIR, fname)
    if not os.path.exists(path):
        logger.warning("dataset preset not found: %s — using agnostic default", path)
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("failed to load dataset preset %s: %s — using default", path, e)
        return {}

# Severity levels
PASS = "pass"
WARN = "warn"
FAIL = "fail"

# Flag codes (stable identifiers for automation / monitoring)
F_NAN = "dataset.nan"                       # NaN/inf present
F_DEGENERATE = "dataset.degenerate"         # zero/negligible variance
F_FOLDABLE = "dataset.foldable"             # engine proof coverage (THE flag)
F_RESOLUTION = "embedding.resolution"       # top1-topK gap (embedding quality)
F_DRIFT = "embedding.provider_drift"        # same provider, space drifted
F_CROSS_PROVIDER = "embedding.cross_provider"  # provider switched (different space)
F_DIM_SHIFT = "embedding.dim_shift"         # dimension changed between batches
F_NORM = "embedding.norm"                   # norm distribution abnormal
F_COLLAPSE = "embedding.anisotropy"         # uint8-embedding trap / collapse
F_ALIGN = "integrity.corpus_alignment"      # corpus vs reference misaligned
F_GT_PROX = "integrity.query_corpus_proximity"  # query index collides with corpus

# Defaults (calibrated against the engine's measured proof coverage).
_FOLD_BOUND_FRAC = 0.50     # bound_frac ≥ this → strongly foldable
_FOLD_BOUND_FRAC_LOW = 0.20  # bound_frac below this → probe/fold gate
_RESOLUTION_GAP_WARN = 0.10  # top1-topK cos gap below this → weak embedding


@dataclass
class QualityConfig:
    """Thresholds for the quality checks (env: WINNEX_AI_NORMALIZE_QUALITY_*).

    Motor e normalize são AGNÓSTICOS: os knobs por dataset vêm do preset JSON
    (configs/dataset_<name>.json), carregado por `QualityConfig.from_dataset()`.
    Isto externaliza a config (basis, pca_iterations, k1_fraction, probe_pca,
    ...) por dataset — o código não hardcodeia valores de dataset.
    """
    enabled: bool = True
    k: int = 10                 # top-K used by the validation searches
    n_seed_queries: int = 8     # seed queries launched to trigger the proof
    probe_pca: bool = True      # if random basis proves little, probe with
                                # pca_corpus (the engine's own build) before
                                # declaring the corpus non-foldable
    fold_bound_frac: float = _FOLD_BOUND_FRAC
    fold_bound_frac_low: float = _FOLD_BOUND_FRAC_LOW
    resolution_gap_warn: float = _RESOLUTION_GAP_WARN
    # Phase 2 (GAIA): when True, a low embedding resolution (top-1 vs top-K
    # cosine gap below resolution_gap_warn) is escalated from WARN to FAIL,
    # blocking the build via QualityGateError. Default False — the current
    # behavior is unchanged (WARN only). Enable per-contract when the operator
    # wants a hard floor on embedding semantic quality ("blurry photo" guard).
    fail_on_resolution: bool = False
    drift_cos_warn: float = 0.90
    drift_cos_fail: float = 0.80
    # Política de NaN/inf (um knob do config, NÃO uma regra hardcoded — motor
    # e normalize agnósticos). O `pca_corpus` AMPLIFICA corrupção de dados:
    # um único NaN na matriz de covariância C=A.T@A vira autovetor NaN →
    # set_basis(P1) com base NaN → bound "certeiro" mas recall despenca
    # (medido: 1 linha NaN/300d → pca recall 0.042 vs random 1.0). O que o
    # preset decide sobre isso é política, não engenharia do motor.
    #   "block_pca"  → NaN presente ⇒ nunca pca_corpus; roteia random + k1 alto
    #                  (protege recall; o corpus degradado vira não-foldable).
    #   "block_build" → NaN presente ⇒ bloqueia o build inteiro (FAIL estrito;
    #                  requer allow_unsafe para prosseguir).
    #   "ignore"     → roda sem proteção (para medir a degradação / debug).
    nan_policy: str = "block_pca"

    # Config do MOTOR sugerida pelo preset do dataset (engine_kwargs aplicados
    # no build_engine). Vazia = agnóstica (o roteador decide pela prova).
    # Ex.: {"basis": "pca_corpus", "pca_iterations": 30, "stage1_dim": 128}
    engine_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dataset(cls, dataset: Optional[str] = None) -> "QualityConfig":
        """Load a QualityConfig from the per-dataset preset JSON (deep-merged
        over the agnostic default). Os knobs (probe_pca, n_seed_queries, k,
        thresholds e engine_kwargs) vêm do arquivo configs/dataset_<name>.json.

        O motor e o normalize permanecem agnósticos: recebem os knobs do
        config, não os hardcodeiam. `dataset=None` → preset default (roteador).
        """
        preset = load_dataset_preset(dataset)
        q = preset.get("quality", {})
        eng = preset.get("engine", {})
        cfg = cls(
            k=int(q.get("k", cls.k)),
            n_seed_queries=int(q.get("n_seed_queries", cls.n_seed_queries)),
            probe_pca=bool(q.get("probe_pca", cls.probe_pca)),
            fold_bound_frac=float(q.get("fold_bound_frac", cls.fold_bound_frac)),
            fold_bound_frac_low=float(q.get("fold_bound_frac_low", cls.fold_bound_frac_low)),
            resolution_gap_warn=float(q.get("resolution_gap_warn", cls.resolution_gap_warn)),
            fail_on_resolution=bool(q.get("fail_on_resolution", cls.fail_on_resolution)),
            nan_policy=str(q.get("nan_policy", cls.nan_policy)),
        )
        # engine_kwargs do preset: ignora valores null (o roteador decide).
        cfg.engine_kwargs = {k: v for k, v in eng.items() if v is not None}
        return cfg


@dataclass
class Flag:
    """A single quality finding. severity ∈ {pass, warn, fail}."""
    code: str
    severity: str
    message: str
    metric: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


@dataclass
class EmbeddingFingerprint:
    """A compact fingerprint of one embedding batch (for drift detection)."""
    provider: str
    dim: int
    centroid: np.ndarray            # (d,) mean vector
    norm_mean: float = 1.0
    batch_id: int = 0


@dataclass
class QualityReport:
    """The audit result: flags + the captured excluded seed set + a SUGGESTED
    engine configuration.

    The suggestion is the ROUTER'S decision based on the engine's own proof
    coverage — the engine still reports its pruned_by_bound / pruned_by_prefilter
    so the operator can verify where recall actually lives.
    """
    n: int
    dim: int
    provider: Optional[str] = None
    flags: List[Flag] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    # The captured seed set (the flag response): documents the engine PROVED
    # to be outside the top-K, for each seed query. Deterministic.
    excluded_seed_set: list = field(default_factory=list)

    # Suggested engine config (what build_quality_engine applies)
    basis: str = "random"
    k1_fraction: float = 0.05
    quant: str = "none"
    stage1_dim: int = 64
    stage2_dim: int = 0
    early_exit: bool = False
    metric: str = "cosine"

    # Optional engine already built with the suggested config (probe reuse)
    engine: object = None

    # Per-query semantic resolution (Phase 2 / GAIA): the top-1 vs top-K exact
    # cosine gap for EACH seed query, captured by the motor's own search. A low
    # gap means the provider's embeddings barely discriminate top-K from top-1
    # ("blurry photo"): recall is bounded by the embedding quality, not the
    # engine. Exposed so operators can route per-query or enforce a floor.
    query_resolution: list = field(default_factory=list)  # list[dict] per seed

    @property
    def fail_flags(self) -> List[Flag]:
        return [f for f in self.flags if f.severity == FAIL]

    @property
    def warn_flags(self) -> List[Flag]:
        return [f for f in self.flags if f.severity == WARN]

    @property
    def has_fail(self) -> bool:
        return any(f.severity == FAIL for f in self.flags)

    def add(self, flag: Flag) -> None:
        self.flags.append(flag)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "dim": self.dim,
            "provider": self.provider,
            "flags": [f.to_dict() for f in self.flags],
            "metrics": {k: round(float(v), 6) if isinstance(v, (int, float, np.floating))
                        else v for k, v in self.metrics.items()},
            "suggested_config": {
                "basis": self.basis,
                "k1_fraction": self.k1_fraction,
                "quant": self.quant,
                "stage1_dim": self.stage1_dim,
                "stage2_dim": self.stage2_dim,
                "early_exit": self.early_exit,
                "metric": self.metric,
            },
            "excluded_seed_set": self.excluded_seed_set[:50],
            "query_resolution": [dict(g) for g in self.query_resolution],
            "nan_fraction": round(float(self.metrics.get("nan_fraction", 0.0)), 8),
            "verdict": "fail" if self.has_fail else "warn" if self.warn_flags else "pass",
        }

    def summary(self) -> str:
        """One-line human summary (for logs / CLI)."""
        verdict = "FAIL" if self.has_fail else "WARN" if self.warn_flags else "PASS"
        nf = len(self.fail_flags)
        nw = len(self.warn_flags)
        return (f"quality[{verdict}] N={self.n} d={self.dim} "
                f"flags: {nf} fail, {nw} warn | "
                f"suggest basis={self.basis} k1={self.k1_fraction} "
                f"quant={self.quant} | "
                f"proof={self.metrics.get('proof_ratio', float('nan')):.2f} "
                f"bound_frac={self.metrics.get('bound_fraction', float('nan')):.2f} "
                f"prefilter={self.metrics.get('prefilter_fraction', float('nan')):.2f} "
                f"excluded_seed={len(self.excluded_seed_set)}")


class QualityGateError(Exception):
    """Raised when the audit finds FAIL flags and allow_unsafe=False."""

    def __init__(self, report: QualityReport):
        self.report = report
        fails = "; ".join(f"{f.code}: {f.message}" for f in report.fail_flags)
        super().__init__(f"quality gate FAILED — {fails}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _l2norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Drift detection (embedding sets from the same/different providers)
# ---------------------------------------------------------------------------
def check_embedding_drift(prev: Optional[EmbeddingFingerprint],
                          new: EmbeddingFingerprint,
                          cfg: Optional[QualityConfig] = None) -> List[Flag]:
    """Compare two embedding batches for space drift / dimension shift.

    - Different provider names → informational cross-provider flag (the two
      sets are in different vector spaces; mixing them corrupts recall).
    - Same provider, dimension changed → FAIL.
    - Same provider, centroid cosine dropped → WARN/FAIL drift (the provider's
      embedding space moved between calls).
    """
    qc = cfg or QualityConfig()
    flags: List[Flag] = []
    if prev is None or new is None:
        return flags
    if prev.provider and new.provider and prev.provider != new.provider:
        flags.append(Flag(
            F_CROSS_PROVIDER, WARN,
            f"provider switched {prev.provider} → {new.provider}: the two "
            "embedding sets live in different vector spaces; mixing them into "
            "one corpus corrupts semantic recall (0 bound violations do not "
            "cover this).",
        ))
        return flags
    if prev.dim != new.dim:
        flags.append(Flag(
            F_DIM_SHIFT, FAIL,
            f"provider {new.provider} changed dimension {prev.dim} → {new.dim} "
            "between batches — indices/queries built on the old dim are invalid.",
            metric=float(new.dim), threshold=float(prev.dim),
        ))
        return flags
    pc = prev.centroid
    nc = new.centroid
    denom = (np.linalg.norm(pc) + 1e-9) * (np.linalg.norm(nc) + 1e-9)
    cos = float(pc @ nc / denom)
    if cos < qc.drift_cos_fail:
        flags.append(Flag(
            F_DRIFT, FAIL,
            f"provider {new.provider} drifted: batch-centroid cosine {cos:.3f} "
            f"(< {qc.drift_cos_fail}) vs the previous batch — the embedding "
            "space moved; do not index old + new together.",
            metric=cos, threshold=qc.drift_cos_fail,
        ))
    elif cos < qc.drift_cos_warn:
        flags.append(Flag(
            F_DRIFT, WARN,
            f"provider {new.provider} drifted: batch-centroid cosine {cos:.3f} "
            f"(< {qc.drift_cos_warn}) vs the previous batch — verify the "
            "provider/model did not change.",
            metric=cos, threshold=qc.drift_cos_warn,
        ))
    return flags


# ---------------------------------------------------------------------------
# The QualityValidator — launches the MOTOR's own Cauchy-Schwarz validation
# on seed queries and captures the excluded set (the flag response).
# ---------------------------------------------------------------------------
class QualityValidator:
    """Validate a corpus via the Madhava engine's own per-document proof.

    For each SEED QUERY the engine's search() runs the Cauchy-Schwarz bound
    over all N vectors and emits the documents it PROVES to be outside the
    top-K (UB < threshold). We capture that excluded seed set (audit_ids,
    pruned_by_bound, pruned_by_prefilter, audit_threshold) — the captured set
    IS the flag response:

      - proof coverage  = pruned_by_bound / N  (fraction of the corpus the
        bound mathematically excludes from top-K)
      - prefilter share = pruned_by_prefilter / N  (the heuristic cut, no
        proof)

    Routing decision:
      - proof coverage ≥ 0.50        → foldable, strong: basis=pca_corpus, k1=0.05
      - 0.20 ≤ coverage < 0.50       → foldable, moderate: basis=random, k1=0.10
      - coverage < 0.20              → loose bound: PROBE with pca_corpus (the
        engine's own tight basis). If the PCA basis proves ≥0.50 → the corpus
        is foldable under a tight basis → basis=pca_corpus. Else → genuinely
        non-foldable (isotropic / noise): basis=random, k1=0.20 (the prefilter
        is the real recall gate).

    The optional PCA probe uses the ENGINE's pca_corpus build — the engine's
    own validation, not a reimplementation. probe_pca=False skips it (avoids
    the one-time O(d³) build cost) and conservatively raises k1.
    """

    def __init__(self, k: int = 10, n_seed_queries: int = 8, seed: int = 42,
                 probe_pca: bool = True, engine_kwargs: Optional[dict] = None,
                 fail_on_resolution: bool = False,
                 pca_iterations: int = 30,
                 nan_policy: str = "block_pca"):
        self.k = k
        self.n_seed_queries = n_seed_queries
        self.seed = seed
        self.probe_pca = probe_pca
        self.engine_kwargs = engine_kwargs or {}
        self.nan_policy = nan_policy
        # G1 FIX (2026-08-31): pca_iterations=30 como default (o knob vem do
        # preset dataset_default.json). O motor é agnóstico — o knob é do
        # chamador. 30 converge o subespaço dominante (subspace sim=1.0 vs 200,
        # medido) e corta ~3s do build PCA em d=1536 (que é O(D²·s·iters),
        # ~21-24s). O chamador pode sobrescrever via engine_kwargs.
        self.pca_iterations = int(pca_iterations)
        # Phase 2 (GAIA): escalate low embedding resolution from WARN to FAIL,
        # blocking the build via QualityGateError. Off by default.
        self.fail_on_resolution = fail_on_resolution

    def _build_engine(self, corpus, dim, metric, quant, basis, stage1_dim):
        import winnex_madhava as wm
        kw = dict(
            metric=metric, basis=basis, quant=quant,
            stage1_dim=stage1_dim, stage2_dim=0,
            k=self.k,
            k1_fraction=0.05, k2_fraction=0.05, k2_max=2000,
            normalize_input=(metric == "cosine"),
            early_exit=False,       # the P0 fix: never degrade recall at dim ≥ 384
            postfilter=True,
            pca_iterations=self.pca_iterations,
            seed=self.seed,
        )
        kw.update(self.engine_kwargs)
        return wm.build_engine(corpus, dim=dim, **kw)

    def _run(self, eng, corpus, norms, dim, is_float, seed_idx, max_captured=1000):
        """Run the seed queries against the engine; capture the excluded set.

        Returns (bound_count, prefilter_count, thresholds, gaps, captured).
        `gaps` is a list of dicts {seed_query, gap} — the per-query top-1 vs
        top-K exact cosine gap (Phase 2: exposed on QualityReport).
        """
        import winnex_madhava as wm
        n = len(corpus)
        nq = len(seed_idx)
        total_bound = 0
        total_pre = 0
        thresholds: List[float] = []
        gaps: List[dict] = []
        captured: List[dict] = []
        for qi in seed_idx:
            q = np.ascontiguousarray(corpus[qi], dtype=np.float32)
            r = eng.search(q)                       # the motor's own proof fires
            total_bound += int(r.pruned_by_bound)
            total_pre += int(r.pruned_by_prefilter)
            thresholds.append(float(r.audit_threshold))
            # capture the excluded seed set (deterministic, bounded)
            for doc_id, ub in zip(r.audit_ids[:max_captured], r.audit_ubs[:max_captured]):
                captured.append({
                    "doc_id": int(doc_id),
                    "upper_bound": float(ub),
                    "threshold": float(r.audit_threshold),
                    "seed_query": int(qi),
                })
            # embedding resolution: top-1 vs top-K exact cosine. Exposed per
            # query so a "blurry" query is distinguishable from a good one.
            if is_float and len(r.indices) >= 2 and norms is not None:
                qn = float(np.linalg.norm(q))
                if qn > 0:
                    c1 = float(corpus[r.indices[0]] @ q) / (norms[r.indices[0]] * qn)
                    ck = float(corpus[r.indices[-1]] @ q) / (norms[r.indices[-1]] * qn)
                    gaps.append({"seed_query": int(qi), "gap": c1 - ck})
        return total_bound, total_pre, thresholds, gaps, captured

    def validate(self, corpus, dim: Optional[int] = None, *,
                 reference=None, provider: Optional[str] = None) -> QualityReport:
        """Run the engine's Cauchy-Schwarz validation over seed queries.

        Returns a QualityReport whose excluded_seed_set is the captured proof
        (the flag response) and whose suggested_config is the router decision.
        """
        arr = np.ascontiguousarray(corpus)
        n = int(arr.shape[0]) if arr.ndim >= 1 else 0
        d = int(arr.shape[1]) if arr.ndim == 2 else 0
        is_float = arr.dtype in (np.float32, np.float64)
        is_uint8 = arr.dtype == np.uint8
        report = QualityReport(n=n, dim=d, provider=provider)
        if n == 0:
            report.add(Flag(F_DEGENERATE, FAIL, "empty corpus"))
            return report

        # --- dimension contract ---
        if dim is not None and d != dim:
            report.add(Flag(F_DIM_SHIFT, FAIL,
                            f"dimension mismatch: expected {dim}, got {d}",
                            metric=float(d), threshold=float(dim)))
            report.dim = d
            return report

        # --- NaN / inf (float only; uint8 cannot hold NaN) ---
        nan_frac = 0.0
        if is_float:
            nan_frac = float(1.0 - np.isfinite(arr).mean())
            if nan_frac > 0.0:
                n_bad = int(round(nan_frac * arr.size))
                report.add(Flag(
                    F_NAN, FAIL,
                    f"{n_bad} of {arr.size} values are NaN/inf "
                    f"({nan_frac:.4%}) — the engine would silently produce "
                    "garbage scores with 0 bound violations.",
                    metric=nan_frac))
        has_nan = nan_frac > 0.0
        report.metrics["nan_fraction"] = nan_frac

        # --- degeneracy (RAW space, before normalization) ---
        if is_float:
            raw_std = float(arr.std())
            report.metrics["raw_std"] = raw_std
            if raw_std < 1e-9:
                report.add(Flag(
                    F_DEGENERATE, FAIL,
                    f"near-zero variance (raw std={raw_std:.2e}) — all vectors "
                    "are ~constant; indexing is meaningless.",
                    metric=raw_std))

        # --- raw uint8 corpus of embeddings → the G4 trap (informational) ---
        if is_uint8:
            report.quant = "int8"
            report.metric = "l2"
            report.add(Flag(
                F_COLLAPSE, WARN,
                "uint8 corpus: if these bytes encode float32 embeddings (the "
                "quantize_corpus path), the [-1,1] embedding domain is lost "
                "and cosine recall degrades. Prefer feeding float32 and "
                "letting the engine use build_float32.",))
            # NOTE: the Cauchy-Schwarz proof still RUNS on uint8/L2 below —
            # the bound is metric-agnostic (⟨v,q⟩ ≤ ⟨Pv,Pq⟩ + e(v)e(q)).

        # --- norm distribution ---
        norms = np.linalg.norm(arr, axis=1)
        norm_mean = float(norms.mean())
        report.metrics["norm_mean"] = norm_mean
        if norm_mean > 1e-6:
            cv = float(norms.std() / (norm_mean + 1e-9))
            if cv > 0.5:
                report.add(Flag(
                    F_NORM, WARN,
                    f"norm distribution is wide (mean {norm_mean:.2f}, CV {cv:.2f}) "
                    "— the cosine contract expects unit-norm rows; "
                    "normalize_l2() is recommended.",
                    metric=cv))

        # --- reference drift / alignment (multiple embedding sets) ---
        if reference is not None:
            ref = np.ascontiguousarray(reference, dtype=np.float32)
            if ref.ndim == 2 and ref.shape[1] == d:
                ccos = float((_l2norm(arr.mean(axis=0, keepdims=True)) @
                              _l2norm(ref.mean(axis=0, keepdims=True)).T).item())
                report.metrics["reference_centroid_cos"] = ccos
                if ccos < 0.80:
                    report.add(Flag(
                        F_DRIFT, FAIL,
                        f"corpus vs reference batch: centroid cosine {ccos:.3f} "
                        "— the two embedding sets are in different spaces; do "
                        "not mix them into one index.",
                        metric=ccos, threshold=0.80))
                elif ccos < 0.90:
                    report.add(Flag(
                        F_DRIFT, WARN,
                        f"corpus vs reference batch: centroid cosine {ccos:.3f} "
                        "— verify they come from the same provider/model.",
                        metric=ccos, threshold=0.90))
            else:
                report.add(Flag(
                    F_ALIGN, FAIL,
                    f"reference batch dimension {ref.shape[1] if ref.ndim==2 else '?'} "
                    f"≠ corpus dim {d} — cannot compare the embedding sets.",
                    metric=float(ref.shape[1]) if ref.ndim == 2 else None,
                    threshold=float(d)))

        # ===================================================================
        # THE FLAG: launch the MOTOR's own Cauchy-Schwarz validation on the
        # seed queries and capture the excluded set.
        # ===================================================================
        metric = "cosine" if is_float else "l2"
        quant = "none" if is_float else "int8"
        rng = np.random.default_rng(self.seed)
        n_seed = min(self.n_seed_queries, n)
        seed_idx = rng.choice(n, n_seed, replace=False)

        # 1) measure with the RANDOM basis (fast, honest worst case)
        s1 = min(64, d)
        try:
            eng_random = self._build_engine(arr, d, metric, quant, "random", s1)
        except Exception as e:  # engine unavailable (not installed) → no flag
            logger.warning(f"quality validator: engine build failed ({e}) — "
                           "skipping the Cauchy-Schwarz proof coverage flag")
            report.basis = "random"
            report.k1_fraction = 0.20
            return report
        tb, tp, thr, gaps, captured = self._run(eng_random, arr, norms, d,
                                                is_float, seed_idx)
        n_tot = n * n_seed
        bound_frac = tb / n_tot if n_tot else 0.0
        pre_frac = tp / n_tot if n_tot else 0.0
        proof_ratio = tb / max(tb + tp, 1)
        mean_thr = float(np.mean(thr)) if thr else 0.0
        mean_gap = float(np.mean([g["gap"] for g in gaps])) if gaps else 0.0
        report.metrics.update({
            "proof_ratio": proof_ratio,
            "bound_fraction": bound_frac,
            "prefilter_fraction": pre_frac,
            "mean_kth_threshold": mean_thr,
            "top1_topk_gap": mean_gap,
            "seed_queries": int(n_seed),
            "basis_probed": "random",
        })
        # Phase 2: expose the per-query resolution (top-1 vs top-K gap) so the
        # operator can route a "blurry" query to a stronger provider or block.
        report.query_resolution = [dict(g) for g in gaps]

        # 2) routing decision
        # nan_policy (knob do config, agnóstico): quando o corpus tem NaN/inf e
        # a política bloqueia pca, o roteador NUNCA escolhe pca_corpus — o PCA
        # amplifica a corrupção (1 NaN na covariância → autovetor NaN → recall
        # despenca, medido 0.042). "block_pca" força random + k1 alto; o FAIL
        # de dataset.nan já está no report (bloqueado por build_quality_engine
        # a menos que allow_unsafe). "ignore" desliga a proteção.
        if has_nan and self.nan_policy == "block_pca":
            report.basis = "random"
            report.k1_fraction = 0.20
            report.engine = eng_random
            report.add(Flag(
                F_FOLDABLE, WARN,
                f"corpus has NaN/inf ({nan_frac:.4%}) and nan_policy="
                f"'block_pca' — pca_corpus would amplify the corruption "
                "(covariance → NaN eigenvectors → recall collapse, measured); "
                "forcing basis=random, k1=0.20. Re-audit after cleaning.",
                metric=nan_frac, threshold=0.0))
            # (não muta self.probe_pca — este early-return já pula o probe e
            #  não afeta chamadas subsequentes do validator)
            report.excluded_seed_set = captured
            report.stage1_dim = s1
            return report
        if bound_frac >= 0.50:
            report.basis = "pca_corpus"
            report.k1_fraction = 0.05
            report.engine = eng_random
            report.add(Flag(
                F_FOLDABLE, PASS,
                f"Cauchy-Schwarz PROVED {bound_frac:.0%} of the corpus is "
                f"outside top-{self.k} (random basis) — the corpus is "
                "foldable; basis=pca_corpus tightens the bound further.",
                metric=bound_frac, threshold=0.50))
        elif bound_frac >= 0.20:
            report.basis = "random"
            report.k1_fraction = 0.10
            report.engine = eng_random
            report.add(Flag(
                F_FOLDABLE, PASS,
                f"Cauchy-Schwarz proved {bound_frac:.0%} of the corpus "
                f"(prefilter {pre_frac:.0%}) — moderately foldable; "
                "k1=0.10 keeps recall safe.",
                metric=bound_frac, threshold=0.20))
        else:
            # Loose bound on the random basis. Probe with the engine's OWN
            # pca_corpus basis to decide foldable-under-tight-basis vs
            # genuinely isotropic.
            pca_proved = None
            pca_engine = None
            if self.probe_pca and d > 64:
                try:
                    s1p = min(192, d)
                    eng_pca = self._build_engine(arr, d, metric, quant,
                                                 "pca_corpus", s1p)
                    tb_p, _, _, _, _ = self._run(eng_pca, arr, norms, d,
                                                 is_float, seed_idx)
                    pca_proved = tb_p / n_tot if n_tot else 0.0
                    pca_engine = eng_pca
                    report.metrics["bound_fraction_pca"] = pca_proved
                    report.metrics["basis_probed"] = "random+pca_corpus"
                except Exception as e:  # pragma: no cover
                    logger.warning(f"PCA probe failed ({e}) — keeping random")
            if pca_proved is not None and pca_proved >= 0.50:
                report.basis = "pca_corpus"
                report.k1_fraction = 0.05
                report.engine = pca_engine
                # FIX (2026-08-31): o stage1_dim do report deve refletir o probe
                # pca (s1p), não o probe random (64). Sem isto, o build final
                # usaria stage1=64 (do random) em vez de 192 (do pca), e o
                # cfg_match do build_quality_engine não reutilizaria o probe.
                report.stage1_dim = s1p
                report.add(Flag(
                    F_FOLDABLE, PASS,
                    f"random basis proved only {bound_frac:.0%} (loose at "
                    f"d={d}), but the PCA basis PROVED {pca_proved:.0%} — the "
                    "corpus is foldable under a tight basis; "
                    "basis=pca_corpus restores proof-based pruning.",
                    metric=pca_proved, threshold=0.50))
            else:
                report.basis = "random"
                report.k1_fraction = 0.20
                report.engine = eng_random
                reason = (f"PCA basis proved only {pca_proved:.0%}" if pca_proved is not None
                          else "PCA probe skipped")
                report.add(Flag(
                    F_FOLDABLE, WARN,
                    f"Cauchy-Schwarz proved only {bound_frac:.0%} of the corpus "
                    f"(prefilter heuristic {pre_frac:.0%}; {reason}) — the "
                    "bound is loose here and the PREFILTER is the real recall "
                    "gate. k1_fraction raised to 0.20.",
                    metric=bound_frac, threshold=0.20))

        # 3) embedding resolution (the third-party quality): if even the top-K
        #    are barely more similar than the top-1, the provider's embeddings
        #    have poor semantic discrimination. WARN by default; escalated to
        #    FAIL when fail_on_resolution is set (Phase 2 / GAIA: a hard floor
        #    on embedding semantic quality, blocking via QualityGateError).
        if mean_gap >= 0 and mean_gap < 0.10:
            sev = FAIL if self.fail_on_resolution else WARN
            report.add(Flag(
                F_RESOLUTION, sev,
                f"top-1 vs top-{self.k} exact-cosine gap {mean_gap:.3f} (< 0.10) "
                "— the embedding set has low semantic resolution; recall is "
                "bounded by the provider's quality, not the engine.",
                metric=mean_gap, threshold=0.10))

        report.excluded_seed_set = captured
        report.stage1_dim = s1
        return report


# ---------------------------------------------------------------------------
# Facade: audit_corpus + gated engine builder
# ---------------------------------------------------------------------------
def audit_corpus(corpus, dim: Optional[int] = None, *,
                 reference=None, provider: Optional[str] = None,
                 k: Optional[int] = None, n_seed_queries: Optional[int] = None,
                 probe_pca: Optional[bool] = None,
                 cfg: Optional[QualityConfig] = None) -> QualityReport:
    """Validate a corpus using the MOTOR's own Cauchy-Schwarz proof.

    Launches the engine over seed queries and captures the excluded set — the
    flag response. Raises nothing; FAIL flags are read from report.has_fail.

    Parameters
    ----------
    corpus : np.ndarray
        (n, dim) float32/float64 embeddings, or uint8 raw bytes.
    dim : int, optional
        Expected dimensionality (mismatch → FAIL flag).
    reference : np.ndarray, optional
        A reference embedding batch to detect provider drift / alignment
        problems (the "multiple embedding sets" case).
    provider : str, optional
        Provider name that produced the corpus (for reporting).
    k : int, optional
        Top-K used by the validation searches.
    n_seed_queries, probe_pca : optional
        Validator overrides (seed count / PCA probe).
    cfg : QualityConfig, optional
        Threshold overrides.

    Returns
    -------
    QualityReport with flags, the captured excluded seed set, and a suggested
    engine configuration.
    """
    qc = cfg or QualityConfig()
    # engine_kwargs do preset (config do motor por dataset, via JSON). O motor
    # permanece agnóstico — recebe os knobs do config, não os hardcodeia.
    preset_engine = dict(qc.engine_kwargs)
    validator = QualityValidator(
        k=k or qc.k,
        n_seed_queries=n_seed_queries or qc.n_seed_queries,
        probe_pca=qc.probe_pca if probe_pca is None else probe_pca,
        fail_on_resolution=qc.fail_on_resolution,
        engine_kwargs=preset_engine,
        pca_iterations=int(preset_engine.get("pca_iterations", 30)),
        nan_policy=qc.nan_policy,
    )
    return validator.validate(corpus, dim=dim, reference=reference, provider=provider)


def build_quality_engine(corpus, dim=None, *, k=10, reference=None,
                         provider=None, allow_unsafe=False, return_report=False,
                         engine_kwargs: Optional[dict] = None,
                         cfg: Optional[QualityConfig] = None,
                         dataset: Optional[str] = None):
    """Run the quality gate (the engine's own validation), adapt the engine
    config, and build the engine.

    Raises QualityGateError when FAIL flags are present and allow_unsafe=False.
    Returns (engine, QualityReport) when return_report=True, else the engine.

    dataset : str, optional
        Nome do preset por dataset (configs/dataset_<name>.json). O motor e o
        normalize são AGNÓSTICOS — os knobs por dataset vêm do arquivo, não do
        código. Ex.: 'arxiv' → pca_corpus/pca_iterations=30; 'word2vec' →
        random/k1=0.20 (onde pca degrada recall); 'isotropic' → probe_pca=false
        (economiza o probe de ~21-24s em d=1536). Se None, usa o roteador
        agnóstico (dataset_default.json).
    """
    if cfg is None and dataset is not None:
        cfg = QualityConfig.from_dataset(dataset)
    report = audit_corpus(corpus, dim=dim, reference=reference, provider=provider,
                          k=k, cfg=cfg)
    if report.has_fail and not allow_unsafe:
        raise QualityGateError(report)

    # engine_kwargs: mescla o preset do dataset (cfg.engine_kwargs, do JSON) com
    # os overrides explícitos do chamador (engine_kwargs param). O motor é
    # agnóstico — recebe os knobs do config, não os hardcodeia.
    preset_kwargs = dict(cfg.engine_kwargs) if cfg is not None else {}
    explicit = {kk: vv for kk, vv in (engine_kwargs or {}).items() if vv is not None}
    preset_kwargs.update(explicit)
    caller_kwargs = preset_kwargs

    build_kwargs = {
        "metric": report.metric,
        "basis": report.basis,
        "k1_fraction": report.k1_fraction,
        "quant": report.quant,
        "stage1_dim": report.stage1_dim,
        "stage2_dim": report.stage2_dim,
        "early_exit": report.early_exit,
        "k": k,
    }
    # nan_policy='block_pca': o basis FORÇADO (do preset ou do chamador) não
    # pode contornar a proteção — pca_corpus sobre corpus NaN amplifica a
    # corrupção (medido: recall 0.042 vs random 1.0). O roteador já forçou
    # random; aqui garantimos que um basis vindo de engine_kwargs não reforce
    # pca_corpus por cima da decisão da política.
    nan_frac = float(report.metrics.get("nan_fraction", 0.0))
    nan_blocked = nan_frac > 0.0 and (cfg.nan_policy if cfg is not None else "block_pca") == "block_pca"
    if nan_blocked:
        for kw in ("basis",):
            caller_kwargs = dict(caller_kwargs)
            caller_kwargs[kw] = "random"
    build_kwargs.update({kk: vv for kk, vv in caller_kwargs.items() if vv is not None})

    engine = None
    if report.engine is not None:
        # Reuse the probe engine ONLY when the suggested config matches the
        # desired build — including basis and pca_iterations. BUG FIX
        # (2026-08-31): o cfg_match antigo NÃO comparava basis, então forçar
        # basis='pca_corpus' via engine_kwargs era silenciosamente ignorado e o
        # motor do probe (random) era reutilizado — as colunas random/pca_corpus
        # de benchmarks saíam idênticas. Agora a reutilização exige que o basis
        # e o pca_iterations do motor do probe sejam os desejados.
        rcfg = report.engine.config()
        rdim = report.engine.dim()
        # NOTA (2026-08-31): build_engine(float32, pca_corpus) constrói com
        # cfg.basis=RANDOM e aplica a base PCA via set_basis() — config().basis
        # NÃO reflete a base real. O discriminador confiável é stage1_dim
        # (probe random=64, probe pca=192) + k1_fraction + metric.
        dim_ok = (rdim == dim) if dim is not None else True
        same_stage1 = int(rcfg.stage1_dim) == int(build_kwargs["stage1_dim"])
        same_k1 = int(rcfg.k1_fraction * 1000) == int(build_kwargs["k1_fraction"] * 1000)
        metric_ok = str(rcfg.metric).lower() in ("cosine", "cosine")
        cfg_match = dim_ok and same_stage1 and same_k1 and metric_ok
        if cfg_match:
            engine = report.engine

    if engine is None:
        import winnex_madhava as wm
        engine = wm.build_engine(corpus, dim=dim, **build_kwargs)

    if return_report:
        return engine, report
    return engine
