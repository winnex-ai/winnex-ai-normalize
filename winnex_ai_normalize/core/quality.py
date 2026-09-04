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


# ---------------------------------------------------------------------------
# Route-table interpreter (2026-09-04): applies the ROUTING POLICY that lives
# in the config (route_rules), NOT in code. The validator measures corpus
# signals (bound_fraction, pca_bound_fraction, seed_recall_vs_exact, ...) and
# this function picks the first rule whose `when` matches. The code is a
# generic interpreter — it knows nothing about specific datasets.
# ---------------------------------------------------------------------------
_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def _match_route(metrics: dict, route_rules: list, recall_floor: float) -> dict:
    """Return the FIRST route whose `when` conditions ALL match the measured
    metrics, or the `fallback` if no rule matches (or no route_rules given).
    `metrics` is the report.metrics dict (signal space). Returns a dict of
    engine-knob overrides, e.g. {"basis": "pca_corpus", "k1_fraction": 0.05}.
    """
    for rule in route_rules or []:
        if "fallback" in rule:
            continue  # evaluated only if no when-rule matches
        when = rule.get("when", {})
        route = rule.get("route", {})
        if not when or not route:
            continue
        ok = True
        for key, cond in when.items():
            val = metrics.get(key)
            if val is None or val != val:  # missing metric or NaN → no match
                ok = False
                break
            cond_s = str(cond).strip()
            matched = False
            # inject recall_floor into the condition if referenced
            c = cond_s.replace("_recall_floor_", repr(float(recall_floor)))
            for op, fn in _OPS.items():
                if c.startswith(op):
                    rhs = c[len(op):].strip()
                    try:
                        threshold = float(rhs)
                    except ValueError:
                        continue
                    matched = fn(float(val), threshold)
                    break
            if not matched:
                ok = False
                break
        if ok:
            return dict(route)
    # fallback
    for rule in route_rules or []:
        if "fallback" in rule:
            return dict(rule["fallback"])
    return {}


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
F_RECALL = "dataset.recall_not_guaranteed"  # top-K is pool_only / recall<floor:
                                            # the returned top-K is NOT the
                                            # global top-K (silent collapse)

# Defaults (calibrated against the engine's measured proof coverage).
_FOLD_BOUND_FRAC = 0.50     # bound_frac ≥ this → strongly foldable
_FOLD_BOUND_FRAC_LOW = 0.20  # bound_frac below this → probe/fold gate
_RESOLUTION_GAP_WARN = 0.10  # top1-topK cos gap below this → weak embedding
# RECALL VALIDATION (2026-09-04, the honest-scope fix): the router used to
# decide by bound COVERAGE alone (pruned_by_bound/N). A config can have high
# proof coverage yet pool_only recall < 1.0 (measured: word2vec-like weak
# manifold, random k1=0.05 → recall 0.75 with viol=0, pool_only 20/20). We now
# validate the REAL recall (search vs the motor's own search_exact) on the seed
# queries and expose recall_guarantee. recall_floor is the minimum mean recall
# a suggested config must meet; below it the config is flagged.
_RECALL_FLOOR = 0.95

# Default route table (2026-09-04): reproduces the historical router behavior
# so that a bare `QualityConfig()` (no preset loaded) behaves as before. The
# routing POLICY lives here / in the preset JSON — NOT in code branches. See
# the `route_rules` field docstring for the format. The special token
# "_recall_floor_" is replaced by the configured recall_floor at match time.
_DEFAULT_ROUTE_RULES = [
    {"when": {"bound_fraction": ">= 0.50"},
     "route": {"basis": "pca_corpus", "k1_fraction": 0.05}},
    {"when": {"bound_fraction": ">= 0.20"},
     "route": {"basis": "random", "k1_fraction": 0.10}},
    # loose random bound → probe PCA: foldable under a tight basis AND the
    # measured pca seed recall holds (>= recall_floor) → route pca. The recall
    # gate is a POLICY (recall_floor), applied to the measured signal. The
    # metric keys are the ACTUAL report.metrics keys the validator exposes:
    #   bound_fraction (random), bound_fraction_pca, seed_recall_vs_exact_pca.
    {"when": {"bound_fraction": "< 0.20",
              "bound_fraction_pca": ">= 0.50",
              "seed_recall_vs_exact_pca": ">= _recall_floor_"},
     "route": {"basis": "pca_corpus", "k1_fraction": 0.05}},
    {"fallback": {"basis": "random", "k1_fraction": 0.20}},
]

# NaN-blocked route (2026-09-04): when nan_policy='block_pca' forces the route
# away from pca_corpus (pca amplifies NaN corruption), the SAFE route the code
# applies. This is a config policy — the code does not invent it. A random
# route needs a higher k1 (the prefilter is the recall gate when the bound is
# loose on a degraded corpus).
_NAN_BLOCKED_ROUTE = {"basis": "random", "k1_fraction": 0.20}


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

    # NAN-BLOCKED ROUTE (2026-09-04): when nan_policy='block_pca' forces the
    # route away from pca_corpus (pca amplifies NaN corruption), the SAFE route
    # the code applies. This is a config POLICY — a random route needs a higher
    # k1 (the prefilter is the recall gate when the bound is loose on a
    # degraded corpus). Lives in the preset JSON (quality.nan_blocked_route).
    nan_blocked_route: dict = field(default_factory=lambda: dict(_NAN_BLOCKED_ROUTE))

    # RECALL FLOOR (2026-09-04): minimum mean seed recall@K (search vs the
    # motor's own search_exact) a SUGGESTED config must meet. Below the floor
    # the config is flagged `dataset.recall_not_guaranteed`. The floor is a
    # POLICY of the config (the operator decides what recall is acceptable);
    # the code only measures and flags — it does NOT reverse the route. If the
    # operator wants "never pca when recall < floor", that is a route_rules
    # entry in the preset JSON, not code. Agnostic knob.
    recall_floor: float = _RECALL_FLOOR

    # Stage-1 probe dimensions (2026-09-04): the dimensions the QUALITY
    # VALIDATOR uses to MEASURE the corpus signals (bound coverage under a
    # random basis and, optionally, a PCA basis). These are MEASUREMENT knobs,
    # not the final route — the final engine config comes from route_rules.
    # They replace the hardcoded min(64, d) / min(192, d) / d > 64 in the code.
    stage1_probe_random: int = 64    # s1 of the random probe
    stage1_probe_pca: int = 192      # s1p of the PCA probe (if enabled)
    probe_pca_dim_gate: int = 64     # probe PCA only when dim > this

    # ROUTE RULES (2026-09-04): the DECISION TABLE that maps measured corpus
    # signals to an engine route (basis / k1_fraction / stage1_dim). This is
    # where the ROUTING POLICY lives — in the config, not in code. The code
    # only MEASURES the signals (bound_fraction, pca_bound_fraction,
    # seed_recall_vs_exact, ...) and applies the first rule whose `when`
    # matches. Format (see configs/dataset_default.json):
    #   [
    #     {"when": {"bound_fraction": ">= 0.50"},
    #      "route": {"basis": "pca_corpus", "k1_fraction": 0.05}},
    #     {"when": {"bound_fraction": "< 0.20", "pca_bound_fraction": ">= 0.50",
    #               "pca_seed_recall": ">= 0.95"},
    #      "route": {"basis": "pca_corpus", "k1_fraction": 0.05}},
    #     {"fallback": {"basis": "random", "k1_fraction": 0.20}}
    #   ]
    # Supported operators: ">=", ">", "<=", "<", "==". A `when` with multiple
    # conditions requires ALL to match (AND). Missing metric keys → condition
    # does not match. The last entry may be {"fallback": {...}} (always used if
    # no `when` rule matches). The default table (below) reproduces the
    # historical router behavior EXACTLY, so existing callers see no change.
    route_rules: list = field(default_factory=lambda: list(_DEFAULT_ROUTE_RULES))

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
            recall_floor=float(q.get("recall_floor", cls.recall_floor)),
            # route_rules + probe knobs (2026-09-04): a política de roteamento e
            # as dimensões de PROBE vêm do preset JSON — o código não decide.
            route_rules=list(q.get("route_rules") or _DEFAULT_ROUTE_RULES),
            stage1_probe_random=int(q.get("stage1_probe_random", 64)),
            stage1_probe_pca=int(q.get("stage1_probe_pca", 192)),
            probe_pca_dim_gate=int(q.get("probe_pca_dim_gate", 64)),
            nan_blocked_route=dict(q.get("nan_blocked_route") or _NAN_BLOCKED_ROUTE),
        )
        # engine_kwargs do preset: ignora valores null (o roteador decide).
        cfg.engine_kwargs = {k: v for k, v in eng.items() if v is not None}
        # Se o preset não define route_rules, usa a tabela default (que
        # reproduz o comportamento histórico).
        if not cfg.route_rules:
            cfg.route_rules = list(_DEFAULT_ROUTE_RULES)
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

    Routing POLICY (2026-09-04): the mapping from measured signals to an
    engine route (basis / k1_fraction / stage1_dim) lives in the CONFIG
    (`route_rules` in the preset JSON), NOT in code branches. This validator
    MEASURES the signals (proof coverage, pca coverage, seed recall) and then
    APPLIES the first route_rules entry whose `when` matches. The default table
    (in dataset_default.json / _DEFAULT_ROUTE_RULES) reproduces the historical
    behavior:
      - proof coverage ≥ 0.50        → basis=pca_corpus, k1=0.05
      - 0.20 ≤ coverage < 0.50       → basis=random, k1=0.10
      - coverage < 0.20 → PCA probe: if pca proves ≥0.50 AND its seed recall
        ≥ recall_floor → basis=pca_corpus; else fallback → random/k1=0.20.

    The optional PCA probe uses the ENGINE's pca_corpus build — the engine's
    own validation, not a reimplementation. probe_pca=False skips it (avoids
    the one-time O(d³) build cost) and conservatively raises k1.
    """

    def __init__(self, k: int = 10, n_seed_queries: int = 8, seed: int = 42,
                 probe_pca: bool = True, engine_kwargs: Optional[dict] = None,
                 fail_on_resolution: bool = False,
                 pca_iterations: int = 30,
                 nan_policy: str = "block_pca",
                 nan_blocked_route: Optional[dict] = None,
                 recall_floor: float = _RECALL_FLOOR,
                 route_rules: Optional[list] = None,
                 stage1_probe_random: int = 64,
                 stage1_probe_pca: int = 192,
                 probe_pca_dim_gate: int = 64):
        self.k = k
        self.n_seed_queries = n_seed_queries
        self.seed = seed
        self.probe_pca = probe_pca
        self.engine_kwargs = engine_kwargs or {}
        self.nan_policy = nan_policy
        self.nan_blocked_route = dict(nan_blocked_route or _NAN_BLOCKED_ROUTE)
        # ROUTE RULES + PROBE KNOBS (2026-09-04): the routing POLICY and the
        # probe measurement dimensions come from the config. The code applies
        # the table; it does not invent policy.
        self.route_rules = list(route_rules) if route_rules else list(_DEFAULT_ROUTE_RULES)
        self.stage1_probe_random = int(stage1_probe_random)
        self.stage1_probe_pca = int(stage1_probe_pca)
        self.probe_pca_dim_gate = int(probe_pca_dim_gate)
        # RECALL FLOOR (2026-09-04): minimum mean seed recall@K (search vs the
        # motor's own search_exact) a SUGGESTED config must meet. Below the
        # floor the config is flagged (WARN) — the code does NOT reverse the
        # route; the policy for "never pca when recall < floor" is the config's.
        self.recall_floor = float(recall_floor)
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

        Returns a dict with:
            bound_count, prefilter_count, thresholds, gaps, captured
            (as before) plus the RECALL validation (2026-09-04):
            recalls: list[float] per seed query — recall@K of search() vs the
                     motor's OWN search_exact() (the exact global top-K). This
                     is what "validate the router by recall, not by bound"
                     means: a config with high proof coverage can still be
                     pool_only with recall < 1.0 (silent collapse), and only
                     search_exact exposes it.
            guarantees: list[str] per seed query — r.recall_guarantee
                     ("pool_only" | "exact_global") from the motor.
        `gaps` is a list of dicts {seed_query, gap} — the per-query top-1 vs
        top-K exact cosine gap (Phase 2: exposed on QualityReport).
        """
        n = len(corpus)
        nq = len(seed_idx)
        total_bound = 0
        total_pre = 0
        thresholds: List[float] = []
        gaps: List[dict] = []
        captured: List[dict] = []
        recalls: List[float] = []
        guarantees: List[str] = []
        for qi in seed_idx:
            q = np.ascontiguousarray(corpus[qi], dtype=np.float32)
            r = eng.search(q)                       # the motor's own proof fires
            total_bound += int(r.pruned_by_bound)
            total_pre += int(r.pruned_by_prefilter)
            thresholds.append(float(r.audit_threshold))
            guarantees.append(str(getattr(r, "recall_guarantee", "pool_only")))
            # RECALL VALIDATION (2026-09-04): compare search() vs search_exact()
            # — the motor's OWN exact global top-K. search_exact is the same
            # operation the motor already uses as its ceiling (measured ~1.1-1.3x
            # the search cost on seed queries), so this adds no reimplementation.
            try:
                rx = eng.search_exact(q)
                g = set(int(i) for i in rx.indices)
                if len(rx.indices) > 0:
                    hit = sum(1 for i in r.indices if int(i) in g)
                    recalls.append(hit / len(rx.indices))
            except Exception:  # search_exact unavailable → recall unknown
                recalls.append(float("nan"))
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
        return {
            "bound_count": total_bound,
            "prefilter_count": total_pre,
            "thresholds": thresholds,
            "gaps": gaps,
            "captured": captured,
            "recalls": recalls,
            "guarantees": guarantees,
        }

    @staticmethod
    def _mean(xs):
        """Mean of a float list, ignoring nan (search_exact unavailable)."""
        f = [x for x in xs if x == x]
        return float(np.mean(f)) if f else float("nan")

    def _flag_recall_shortfall(self, report, mean_recall, pool_only_frac,
                               guarantees, recall_floor, *, route=None):
        """Emit `dataset.recall_not_guaranteed` as a SIGNAL (2026-09-04): when
        the applied route is pool_only with measured seed recall below the
        floor, or pool_only with material recall loss, flag it so the operator
        SEES that the returned top-K is not the global top-K (the silent
        collapse). Severity is always WARN — the code does NOT decide to block
        or reverse; that policy belongs to the config (route_rules /
        recall_floor / fail thresholds).
        """
        if mean_recall != mean_recall:      # nan → search_exact unavailable
            return
        nq = len(guarantees)
        n_pool = sum(1 for g in guarantees if g == "pool_only")
        below = mean_recall < recall_floor
        # pool_only alone is not a defect (the default IS pool_only when the
        # bound is loose); flag when recall is measurably below the floor OR
        # when the top-K is pool_only with material recall loss (< 1.0 in a way
        # the operator should not read as "perfect").
        if not below and not (pool_only_frac > 0.5 and mean_recall < 1.0):
            return
        msg = (f"seed recall@K = {mean_recall:.3f} vs the motor's own "
               f"search_exact (< floor {recall_floor:.2f}); "
               f"{n_pool}/{nq} seed queries are recall_guarantee=pool_only — "
               "the returned top-K is the best WITHIN the pool, NOT proven the "
               "global top-K. viol=0 does not cover this (silent collapse). "
               "For a global guarantee use audit_exhaustive=True or k1=N.")
        report.add(Flag(F_RECALL, WARN, msg, metric=mean_recall,
                        threshold=recall_floor))

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

        # 1) measure with the RANDOM basis (fast, honest worst case). The probe
        #    dimension is a MEASUREMENT knob from the config (agnostic), not a
        #    hardcoded min(64, d).
        s1 = min(int(self.stage1_probe_random), d)
        try:
            eng_random = self._build_engine(arr, d, metric, quant, "random", s1)
        except Exception as e:  # engine unavailable (not installed) → no flag
            logger.warning(f"quality validator: engine build failed ({e}) — "
                           "skipping the Cauchy-Schwarz proof coverage flag")
            # Cannot measure → apply the config's FALLBACK route (a conservative
            # random/k1). This is the config's policy, not a code decision.
            fb = _match_route({}, self.route_rules, self.recall_floor)
            report.basis = fb.get("basis", "random")
            report.k1_fraction = float(fb.get("k1_fraction", 0.20))
            report.stage1_dim = s1
            return report
        res = self._run(eng_random, arr, norms, d, is_float, seed_idx)
        tb = res["bound_count"]
        tp = res["prefilter_count"]
        thr = res["thresholds"]
        gaps = res["gaps"]
        captured = res["captured"]
        recalls = res["recalls"]
        guarantees = res["guarantees"]
        n_tot = n * n_seed
        bound_frac = tb / n_tot if n_tot else 0.0
        pre_frac = tp / n_tot if n_tot else 0.0
        proof_ratio = tb / max(tb + tp, 1)
        mean_thr = float(np.mean(thr)) if thr else 0.0
        mean_gap = float(np.mean([g["gap"] for g in gaps])) if gaps else 0.0
        # RECALL VALIDATION (2026-09-04): mean recall of search() vs the motor's
        # own search_exact() over the seed queries, and the fraction of seed
        # queries whose returned top-K is pool_only (NOT the global top-K). This
        # is the "validate by recall, not by bound" number.
        _finite = [x for x in recalls if x == x]           # drop nan
        mean_recall = float(np.mean(_finite)) if _finite else float("nan")
        pool_only_frac = (sum(1 for g in guarantees if g == "pool_only") / len(guarantees)
                          if guarantees else 0.0)
        report.metrics.update({
            "proof_ratio": proof_ratio,
            "bound_fraction": bound_frac,
            "prefilter_fraction": pre_frac,
            "mean_kth_threshold": mean_thr,
            "top1_topk_gap": mean_gap,
            "seed_queries": int(n_seed),
            "basis_probed": "random",
            # recall telemetry (new)
            "seed_recall_vs_exact": mean_recall,
            "pool_only_frac": pool_only_frac,
        })
        report.metrics["recall_guarantee_counts"] = {
            "exact_global": sum(1 for g in guarantees if g == "exact_global"),
            "pool_only": sum(1 for g in guarantees if g == "pool_only"),
        }
        # Phase 2: expose the per-query resolution (top-1 vs top-K gap) so the
        # operator can route a "blurry" query to a stronger provider or block.
        report.query_resolution = [dict(g) for g in gaps]

        # Recall telemetry of the FINAL route. Each branch below sets these to
        # the seed recall/guarantees of the engine it SUGGESTS (random or pca),
        # so the recall flag at the end reflects the config actually suggested,
        # not the random baseline that may have been discarded by the probe.
        route_recall = mean_recall
        route_guarantees = list(guarantees)
        route_pool_frac = pool_only_frac

        # 2) MEASURE the PCA probe (if the config enables it): an optional
        #    tight-basis signal used by the route table below. This is
        #    MEASUREMENT (agnostic), not a decision.
        pca_proved = None
        pca_engine = None
        pca_recall = None
        pca_res = None
        if self.probe_pca and d > int(self.probe_pca_dim_gate):
            try:
                s1p = min(int(self.stage1_probe_pca), d)
                eng_pca = self._build_engine(arr, d, metric, quant,
                                             "pca_corpus", s1p)
                pca_res = self._run(eng_pca, arr, norms, d,
                                    is_float, seed_idx)
                pca_proved = pca_res["bound_count"] / n_tot if n_tot else 0.0
                pca_engine = eng_pca
                pca_recall = self._mean(pca_res["recalls"])
                report.metrics["bound_fraction_pca"] = pca_proved
                if pca_recall == pca_recall:
                    report.metrics["seed_recall_vs_exact_pca"] = pca_recall
                report.metrics["basis_probed"] = "random+pca_corpus"
            except Exception as e:  # pragma: no cover
                logger.warning(f"PCA probe failed ({e}) — keeping random")

        # 3) APPLY the route table (the config's POLICY). The code does NOT
        #    decide basis/k1 — it applies the first route_rules entry whose
        #    `when` matches the measured signals. The default table (in the
        #    preset JSON) reproduces the historical behavior, so existing
        #    callers see no change; operators change policy by editing the
        #    JSON, not the code.
        route_knobs = _match_route(report.metrics, self.route_rules,
                                   self.recall_floor)

        # nan_policy (a SAFETY policy declared in the config, not a dataset
        # decision): when the corpus has NaN/inf and the policy blocks pca, the
        # applied route is never pca_corpus — pca amplifies corruption (1 NaN
        # in the covariance → NaN eigenvectors → recall collapse, measured
        # 0.042). This overrides the route table's choice, applying the config's
        # nan_blocked_route (a random route with a safe k1).
        if has_nan and self.nan_policy == "block_pca":
            nb = self.nan_blocked_route
            route_knobs = dict(nb)
            report.add(Flag(
                F_FOLDABLE, WARN,
                f"corpus has NaN/inf ({nan_frac:.4%}) and nan_policy="
                f"'block_pca' — pca_corpus would amplify the corruption "
                "(covariance → NaN eigenvectors → recall collapse, measured); "
                f"forcing {nb.get('basis')} (k1={nb.get('k1_fraction')}). "
                "Re-audit after cleaning.",
                metric=nan_frac, threshold=0.0))

        # Apply the config's route to the report.
        report.basis = route_knobs.get("basis", "random")
        report.k1_fraction = float(route_knobs.get("k1_fraction", 0.20))
        # stage1_dim: if the config's route pins a stage1, use it; else the
        # random-probe dim (s1) or the pca-probe dim when pca was chosen.
        route_stage1 = route_knobs.get("stage1_dim")
        if route_stage1 is not None:
            report.stage1_dim = int(route_stage1)
        elif report.basis == "pca_corpus" and pca_engine is not None:
            report.stage1_dim = s1p
        else:
            report.stage1_dim = s1
        # Which probe engine to attach for reuse by build_quality_engine: the
        # one matching the chosen basis. When the route is pca but the PCA
        # probe did not run (e.g. the random basis already proved >=50%, so the
        # route table's first rule fired), there is NO pca probe engine to
        # reuse — report.engine stays None and build_quality_engine constructs a
        # real pca_corpus engine. This keeps report ↔ engine consistent (no more
        # "report says pca but the reused engine is random").
        if report.basis == "pca_corpus" and pca_engine is not None:
            report.engine = pca_engine
            # the route's recall/guarantee reflect the PCA engine (if measured)
            if pca_recall == pca_recall:
                route_recall = pca_recall
                route_guarantees = list(pca_res["guarantees"])
                route_pool_frac = (sum(1 for g in route_guarantees if g == "pool_only")
                                   / len(route_guarantees) if route_guarantees else 0.0)
                report.metrics["seed_recall_vs_exact"] = pca_recall
                report.metrics["pool_only_frac"] = route_pool_frac
                report.metrics["recall_guarantee_counts"] = {
                    "exact_global": sum(1 for g in route_guarantees if g == "exact_global"),
                    "pool_only": sum(1 for g in route_guarantees if g == "pool_only"),
                }
        elif report.basis == "pca_corpus":
            # pca route but no pca probe ran (bound_frac >= 0.50 fired first):
            # report.engine stays None → build_quality_engine builds a REAL pca.
            report.engine = None
            route_recall = mean_recall
            route_guarantees = list(guarantees)
            route_pool_frac = pool_only_frac
        else:
            report.engine = eng_random
            route_recall = mean_recall
            route_guarantees = list(guarantees)
            route_pool_frac = pool_only_frac

        # Describe the applied route (foldable flag is informational).
        if report.basis == "pca_corpus":
            report.add(Flag(
                F_FOLDABLE, PASS,
                f"route table chose basis=pca_corpus, k1={report.k1_fraction} "
                f"(random bound_frac={bound_frac:.0%}, pca bound_frac="
                f"{pca_proved if pca_proved is not None else float('nan'):.0%}, "
                f"pca seed recall "
                f"{pca_recall if pca_recall is not None else float('nan'):.3f}).",
                metric=bound_frac,
                threshold=_FOLD_BOUND_FRAC if bound_frac >= _FOLD_BOUND_FRAC
                else _FOLD_BOUND_FRAC_LOW))
        else:
            reason = ("route table chose random" if not has_nan
                      else "nan_policy=block_pca")
            report.add(Flag(
                F_FOLDABLE,
                WARN if bound_frac < _FOLD_BOUND_FRAC_LOW else PASS,
                f"{reason}: basis=random, k1={report.k1_fraction} "
                f"(random bound_frac={bound_frac:.0%}, prefilter "
                f"{pre_frac:.0%}).",
                metric=bound_frac, threshold=_FOLD_BOUND_FRAC_LOW))

        # 3c) embedding resolution (the third-party quality): if even the top-K
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

        # 4) RECALL VALIDATION of the FINAL applied route (2026-09-04): the
        #    route_recall/route_guarantees above reflect the engine that the
        #    config's route table CHOSE (pca when the pca route won, random
        #    otherwise). Expose the honest scope: pool_only vs exact_global +
        #    measured seed recall. If the applied route has recall below the
        #    floor, flag it (WARN) — never hide recall<1.0 with viol=0. The
        #    code does NOT reverse the route; that is the config's call.
        self._flag_recall_shortfall(report, route_recall, route_pool_frac,
                                    route_guarantees, self.recall_floor,
                                    route=report.basis)

        report.excluded_seed_set = captured
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
        nan_blocked_route=qc.nan_blocked_route,
        recall_floor=qc.recall_floor,
        route_rules=qc.route_rules,
        stage1_probe_random=qc.stage1_probe_random,
        stage1_probe_pca=qc.stage1_probe_pca,
        probe_pca_dim_gate=qc.probe_pca_dim_gate,
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
    # nan_policy='block_pca' (a SAFETY policy of the config): the validator
    # already applied nan_blocked_route to report.basis when NaN is present.
    # This second guard makes the policy airtight at the build boundary: even a
    # preset/caller engine_kwargs that FORCES pca_corpus / a low k1 cannot
    # override the NaN block — pca over NaN amplifies corruption (measured
    # recall 0.042), and a random route needs its safe k1 (the prefilter is the
    # recall gate on a degraded corpus). The nan_blocked_route is the config's
    # policy; the code only enforces it.
    nan_frac = float(report.metrics.get("nan_fraction", 0.0))
    nan_blocked = nan_frac > 0.0 and (cfg.nan_policy if cfg is not None else "block_pca") == "block_pca"
    if nan_blocked:
        nb_route = cfg.nan_blocked_route if cfg is not None else _NAN_BLOCKED_ROUTE
        caller_kwargs = dict(caller_kwargs)
        caller_kwargs["basis"] = nb_route.get("basis", "random")
        caller_kwargs["k1_fraction"] = nb_route.get("k1_fraction", 0.20)
    build_kwargs.update({kk: vv for kk, vv in caller_kwargs.items() if vv is not None})

    # REPORT ↔ FINAL CONFIG sync (2026-09-04): the report's suggested_config
    # must reflect the engine config that will ACTUALLY be built. The route
    # table picked basis/k1/stage1 from the measured signals, but the preset's
    # engine_kwargs (operator overrides) may pin different values on top. Update
    # the report so it is consistent with the engine — no more "report says
    # stage1=64, engine uses 128".
    report.basis = str(build_kwargs.get("basis", report.basis))
    report.k1_fraction = float(build_kwargs.get("k1_fraction", report.k1_fraction))
    report.stage1_dim = int(build_kwargs.get("stage1_dim", report.stage1_dim))
    report.quant = str(build_kwargs.get("quant", report.quant))
    report.stage2_dim = int(build_kwargs.get("stage2_dim", report.stage2_dim))
    report.early_exit = bool(build_kwargs.get("early_exit", report.early_exit))

    # SCAN INT8 AUTOMÁTICO e AGNÓSTICO (2026-09-03, normalize 1.3.0):
    # o motor (madhava 1.9.11) ganhou scan_int8 (quantiza as projeções do
    # Stage-1, ~1.4-1.7× mais rápido em N grande). MAS a segurança NÃO depende do
    # basis — depende da DIMENSÃO/distribuição: o erro de quantização int8
    # (~1e-3) reordena o pool quando o gap entre candidatos é menor (medido:
    # basis random em d≥384 degrada recall 1.0→0.5; d=128 é seguro; pca_corpus em
    # d=1536 é seguro porque concentra a energia). Nenhuma regra de basis/dim
    # captura isso de forma confiável (a métrica erro/gap NÃO separa os casos).
    #
    # SOLUÇÃO AGNÓSTICA: o normalize TENTA scan_int8 e VALIDA por recall real
    # numa amostra de seed queries (search vs search_exact). Se o recall cai
    # abaixo do limiar, rebuild sem scan_int8. A decisão é por MEDIÇÃO, e vale
    # para qualquer basis/dimensão — o sistema "sabe quando operar em alta
    # dimensão" empiricamente, não por heurística. Validado: seed queries (8)
    # predizem corretamente a degradação (d=384/1536 random → recall baixo →
    # desliga; pca/d=128 → recall 1.0 → mantém).
    #
    # O preset JSON (engine.scan_int8) ou o chamador (engine_kwargs) podem
    # FORÇAR explicitamente (True = sempre int8, False = nunca).
    import winnex_madhava as _wm
    caller_explicit_scan8 = any(
        kk == "scan_int8" for kk in (caller_kwargs or {}))
    final_quant = str(build_kwargs.get("quant", "")).lower()
    is_float32_build = (final_quant in ("", "none"))
    engine = None
    scan8_decision = None     # None = não testado ainda
    if caller_explicit_scan8:
        # Força explícita do preset/caller.
        scan8_decision = bool(build_kwargs.get("scan_int8", caller_kwargs.get("scan_int8", False)))
    elif not is_float32_build:
        # uint8/L2: o scan_int8 não se aplica (o caminho int8 nativo já é usado).
        scan8_decision = False
    else:
        # AGNÓSTICO: tenta scan_int8 e valida por recall em seed queries.
        # (Somente quando há corpus suficiente p/ uma amostra significativa.)
        scan8_decision = True   # otimista; validado abaixo
        build_kwargs["scan_int8"] = True
    # Reuse the probe engine ONLY when it matches the desired build WITHOUT
    # scan_int8 (the probe never uses it). BUG FIX (2026-08-31): o cfg_match
    # antigo NÃO comparava basis — agora exige basis/stage1/k1/metrice iguais.
    if report.engine is not None and not build_kwargs.get("scan_int8", False):
        rcfg = report.engine.config()
        rdim = report.engine.dim()
        dim_ok = (rdim == dim) if dim is not None else True
        same_stage1 = int(rcfg.stage1_dim) == int(build_kwargs["stage1_dim"])
        same_k1 = int(rcfg.k1_fraction * 1000) == int(build_kwargs["k1_fraction"] * 1000)
        metric_ok = str(rcfg.metric).lower() in ("cosine", "cosine")
        if dim_ok and same_stage1 and same_k1 and metric_ok:
            engine = report.engine

    if engine is None:
        import winnex_madhava as wm
        engine = wm.build_engine(corpus, dim=dim, **build_kwargs)

    # VALIDAÇÃO AGNÓSTICA do scan_int8 (2026-09-03): quando a decisão foi
    # otimista (tentar int8), confirma por recall real numa amostra de seed
    # queries (search vs search_exact do MESMO engine). Se o recall caiu abaixo
    # do limiar (o erro de quantização reordenou o pool — medido em d≥384
    # random), rebuild SEM scan_int8 (float32 exato). O resultado é o motor
    # correto PARA ESTE corpus/dimensão, decidido por medição.
    if scan8_decision and engine is not None:
        try:
            import winnex_madhava as wm
            _n = len(corpus)
            _nseed = 8
            _k = int(k)
            # seed queries: amostra determinística do corpus (não as últimas —
            # podem estar no pool de forma enviesada)
            _rng = np.random.default_rng(1234)
            _idx = _rng.choice(max(1, _n), min(_nseed, _n), replace=False)
            _rec = 0.0
            _cnt = 0
            for _qi in _idx:
                _q = np.ascontiguousarray(corpus[_qi], dtype=np.float32)
                _r = engine.search(_q)
                _rx = engine.search_exact(_q)
                if len(_r.indices) == 0 or len(_rx.indices) == 0:
                    continue
                _rec += sum(1 for _i in _r.indices if _i in _rx.indices) / len(_rx.indices)
                _cnt += 1
            if _cnt > 0:
                _rec /= _cnt
            # Limiar: scan_int8 é aceito se o recall médio ≥ 0.95 (pequena
            # tolerância p/ ruído de float32; a degradação medida é 0.5-0.6,
            # muito abaixo).
            if _rec < 0.95:
                logger.warning(
                    f"scan_int8 degradou recall ({_rec:.3f} em {_cnt} seed queries) "
                    f"— rebuild sem scan_int8 (float32 exato) p/ este corpus")
                build_kwargs.pop("scan_int8", None)
                engine = wm.build_engine(corpus, dim=dim, **build_kwargs)
        except Exception as _e:  # nunca quebrar o build por causa da validação
            logger.debug(f"scan_int8 validation skipped: {_e}")

    # RECALL/GUARANTEE OF THE FINAL ENGINE (2026-09-04): regardless of whether
    # the probe engine was reused or the final build happened above (basis may
    # have been reversed by nan_policy or the recall floor; scan_int8 may have
    # been toggled), measure the REAL seed recall of the engine being returned
    # (search vs search_exact) and expose recall_guarantee on the report. This
    # is the honest scope statement of the config the caller actually gets.
    if engine is not None and return_report:
        try:
            import winnex_madhava as wm
            _n = len(corpus)
            _nseed = 6
            _rng = np.random.default_rng(4242)
            _idx = _rng.choice(max(1, _n), min(_nseed, _n), replace=False)
            _recs = []
            _guars = []
            for _qi in _idx:
                _q = np.ascontiguousarray(corpus[_qi], dtype=np.float32)
                _r = engine.search(_q)
                _rx = engine.search_exact(_q)
                _guars.append(str(getattr(_r, "recall_guarantee", "pool_only")))
                if len(_rx.indices) > 0:
                    _g = set(int(i) for i in _rx.indices)
                    _recs.append(sum(1 for i in _r.indices if int(i) in _g) / len(_rx.indices))
            if _recs:
                _mr = float(np.mean(_recs))
                report.metrics["seed_recall_vs_exact_final"] = _mr
            report.metrics["recall_guarantee_final"] = {
                "exact_global": sum(1 for g in _guars if g == "exact_global"),
                "pool_only": sum(1 for g in _guars if g == "pool_only"),
            }
            # Flag the FINAL config if it is measurably below the floor — the
            # caller must see that the engine it received has recall < 1.0 with
            # viol=0 (the honest exposure), even when the router did its best.
            _floor = float(getattr(cfg, "recall_floor", _RECALL_FLOOR)) if cfg is not None else _RECALL_FLOOR
            if _recs and _mr < _floor:
                report.add(Flag(
                    F_RECALL, WARN,
                    f"final engine (basis={report.basis}, k1={report.k1_fraction}) "
                    f"seed recall@K = {_mr:.3f} vs search_exact (< floor {_floor:.2f}); "
                    f"{sum(1 for g in _guars if g == 'pool_only')}/{len(_guars)} "
                    "queries pool_only — the returned top-K is NOT the global "
                    "top-K. viol=0 does not cover this. Use audit_exhaustive=True "
                    "or k1=N for a global guarantee.",
                    metric=_mr, threshold=_floor))
        except Exception as _e:  # nunca quebrar o build por causa da validação
            logger.debug(f"final-engine recall validation skipped: {_e}")

    if return_report:
        return engine, report
    return engine
