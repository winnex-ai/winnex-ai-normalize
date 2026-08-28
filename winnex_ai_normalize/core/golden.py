"""
winnex-ai-normalize — golden set / retrieval model card (Phase 2 / GAIA).

The Madhava guarantee is soundness of pruning — it does NOT cover embedding
semantic quality. A weak provider ("blurry photo") can return the WRONG
document with a perfect proof. This module makes embedding quality MEASURED
and MONITORED per provider, so the "blurry photo" is quantified instead of
assumed:

  - A golden set = seed queries + ground-truth relevant documents, per domain.
  - eval_provider() runs the FULL pipeline (embed → index → search) over the
    golden set and produces a retrieval MODEL CARD: semantic recall@K,
    per-query resolution (top-1 vs top-K exact cosine gap), and drift flags.
  - The model card is the contract floor: an operator can require
    "recall@5 >= 0.8 on the legal golden set" before promoting a provider to
    production — converting "the embedding is good/bad" from an opinion into
    a verifiable number.

The golden sets ship as TEXT (queries + candidate docs) so they are provider-
agnostic: any embedding service can be evaluated against the same ground truth.
This is the honest answer to the GAIA "foto borrada": we measure the photo's
resolution, not just guarantee the item-by-item proof.

License: Business Source License 1.1 (BSL 1.1)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger("winnex-ai-normalize.golden")


# ---------------------------------------------------------------------------
# Golden sets (seed queries + ground-truth relevant doc indices)
# ---------------------------------------------------------------------------
# Structure: {domain: {"queries": [str], "documents": [str], "relevant": [[idx...] per query]}}
# Documents are short domain texts; relevant[i] lists the indices of documents
# that MUST be retrieved for query i. Built to be provider-agnostic (text, not
# vectors) so any embedding model can be scored against the same truth.

GOLDEN_SETS: Dict[str, Dict[str, Any]] = {
    "legal": {
        "documents": [
            "cláusula de rescisão contratual por inadimplemento do fornecedor",
            "prazo de prescrição aplicável às ações de cobrança de honorários",
            "responsabilidade civil do Estado por ato omissivo em serviço público",
            "requisitos da citação válida no processo civil brasileiro",
            "tese da prescrição intercorrente no cumprimento de sentença",
            "distinção entre dano moral e dano material na responsabilidade",
            "competência territorial para ação de cobrança de aluguéis",
            "honorários advocatícios sucumbenciais e sua base de cálculo",
        ],
        "queries": [
            "prazo para cobrar honorários advocatícios prescritos",
            "quando o Estado responde por omissão em serviço público",
            "como citar o réu no processo civil",
        ],
        "relevant": [
            [1, 7],      # query 0 → prescrição + honorários
            [2],         # query 1 → responsabilidade do Estado
            [3],         # query 2 → citação
        ],
    },
    "medical": {
        "documents": [
            "hipoglicemia neonatal: manejo imediato e monitoramento glicêmico",
            "síndrome do desconforto respiratório do recém-nascido",
            "hemorragia pós-parto: protocolo de tratamento com ocitocina",
            "sepse neonatal precoce: critérios diagnósticos e antibioticoterapia",
            "icterícia neonatal: avaliação de bilirrubina e fototerapia",
            "trabalho de parto prematuro: tocólise e corticoterapia",
            "apgar neonatal: interpretação e manejo de reanimação",
            "amamentação: complicações comuns e suporte",
        ],
        "queries": [
            "recém-nascido com falta de oxigênio como reanimar",
            "tratamento de infecção grave no bebê",
            "bebê amarelo precisa de luz",
        ],
        "relevant": [
            [6, 1],   # query 0 → apgar/reanimação + desconforto respiratório
            [3],      # query 1 → sepse
            [4],      # query 2 → icterícia
        ],
    },
    "financial": {
        "documents": [
            "contabilidade de instrumentos financeiros derivativos",
            "reconhecimento de receita sob IFRS 15",
            "gestão de risco cambial em empresas exportadoras",
            "teste de impairment de ativos de longo prazo",
            "apuração de lucro real e presunção fiscal",
            "cobrança de aluguéis inadimplentes em fundos imobiliários",
            "due diligence fiscal em aquisição de empresas",
            "provisão para créditos de liquidação duvidosa",
        ],
        "queries": [
            "como contabilizar derivativos de câmbio",
            "quando reconhecer receita de contrato longo",
            "risco de câmbio em empresa que exporta",
        ],
        "relevant": [
            [0],      # query 0 → derivativos
            [1],      # query 1 → receita
            [2],      # query 2 → risco cambial
        ],
    },
}

# Default: how many documents to retrieve when scoring (recall@K floor).
DEFAULT_K = 3


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
@dataclass
class RetrievalModelCard:
    """The measured quality of a provider on a golden set.

    This is the contract floor for "is this embedding good enough": it turns
    the "blurry photo" concern (Phase 1 / GAIA) into a verifiable number.
    """
    provider: str
    model: str
    domain: str
    k: int
    semantic_recall: float = 0.0        # mean recall@k over the golden queries
    query_resolutions: List[dict] = field(default_factory=list)  # per-query gap
    mean_resolution: float = 0.0        # top-1 vs top-K exact cosine gap
    bound_violations: int = 0            # the engine's proof must be 0
    n_queries: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"model-card[{self.provider}:{self.model}] domain={self.domain} "
                f"k={self.k} semantic_recall@{self.k}={self.semantic_recall:.3f} "
                f"mean_resolution={self.mean_resolution:.3f} "
                f"bound_violations={self.bound_violations} queries={self.n_queries}")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
def eval_provider(provider, model: str, domain: str = "legal",
                  k: int = DEFAULT_K, seed: int = 42) -> RetrievalModelCard:
    """Score an embedding provider against a golden set → RetrievalModelCard.

    Runs the real pipeline: provider.embed(texts) → L2-normalize → build the
    Madhava engine (basis='pca_corpus' for a tight bound) → search each golden
    query → measure semantic recall@k + per-query resolution + bound violations.

    Parameters
    ----------
    provider : object
        Anything with `.embed(texts) -> (n,d) float32` (e.g. EmbeddingProvider,
        EmbeddingService, or a duck-typed local embedder). L2-normalization is
        applied inside.
    model : str
        Provider/model identifier for the card (e.g. 'qwen3' / 'text-embedding-3-small').
    domain : str
        Which golden set to use ('legal', 'medical', 'financial').
    k : int
        Recall@k floor (default 3 — the Maestro's RAG context size).
    seed : int
        Determinism seed for the engine build.

    Returns
    -------
    RetrievalModelCard — the measured quality, ready to be logged or asserted
    against a contract floor ("recall@3 >= 0.8 before promoting this model").
    """
    import winnex_madhava as wm

    if domain not in GOLDEN_SETS:
        raise ValueError(f"unknown golden domain {domain!r}; choose from {list(GOLDEN_SETS)}")
    gs = GOLDEN_SETS[domain]
    docs = gs["documents"]
    queries = gs["queries"]
    relevant = gs["relevant"]

    # Embed once for the index and once per query.
    doc_vecs = np.asarray(provider.embed(docs), dtype=np.float32)
    doc_vecs = doc_vecs / np.maximum(np.linalg.norm(doc_vecs, axis=1, keepdims=True), 1e-12)
    d = doc_vecs.shape[1]
    n = doc_vecs.shape[0]

    # Build the engine with a tight basis (pca_corpus) so the bound is active.
    eng = wm.build_engine(doc_vecs, dim=d, metric="cosine", quant="none",
                          basis="pca_corpus", stage1_dim=min(192, d),
                          stage2_dim=0, k=k, seed=seed, early_exit=False)

    recalls = []
    resolutions = []
    bound_viols = 0
    for qi, qtext in enumerate(queries):
        qv = np.asarray(provider.embed([qtext]), dtype=np.float32)[0]
        qv = qv / np.maximum(np.linalg.norm(qv), 1e-12)
        r = eng.search(qv.astype(np.float32))
        gt = set(relevant[qi])
        hit = len(set(r.indices) & gt) / len(gt) if gt else 0.0
        recalls.append(hit)
        bound_viols += int(r.bound_violations)
        # resolution: top-1 vs top-K exact cosine gap (semantic separation)
        if len(r.indices) >= 2:
            c1 = float(doc_vecs[r.indices[0]] @ qv)
            ck = float(doc_vecs[r.indices[-1]] @ qv)
            resolutions.append({"seed_query": qi, "query": qtext[:40], "gap": c1 - ck})

    card = RetrievalModelCard(
        provider=provider.name if hasattr(provider, "name") else "?",
        model=model,
        domain=domain,
        k=k,
        semantic_recall=float(np.mean(recalls)) if recalls else 0.0,
        query_resolutions=resolutions,
        mean_resolution=float(np.mean([x["gap"] for x in resolutions])) if resolutions else 0.0,
        bound_violations=bound_viols,
        n_queries=len(queries),
    )
    logger.info("golden: %s", card.summary())
    return card


def check_contract(card: RetrievalModelCard, min_recall: float = 0.8,
                   min_resolution: float = 0.05) -> bool:
    """Contract floor check: is this provider good enough to promote?

    The default floor (recall@3 >= 0.8 AND mean_resolution >= 0.05) is a
    sensible baseline; operators override per domain. A provider that fails
    the floor is a "blurry photo" — the proof still runs (0 violations), but
    the retrieval can be wrong. This is the Phase 2 answer to GAIA: we make
    the embedding-quality dependency MEASURED, MONITORED and BLOCKABLE instead
    of assumed.
    """
    ok_recall = card.semantic_recall >= min_recall
    ok_resolution = card.mean_resolution >= min_resolution
    ok_proof = card.bound_violations == 0
    return ok_recall and ok_resolution and ok_proof


# ---------------------------------------------------------------------------
# CLI convenience: python -m winnex_ai_normalize.core.golden <domain>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import winnex_madhava as wm

    dom = sys.argv[1] if len(sys.argv) > 1 else "legal"

    # A synthetic "local embedder" (SpectralTokenizer-like) so the golden set
    # can be exercised without an HTTP provider. This is a STAND-IN for a real
    # provider — it shows the card format, not a real semantic model.
    class _SyntheticEmbedder:
        name = "synthetic-local"
        def embed(self, texts):
            # character-bigram hashing — low semantic quality (deliberately,
            # to show a weak provider on the card)
            rng = np.random.default_rng(0)
            v = np.zeros((len(texts), 64), dtype=np.float32)
            for i, t in enumerate(texts):
                for ch in t:
                    h = (hash(ch) % 1000)
                    v[i, h % 64] += 1.0
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            return v

    card = eval_provider(_SyntheticEmbedder(), "synthetic-char-bigram", domain=dom)
    print(json.dumps(card.to_dict(), indent=2, ensure_ascii=False))
    print("contract(0.8, 0.05):", check_contract(card))
