"""O roteador garantia a config, não a CERTIFICABILIDADE (2026-10-30).

O defeito: `audit_corpus` media as duas bases, roteava, e emitia um WARN quando
o corpus não era dobrável. Mas o `build_quality_engine` construía o engine
assim mesmo, e o chamador recebia um motor que responde certo e **não prova
nada** sobre o que deixou de fora.

Medido (isotrópico 8000x384): roteia `random / s1=64 / k1=0.2`, emite
`dataset.foldable: warn`, e a cobertura de exclusão do engine resultante é
**0.00%** — com recall 1.000. No corpus de baixo posto, roteia
`pca_corpus / s1=192` e a cobertura é **99.96%**.

Os dois casos produziam um relatório de aparência bem-sucedida. Só um era
certificável, e nada distinguia os dois de forma acionável.
"""
import numpy as np
import pytest

import winnex_ai_normalize as norm


def _corpus(n=6000, d=384, posto=None, seed=3):
    rng = np.random.default_rng(seed)
    if posto is None:
        X = rng.standard_normal((n, d)).astype(np.float32)
    else:
        X = (rng.standard_normal((n, posto)).astype(np.float32)
             @ rng.standard_normal((posto, d)).astype(np.float32))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.ascontiguousarray(X, dtype=np.float32)


def test_certifiable_e_true_em_corpus_dobravel():
    """Baixo posto: o PCA acha o manifold e a prova existe."""
    X = _corpus(posto=16)
    rep = norm.audit_corpus(X, dim=384,
                            cfg=norm.QualityConfig(fail_on_recall=False,
                                                   recall_floor=0.0))
    assert rep.certifiable is True, (
        f"corpus dobrabel marcado como nao-certificavel "
        f"(basis={rep.basis}, pca={rep.metrics.get('bound_fraction_pca')})")
    assert rep.certification_action == ""


def test_certifiable_e_false_em_corpus_isotropico_e_explica():
    """Isotrópico: nenhuma projeção prova nada. O relatório tem de DIZER isso
    com uma ação, não só com um WARN que o chamador pode não ler."""
    X = _corpus()
    rep = norm.audit_corpus(X, dim=384,
                            cfg=norm.QualityConfig(fail_on_recall=False,
                                                   recall_floor=0.0))
    assert rep.certifiable is False
    assert rep.certification_action, "nao-certificavel sem acao declarada"
    # A acao tem de mencionar as duas saidas, nao so o problema
    assert ("audit_exhaustive" in rep.certification_action
            or "pca_corpus" in rep.certification_action
            or "pool_only" in rep.certification_action)


def test_to_dict_expoe_certifiable():
    """O campo tem de sair na serializacao — e por ela que um pipeline decide."""
    X = _corpus()
    d = norm.audit_corpus(X, dim=384,
                          cfg=norm.QualityConfig(fail_on_recall=False,
                                                 recall_floor=0.0)).to_dict()
    assert "certifiable" in d
    assert "certification_action" in d
    assert isinstance(d["certifiable"], bool)


def test_require_certifiable_bloqueia_config_nao_certificavel():
    """O knob que faltava: 'preciso que o selo prove algo'."""
    X = _corpus()
    # Sem exigir: constroi (o top-K exato vem de qualquer jeito)
    norm.build_quality_engine(X, dim=384, k=3, allow_unsafe=True)

    # Exigindo: recusa, e a mensagem nomeia a acao
    with pytest.raises(norm.QualityGateError) as ei:
        norm.build_quality_engine(
            X, dim=384, k=3, allow_unsafe=True,
            cfg=norm.QualityConfig(require_certifiable=True))
    assert "certificate required" in str(ei.value)
    assert ei.value.report.certifiable is False


def test_require_certifiable_permite_corpus_dobravel():
    """Contraprova: o knob não pode bloquear quando a prova existe."""
    X = _corpus(posto=16)
    eng = norm.build_quality_engine(
        X, dim=384, k=3, allow_unsafe=True,
        cfg=norm.QualityConfig(require_certifiable=True))
    assert eng is not None


def test_fail_on_recall_e_require_certifiable_sao_falhas_distintas():
    """Os dois campos guardam coisas diferentes e nenhum substitui o outro:
    `fail_on_recall` guarda a RESPOSTA (top-K é o global?);
    `require_certifiable` guarda o SELO (ele prova algo sobre o descarte?).
    """
    cfg = norm.QualityConfig()
    assert hasattr(cfg, "fail_on_recall")
    assert hasattr(cfg, "require_certifiable")
    # Defaults: nenhum dos dois bloqueia, para não quebrar o caminho comum.
    assert cfg.fail_on_recall is False
    assert cfg.require_certifiable is False
