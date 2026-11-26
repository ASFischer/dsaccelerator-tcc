import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as ft
import policy as po
import train as tr

OBS = pd.Timestamp("2025-02-22")


def test_submissao_uma_linha_por_cliente_com_features():
    sub = pd.DataFrame({"id_cliente": [1, 2, 3], "tipo_contrato": ["a"] * 3, "valor_credito": [10., 20., 30.],
                        "valor_bem": [9., 19., 29.], "valor_parcela": [1., 2., 3.], "dia_semana_solicitacao": ["x"] * 3,
                        "hora_solicitacao": ["10:00:00"] * 3, "data_solicitacao": pd.to_datetime(["2024-05-01", None, "2024-06-01"])})
    cad = pd.DataFrame({"id_cliente": [1, 2, 3], "data_nascimento": pd.to_datetime(["1980-01-01"] * 3)})
    out = ft.base_submissao(sub, cad, OBS)
    assert len(out) == 3 and out.id_cliente.is_unique
    assert out.idade.notna().all()  # data de solicitação nula -> usa a data de observação
    assert out.pct_valor_credito.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_categoria_nova_avisa_e_vira_nula():
    treino = pd.DataFrame({"sexo": ["F", "M", "F"], "idade": [30, 40, 50]})
    cats = tr.categorias_do_treino(treino)
    novo = pd.DataFrame({"sexo": ["F", "X"], "idade": [30, 31]})
    with pytest.warns(UserWarning, match="sexo"):
        X = tr.preparar(novo, ["sexo", "idade"], cats)
    assert X.sexo.isna().tolist() == [False, True]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tr.preparar(treino, ["sexo", "idade"], cats)  # sem categoria nova, sem aviso


def test_cortes_da_politica():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    p1 = po.corte_p1(y, p, 0.95)
    assert y[p >= p1].sum() / y.sum() >= 0.95 and y[p >= p1 + 0.01].sum() / y.sum() < 0.95
    p2 = po.corte_p2(y, p, p1)
    assert p2 > p1
    cf = po.cortes_faixas(p, p1, p2)
    assert cf == sorted(cf)
    assert set(po.faixa(p, cf)) == set(po.FAIXAS)
    with pytest.raises(AssertionError, match="zona de ajuste vazia"):
        po.cortes_faixas(p, p2, p2)


def test_calibrador_monotono_e_wilson():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p ** 2).astype(int)
    for metodo in ("platt", "isotonica"):
        q = po.calibrador(metodo, p, y)(np.sort(p))
        assert np.all(np.diff(q) >= 0) and q.min() >= 0 and q.max() <= 1
    lo, hi = po.wilson(10, 100)
    assert lo < 0.1 < hi and 0 < lo and hi < 1


def test_previsoes_geradas_consistentes():
    f = Path(__file__).resolve().parents[1] / "submissao" / "previsoes.csv"
    if not f.exists():
        pytest.skip("rode o nb04 para gerar submissao/previsoes.csv")
    d = pd.read_csv(f)
    assert d.id_cliente.is_unique and d.probabilidade.between(0, 1).all()
    assert (d.decisao == d.faixa_gestao.map(po.DECISAO)).all()


def test_modelo_final_reproduz_previsoes():
    raiz = Path(__file__).resolve().parents[1]
    arq, prev, base = raiz / "models" / "modelo_final.joblib", raiz / "submissao" / "previsoes.csv", raiz / "data" / "base_submissao_features.parquet"
    if not (arq.exists() and prev.exists() and base.exists()):
        pytest.skip("rode os notebooks para gerar o modelo, as previsões e a base de submissão")
    import joblib
    a, bs, d = joblib.load(arq), pd.read_parquet(base), pd.read_csv(prev)
    q = a["calibrador"](a["modelo"].predict_proba(tr.preparar(bs, a["features"], a["categorias"]))[:, 1])
    assert np.allclose(q, d.probabilidade.values)
    assert a["p1"] < a["p2"]
