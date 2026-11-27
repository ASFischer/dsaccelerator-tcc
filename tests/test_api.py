"""API de score: contrato de entrada (422), reprodução de submissao/previsoes.csv, avisos e explicação."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if not (RAIZ / "models" / "modelo_final.joblib").exists():
    pytest.skip("rode o nb04 para gerar models/modelo_final.joblib", allow_module_level=True)
sys.path.insert(0, str(RAIZ))
from fastapi.testclient import TestClient  # noqa: E402

from api.main import EXEMPLO, app  # noqa: E402

api = TestClient(app)


def test_saude():
    r = api.get("/saude").json()
    assert r["status"] == "ok" and r["p1"] < r["p2"]


@pytest.fixture(scope="module")
def submissao():
    import data_loader as dl
    import features as ft
    from config import DATA_OBSERVACAO
    bs = ft.base_submissao(dl.carregar_submissao(), dl.carregar_cadastral(), pd.Timestamp(DATA_OBSERVACAO))
    prev = pd.read_csv(RAIZ / "submissao" / "previsoes.csv").set_index("id_cliente")
    return bs, prev


def test_score_reproduz_previsoes(submissao):
    bs, prev = submissao
    linhas = bs.sample(20, random_state=0)
    payload = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in linhas[list(EXEMPLO)].to_dict("records")]
    r = api.post("/score", json=payload).json()["clientes"]
    esperado = prev.loc[linhas.id_cliente]
    assert np.allclose([c["probabilidade"] for c in r], esperado.probabilidade)
    assert [c["decisao"] for c in r] == esperado.decisao.tolist()


def test_entrada_invalida_da_422():
    assert api.post("/score", json=[{**EXEMPLO, "idade": "trinta"}]).status_code == 422
    assert api.post("/score", json=[{**EXEMPLO, "idadee": 30}]).status_code == 422
    assert api.post("/score", json=[]).status_code == 422
    assert api.post("/score", json=[EXEMPLO] * 1001).status_code == 422


def test_categoria_nova_avisa_e_explicar():
    r = api.post("/score?explicar=true", json=[{**EXEMPLO, "sexo": "X"}, {}]).json()
    assert any("sexo" in a for a in r["avisos"])
    assert len(r["clientes"]) == 2 and len(r["clientes"][0]["contribuicoes"]) == 6
    assert "contribuicoes" not in api.post("/score", json=[EXEMPLO]).json()["clientes"][0]
