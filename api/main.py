"""API de score (FastAPI): como um sistema de crédito consumiria o modelo.

Rodar da raiz do projeto: `uvicorn api.main:app --port 8000` e abrir http://localhost:8000/docs
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # o pickle do modelo precisa de `policy`
import scoring as sc  # noqa: E402
from config import MODELOS  # noqa: E402

# cliente 100023 da submissão (probabilidade 0,1179 -> aprovar com ajuste, no degrau de p1)
EXEMPLO = {"tipo_contrato": "Financiamento de consumo", "dia_semana_solicitacao": "segunda-feira", "hora": 12,
           "sexo": "Feminino", "idade": 31.07, "qtd_filhos": 1, "qtd_membros_familia": 2, "renda_anual": 90000,
           "tipo_renda": "Servidor público", "ocupacao": "Equipe administrativa",
           "tipo_organizacao": "Educação infantil", "nivel_educacao": "Ensino superior", "estado_civil": "Solteiro",
           "tipo_moradia": "Casa ou apartamento", "possui_carro": "Não", "possui_imovel": "Sim",
           "nota_regiao_cliente": 2, "nota_regiao_cliente_cidade": 2}


class Cliente(BaseModel):
    """Variáveis já derivadas (como em `features.base_submissao`). Campo ausente ou nulo = valor faltante."""
    model_config = ConfigDict(extra="forbid")  # campo com nome errado dá 422 em vez de virar nulo calado
    tipo_contrato: str | None = None
    dia_semana_solicitacao: str | None = None
    hora: float | None = Field(None, ge=0, le=23)
    sexo: str | None = None
    idade: float | None = Field(None, description="anos na data da solicitação")
    qtd_filhos: float | None = None
    qtd_membros_familia: float | None = None
    renda_anual: float | None = None
    tipo_renda: str | None = None
    ocupacao: str | None = None
    tipo_organizacao: str | None = None
    nivel_educacao: str | None = None
    estado_civil: str | None = None
    tipo_moradia: str | None = None
    possui_carro: str | None = None
    possui_imovel: str | None = None
    nota_regiao_cliente: float | None = None
    nota_regiao_cliente_cidade: float | None = None


app = FastAPI(title="Score de inadimplência", version="1.0",
              description="LightGBM + calibração isotônica; política p1/p2 do notebook 04.")


@app.get("/saude")
def saude():
    a = sc.carregar()
    data = datetime.fromtimestamp((MODELOS / "modelo_final.joblib").stat().st_mtime).isoformat(timespec="seconds")
    return {"status": "ok", "algoritmo": a["algoritmo"], "metodo_calibracao": a["metodo_calibracao"],
            "versoes_bibliotecas": a["versoes"], "data_modelo": data, "p1": a["p1"], "p2": a["p2"],
            "metricas": a["metricas"]}


@app.post("/score")
def score(clientes: Annotated[list[Cliente], Body(min_length=1, max_length=1000, examples=[[EXEMPLO]])],
          explicar: bool = False):
    """Probabilidade calibrada, decisão e faixa. `explicar=true` adiciona as 6 maiores contribuições SHAP
    (log-odds do score bruto: não somam à probabilidade calibrada)."""
    df = pd.DataFrame([c.model_dump() for c in clientes])
    df = df.astype({c: float for c in df if c not in sc.carregar()["categorias"]})  # coluna toda nula -> NaN, não object
    r, avisos = sc.pontuar(df)
    saida = r.to_dict("records")
    if explicar:
        for s, e in zip(saida, sc.explicar(df)):
            s["contribuicoes"] = e
    return {"clientes": saida, "avisos": avisos}
