"""Núcleo de pontuação usado pela API (api/main.py) e pelo app (app/streamlit_app.py).

Recebe um DataFrame com as variáveis de `features.FEATURES["A"]` (já derivadas, como em
`features.base_submissao`) e devolve probabilidade calibrada, decisão da política e avisos.
"""
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import policy as po
import train as tr
from config import MODELOS

TOL_DEGRAU = 1e-4  # 38% da submissão fica no degrau da isotônica em ≈ p1 (DECISOES 4.1/4.4)


@lru_cache
def carregar(caminho=MODELOS / "modelo_final.joblib") -> dict:
    """Artefato do nb04 (pickle: carregue só arquivos gerados por você)."""
    return joblib.load(caminho)


def _x(df: pd.DataFrame):
    a = carregar()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X = tr.preparar(df, a["features"], a["categorias"])
    return X, [str(x.message) for x in w if issubclass(x.category, UserWarning)]


def pontuar(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """(tabela por cliente, avisos). Avisos = categorias fora do treino que viraram nulo."""
    a = carregar()
    X, avisos = _x(df)
    bruto = a["modelo"].predict_proba(X)[:, 1]
    prob = a["calibrador"](bruto)
    faixa = po.faixa(prob, a["cortes_faixas"]).values
    return pd.DataFrame({"probabilidade": prob, "score_bruto": bruto,
                         "no_degrau_p1": np.abs(prob - a["p1"]) < TOL_DEGRAU,
                         "decisao": pd.Series(faixa).map(po.DECISAO).values, "faixa_gestao": faixa},
                        index=df.index), avisos


@lru_cache
def _explicador():
    import shap  # import pesado: só quando pedem explicação
    return shap.TreeExplainer(carregar()["modelo"])


def explicar(df: pd.DataFrame, top: int = 6) -> list[dict]:
    """Por cliente, as `top` maiores contribuições SHAP em log-odds do **score bruto** (não somam à probabilidade calibrada)."""
    X, _ = _x(df)
    v = _explicador()(X).values
    v = v[:, :, 1] if v.ndim == 3 else v
    return [pd.Series(linha, index=X.columns).sort_values(key=abs, ascending=False).head(top).round(4).to_dict()
            for linha in v]
