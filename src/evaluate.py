"""Métricas e diagnósticos de avaliação (cresce nas etapas seguintes)."""
import numpy as np
import pandas as pd


def psi(esperado, atual, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index. Numérica: faixas por quantis do `esperado`; categórica: por categoria.

    Nulos formam uma faixa própria. Regra usual: <0,10 estável, 0,10-0,25 atenção, >0,25 mudança relevante.
    """
    e, a = pd.Series(esperado).reset_index(drop=True), pd.Series(atual).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(e):
        corte = np.unique(np.quantile(e.dropna(), np.linspace(0, 1, bins + 1)))
        corte[0], corte[-1] = -np.inf, np.inf
        e, a = (pd.cut(s, corte).astype(object) for s in (e, a))
    pe = e.fillna("nulo").astype(str).value_counts(normalize=True)
    pa = a.fillna("nulo").astype(str).value_counts(normalize=True)
    pe, pa = pe.reindex(pe.index.union(pa.index), fill_value=0).clip(eps), pa.reindex(pe.index.union(pa.index), fill_value=0).clip(eps)
    return float(((pa - pe) * np.log(pa / pe)).sum())
