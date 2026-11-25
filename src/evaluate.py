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


def iv_woe(x, y, bins: int = 10):
    """Information Value e tabela WoE de `x` contra o target binário `y`.

    Numérica com mais de `bins` valores: quantis (f01, f02, ... em ordem crescente); nulo é faixa própria.
    Regra usual de IV: < 0,02 sem poder; 0,02-0,1 fraco; 0,1-0,3 médio; > 0,3 forte.
    """
    x, y = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True).astype(int)
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        x = pd.qcut(x, bins, labels=False, duplicates="drop").map(lambda i: f"f{int(i) + 1:02d}" if pd.notna(i) else None)
    g = pd.crosstab(x.fillna("nulo").astype(str), y).reindex(columns=[0, 1], fill_value=0)
    d = (g + 0.5) / (g + 0.5).sum()
    woe = np.log(d[0] / d[1])
    return float(((d[0] - d[1]) * woe).sum()), pd.DataFrame({"n": g.sum(axis=1), "taxa_inad": g[1] / g.sum(axis=1), "woe": woe})
