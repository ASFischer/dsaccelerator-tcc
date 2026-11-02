import numpy as np
import pandas as pd

# cortes sobre a probabilidade de inadimplência (>=30d nas 12 primeiras parcelas), escolhidos na validação e conferidos no teste (nb 06)
CORTE_APROVAR = 0.10
CORTE_REJEITAR = 0.25


def decisao(p) -> np.ndarray:
    return np.select([p < CORTE_APROVAR, p < CORTE_REJEITAR], ["aprovar", "aprovar com ajuste"], "rejeitar")


def resumo(y, p) -> pd.DataFrame:
    d = pd.DataFrame({"decisao": decisao(np.asarray(p)), "y": np.asarray(y)})
    t = d.groupby("decisao").agg(contratos=("y", "size"), inadimplencia=("y", "mean"))
    t["pct_carteira"] = t.contratos / len(d)
    return t.loc[["aprovar", "aprovar com ajuste", "rejeitar"]]


def curva(y, p, cortes=np.arange(0.04, 0.41, 0.02)) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    return pd.DataFrame({"corte": cortes, "aprovacao": [(p < c).mean() for c in cortes],
                         "inad_aprovados": [y[p < c].mean() for c in cortes]})
