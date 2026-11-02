import itertools

import numpy as np
import pandas as pd

from src.policy import CORTE_APROVAR, CORTE_REJEITAR, decisao

ROTULOS = list("ABCDE")
MIN_PCT = 0.03  # nenhuma faixa com menos de 3% dos contratos


def faixa(p, cortes) -> np.ndarray:
    """A = menor risco. cortes: 4 limites crescentes sobre a probabilidade de inadimplência."""
    return np.asarray(ROTULOS)[np.searchsorted(cortes, np.asarray(p), side="right")]


def _wilson(k, n, z=1.96):
    ph = k / n
    c = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return c - h, c + h


def tabela_faixas(y, p, cortes) -> pd.DataFrame:
    d = pd.DataFrame({"faixa": faixa(p, cortes), "y": np.asarray(y), "p": np.asarray(p)})
    t = d.groupby("faixa").agg(contratos=("y", "size"), prob_media=("p", "mean"), inadimplencia=("y", "mean"))
    t = t.reindex(ROTULOS)
    t["pct_carteira"] = t.contratos / len(d)
    t["ic95_inf"], t["ic95_sup"] = _wilson(t.inadimplencia * t.contratos, t.contratos)
    t["lift"] = t.inadimplencia / d.y.mean()
    t["decisao"] = decisao(t.prob_media.values)
    return t


def valido(t: pd.DataFrame) -> bool:
    """Todas as faixas com volume mínimo, taxa crescente e IC95% sem sobreposição entre vizinhas."""
    if t.contratos.isna().any() or (t.pct_carteira < MIN_PCT).any():
        return False
    return bool((t.ic95_inf.values[1:] > t.ic95_sup.values[:-1]).all())


def escolher_cortes(y, p, baixo=np.arange(0.03, 0.081, 0.01), medio=np.arange(0.12, 0.221, 0.01)):
    """Mantém 0,10 e 0,25 da política e procura os outros dois cortes; vence o maior menor-salto de taxa entre faixas vizinhas."""
    melhor, salto = None, -1
    for c1, c2 in itertools.product(baixo, medio):
        cortes = [round(float(c1), 2), CORTE_APROVAR, round(float(c2), 2), CORTE_REJEITAR]
        t = tabela_faixas(y, p, cortes)
        if valido(t) and (s := np.diff(t.inadimplencia.values).min()) > salto:
            melhor, salto = cortes, s
    assert melhor is not None, "nenhuma combinação de cortes respeita as restrições"
    return melhor
