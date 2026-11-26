"""Calibração, cortes da política de crédito (p1, p2), faixas de risco A–E e simulação de perda esperada."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from config import RECALL_ALVO

FAIXAS = ["A", "B", "C", "D", "E"]
DECISAO = {"A": "aprovar", "B": "aprovar", "C": "aprovar com ajuste", "D": "aprovar com ajuste", "E": "rejeitar"}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrador:
    """Ajusta `platt` (logística sobre o logit, monótona e sem empates) ou `isotonica` (com desempate pelo score bruto).

    O objeto é chamável (score -> probabilidade calibrada) e pode ser salvo com joblib.
    """

    def __init__(self, metodo: str, p, y):
        self.metodo = metodo
        self.m = (LogisticRegression(C=1e6).fit(_logit(p), y) if metodo == "platt"
                  else IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y))

    def __call__(self, q):
        q = np.asarray(q)
        if self.metodo == "platt":
            return self.m.predict_proba(_logit(q))[:, 1]
        # desempate pelo score bruto (1e-6): os degraus da isotônica criam empates que travam os cortes
        return np.clip(self.m.predict(q) + 1e-6 * q, 0, 1)


def ece(y, p, bins: int = 10) -> float:
    """Erro de calibração esperado com faixas de mesmo volume."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["b"] = pd.qcut(d.p, bins, labels=False, duplicates="drop")
    g = d.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g.n * (g.y - g.p).abs()).sum() / g.n.sum())


def wilson(k, n, z: float = 1.96):
    """IC de Wilson para proporção (k sucessos em n)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    n = np.where(n == 0, np.nan, n)
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def corte_p1(y, p, recall_alvo: float = RECALL_ALVO) -> float:
    """Maior corte tal que 'p >= corte' captura pelo menos `recall_alvo` dos inadimplentes."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    o = np.argsort(-p)
    rec = np.cumsum(y[o]) / y.sum()
    return float(p[o][np.searchsorted(rec, recall_alvo)])


def corte_p2(y, p, p1: float) -> float:
    """Corte de máximo F1 (rejeitar = prever inadimplente), restrito a cortes >= p1."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    f1[thr < p1] = -1
    return float(thr[np.argmax(f1)])


def cortes_faixas(p, p1: float, p2: float) -> list:
    """[A|B, p1, C|D, p2]: A/B dividem a zona 'aprovar' e C/D a zona 'ajuste' na mediana do volume (referência: `p`)."""
    assert p1 < p2, "zona de ajuste vazia (p1 >= p2): revise o recall-alvo"
    p = np.asarray(p)
    return [float(np.median(p[p < p1])), p1, float(np.median(p[(p >= p1) & (p < p2)])), p2]


def faixa(p, cortes: list) -> pd.Series:
    return pd.Series(np.array(FAIXAS)[np.searchsorted(cortes, np.asarray(p), side="right")])


def tabela_faixas(y, p, cortes: list) -> pd.DataFrame:
    """Volume, taxa de inadimplência e IC de Wilson por faixa; `sobrepoe` marca IC que toca o da faixa seguinte."""
    f = faixa(p, cortes)
    g = pd.DataFrame({"y": np.asarray(y).astype(int), "f": f}).groupby("f").y.agg(n="size", inad="sum").reindex(FAIXAS)
    g["taxa"] = g.inad / g.n
    g["ic_inf"], g["ic_sup"] = wilson(g.inad, g.n)
    g["decisao"] = [DECISAO[x] for x in g.index]
    g["sobrepoe_proxima"] = g.ic_sup.values >= np.r_[g.ic_inf.values[1:], np.inf]
    return g


def resumo_corte(y, p, p1: float, p2: float) -> dict:
    """Métricas da política: 'barrar direto' = p >= p1 (ajuste + rejeição); 'rejeitar' = p >= p2."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    aprov, rej = p < p1, p >= p2
    barra = ~aprov
    return dict(taxa_aprovacao_direta=aprov.mean(), taxa_ajuste=(barra & ~rej).mean(), taxa_rejeicao=rej.mean(),
                inad_aprovados_diretos=y[aprov].mean(), inad_ajuste=y[barra & ~rej].mean() if (barra & ~rej).any() else np.nan,
                inad_rejeitados=y[rej].mean() if rej.any() else np.nan,
                recall_nao_aprovar=y[barra].sum() / y.sum(), precision_nao_aprovar=y[barra].mean(),
                recall_rejeitar=y[rej].sum() / y.sum(), precision_rejeitar=y[rej].mean(),
                bons_barrados_por_mau_barrado=(barra & (y == 0)).sum() / max((barra & (y == 1)).sum(), 1))


def perda_esperada(p, ead, lgd: float, mascara=None) -> float:
    """EL = soma(PD x LGD x EAD) sobre `mascara` (default: todos)."""
    p, ead = np.asarray(p), np.asarray(ead)
    m = np.ones(len(p), bool) if mascara is None else np.asarray(mascara)
    return float((p[m] * ead[m]).sum() * lgd)
