"""Métricas e diagnósticos de avaliação (cresce nas etapas seguintes)."""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve


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


# --- métricas, limiar e comparação de modelos -----------------------------------------------
def limiar_max_f1(y, p) -> float:
    """Limiar que maximiza o F1 da classe positiva (escolhido na validação, nunca no teste)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.nanargmax(f1)])


def metricas(y, p, limiar: float) -> dict:
    """Recall/F1 (principais), precision e F2 no `limiar`; auxiliares: ROC-AUC, Gini, KS, PR-AUC e Brier."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    pred = p >= limiar
    tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), ((~pred) & (y == 1)).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    def f(b):
        return (1 + b * b) * prec * rec / max(b * b * prec + rec, 1e-12)

    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return dict(recall=rec, precision=prec, f1=f(1), f2=f(2), auc=auc, gini=2 * auc - 1, ks=float((tpr - fpr).max()),
                pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, np.clip(p, 0, 1)), taxa_aprovacao=1 - pred.mean())


def delong(y, p1, p2):
    """Teste de DeLong pareado para a diferença de AUC: (auc1, auc2, z, p-valor bilateral)."""
    y = np.asarray(y).astype(int)
    pos = y == 1
    m, n = pos.sum(), (~pos).sum()
    P = np.vstack([p1, p2])
    X, Y = P[:, pos], P[:, ~pos]
    tx = np.vstack([rankdata(r) for r in X])
    ty = np.vstack([rankdata(r) for r in Y])
    tz = np.vstack([rankdata(np.concatenate([a, b])) for a, b in zip(X, Y)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2 / n
    s = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1 - (tz[:, m:] - ty) / m) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0  # scores idênticos: sem diferença
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


def bootstrap_pareado(y, p1, p2, lim1, lim2, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """IC 95% da diferença (modelo 2 − modelo 1) em AUC, recall e F1, com a mesma reamostragem para os dois."""
    y, p1, p2 = np.asarray(y).astype(int), np.asarray(p1), np.asarray(p2)
    rng, out = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        a, b = metricas(y[i], p1[i], lim1), metricas(y[i], p2[i], lim2)
        out.append({k: b[k] - a[k] for k in ("auc", "recall", "f1")})
    d = pd.DataFrame(out)
    return pd.DataFrame({"delta_medio": d.mean(), "ic95_inf": d.quantile(.025), "ic95_sup": d.quantile(.975)})
