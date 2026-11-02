import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ks(y, p) -> float:
    d = pd.DataFrame({"y": y, "p": p}).sort_values("p")
    return float(np.max(np.abs(d.y.cumsum() / d.y.sum() - (1 - d.y).cumsum() / (1 - d.y).sum())))


def metricas(y, p, n_boot=200, seed=42) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    auc = roc_auc_score(y, p)
    return {"auc": auc, "auc_ic95": (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)),
            "gini": 2 * auc - 1, "ks": ks(y, p), "brier": float(np.mean((p - y) ** 2)), "taxa": float(y.mean())}


def decis(y, p, n=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decil"] = pd.qcut(d.p.rank(method="first"), n, labels=range(1, n + 1))
    t = d.groupby("decil", observed=True).agg(n=("y", "size"), prob_media=("p", "mean"), taxa_real=("y", "mean"))
    t["lift"] = t.taxa_real / d.y.mean()
    return t


def psi(ref, new, n=10) -> float:
    cortes = np.unique(np.nanpercentile(ref, np.linspace(0, 100, n + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    a = np.histogram(ref[~np.isnan(ref)], cortes)[0] / max(len(ref[~np.isnan(ref)]), 1)
    b = np.histogram(new[~np.isnan(new)], cortes)[0] / max(len(new[~np.isnan(new)]), 1)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))
