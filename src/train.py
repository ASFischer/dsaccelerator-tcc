import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import INTERIM, MODELS, SEED
from src.evaluate import metricas

CORTE_VALID, CORTE_TESTE = pd.Timestamp("2022-07-01"), pd.Timestamp("2023-04-01")


def carregar():
    d = pd.read_parquet(INTERIM / "train.parquet")
    return d.drop(columns=["target", "id_cliente", "data_decisao"]), d.target, d.id_cliente, d.data_decisao


def categorias(X: pd.DataFrame) -> dict:
    return {c: sorted(X[c].dropna().unique()) for c in X.select_dtypes("object")}


def aplicar_cat(X: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = X.copy()
    for c, v in cats.items():
        X[c] = pd.Categorical(X[c], categories=v)
    return X


def split(data: pd.Series):
    return data < CORTE_VALID, (data >= CORTE_VALID) & (data < CORTE_TESTE), data >= CORTE_TESTE


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, verbose=-1)


def treinar(X, y, tr, va, n_max=3000):
    m = lgb.LGBMClassifier(n_estimators=n_max, **PARAMS)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def rodar(excluir=()):
    X, y, cli, data = carregar()
    com_hist = (X.tem_historico == 1).values  # perfil de quase toda a submissão; calculado antes das ablações
    X = X.drop(columns=list(excluir))
    cats = categorias(X)
    X = aplicar_cat(X, cats)
    tr, va, te = split(data)
    m = treinar(X, y, tr, va)
    p = m.predict_proba(X[te])[:, 1]
    r = metricas(y[te], p)
    novo = ~cli[te].isin(set(cli[tr | va])).values  # clientes que o modelo nunca viu
    com_hist = com_hist[te.values]
    r.update(n_clientes_novos=int(novo.sum()), auc_clientes_novos=roc_auc_score(y[te][novo], p[novo]),
             auc_com_historico=roc_auc_score(y[te][com_hist], p[com_hist]))
    return m, cats, r, (X, y, tr, va, te, p)


if __name__ == "__main__":
    m, cats, r, (X, y, tr, va, te, p) = rodar()
    print("teste OOT:", r, "iter", m.best_iteration_)
    # modelo final: todas as safras, nº de árvores do early stopping
    final = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ * 1.1), **PARAMS).fit(X, y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "cats": cats, "cols": list(X.columns)}, MODELS / "lgbm.joblib")
