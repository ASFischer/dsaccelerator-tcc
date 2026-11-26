"""Preparação dos dados, tuning (Optuna) e treino de LightGBM, XGBoost e baseline logístico."""
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import SEED
from features import CATEGORICAS, SEM_PODER_IV, VALORES

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_ARVORES_MAX, PARADA = 3000, 100


def preparar(df: pd.DataFrame, feats: list, categorias: dict) -> pd.DataFrame:
    """Seleciona `feats` e fixa as categorias do treino; categoria nova vira NaN **com aviso** (nunca em silêncio)."""
    X = df[feats].copy()
    for c in [c for c in feats if c in categorias]:
        novas = set(X[c].dropna().unique()) - set(categorias[c])
        if novas:
            warnings.warn(f"{c}: categorias fora do treino viraram nulo: {sorted(novas)}")
        X[c] = pd.Categorical(X[c].where(X[c].isin(categorias[c])), categories=categorias[c])
    return X


def categorias_do_treino(df: pd.DataFrame) -> dict:
    return {c: sorted(df[c].dropna().unique()) for c in CATEGORICAS if c in df}


def _espaco(trial, algoritmo):
    if algoritmo == "lgbm":
        return dict(num_leaves=trial.suggest_int("num_leaves", 15, 127),
                    learning_rate=trial.suggest_float("learning_rate", .01, .2, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 20, 300),
                    colsample_bytree=trial.suggest_float("colsample_bytree", .5, 1),
                    subsample=trial.suggest_float("subsample", .5, 1), subsample_freq=1,
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    return dict(max_depth=trial.suggest_int("max_depth", 3, 9),
                learning_rate=trial.suggest_float("learning_rate", .01, .2, log=True),
                min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
                colsample_bytree=trial.suggest_float("colsample_bytree", .5, 1),
                subsample=trial.suggest_float("subsample", .5, 1), gamma=trial.suggest_float("gamma", 0, 5),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))


def _modelo(algoritmo, params, n_arvores=N_ARVORES_MAX, parada=PARADA, peso=1.0):
    if algoritmo == "lgbm":
        return lgb.LGBMClassifier(n_estimators=n_arvores, random_state=SEED, n_jobs=-1, verbose=-1,
                                  scale_pos_weight=peso, **params)
    return xgb.XGBClassifier(n_estimators=n_arvores, random_state=SEED, n_jobs=-1, tree_method="hist",
                             enable_categorical=True, early_stopping_rounds=parada, eval_metric="aucpr",
                             scale_pos_weight=peso, **params)


def _fit(m, algoritmo, X, y, Xv=None, yv=None):
    if Xv is None:
        return m.fit(X, y)
    if algoritmo == "lgbm":
        return m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="average_precision",
                     callbacks=[lgb.early_stopping(PARADA, verbose=False)])
    return m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)


def _melhor_iteracao(m, algoritmo):
    return int(m.best_iteration_ if algoritmo == "lgbm" else m.best_iteration + 1)


def tunar(algoritmo, X, y, grupos, n_trials=25, n_folds=5, peso=1.0):
    """Optuna (TPE, seed fixa) maximizando o PR-AUC médio em StratifiedGroupKFold agrupado por cliente.

    Devolve (melhores parâmetros, nº de árvores = média das melhores iterações × 1,1, PR-AUC por fold do melhor trial).
    O nº de árvores sai do CV: a validação não participa do tuning.
    """
    folds = list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=SEED).split(X, y, grupos))

    def objetivo(trial):
        params, ap, its = _espaco(trial, algoritmo), [], []
        for a, b in folds:
            m = _fit(_modelo(algoritmo, params, peso=peso), algoritmo, X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b])
            ap.append(average_precision_score(y.iloc[b], m.predict_proba(X.iloc[b])[:, 1]))
            its.append(_melhor_iteracao(m, algoritmo))
        trial.set_user_attr("aps", ap)
        trial.set_user_attr("iteracoes", its)
        return float(np.mean(ap))

    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    estudo.optimize(objetivo, n_trials=n_trials)
    t = estudo.best_trial
    return t.params, int(np.mean(t.user_attrs["iteracoes"]) * 1.1), pd.Series(t.user_attrs["aps"], name="pr_auc_fold")


def treinar(algoritmo, params, n_arvores, X, y, peso=1.0):
    """Treino final no conjunto de treino, sem early stopping; devolve (modelo, segundos)."""
    t = time.perf_counter()
    m = _modelo(algoritmo, params, n_arvores=n_arvores, peso=peso)
    if algoritmo == "xgb":
        m.set_params(early_stopping_rounds=None)
    return _fit(m, algoritmo, X, y), time.perf_counter() - t


def _log1p(a):
    return np.log1p(np.clip(a, 0, None))


def baseline_logistico(X: pd.DataFrame, y: pd.Series):
    """Regressão logística sem as variáveis de IV < 0,02: mediana/moda nos nulos, escala nas numéricas (log nos valores), one-hot nas categóricas."""
    X = X.drop(columns=[c for c in SEM_PODER_IV if c in X])
    cat = [c for c in X.columns if c in CATEGORICAS]
    logs = [c for c in X.columns if c in VALORES]
    num = [c for c in X.columns if c not in cat + logs]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("log", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(_log1p), StandardScaler()), logs),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=50)), cat)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5, random_state=SEED)).fit(
        X.astype({c: object for c in cat}), y)
