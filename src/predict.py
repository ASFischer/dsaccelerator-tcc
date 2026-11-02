import joblib
import pandas as pd

from src.config import INTERIM, MODELS, SUBMISSAO
from src.data_loader import load
from src.policy import decisao
from src.train import aplicar_cat


def main():
    art = joblib.load(MODELS / "lgbm.joblib")
    raw = pd.read_parquet(INTERIM / "submission.parquet")
    for c, v in art["cats"].items():  # categoria que o modelo nunca viu viraria NaN em silêncio
        novas = (raw[c].notna() & ~raw[c].isin(v)).mean()
        assert novas <= 0.005, f"{c}: {novas:.1%} das linhas com categoria nova"
    X = aplicar_cat(raw, art["cats"])[art["cols"]]
    out = X[[]].copy()
    out["prob_inadimplencia"] = art["model"].predict_proba(X)[:, 1]
    out["decisao"] = decisao(out.prob_inadimplencia.values)
    out.index.name = "id_cliente"
    assert out.index.is_unique and out.prob_inadimplencia.notna().all()
    assert set(out.index) == set(load("base_submissao").id_cliente), "submissão desatualizada: rode src.build_dataset"
    SUBMISSAO.mkdir(exist_ok=True)
    out.reset_index().to_csv(SUBMISSAO / "submission_predictions.csv", index=False)
    return out


if __name__ == "__main__":
    o = main()
    print(len(o), o.prob_inadimplencia.describe().round(3).to_dict(), o.decisao.value_counts(normalize=True).round(3).to_dict(), sep="\n")
