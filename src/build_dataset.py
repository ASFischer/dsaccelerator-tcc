from src.config import INTERIM
from src.data_loader import load
from src.features import montar
from src.target import construir_target


def main():
    e, p, cad, s = [load(n) for n in ["historico_emprestimos", "historico_parcelas", "base_cadastral", "base_submissao"]]
    t = construir_target(e, p, p.data_real_pagamento.max())
    X, Xs = montar(t, cad, e, p, True), montar(s, cad, e, p, False)
    assert set(X.columns) == set(Xs.columns)  # treino só usa colunas disponíveis na submissão
    ti = t.set_index("id_contrato")
    INTERIM.mkdir(parents=True, exist_ok=True)
    X.assign(target=ti.target, id_cliente=ti.id_cliente, data_decisao=ti.data_decisao).to_parquet(INTERIM / "train.parquet")
    Xs.to_parquet(INTERIM / "submission.parquet")
    return X, Xs, t


if __name__ == "__main__":
    X, Xs, t = main()
    print(f"treino {X.shape} (inadimplência {t.target.mean():.3f}) | submissão {Xs.shape}")
