import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_loader as dl


def test_padronizar_categorias():
    df = pd.DataFrame({"x": ["Não"] * 5 + ["nao", "Não ", "sexta-feira"] + [None]})
    out = dl.padronizar_categorias(df)["x"]
    assert out.iloc[:7].eq("Não").all() and out.iloc[7] == "sexta-feira" and out.isna().iloc[8]


def test_loaders_chaves_unicas_e_sem_duplicata_exata():
    assert dl.carregar_cadastral().id_cliente.is_unique
    sub = dl.carregar_submissao()
    assert sub.id_cliente.is_unique and sub.data_solicitacao.isna().sum() == 99
    assert not dl.carregar_parcelas().duplicated().any()


def test_psi():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from evaluate import psi
    x = pd.Series(range(1000))
    assert psi(x, x) < 1e-9 and psi(x, x + 5000) > 0.25
