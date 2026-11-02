import pandas as pd

from src.config import DADOS

DATAS = {
    "base_cadastral": ["data_nascimento"],
    "base_submissao": ["data_solicitacao"],
    "historico_emprestimos": ["data_decisao", "data_liberacao", "data_primeiro_vencimento",
                              "data_ultimo_vencimento_original", "data_ultimo_vencimento", "data_encerramento"],
    "historico_parcelas": ["data_prevista_pagamento", "data_real_pagamento"],
}


def _to_date(s: pd.Series) -> pd.Series:
    # formatos mistos: ISO (yyyy-mm-dd) e dd/mm/yyyy
    s = s.astype("object")
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    br = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    swap = pd.to_datetime(s, format="%Y-%d-%m", errors="coerce")  # ex.: 2025-13-01 (aaaa-dd-mm invertido)
    return iso.fillna(br).fillna(swap)


def normalizar(s: pd.Series) -> pd.Series:
    """minúsculas, sem espaços nas pontas e sem acento: 'Crédito rotativo ' e 'credito rotativo' viram a mesma categoria."""
    s = s.astype("object").str.strip().str.lower().str.normalize("NFKD")
    return s.str.encode("ascii", "ignore").str.decode("ascii")


def load(nome: str) -> pd.DataFrame:
    df = pd.read_parquet(DADOS / f"{nome}.parquet")
    for c in DATAS[nome]:
        antes = df[c].notna().sum()
        df[c] = _to_date(df[c])
        # datas impossíveis (ex.: 31/02/1990) viram NaT; tolera até 0,1%
        assert antes - df[c].notna().sum() <= 0.001 * len(df), f"{nome}.{c}: datas não parseadas"
    for c in df.select_dtypes(exclude=["number", "datetime"]).columns:
        df[c] = normalizar(df[c])
    if nome == "base_cadastral":
        df["renda_anual"] = pd.to_numeric(
            df["renda_anual"].astype("object").str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")  # "não informado" -> NaN
        df.loc[df["renda_anual"] <= 0, "renda_anual"] = pd.NA
    if nome in ("base_cadastral", "base_submissao"):
        # cópias ruidosas anexadas ao fim do arquivo: mantém a 1ª ocorrência (original)
        df = df.drop_duplicates("id_cliente", keep="first").reset_index(drop=True)
    if nome == "historico_emprestimos":
        assert df.id_contrato.is_unique
    return df
