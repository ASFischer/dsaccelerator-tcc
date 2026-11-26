"""Leitura e limpeza (só formato e duplicatas) das quatro bases. Nada é imputado."""
import unicodedata

import pandas as pd

from config import DADOS

COLUNAS_DATA = {
    "base_cadastral": ["data_nascimento"],
    "base_submissao": ["data_solicitacao"],
    "historico_emprestimos": ["data_decisao", "data_liberacao", "data_primeiro_vencimento",
                              "data_ultimo_vencimento_original", "data_ultimo_vencimento",
                              "data_encerramento"],
    "historico_parcelas": ["data_prevista_pagamento", "data_real_pagamento"],
}


def _chave(valor: str) -> str:
    """'Sexta feira ' e 'sexta-feira' -> 'sexta feira'; 'Não' e 'nao' -> 'nao'."""
    sem_acento = unicodedata.normalize("NFKD", valor).encode("ascii", "ignore").decode()
    return " ".join(sem_acento.lower().replace("-", " ").split())


def padronizar_categorias(df: pd.DataFrame, max_categorias: int = 100) -> pd.DataFrame:
    """Mapeia as variantes de grafia de cada coluna de texto para a forma mais frequente.

    shortcut: colunas de texto com mais de `max_categorias` valores (hora, ids em texto)
    não são tratadas como categoria; subir o limite se surgir categoria maior.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].nunique() > max_categorias:
            continue
        limpo = df[col].str.strip()
        chaves = limpo.dropna().map(_chave)
        canonica = limpo.dropna().groupby(chaves).agg(lambda s: s.value_counts().index[0])
        df[col] = limpo.map(lambda v: v if pd.isna(v) else canonica[_chave(v)])
    return df


def _ler(nome: str) -> pd.DataFrame:
    df = pd.read_parquet(DADOS / f"{nome}.parquet")
    for col in COLUNAS_DATA[nome]:
        # valores fora do padrão AAAA-MM-DD (ex.: '2025-13-01', '24/02/2025') viram NaT
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    return padronizar_categorias(df)


def carregar_cadastral() -> pd.DataFrame:
    df = _ler("base_cadastral").drop_duplicates("id_cliente", keep="first")
    # 'R$ 112.500,00', '135000,00' e '90.000,00' -> float; 'não informado' -> NaN
    renda = pd.to_numeric(df["renda_anual"].str.replace(r"[R$\s.]", "", regex=True)
                          .str.replace(",", ".", regex=False), errors="coerce")
    df["renda_anual"] = renda.mask(renda < 0)  # renda negativa é impossível: NaN (confirmado)
    return df.reset_index(drop=True)


def carregar_submissao() -> pd.DataFrame:
    return _ler("base_submissao").drop_duplicates("id_cliente", keep="first").reset_index(drop=True)


def carregar_emprestimos() -> pd.DataFrame:
    return _ler("historico_emprestimos")


def carregar_parcelas() -> pd.DataFrame:
    # só linhas idênticas em todas as colunas (inclui contrato, parcela, versão, data e valor)
    return _ler("historico_parcelas").drop_duplicates().reset_index(drop=True)
