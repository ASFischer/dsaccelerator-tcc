"""Target de inadimplência e montagem da base de modelagem (regras em docs/PLANO.md e docs/DECISOES.md)."""
import numpy as np
import pandas as pd

from config import DPD_ALTERNATIVO, DPD_PRINCIPAL, N_PARCELAS_JANELA, TOL_QUITACAO

K = ["id_contrato", "numero_parcela"]


def consolidar_parcelas(par: pd.DataFrame) -> pd.DataFrame:
    """1 linha por (contrato, parcela): vencimento, previsto, pago, quitada e data de conclusão.

    - previsto = soma dos previstos das versões (a parcela é fracionada entre versões);
    - pago = pago da versão com mais pagamento (o pago se repete em cada versão), somando parciais;
    - conclusão = data do pagamento que completa o previsto, nas linhas dessa versão.
    """
    v = par.groupby(K + ["versao_parcela"], as_index=False).agg(
        prev_v=("valor_previsto_parcela", "max"), pago_v=("valor_pago_parcela", "sum"),
        venc=("data_prevista_pagamento", "min"))
    v = v.sort_values(K + ["pago_v", "versao_parcela"], ascending=[True, True, False, True])
    escolhida = v.drop_duplicates(K)[K + ["versao_parcela", "pago_v"]].rename(
        columns={"versao_parcela": "versao_pago", "pago_v": "pago"})
    out = v.groupby(K, as_index=False).agg(previsto=("prev_v", "sum"), venc=("venc", "min")).merge(escolhida, on=K)
    out["quitada"] = out.pago >= out.previsto - TOL_QUITACAO

    r = par.merge(out[K + ["versao_pago", "previsto"]], on=K)
    r = r[r.versao_parcela == r.versao_pago].sort_values(K + ["data_real_pagamento"])
    r["acum"] = r.groupby(K).valor_pago_parcela.cumsum()
    concl = (r[r.acum >= r.previsto - TOL_QUITACAO].groupby(K).data_real_pagamento.min()
             .rename("data_conclusao").reset_index())
    out = out.merge(concl, on=K, how="left")
    out["data_conclusao"] = out.data_conclusao.where(out.quitada).fillna(out.venc.where(out.quitada))
    return out


def construir_target(parcelas: pd.DataFrame, obs: pd.Timestamp, n: int = N_PARCELAS_JANELA) -> pd.DataFrame:
    """1 linha por contrato com DPD máximo na janela das `n` primeiras parcelas e os dois targets.

    `parcelas` é a saída de `consolidar_parcelas`. Parcela quitada: DPD = conclusão - vencimento;
    não quitada: DPD = observação - vencimento. O target só existe (não-nulo) para contratos
    maduros: a parcela `n` existe e venceu há mais de X dias (X = corte do DPD).
    """
    p = parcelas[parcelas.numero_parcela <= n].copy()
    p["dpd"] = np.where(p.quitada, (p.data_conclusao - p.venc).dt.days, (obs - p.venc).dt.days)
    c = p.groupby("id_contrato").agg(dpd_max=("dpd", "max"), n_parcelas_janela=("numero_parcela", "nunique"))
    c["venc_parcela_n"] = p[p.numero_parcela == n].set_index("id_contrato").venc
    dias_desde_n = (obs - c.venc_parcela_n).dt.days
    for corte, nome in [(DPD_PRINCIPAL, "target_dpd5"), (DPD_ALTERNATIVO, "target_dpd30")]:
        c[nome] = (c.dpd_max > corte).astype(float).where(dias_desde_n > corte)
    return c.reset_index()


# --- base de modelagem: só o que se conhece na solicitação -----------------------------------
VALORES = ["valor_credito", "valor_bem", "valor_parcela"]
CONTRATO_SEM_VALOR = ["tipo_contrato", "dia_semana_solicitacao", "hora"]
CADASTRO = ["sexo", "idade", "qtd_filhos", "qtd_membros_familia", "renda_anual", "tipo_renda", "ocupacao",
            "tipo_organizacao", "nivel_educacao", "estado_civil", "tipo_moradia", "possui_carro",
            "possui_imovel", "nota_regiao_cliente", "nota_regiao_cliente_cidade"]
FEATURES = {  # candidatos do nb03 (docs/DECISOES.md, item 1.10)
    "A": CONTRATO_SEM_VALOR + CADASTRO,
    "B": CONTRATO_SEM_VALOR + CADASTRO + ["pct_" + c for c in VALORES],
    "C": CONTRATO_SEM_VALOR + CADASTRO + VALORES,
}
CATEGORICAS = ["tipo_contrato", "dia_semana_solicitacao", "sexo", "tipo_renda", "ocupacao", "tipo_organizacao",
               "nivel_educacao", "estado_civil", "tipo_moradia", "possui_carro", "possui_imovel"]


def _features(contratos: pd.DataFrame, cad: pd.DataFrame, data_ref: pd.Series, grupo) -> pd.DataFrame:
    """Colunas do contrato + cadastro + idade na `data_ref` + percentil dos valores dentro de `grupo`."""
    f = contratos.reset_index(drop=True).copy()
    grupo = pd.Series(grupo).reset_index(drop=True)
    for c in VALORES:
        f["pct_" + c] = f[c].groupby(grupo).rank(pct=True)
    f["hora"] = pd.to_datetime(f.hora_solicitacao, format="%H:%M:%S", errors="coerce").dt.hour
    f = f.merge(cad, on="id_cliente", how="left", validate="m:1")
    f["idade"] = (pd.Series(data_ref).reset_index(drop=True) - f.data_nascimento).dt.days / 365.25
    return f


def base_modelagem(emp: pd.DataFrame, cad: pd.DataFrame, alvo: pd.DataFrame, nome_alvo: str = "target_dpd5") -> pd.DataFrame:
    """Contratos aprovados e maduros com target, `safra` (mês da decisão) e todas as features."""
    ap = emp[emp.status_contrato == "Aprovado"].reset_index(drop=True)
    # percentil só entre aprovados do mesmo mês, antes de filtrar os maduros (decisão 1.10)
    f = _features(ap, cad, ap.data_decisao, ap.data_decisao.dt.to_period("M"))
    f["safra"] = f.data_decisao.dt.to_period("M").dt.to_timestamp()
    f = f.merge(alvo[["id_contrato", "dpd_max", "target_dpd5", "target_dpd30"]], on="id_contrato", how="inner")
    return f[f[nome_alvo].notna()].reset_index(drop=True)


def base_submissao(sub: pd.DataFrame, cad: pd.DataFrame, obs: pd.Timestamp) -> pd.DataFrame:
    """Features da submissão; o percentil é calculado dentro da própria submissão."""
    return _features(sub, cad, sub.data_solicitacao.fillna(obs), np.zeros(len(sub)))


def separar_oot(data_decisao: pd.Series) -> pd.Series:
    """Rótulo 'treino' / 'valid' / 'teste' pela data de decisão (cortes no config)."""
    from config import DATA_CORTE_TREINO, DATA_CORTE_VALID
    return pd.Series(np.select([data_decisao <= DATA_CORTE_TREINO, data_decisao <= DATA_CORTE_VALID],
                               ["treino", "valid"], "teste"), index=data_decisao.index)
