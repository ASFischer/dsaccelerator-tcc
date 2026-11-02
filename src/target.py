import numpy as np
import pandas as pd

from src.config import DPD_LIMITE, JANELA_PARCELAS, TOLERANCIA_PAGO


def parcelas_janela(p: pd.DataFrame, janela: int = JANELA_PARCELAS) -> pd.DataFrame:
    """1 linha por (contrato, parcela<=janela): versão vigente, soma paga e data do último pagamento."""
    p = p[p.numero_parcela <= janela]
    p = p[p.versao_parcela == p.groupby(["id_contrato", "numero_parcela"]).versao_parcela.transform("max")]
    # cópias do mesmo pagamento (mesma data) não somam duas vezes; pagamentos parciais têm datas distintas
    p = p.drop_duplicates(["id_contrato", "numero_parcela", "data_real_pagamento"])
    return (p.groupby(["id_contrato", "numero_parcela"])
              .agg(venc=("data_prevista_pagamento", "first"), previsto=("valor_previsto_parcela", "first"),
                   pago=("valor_pago_parcela", "sum"), data_pag=("data_real_pagamento", "max"))
              .reset_index())


def construir_target(e: pd.DataFrame, p: pd.DataFrame, obs: pd.Timestamp,
                     janela: int = JANELA_PARCELAS, dpd: int = DPD_LIMITE) -> pd.DataFrame:
    """Contratos aprovados maduros -> (id_contrato, id_cliente, target, max_dpd).
    Maduro = todas as parcelas da janela venceram há >= `dpd` dias em `obs`."""
    ap = e[e.status_contrato == "aprovado"].copy()
    n = ap.qtd_parcelas_planejadas.where(ap.qtd_parcelas_planejadas > 0, janela).clip(upper=janela)
    # shortcut: aproxima o vencimento da última parcela da janela em 30d*(n-1); parcelas sem linha são ignoradas
    primeiro = (p.data_prevista_pagamento - pd.to_timedelta(30 * (p.numero_parcela - 1), unit="D")).groupby(p.id_contrato).min()
    primeiro = ap.data_primeiro_vencimento.fillna(ap.id_contrato.map(primeiro))  # 6,8% dos aprovados não têm a data
    ult = primeiro + pd.to_timedelta(30 * (n - 1), unit="D")
    ap = ap[(ult + pd.Timedelta(days=dpd) <= obs) & ap.id_contrato.isin(p.id_contrato)]

    q = parcelas_janela(p, janela)
    q = q[q.id_contrato.isin(ap.id_contrato)]
    quitada = q.pago >= TOLERANCIA_PAGO * q.previsto
    q["dpd"] = np.where(quitada, (q.data_pag - q.venc).dt.days, (obs - q.venc).dt.days)
    q.loc[quitada & q.data_pag.isna(), "dpd"] = 0
    r = q.groupby("id_contrato").dpd.max().clip(lower=0).rename("max_dpd").reset_index()
    r["target"] = (r.max_dpd >= dpd).astype(int)
    return ap[["id_contrato", "id_cliente", "data_decisao"]].merge(r, on="id_contrato")
