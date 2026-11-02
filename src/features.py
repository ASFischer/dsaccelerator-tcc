import numpy as np
import pandas as pd

from src.target import parcelas_janela


def _contrato_atual(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["tipo_contrato", "valor_credito", "valor_bem", "valor_parcela", "dia_semana_solicitacao"]].copy()
    X["hora"] = pd.to_datetime(df.hora_solicitacao, format="%H:%M:%S", errors="coerce").dt.hour
    return X


def _hist(K: pd.DataFrame, e: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """K: key, id_cliente, corte (+ id_contrato no treino). Só usa contratos decididos e pagamentos feitos antes do corte."""
    pe = K[["key", "id_cliente", "corte"]].merge(
        e[["id_contrato", "id_cliente", "data_decisao", "status_contrato", "valor_credito", "qtd_parcelas_planejadas"]]
        .rename(columns={"id_contrato": "c_prev"}), on="id_cliente")
    pe = pe[pe.data_decisao < pe.corte]
    if "id_contrato" in K:
        pe = pe[pe.c_prev != pe.key]  # na chave de treino key == id_contrato
    g = pe.groupby("key")
    h = pd.DataFrame({"h_n_contratos": g.size(),
                      "h_dias_ult_decisao": (pe.corte - g.data_decisao.transform("max")).dt.days.groupby(pe.key).first()})
    # só a recusa: aprovado/cancelado são status finais de hoje, não do corte
    h["h_n_recusado"] = (pe.status_contrato == "recusado").groupby(pe.key).sum()
    h["h_pct_recusado"] = h.h_n_recusado / h.h_n_contratos
    h["h_credito_medio"] = g.valor_credito.mean()
    h["h_prazo_medio"] = g.qtd_parcelas_planejadas.mean()

    q = parcelas_janela(p, 999)
    q = pe[["key", "c_prev", "corte"]].merge(q, left_on="c_prev", right_on="id_contrato")
    pago_antes = q.data_pag < q.corte  # pagamento depois do corte (ou nunca) é desconhecido no corte
    q = q[pago_antes | (q.venc < q.corte)]  # vencida e não paga até o corte = em atraso
    pago_antes = pago_antes[q.index]
    q["dpd"] = np.where(pago_antes, (q.data_pag - q.venc).dt.days, (q.corte - q.venc).dt.days).clip(min=0)
    q["razao"] = np.where(pago_antes, q.pago / q.previsto.replace(0, np.nan), 0.0)
    gq = q.groupby("key")
    h["h_n_parcelas"] = gq.size()
    h["h_dpd_medio"] = gq.dpd.mean()
    h["h_dpd_max"] = gq.dpd.max()
    h["h_pct_atraso"] = (q.dpd > 0).groupby(q.key).mean()
    h["h_pct_atraso30"] = (q.dpd >= 30).groupby(q.key).mean()
    h["h_razao_paga"] = gq.razao.mean()
    return h


def montar(base: pd.DataFrame, cad: pd.DataFrame, e: pd.DataFrame, p: pd.DataFrame, treino: bool) -> pd.DataFrame:
    """base: treino -> target (id_contrato, id_cliente, data_decisao); submissão -> base_submissao.
    Índice do resultado = key (id_contrato no treino, id_cliente na submissão)."""
    if treino:
        d = base.merge(e.drop(columns=["id_cliente", "data_decisao"]), on="id_contrato")
        K = base.assign(key=base.id_contrato, corte=base.data_decisao)
    else:
        d = base
        K = base.assign(key=base.id_cliente, corte=base.data_solicitacao)
    X = _contrato_atual(d).set_axis(K.key.values)
    X = X.join(_hist(K, e, p))
    c = K[["key", "id_cliente", "corte"]].merge(cad, on="id_cliente", how="left").set_index("key")
    X["idade"] = (c.corte - c.data_nascimento).dt.days / 365.25
    X["renda_mensal"] = c.renda_anual / 12
    for col in cad.columns.drop(["id_cliente", "data_nascimento", "renda_anual"]):
        X[col] = c[col]
    X["credito_sobre_bem"] = X.valor_credito / X.valor_bem
    X["prazo_implicito"] = X.valor_credito / X.valor_parcela
    X["renda_per_capita"] = X.renda_mensal / c.qtd_membros_familia
    # valores absolutos do contrato atual têm escala 3-4x maior na submissão que no histórico: só razões ficam
    X = X.drop(columns=["valor_credito", "valor_bem", "valor_parcela"]).replace([np.inf, -np.inf], np.nan)
    X["tem_historico"] = X.h_n_contratos.notna().astype(int)
    return X
