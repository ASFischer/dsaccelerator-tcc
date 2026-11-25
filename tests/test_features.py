import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as ft

OBS = pd.Timestamp("2025-02-22")


def _par(contrato, inicio, pagamentos, versoes=None):
    """6 parcelas mensais de 100. pagamentos[n] = lista de (atraso_em_dias, valor); None = não paga."""
    linhas = []
    for n in range(1, 7):
        venc = pd.Timestamp(inicio) + pd.DateOffset(months=n - 1)
        for atraso, valor in pagamentos.get(n, [(0, 100)]):
            for v, prev in (versoes or {}).get(n, [(1, 100)]):
                linhas.append(dict(id_contrato=contrato, id_cliente=1, versao_parcela=v, numero_parcela=n,
                                   data_prevista_pagamento=venc, valor_previsto_parcela=prev,
                                   data_real_pagamento=None if atraso is None else venc + pd.Timedelta(days=atraso),
                                   valor_pago_parcela=None if atraso is None else valor))
    return pd.DataFrame(linhas)


def test_target():
    par = pd.concat([
        _par(1, "2024-01-10", {3: [(5, 100)]}),                     # DPD 5: não é inadimplente
        _par(2, "2024-01-10", {3: [(6, 100)]}),                     # DPD 6: inadimplente
        _par(3, "2024-01-10", {2: [(0, 60), (10, 40)]}),            # parcial: quitada só no 2º pagamento (DPD 10)
        _par(4, "2024-01-10", {6: [(None, None)]}),                 # não paga e vencida: DPD alto
        _par(5, "2024-09-19", {}),                                  # 6ª parcela venceu há 3 dias: imaturo
        _par(6, "2024-01-10", {2: [(0, 100)]}, {2: [(1, 40), (2, 60)]}),  # versões: pago repetido, não soma
    ])
    t = ft.construir_target(ft.consolidar_parcelas(par), OBS).set_index("id_contrato")
    assert t.target_dpd5.loc[1] == 0 and t.target_dpd5.loc[2] == 1
    assert t.dpd_max.loc[3] == 10 and t.target_dpd5.loc[3] == 1 and t.target_dpd30.loc[3] == 0
    assert t.target_dpd5.loc[4] == 1 and t.target_dpd30.loc[4] == 1
    assert np.isnan(t.target_dpd5.loc[5])
    assert t.target_dpd5.loc[6] == 0 and t.dpd_max.loc[6] <= 0


def test_base_so_tem_colunas_da_submissao():
    pos_decisao = {"status_contrato", "data_liberacao", "data_encerramento", "taxa_juros_padrao", "dpd_max"}
    usadas = set(sum(ft.FEATURES.values(), []))
    assert not usadas & pos_decisao
    sub_cols = {"tipo_contrato", "valor_credito", "valor_bem", "valor_parcela", "dia_semana_solicitacao", "hora_solicitacao"}
    assert {"tipo_contrato", "dia_semana_solicitacao"} <= sub_cols and set(ft.VALORES) <= sub_cols
