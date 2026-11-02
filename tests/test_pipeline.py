import pandas as pd

from src.features import _hist
from src.target import construir_target

D = pd.Timestamp


def _parcelas(c, pagas):
    """3 parcelas mensais a partir de 2024-01-10; pagas = data real (None = não paga)."""
    venc = [D("2024-01-10"), D("2024-02-09"), D("2024-03-10")]
    return pd.DataFrame({"id_contrato": c, "id_cliente": c, "versao_parcela": 0.0, "numero_parcela": [1, 2, 3],
                         "data_prevista_pagamento": venc, "data_real_pagamento": [D(x) if x else pd.NaT for x in pagas],
                         "valor_previsto_parcela": 100.0, "valor_pago_parcela": [100.0 if x else None for x in pagas]})


def _contratos(ids):
    return pd.DataFrame({"id_contrato": ids, "id_cliente": ids, "status_contrato": "aprovado", "qtd_parcelas_planejadas": 3.0,
                         "data_primeiro_vencimento": D("2024-01-10"), "data_decisao": D("2023-12-20"),
                         "valor_credito": 1000.0})


def test_target():
    p = pd.concat([_parcelas(1, ["2024-01-10", "2024-02-09", "2024-03-10"]),   # em dia
                   _parcelas(2, ["2024-01-10", "2024-03-20", "2024-03-10"]),   # parcela 2 paga com 40 dias de atraso
                   _parcelas(3, ["2024-01-10", "2024-02-09", None])])          # última nunca paga
    t = construir_target(_contratos([1, 2, 3]), p, D("2025-01-01")).set_index("id_contrato").target
    assert t.to_dict() == {1: 0, 2: 1, 3: 1}


def test_target_contrato_sem_primeiro_vencimento_nao_some():
    e = _contratos([1]).assign(data_primeiro_vencimento=pd.NaT)
    p = _parcelas(1, ["2024-01-10", "2024-02-09", "2024-03-10"])
    assert len(construir_target(e, p, D("2025-01-01"))) == 1


def test_historico_nao_usa_futuro():
    corte = D("2024-06-01")
    K = pd.DataFrame({"key": [99], "id_cliente": [1], "corte": corte, "id_contrato": [99]})
    e = _contratos([1]).assign(id_cliente=1)
    antes = _hist(K, e, _parcelas(1, ["2024-01-10", "2024-02-09", "2024-03-10"])).h_dpd_max.iloc[0]
    # a parcela 3 foi paga só depois do corte: no corte ela estava vencida e em aberto (82 dias)
    depois = _hist(K, e, _parcelas(1, ["2024-01-10", "2024-02-09", "2024-09-01"])).h_dpd_max.iloc[0]
    nunca = _hist(K, e, _parcelas(1, ["2024-01-10", "2024-02-09", None])).h_dpd_max.iloc[0]
    assert antes == 0 and depois == nunca == (corte - D("2024-03-10")).days


def test_parcelas_copia_nao_soma_duas_vezes_mas_pagamento_parcial_soma():
    from src.target import parcelas_janela
    base = dict(id_contrato=1, id_cliente=1, versao_parcela=0.0, numero_parcela=1,
                data_prevista_pagamento=D("2024-01-10"), valor_previsto_parcela=100.0)
    p = pd.DataFrame([dict(base, data_real_pagamento=D("2024-01-10"), valor_pago_parcela=100.0),
                      dict(base, data_real_pagamento=D("2024-01-10"), valor_pago_parcela=130.0)])  # cópia ruidosa
    assert parcelas_janela(p).pago.iloc[0] == 100.0
    parcial = pd.DataFrame([dict(base, data_real_pagamento=D("2024-01-10"), valor_pago_parcela=40.0),
                            dict(base, data_real_pagamento=D("2024-01-20"), valor_pago_parcela=60.0)])
    assert parcelas_janela(parcial).pago.iloc[0] == 100.0


def test_datas_e_acentos():
    from src.data_loader import _to_date, normalizar
    d = _to_date(pd.Series(["2025-02-24", "24/02/2025", "2025-13-01", "31/02/1990"]))
    assert d.iloc[0] == d.iloc[1] == D("2025-02-24") and d.iloc[2] == D("2025-01-13") and pd.isna(d.iloc[3])
    assert normalizar(pd.Series(["Crédito rotativo ", "credito rotativo"])).nunique() == 1


def test_faixas():
    import numpy as np
    from src.segmentation import faixa, tabela_faixas
    c = [0.05, 0.10, 0.19, 0.25]
    assert list(faixa([0.0, 0.049, 0.05, 0.10, 0.189, 0.19, 0.25, 0.9], c)) == list("AABCCDEE")
    p = np.repeat([0.02, 0.07, 0.15, 0.22, 0.4], 100)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(100 - k)] for k in (2, 7, 15, 22, 40)])
    t = tabela_faixas(y, p, c)
    assert list(t.index) == list("ABCDE") and (t.contratos == 100).all() and t.inadimplencia.is_monotonic_increasing
