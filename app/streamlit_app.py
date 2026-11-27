"""App de demonstração (Streamlit): pontuar um cliente, simular a política e ver a submissão.

Rodar da raiz do projeto: `streamlit run app/streamlit_app.py` e abrir http://localhost:8501
Chama `src/scoring.py` direto (não depende da API estar no ar).
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_loader as dl  # noqa: E402
import features as ft  # noqa: E402
import policy as po  # noqa: E402
import scoring as sc  # noqa: E402
from config import DATA_OBSERVACAO, LGD_CENARIOS, OUTPUTS  # noqa: E402

TAB, FIG = OUTPUTS / "tables", OUTPUTS / "figures"
COR = {"aprovar": "green", "aprovar com ajuste": "orange", "rejeitar": "red"}
st.set_page_config(page_title="Score de inadimplência", layout="wide")
a = sc.carregar()


@st.cache_data
def submissao() -> pd.DataFrame:
    bs = ft.base_submissao(dl.carregar_submissao(), dl.carregar_cadastral(), pd.Timestamp(DATA_OBSERVACAO))
    return bs.set_index("id_cliente")[a["features"]]


@st.cache_data
def validacao() -> pd.DataFrame:
    return pd.read_parquet(TAB / "nb04_scores_validacao.parquet")


st.title("Score de inadimplência e política de crédito")
aba1, aba2, aba3 = st.tabs(["Pontuar cliente", "Simular política", "Submissão e monitoramento"])

with aba1:
    bs = submissao()
    cid = st.selectbox("Carregar cliente da submissão (id_cliente)", bs.index, index=0)
    base = bs.loc[cid]
    st.caption("Os campos vêm do cliente escolhido e podem ser editados para ver a decisão mudar.")
    cliente, cols = {}, st.columns(3)
    for i, f in enumerate(a["features"]):
        v = None if pd.isna(base[f]) else base[f]
        with cols[i % 3]:
            if f in a["categorias"]:
                ops = [None] + list(a["categorias"][f]) + ([v] if v is not None and v not in a["categorias"][f] else [])
                cliente[f] = st.selectbox(f, ops, index=ops.index(v), key=f"{f}_{cid}",
                                          format_func=lambda x: "(vazio)" if x is None else x)
            else:
                cliente[f] = st.number_input(f, value=None if v is None else float(v), key=f"{f}_{cid}")
    df = pd.DataFrame([cliente]).astype({f: float for f in a["features"] if f not in a["categorias"]})
    r, avisos = sc.pontuar(df)
    r = r.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilidade calibrada", f"{r.probabilidade:.2%}")
    c2.metric("Score bruto (antes da calibração)", f"{r.score_bruto:.4f}")
    c3.markdown(f"**Decisão**<br><span style='color:{COR[r.decisao]};font-size:1.6em'>{r.decisao}</span>",
                unsafe_allow_html=True)
    c4.metric("Faixa de gestão", r.faixa_gestao)
    for x in avisos:
        st.warning(x)
    if r.no_degrau_p1:
        st.info(f"Probabilidade no degrau da calibração isotônica (≈ p₁ = {a['p1']:.4f}): entre aprovar e ajuste, "
                "a decisão é definida por um desempate de 1e-6 pelo score bruto.")
    st.subheader("O que mais pesou (SHAP)")
    st.caption("Contribuições em log-odds do score bruto; positivas aumentam o risco. Não somam à probabilidade calibrada.")
    ctb = pd.Series(sc.explicar(df)[0]).rename("contribuição").rename_axis("variável").reset_index()
    st.altair_chart(alt.Chart(ctb).mark_bar().encode(
        x="contribuição:Q", y=alt.Y("variável:N", sort="-x"),
        color=alt.condition("datum['contribuição'] > 0", alt.value("#d62728"), alt.value("#2ca02c"))),
        width="stretch")

with aba2:
    v = validacao()
    y, q, ead = v.y.values, v.probabilidade.values, v.valor_credito.values
    s1, s2 = st.columns(2)
    p1 = s1.slider("p₁ (abaixo: aprovar direto)", 0.05, 0.35, a["p1"], 0.0001, format="%.4f")
    p2 = s2.slider("p₂ (a partir daqui: rejeitar)", 0.10, 0.60, a["p2"], 0.0001, format="%.4f")
    st.caption(f"Base: validação de 2023 ({len(v):,} contratos). Modelo: p₁ = {a['p1']:.4f}, p₂ = {a['p2']:.4f}. "
               "Mover p₁ para cima de ≈ 0,1179 faz a aprovação direta saltar (degrau da isotônica).")
    if p1 >= p2:
        st.error("p₁ precisa ser menor que p₂ (senão a zona de ajuste fica vazia).")
    else:
        nomes = {"taxa_aprovacao_direta": "aprovação direta", "taxa_ajuste": "aprovar com ajuste",
                 "taxa_rejeicao": "rejeição", "inad_aprovados_diretos": "inadimplência dos aprovados diretos",
                 "inad_ajuste": "inadimplência do ajuste", "inad_rejeitados": "inadimplência dos rejeitados",
                 "recall_nao_aprovar": "recall (barrar inadimplente)", "precision_rejeitar": "precision da rejeição",
                 "bons_barrados_por_mau_barrado": "bons barrados por mau barrado"}
        comp = pd.DataFrame({"escolhido": po.resumo_corte(y, q, p1, p2),
                             "modelo": po.resumo_corte(y, q, a["p1"], a["p2"])}).loc[list(nomes)].rename(index=nomes)
        el = pd.DataFrame({f"LGD {lgd:.0%}": {
            "EL aprovando tudo": po.perda_esperada(q, ead, lgd),
            "EL com a política (escolhido)": po.perda_esperada(q, ead, lgd, q < p2),
            "EL com a política (modelo)": po.perda_esperada(q, ead, lgd, q < a["p2"])} for lgd in LGD_CENARIOS})
        t1, t2 = st.columns(2)
        t1.dataframe(comp.style.format("{:.1%}").format("{:.2f}", subset=pd.IndexSlice[nomes["bons_barrados_por_mau_barrado"], :]))
        t2.dataframe(el.style.format("R$ {:,.0f}"))
        curva = pd.DataFrame([{"p1": t, **po.resumo_corte(y, q, t, max(t, p2))} for t in np.linspace(.05, .35, 61)])
        atual = pd.DataFrame([comp["escolhido"].rename(dict(zip(nomes.values(), nomes)))])
        enc = dict(x=alt.X("taxa_aprovacao_direta:Q", title="taxa de aprovação direta", axis=alt.Axis(format="%")),
                   y=alt.Y("recall_nao_aprovar:Q", title="recall (barrar inadimplente)", axis=alt.Axis(format="%")))
        st.altair_chart(alt.Chart(curva).mark_line(point=True).encode(**enc, tooltip=["p1"])
                        + alt.Chart(atual).mark_point(size=250, color="red", filled=True).encode(**enc),
                        width="stretch")

with aba3:
    dec = pd.read_csv(TAB / "nb04_decisoes_por_base.csv", index_col=0)
    st.subheader("Distribuição das decisões")
    st.bar_chart(dec.T, stack=False)
    st.subheader("Mudança de perfil: teste × submissão")
    st.caption("PSI por variável (> 0,25 = mudança forte) e diferença da contribuição média (SHAP).")
    shift = pd.read_csv(TAB / "nb04_shift_submissao.csv", index_col=0)
    st.dataframe(shift.style.format("{:.4f}").map(lambda x: "background-color:#f8d7da" if x > 0.25 else "",
                                                  subset=["PSI"]))
    f1, f2 = st.columns(2)
    f1.image(str(FIG / "nb04_calibracao.png"), caption="Calibração")
    f2.image(str(FIG / "nb04_shap_global.png"), caption="Importância das variáveis (SHAP)")
