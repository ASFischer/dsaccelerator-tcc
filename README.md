# Modelo de inadimplência e política de crédito

Case WeCogno / DS Accelerator (enunciado em `README - Modelo de inadimplência.pdf`): estimar a probabilidade de inadimplência de cada contrato de crédito para pessoa física, definir uma política de crédito (aprovar, aprovar com ajuste, rejeitar) e pontuar a base de submissão.

## Entregas
| Entrega | Onde |
|---|---|
| Modelo de P(inadimplência) | LightGBM com calibração isotônica (`notebooks/03`, `notebooks/04`) |
| Política de crédito | p₁ = 0,118 e p₂ = 0,250 (`notebooks/04`, `src/policy.py`) |
| Modelo final | `models/modelo_final.joblib` (modelo, calibrador, cortes p₁/p₂, variáveis e categorias) |
| Previsões da submissão | `submissao/previsoes.csv` (`id_cliente`, `probabilidade`, `decisao`, `faixa_gestao`; 1 linha por cliente) |
| Decisões e alternativas | `docs/DECISOES.md` (o `docs/PLANO.md` é o plano original; onde divergem, vale o `DECISOES.md`) |
| Medidas estatísticas explicadas | `docs/METRICAS.md` |

## Problema e dados
- Bases em `dados/`: cadastro (1 linha por cliente), histórico de empréstimos (1 por contrato), histórico de parcelas (1 por pagamento) e a base de submissão (contrato mais recente de cada cliente, sem target).
- **Target:** o contrato é inadimplente se alguma das **6 primeiras parcelas** teve atraso > 5 dias (DPD > 5); DPD > 30 é a alternativa de comparação. Só contratos aprovados cuja 6ª parcela já venceu têm target (98.861 contratos; 22,5% inadimplentes).
- **Variáveis:** só o que se conhece na solicitação (contrato sem valores + cadastro). Os valores do contrato (`valor_credito`, `valor_bem`, `valor_parcela`) mudaram de escala na submissão (PSI até 2,46) e ficaram de fora (conjunto A).
- **Separação fora do tempo:** treino até 2022 (62.580), validação 2023 (22.255) e teste de 2024 em diante (14.026).

## Resultados
| | Validação | Teste |
|---|---|---|
| ROC-AUC (LightGBM) | 0,699 | 0,694 (IC 95%: 0,682–0,707) |
| F1 / recall no limiar de máximo F1 | 0,479 / 0,386 | 0,477 / 0,388 |
| Baseline logístico, sem as variáveis de IV < 0,02 (AUC) | 0,612 | 0,600 |

- **LightGBM × XGBoost:** empate estatístico (DeLong p = 0,44 e 0,89); ficou o LightGBM, mais rápido.
- **Calibração:** o score bruto superestimava o risco (média 0,22 contra 17,5% observado); a isotônica corrige (Brier 0,118 no teste).
- **DPD > 30:** pesos de classe não melhoraram o ranking e distorcem as probabilidades; o modelo segue sem peso.
- **Política:** `p < p₁` aprova direto; `p₁ ≤ p < p₂` aprova com ajuste; `p ≥ p₂` rejeita. p₁ é o maior corte que barra pelo menos 80% dos inadimplentes (validação) e p₂ é o de máximo F1 acima de p₁. Na validação, 33% são aprovados direto (inadimplência 10,5%), 56% aprovados com ajuste (13,1%) e 11% rejeitados (62,9%). O recall de "não aprovar direto" é 80,0% na validação e 78,9% no teste, ao custo de cerca de 3,7 bons clientes barrados por inadimplente. A perda esperada cai 31% (validação) e 35% (teste), só pela rejeição.
- **Submissão:** 29% aprovar, 51% ajuste e 20% rejeitar. `faixa_gestao` (A–E) é só visão de acompanhamento; a decisão vem de p₁ e p₂.

## Limitações
- O poder preditivo é modesto (AUC 0,70). Só a faixa de rejeição se separa com segurança; as faixas A–D não se distinguem estatisticamente e servem só como visão de gestão.
- **Clientes já vistos no treino:** 56% dos contratos de teste e 65% dos clientes da submissão já aparecem no treino com outros contratos. No teste, o AUC é 0,75 para esses clientes e 0,64 para clientes novos (recall de "não aprovar direto" 82% contra 76%). O desempenho esperado para clientes novos é o menor.
- **Viés de seleção:** o modelo só viu contratos aprovados e maduros (≈ 24 mil aprovados ficam sem target).
- **Submissão mais arriscada que o teste** (probabilidade média 0,225 contra 0,176; rejeição de 19,9% contra 11,0%), com mudança de perfil moderada (AUC adversarial 0,61) espalhada em várias variáveis. A taxa de rejeição da submissão supõe que a relação entre perfil e risco se mantém.
- Os ajustes sugeridos (limite, prazo, entrada, garantia, taxa) são recomendações de negócio; o efeito deles não foi medido e exige piloto. A LGD (45%, 60%, 75%) é uma premissa.

## Como rodar
```
python -m venv venv
venv/Scripts/pip install -r requirements.txt          # Windows; em Linux/macOS: venv/bin/pip
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_qualidade_eda.ipynb 02_target_features.ipynb 03_modelagem_lgbm_xgb.ipynb 04_avaliacao_politica.ipynb
cd .. && python -m pytest tests
```
Execute na ordem. O nb02 grava as bases em `data/` (ignorada pelo git), o nb03 faz o tuning (≈ 35 min na primeira vez; reaproveita `data/modelos/*.json` depois) e o nb04 grava `submissao/previsoes.csv` e `models/modelo_final.joblib`. Python 3.12, versões fixadas em `requirements.txt`; seed 42.

## Usar o modelo final
```python
import sys, joblib; sys.path.insert(0, "src")   # o calibrador é da classe policy.Calibrador
import train as tr, policy as po
a = joblib.load("models/modelo_final.joblib")
X = tr.preparar(novos_clientes, a["features"], a["categorias"])   # mesmas colunas de features.FEATURES["A"]
prob = a["calibrador"](a["modelo"].predict_proba(X)[:, 1])
decisao = po.faixa(prob, a["cortes_faixas"]).map(po.DECISAO)      # aprovar / aprovar com ajuste / rejeitar
```
As variáveis do conjunto A são calculadas por `features.base_submissao`. O arquivo é um *pickle*: carregue apenas arquivos gerados por você, e use as versões de `requirements.txt` (as usadas ficam em `a["versoes"]`).

## Estrutura
```
dados/            entrada
notebooks/        01 qualidade e EDA · 02 target e features · 03 modelagem · 04 avaliação e política
src/              config, data_loader, features, train, evaluate, policy
tests/            target, features, categorias novas, estatísticas (`evaluate`), cortes da política, calibrador, previsões e modelo final
models/           modelo final (gerado pelo nb04)
outputs/          figuras e tabelas dos notebooks
docs/             PLANO, DECISOES, METRICAS
submissao/        previsoes.csv
```
