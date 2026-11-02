> Plano original da fase de planejamento. O estado atual e as decisões estão em README.md, REPORT.md e DECISIONS.md.

# Plano: Modelo de Inadimplência + Política de Crédito (DS Accelerator / WeCogno)

## Context
Banco PF quer reduzir inadimplência. Entregáveis do case: (1) modelo que estima P(inadimplência) por contrato; (2) política de crédito (aprovar / rejeitar / ajustar condições) usando o score; (3) previsões para os contratos de `base_submissao.parquet`. O target **não existe**: precisamos defini-lo e justificá-lo a partir de `historico_emprestimos` + `historico_parcelas`.

Arquivos lidos: README (PDF), `dicionario_dados.xlsx`, 4 parquets em `dados/` (só tamanhos; **pyarrow não está instalado** e o `venv/` só tem pip — não pude abrir os parquets, então o EDA é a 1ª etapa real).

Tabelas e chaves:
- `base_cadastral` (1 linha/cliente): demografia, renda, moradia, notas de região.
- `base_submissao` (1 linha/cliente, contrato mais recente): data_solicitacao, tipo_contrato, valor_credito, valor_bem, valor_parcela. **Sem target** — é o que se pontua.
- `historico_emprestimos` (1 linha/contrato): status, datas, valores, taxas, canal, produto etc.
- `historico_parcelas` (1 linha/parcela *paga*): previsto vs. real, versão da parcela.

## Riscos centrais (guiam o desenho)
1. **Definição do target**: parcelas só existem se pagas → atraso/não-pagamento precisa ser inferido (parcela prevista sem pagamento, `data_real_pagamento` > `data_prevista_pagamento`, `valor_pago` < `valor_previsto`). Contratos recusados/cancelados não têm performance.
2. **Vazamento temporal**: features só com informação anterior à `data_solicitacao`/`data_decisao` do contrato. Parcelas do próprio contrato são target, nunca feature. Features de histórico do cliente usam apenas contratos/parcelas anteriores ao ponto de corte.
3. **Descompasso treino × submissão**: submissão traz só 5 campos de contrato (sem taxa, canal, finalidade…). O modelo só pode usar, do contrato atual, esses campos + cadastro + histórico do cliente.
4. **Maturação/censura**: contratos recentes ainda sem janela de observação suficiente → excluir ou usar janela fixa (ex.: atraso > X dias nas primeiras N parcelas).
5. **Versões de parcela** (`versao_parcela`): deduplicar mantendo a versão vigente; checar duplicidades.
6. **Validação**: split **out-of-time** (por data), não aleatório; agrupar por cliente para evitar vazamento entre folds.

## Estrutura de pastas (criada na execução)
```
dsaccelerator-tcc/
├── .claude/                     # contexto e planejamento (memória do projeto)
│   ├── CONTEXT.md              # problema, bases, chaves, decisões tomadas
│   ├── PLAN.md                 # este plano (cópia) + checklist de progresso
│   └── DECISIONS.md              # log: definição de target, cortes, métricas, por quê
├── dados/                       # (existente) dados brutos — somente leitura
├── data/{interim,processed}/    # parquets intermediários e base de modelagem
├── notebooks/
│   ├── 01_eda_data_quality.ipynb
│   ├── 02_target_definition.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation_explainability.ipynb
│   └── 06_credit_policy.ipynb
├── src/
│   ├── config.py                # caminhos, seeds, parâmetros (corte temporal, janela do target)
│   ├── data_loader.py           # leitura + validação de integridade/chaves
│   ├── target.py                # funções de definição do target (variantes)
│   ├── features.py              # features cadastrais, contrato, histórico (point-in-time)
│   ├── train.py                 # pipeline de treino/CV/tuning
│   ├── evaluate.py              # AUC, KS, Gini, calibração, lift/decis, PSI
│   ├── policy.py                # cortes de score, simulação de aprovação/perda
│   └── predict.py               # pontua base_submissao
├── models/                      # modelo serializado + metadados
├── outputs/{figures,tables}/
├── submissao/                   # submission_predictions.csv/parquet (id_cliente, prob, decisão)
├── requirements.txt             # pandas, pyarrow, scikit-learn, lightgbm, shap, matplotlib, seaborn, jupyter, openpyxl
└── README.md
```
Notebooks = exploração e narrativa; lógica reutilizável vive em `src/*.py` e é importada pelos notebooks.

## Plano de ação (fases)
**0. Setup** — instalar deps no `venv` (`pyarrow`, `openpyxl`, etc.), `requirements.txt`, criar pastas, `.gitignore` (venv, data/, models/), `.claude/CONTEXT.md`.

**1. EDA e qualidade (nb 01)** — shape, tipos, nulos, duplicatas, unicidade de chaves, integridade referencial entre as 4 tabelas, outliers/códigos sentinela (ex.: dias = 365243 típicos), distribuição de status_contrato, cobertura temporal (2017→hoje), sobreposição de clientes submissão × histórico.

**2. Target (nb 02, `target.py`)** — construir população: contratos aprovados/liberados com parcelas observáveis. Testar variantes de inadimplência: (a) atraso ≥ 30 dias em qualquer parcela, (b) ≥ 60/90 dias, (c) nas primeiras 6/12 parcelas (janela fixa), (d) proporção paga < previsto. Comparar taxa de eventos, estabilidade no tempo, tamanho do "indeterminado" e curva de safras (vintage). Escolher uma, justificar (equilíbrio evento/maturidade/relevância de negócio). Registrar em `DECISIONS.md`.

**3. Features (nb 03, `features.py`)** — ponto de corte = data da solicitação:
- Cadastro: idade, tempo de renda, razão renda/crédito, renda per capita, flags.
- Contrato: crédito/renda, parcela/renda, crédito/bem, prazo implícito, hora/dia.
- Histórico do cliente (anterior ao corte): nº contratos, % aprovados/recusados, motivos de recusa, atraso médio/máximo, % parcelas pagas com atraso, razão pago/previsto, utilização, tempo desde último contrato, agregados em janelas (3/6/12m).
- Encoding (target/ordinal/one-hot conforme cardinalidade), tratamento de nulos; checagem anti-vazamento.

**4. Modelagem (nb 04, `train.py`)** — baseline: regressão logística (WOE/scorecard) + LightGBM/HistGB. Split out-of-time (treino/validação/teste temporal) + CV agrupado por cliente. Tratar desbalanceamento (pesos, sem SMOTE inicialmente). Tuning leve (Optuna ou random search). Seleção de features por importância/estabilidade.

**5. Avaliação (nb 05, `evaluate.py`)** — AUC, KS, Gini, PR-AUC, Brier, curva de calibração (calibrar com isotônica/Platt se preciso), lift por decil, PSI treino × teste × submissão, SHAP global/local, análise de segmentos (viés por sexo/idade — verificar uso de atributos sensíveis e fairness).

**6. Política de crédito (nb 06, `policy.py`)** — faixas de risco por score (ex.: baixo/médio/alto): aprovar / aprovar com ajuste (reduzir valor, aumentar entrada, menos parcelas, exigir seguro) / rejeitar. Simular trade-off aprovação × inadimplência esperada × perda (hipóteses explícitas de LGD e margem), escolher corte que reduz inadimplência com custo de aprovação aceitável; sensibilidade ao corte.

**7. Submissão (`predict.py`)** — pontuar `base_submissao`, aplicar política, gerar `submissao/submission_predictions.*` com `id_cliente`, probabilidade, faixa e decisão sugerida; validar contagem/chaves e distribuição do score vs. treino.

**8. Documentação** — README com resultados, `DECISIONS.md` atualizado, resumo executivo para o banco.

## Verificação
- Testes de sanidade em `src/` (asserts): unicidade de chaves, nenhum feature com data > corte, sem id duplicado na submissão.
- Rodar notebooks 01→06 em ordem a partir de kernel limpo (`jupyter nbconvert --execute`) sem erro.
- Métricas no conjunto out-of-time reportadas; AUC treino × teste sem gap grande; calibração e PSI aceitáveis.
- `submissao/` com 1 linha por `id_cliente` da base original e sem nulos em probabilidade.

## Revisão (skill data-scientist) — ajustes incorporados
Pontos fortes: split out-of-time, features point-in-time, baseline LR + GBM, calibração, PSI, SHAP. Lacunas corrigidas:
1. **Target sem parcelas não pagas**: `historico_parcelas` só tem parcelas pagas. Reconstruir a grade esperada por contrato (`qtd_parcelas_planejadas`, `data_primeiro_vencimento`, parcelas 1..min(12, planejadas)); parcela esperada e ausente com vencimento já passado na data de observação = inadimplente; parcela sem vencimento ainda ocorrido = censurada. Validar essa regra no nb02 antes de qualquer modelo (é a maior fonte de erro do projeto).
2. **Data de observação** = max das datas nos dados; só entram na modelagem contratos com as 12 parcelas (ou o prazo total, se menor) já vencidas. Registrar % excluído por maturidade e checar se isso enviesa por safra.
3. **Viés de seleção**: só contratos aprovados têm desempenho; o modelo não vê o perfil dos recusados (reject inference fora de escopo). Documentar como limitação e cuidar ao extrapolar política para perfis tipicamente recusados.
4. **Cadastro é snapshot atual**, não point-in-time (renda, moradia podem ter mudado depois do contrato). Documentar; priorizar features estáveis.
5. **Campos do contrato atual**: treino só pode usar os 5 campos presentes na submissão; taxa, canal, finalidade, status etc. entram apenas como histórico de contratos *anteriores* do cliente. Teste automático: conjunto de colunas de treino ⊆ colunas disponíveis na submissão.
6. **Contratos repetidos por cliente**: CV agrupado por cliente; opcional treinar também na amostra "1 contrato por cliente" para espelhar a submissão. Peso maior para safras recentes (drift desde 2017).
7. **Incerteza**: IC por bootstrap para AUC/KS/Gini (em vez de p-valor isolado); estabilidade por safra; calibração feita no conjunto temporal posterior ao treino.
8. **Sexo/idade mantidos (decisão do usuário)**: rodar ablação com/sem sexo e idade, reportar o delta de AUC/KS e a taxa de aprovação por grupo; o uso de sexo em decisão de crédito tem risco regulatório (LGPD e normas de crédito/não discriminação) — registrar o alerta em `DECISIONS.md` e deixar a escolha final explícita.
9. **Política por faixas**: calibrar o nº de faixas pelo lift por decil; incluir curva aprovação × inadimplência esperada para o banco escolher o corte, mesmo sem LGD/margem; regra de ajuste (valor/prazo/entrada) só onde a feature realmente é monotônica no risco (verificar via SHAP/PDP). Acompanhamento pós-implantação: PSI mensal e backtest.
10. Mudança de escopo mínima: `requirements.txt` inclui `optuna` apenas se o tuning for além de random search.

## Decisões confirmadas com o usuário
- Ambiente: instalar deps no `venv` na execução (pyarrow, scikit-learn, lightgbm, shap, openpyxl, jupyter…) e gerar `requirements.txt`.
- Target inicial: atraso ≥ 30 dias nas primeiras 12 parcelas (validado/comparado com 60/90d no nb 02).
- Política de crédito: **somente por faixas de risco** (cortes de score → aprovar / ajustar / rejeitar). Sem simulação financeira no escopo principal; `DECISIONS.md` e o README documentam, como extensão possível, o uso de hipóteses explícitas de LGD/margem com análise de sensibilidade (a Fase 6 acima fica restrita às faixas).
- Atributos sensíveis: **manter sexo e idade** no modelo; ainda assim medir e documentar viés por segmento (nb 05) e registrar o risco regulatório/ético em `DECISIONS.md`.
- Modelo final: LightGBM com regressão logística como baseline interpretável.
