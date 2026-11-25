# Plano — Modelo de Inadimplência + Política de Crédito (Case WeCogno / DS Accelerator)

## Contexto
O banco (crédito para pessoa física) quer reduzir a inadimplência. O README em PDF pede três entregas: (1) um modelo que estime a P(inadimplência) de cada contrato; (2) uma política de crédito (aprovar / ajustar / rejeitar) baseada no score; (3) as previsões para `dados/base_submissao.parquet`. A definição de inadimplência (o target) fica a nosso critério e precisa ser justificada.
Hoje o repositório tem apenas `dados/*.parquet`, `dicionario_dados.xlsx`, o PDF, o `.gitignore` e um `venv/` vazio (Python 3.12, sem pacotes). Decisões do usuário:
- **projeto do zero**;
- estrutura **notebooks + src/**;
- métricas principais: **recall e F1 da classe inadimplente**;
- **sexo e idade mantidos** como features (sem auditoria de viés);
- comparação **LightGBM vs XGBoost**.

Bases, conforme o dicionário:
- `base_cadastral`: 1 linha por cliente.
- `base_submissao`: o contrato mais recente de cada cliente, sem target.
- `historico_emprestimos`: 1 linha por contrato, desde 2017, com status, datas, valores e taxas.
- `historico_parcelas`: 1 linha por pagamento, com previsto vs real e `versao_parcela`.

Chaves: `id_cliente` e `id_contrato`.

## Estrutura do projeto
```
dados/                      # entrada (já existe, não alterar)
notebooks/
  01_qualidade_eda.ipynb            # entendimento, qualidade e análise exploratória
  02_target_features.ipynb          # definição do target + engenharia de features
  03_modelagem_lgbm_xgb.ipynb       # separação, baseline, LightGBM vs XGBoost
  04_avaliacao_politica.ipynb       # avaliação, explicabilidade e política de crédito
src/
  config.py        # caminhos, seeds, janelas, cortes
  data_loader.py   # leitura + limpeza/tipagem de cada base
  features.py      # target + montagem da base com as variáveis conhecidas na solicitação
  train.py         # divisão out-of-time + baseline, LGBM, XGB e tuning
  evaluate.py      # métricas, curvas, IC por bootstrap, PSI
  policy.py        # faixas de risco, regras e pontuação da submissão
tests/test_pipeline.py
outputs/figures, outputs/tables, submissao/
requirements.txt, README.md, docs/DECISOES.md
```
Regra: o notebook narra e mostra os resultados; a lógica reutilizada fica em `src/` e é importada pelos notebooks.

## Etapas

### 0. Setup
- `requirements.txt` com pandas, pyarrow, numpy, scipy, scikit-learn, lightgbm, xgboost, optuna, shap, matplotlib, seaborn, openpyxl, jupyter e pytest, instalados no `venv/`.

### 1. Qualidade e EDA (nb01, `data_loader.py`)
**Qualidade**
- Shape, tipos, nulos e cardinalidade de cada base.
- Duplicatas por chave: `id_cliente` (cadastral e submissão), `id_contrato` (empréstimos) e (contrato, parcela, versão) nas parcelas.
- Tipagem: datas em formatos mistos, renda como texto e categorias com grafia inconsistente.
- Integridade referencial entre as quatro bases. Cliente sem histórico = cold start.
- Colunas reais comparadas com o dicionário.
- **Investigar `versao_parcela`:**
  - quantas parcelas têm mais de uma versão;
  - o que muda entre as versões (vencimento, valor);
  - se a versão nova aparece em parcelas atrasadas (renegociação) ou só em pagamentos parciais.

  O resultado define se o target usa a versão original ou a vigente.

**EDA univariada e temporal**
- Distribuições, outliers (por IQR/percentis) e volume por safra.
- Comparação treino vs submissão com o **PSI** de cada variável, para detectar mudança de escala ou de população.

As análises bivariadas com o target (IV/WoE, χ², KS, VIF) ficam no nb02, porque dependem do target.

### 2. Target + features (nb02, `features.py`)
**Target (definido com o usuário)**
1. População: só contratos **aprovados** (`historico_emprestimos`) que têm parcelas.
2. Para cada parcela, entre as **6 primeiras**: vencimento (`data_prevista_pagamento`), valor previsto, total pago (soma dos pagamentos) e data do último pagamento.
3. Parcela **quitada** = total pago ≥ valor previsto, com tolerância de centavos para arredondamento.
4. DPD da parcela:
   - quitada: data do pagamento que completou a parcela − vencimento;
   - não quitada: data de observação − vencimento.
5. **Inadimplente (`target = 1`)** se alguma das 6 parcelas tiver **DPD > 5 dias** (target principal). Para comparação, também se calcula o target com **DPD > 30 dias**.
6. Maturidade: o contrato só entra se a 6ª parcela venceu há mais de X dias na data de observação (X = 5 ou 30, conforme o target).
   - No nb02, os dois targets são comparados: taxa de inadimplência, nº de contratos maduros, taxa por safra e quanto um contém o outro (todo DPD > 30 também é DPD > 5).
   - No nb03, os modelos podem ser treinados com os dois targets, para ver o efeito na recall/F1.
7. Tratamento de `versao_parcela`: **a decidir no nb01**. A verificação nos dados (se a versão muda quando a parcela atrasa) será feita no início da implementação e define se o target usa a versão original ou a vigente.
8. Janela fixa de 6 parcelas (*performance window*), sem comparação com o contrato inteiro. Justificativa registrada em `docs/DECISOES.md`: maturidade, exposição comparável entre prazos, proximidade da decisão e ambiguidade das parcelas ausentes (calote ou quitação antecipada). O saldo em aberto dentro da janela já é capturado pelo DPD da parcela não quitada.

- No nb02 serão mostradas a taxa de inadimplência resultante e a taxa por safra.
- **Balanceamento das classes**, medido para os dois targets (DPD > 5 e > 30):
  - quantidade e % de 0 e de 1 no total, por safra e nos splits de treino, validação e teste;
  - razão `n_neg / n_pos`, que é o valor de referência do `scale_pos_weight`.
- Regra de decisão, registrada em `docs/DECISOES.md`:
  - classe positiva **≥ ~20%**: desbalanceamento leve, sem `scale_pos_weight`; o corte otimizado na validação já resolve a recall/F1;
  - entre **~5% e 20%**: no nb03, comparar sem peso, com `scale_pos_weight = n_neg/n_pos` e com `√(n_neg/n_pos)`, todos com o corte otimizado;
  - **< ~5%**: usar o peso (ou o meio-termo pela raiz) e calibrar as probabilidades.
- Com peso, as probabilidades ficam infladas e precisam ser calibradas (isotônica/Platt) antes da política, que usa probabilidade.
- Limitação: viés de seleção (só aprovados têm desfecho). Reject inference fica fora do escopo.

**Variáveis (sem criar features novas)**
- Nesta etapa só definimos o target. **Nenhuma feature derivada** é criada: nada de razões, agregados do histórico ou transformações.
- Entram apenas as variáveis **conhecidas no momento da solicitação**, ou seja, as que existem na submissão:
  - do contrato: `tipo_contrato`, `valor_credito`, `valor_bem`, `valor_parcela`, `dia_semana_solicitacao`, `hora_solicitacao`;
  - do cadastro: todas as colunas de `base_cadastral`, incluindo sexo. A idade sai de `data_nascimento` na data do corte; é só a conversão de data para anos, para que a variável possa ser usada.
- Colunas que só existem depois da decisão (status, datas de liberação/encerramento, taxas, etc.) **não entram**. Elas servem apenas para construir o target.
- A base de modelagem é a junção do contrato aprovado e maduro com o target, as colunas acima do contrato e o cadastro do cliente.

**Análise bivariada e seleção**
- Taxa de default por faixa, **IV/WoE**, χ², Mann-Whitney/KS, Spearman e **VIF**.
- Remover variáveis com IV < 0,02, PSI > 0,25 ou colinearidade alta.

### 3. Modelagem LightGBM vs XGBoost (nb03, `train.py`)
**Separação dos dados**
- Divisão **out-of-time**: treino nas safras antigas, validação na seguinte, teste nas mais recentes. O teste é usado uma única vez.
- No treino, `StratifiedGroupKFold` agrupado por cliente.

**Modelos**
- Baseline: regressão logística (one-hot ou WoE, com escala).
- LightGBM e XGBoost nas mesmas condições: mesmas features e splits, mesmo Optuna (mesmo nº de trials e seed), early stopping.
- Desbalanceamento: aplicar a regra definida no nb02 a partir das quantidades medidas. Quando a comparação for necessária, o mesmo teste vale para o LGBM e o XGB (`scale_pos_weight` existe nos dois), e a escolha é pela recall/F1 na validação.

**Comparação**
- Métricas da etapa 4.
- Tempo de treino e de inferência.
- Estabilidade entre folds (média ± desvio).
- Concordância dos scores (Spearman).
- Importância das features.
- Diferença testada com **DeLong** (AUC) e bootstrap pareado (recall e F1). Em caso de empate, vence o modelo mais simples ou mais rápido.

### 4. Avaliação + política (nb04, `evaluate.py`, `policy.py`)
**Métricas**
- **Principais: recall e F1** da classe inadimplente. O corte é otimizado na validação, nunca no teste: o de máximo F1 e o que atinge um recall-alvo.
- Auxiliares: ROC-AUC, Gini, **KS**, PR-AUC, precision e F2.
- Calibração: Brier e curva de confiabilidade, com calibração isotônica ou Platt se necessário.

**Robustez**
- IC 95% por bootstrap.
- Desempenho por safra e por `tipo_contrato`.
- PSI do score entre treino e submissão.

**Explicabilidade**
- SHAP global e local.

**Política de crédito**

*Revisão frente ao que se espera do modelo:*
- O foco em recall vem do pedido de "política segura". Recall alto, porém, também barra bons clientes.
- A política, portanto, não pode ser um corte binário único: ela separa **"barrar o inadimplente"** (recall) de **"perder o bom cliente"** (taxa de aprovação).

*Pré-requisito:*
- A política usa a **probabilidade calibrada** do modelo final. Se houver `scale_pos_weight`, a calibração é obrigatória antes de definir qualquer corte.

*Três decisões e dois cortes, ambos escolhidos na validação:*
- **Corte inferior (p₁) — fronteira do "aprovar direto":**
  - escolhido para que a zona "não aprovar direto" (ajuste + rejeição) atinja a **recall-alvo de 80% ou mais** dos inadimplentes (definida com o usuário);
  - é aqui que o foco em recall entra na política.
- **Corte superior (p₂) — fronteira do "rejeitar":**
  - escolhido pelo **máximo F1** entre os contratos acima de p₁;
  - rejeitar exige alta concentração de maus, para não recusar bons clientes sem necessidade.
- Entre p₁ e p₂: **aprovar com ajuste**.

*Faixas de risco A–E (dentro das três decisões, para gestão):*
- Monotônicas e com IC de Wilson sem sobreposição entre faixas vizinhas.
- Mapa: A/B → aprovar, C/D → ajuste, E → rejeitar. Os cortes das faixas respeitam p₁ e p₂.

*Ajustes (limite, prazo, entrada, garantia, taxa):*
- São **recomendações de negócio**: os dados não mostram o efeito causal de um ajuste.
- Para cada faixa: o ajuste sugerido, o motivo (ex.: reduzir a exposição onde a inadimplência esperada é X%) e a ressalva de que precisa de teste A/B ou piloto.

*Simulação e trade-off:*
- Curva com taxa de aprovação × inadimplência da carteira aprovada × recall × precision, para cada par (p₁, p₂).
- Sensibilidade de perda esperada (EL = PD × LGD × EAD), com LGD como premissa explícita (ex.: 45%, 60%, 75%). EAD = `valor_credito`.
- Validação do par escolhido no teste OOT: as métricas devem ficar dentro do IC da validação.
- Aplicação na submissão: distribuição por faixa e decisão, PSI do score e alerta se a taxa de rejeição divergir do teste.

*Saída:*
- `submissao/previsoes.csv` com id_cliente, probabilidade, faixa e decisão, e 1 linha por cliente.

**Explicação minuciosa de cada decisão da política** (no nb04, no `docs/DECISOES.md` e no deck). Para cada corte (p₁, p₂ e os limites das faixas) e para cada ação, uma ficha com:
1. **O que foi decidido:** o valor do corte ou a regra.
2. **Critério:** a métrica otimizada e a restrição.
3. **Números que sustentam a escolha:** volume, taxa de inadimplência com IC, recall, precision, F1 e taxa de aprovação, na validação e no teste.
4. **Alternativas consideradas:** cortes vizinhos e o que se ganha ou perde com cada um.
5. **Trade-off assumido:** quantos bons clientes são recusados para cada inadimplente barrado.
6. **Impacto esperado:** inadimplência da carteira aprovada e perda esperada sob as premissas de LGD.
7. **Premissas e riscos:** o que pode invalidar a decisão (drift, viés de seleção, efeito do ajuste não medido) e quando revisar.

### 5. Apresentação e documentação
- `README.md` (problema, dados, decisões, resultados, como rodar) e `docs/DECISOES.md` (decisões e limitações).
- Deck executivo (artifact Slides): target, LGBM × XGB, métricas, política e impacto.

## Verificação
- `python -m pytest tests` cobre:
  - target: DPD de 5 contra 6 dias, parcela parcial, parcela não paga, maturidade;
  - base de modelagem: só colunas presentes na submissão (sem campos pós-decisão);
  - submissão: 1 linha por cliente e nenhuma categoria nova passando em silêncio.
- Os 4 notebooks rodam em ordem com `jupyter nbconvert --execute`, sem erro.
- O teste OOT não é usado em tuning nem na escolha de cortes; as métricas do teste ficam dentro do IC da validação; o PSI do score na submissão fica abaixo de 0,25.

## Forma de trabalho
Ordem: Setup → nb01 → nb02 → nb03 → nb04 → documentação/deck.
Uma etapa por vez. Antes de criar cada script ou notebook, apresento o que será introduzido no código (funções, regras, decisões) e debatemos; só implemento depois do seu aceite. Ao fim de cada etapa mostro os resultados e faço um commit próprio na branch `dev`.
