# Decisões e limitações

Registro das decisões do projeto, em ordem de etapa. Cada item diz o que foi decidido, por quê e quem validou.

## Etapa 1 — Qualidade e EDA (nb01, `src/data_loader.py`)

| # | Decisão | Evidência | Validada |
|---|---|---|---|
| 1.1 | Cadastral e submissão: manter a **1ª ocorrência** de cada `id_cliente` | 600 (cadastral) e 400 (submissão) clientes duplicados; a cópia fica no fim do arquivo e tem valores alterados (renda, hora, valor do crédito) | Sim |
| 1.2 | Categorias: tirar espaços, uniformizar caixa e acento, mapear para a forma mais frequente | `sexo` 6→3 categorias, `dia_semana` 10→7, `tipo_contrato` 6→3 | Sim |
| 1.3 | `renda_anual` para número; "não informado" e **renda negativa → nulo** | 4 formatos de texto ("R$ …", sem milhar, etc.), 101 "não informado" e ~67 valores negativos (−2 mil a −50 mil) que não parecem sinal trocado | Sim |
| 1.4 | `data_solicitacao` fora do padrão → **nulo** | Todas as 51 são "2025-13-01" e todas as 50 são "24/02/2025", com dias da semana variados: são valores constantes, não datas recuperáveis. A coluna não é feature | Sim |
| 1.5 | Parcelas: remover só as **linhas idênticas em todas as colunas** (4.976) | Nesses casos a soma bruta dos pagamentos bate com o previsto em 1,3% das parcelas; sem as repetições, em 94,7% | Sim |
| 1.6 | `versao_parcela` **não é renegociação**: é o fracionamento da parcela | Vencimento igual entre versões em 99,7% dos casos; soma dos previstos ≈ pago em 98%; o valor pago aparece repetido em cada versão | Sim |
| 1.7 | Consolidação da parcela (no `features.py`): **previsto = soma dos previstos das versões**; **pago = valor pago de uma versão só**, somando pagamentos parciais dentro da versão; **vencimento = `data_prevista_pagamento`** | Itens 1.5 e 1.6 | Sim |
| 1.8 | Data de observação = **22/02/2025** | É o máximo de `data_real_pagamento` e de vencimento; o arquivo não tem parcelas futuras | Sim |
| 1.9 | Outliers (renda máx. 117 mi; `valor_credito` da submissão máx. 135 mi) **sem tratamento no loader**; decisão no modelo | Máx./p99 = 217× (renda) e 72× (crédito, submissão) | Sim |

### 1.10 Valores do contrato na submissão fora da distribuição do treino
- **Problema:** PSI treino × submissão de `valor_bem` 2,46, `valor_credito` 1,96 e `valor_parcela` 0,95; as demais variáveis têm PSI < 0,05. Mediana de `valor_credito`: 450 mil na submissão, 97 mil no treino e 101 mil nos aprovados de 2025. As razões (crédito/bem, parcela/crédito, crédito/renda) também têm PSI de 0,6 a 2,1, então não é só escala.
- **Diagnóstico (nb01, seção 4.2):** os contratos da submissão não estão no histórico; em relação aos aprovados do próprio cliente o valor é ~4× maior, mas muito disperso (P25 1,9 / P75 8,5). Recusados têm valor mediano maior que aprovados, então parte da diferença pode ser seleção (solicitações × aprovados).
- **Decisão (validada):** manter os valores como feature, mas como **percentil dentro do período**, e comparar três candidatos no nb03:
  - **A:** sem valores (referência segura);
  - **B:** percentil do valor dentro do mês da decisão, calculado **somente entre contratos aprovados**; na submissão, percentil dentro da própria submissão;
  - **C:** valor bruto (só referência, para medir a degradação).
- **Critério de adoção do B** (definido antes dos resultados): (1) supera o A na validação e no teste fora do tempo com ganho maior que o IC 95% do bootstrap; (2) PSI do score na submissão < 0,25; (3) Spearman entre os scores de B e A na submissão ≥ 0,8. Se algum falhar, o modelo final é o A e o ganho possível do B vira limitação.
- **Risco residual:** supõe-se que a posição relativa do contrato carrega o mesmo risco hoje e no passado; os critérios 2 e 3 detectam sintomas dessa falha, mas não a provam.
- **Exceção ao plano:** o percentil é uma feature derivada (o plano dizia "sem features derivadas"); só entra no modelo final se o B for adotado.

## Etapa 2 — Target e features (nb02, `src/features.py`)

| # | Decisão | Evidência | Validada |
|---|---|---|---|
| 2.1 | Target = alguma das **6 primeiras parcelas** com DPD > 5 dias (principal) e > 30 (comparação); só contratos aprovados e **maduros** (6ª parcela venceu há mais de X dias) | Regra do plano; 98.861 contratos maduros, 22,5% de inadimplência (DPD > 5) e 9,4% (DPD > 30); todo DPD > 30 é também DPD > 5 | Sim (plano) |
| 2.2 | DPD da parcela quitada = data de conclusão − vencimento; não quitada = observação (22/02/2025) − vencimento; quitada se pago ≥ previsto − R$ 0,01 | Consolidação do item 1.7; 99,75% das parcelas quitadas | Sim (plano) |
| 2.3 | Janela fixa de 6 parcelas (*performance window*) | Maturidade, exposição comparável, proximidade da decisão e ambiguidade das parcelas ausentes | Sim (plano) |
| 2.4 | Percentil dos valores (modelo B) calculado **só entre os aprovados do mesmo mês**, antes de filtrar os maduros | Item 1.10 | Sim |
| 2.5 | Idade = anos entre `data_nascimento` e a **data da decisão** (treino) ou da solicitação (submissão; nula → data de observação) | Evita usar informação posterior ao pedido | Proposta |
| 2.6 | Separação fora do tempo pela data de decisão: **treino até 31/12/2022 (62.580), validação 2023 (22.255), teste 2024 em diante (14.026)** | Mantém validação e teste parecidos (taxa 17,6% e 17,5%) e treino longo | Sim |
| 2.7 | Sem `scale_pos_weight` para DPD > 5 (≥ 20%); comparar pesos no nb03 para DPD > 30 | Regra de decisão do plano: 25,3% no treino | Sim (plano) |
| 2.8 | **Seleção de variáveis:** LightGBM e XGBoost usam todas as candidatas; só o baseline logístico usa o conjunto reduzido (remove as 9 variáveis com IV < 0,02, mantendo `sexo` por decisão do usuário) | 10 variáveis com IV < 0,02 (`renda_anual`, `tipo_moradia`, `nivel_educacao`, `estado_civil`, `sexo`, `dia_semana_solicitacao`, `possui_imovel`, `hora`, `tipo_contrato`, `possui_carro`); PSI > 0,25 só nos valores brutos (modelo C, só referência); nenhum VIF > 10 | Sim |

### Limitações do target
- **Maturidade e seleção:** 24 mil aprovados (≈ 20% dos que têm parcelas) ficam sem target: 20.450 com 6 ou mais parcelas planejadas mas menos de 6 vencidas (parcelas ausentes: quitação antecipada ou inadimplência?) e 3.528 com menos de 6 parcelas planejadas. A mistura de `tipo_contrato` não muda (38% / 37% / 25%), mas o efeito sobre o risco é desconhecido.
- **Deriva do target:** taxa de 25,3% no treino contra 17,5% no teste; probabilidades precisam de calibração na validação.
- **Safras recentes:** só entram contratos cuja 6ª parcela já venceu, o que pode enviesar 2024 e 2025.

## Limitações conhecidas
- Viés de seleção: só contratos aprovados têm desfecho. Reject inference fora do escopo.
- Das 131.735 contratações aprovadas, 8.868 não têm parcelas (primeira parcela ainda não venceu); concentram-se em 2024 (12%) e 2025 (74%).
- 2.048 clientes da submissão não têm histórico de empréstimos e 3.769 não têm contrato com parcelas. As features do modelo são só cadastro e contrato, então isso não afeta a pontuação.
