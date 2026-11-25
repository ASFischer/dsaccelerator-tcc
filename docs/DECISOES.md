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

## Limitações conhecidas
- Viés de seleção: só contratos aprovados têm desfecho. Reject inference fora do escopo.
- Das 131.735 contratações aprovadas, 8.868 não têm parcelas (primeira parcela ainda não venceu); concentram-se em 2024 (12%) e 2025 (74%).
- 2.048 clientes da submissão não têm histórico de empréstimos e 3.769 não têm contrato com parcelas. As features do modelo são só cadastro e contrato, então isso não afeta a pontuação.
