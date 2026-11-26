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
| 2.5 | Idade = anos entre `data_nascimento` e a **data da decisão** (treino) ou da solicitação (submissão; nula → data de observação) | Evita usar informação posterior ao pedido | Sim |
| 2.6 | Separação fora do tempo pela data de decisão: **treino até 31/12/2022 (62.580), validação 2023 (22.255), teste 2024 em diante (14.026)** | Mantém validação e teste parecidos (taxa 17,6% e 17,5%) e treino longo | Sim |
| 2.7 | Sem `scale_pos_weight` para DPD > 5 (≥ 20%); comparar pesos no nb03 para DPD > 30 | Regra de decisão do plano: 25,3% no treino | Sim (plano) |
| 2.8 | **Seleção de variáveis:** LightGBM e XGBoost usam todas as candidatas; só o baseline logístico usa o conjunto reduzido (remove as 9 variáveis com IV < 0,02, mantendo `sexo` por decisão do usuário) | 10 variáveis com IV < 0,02 (`renda_anual`, `tipo_moradia`, `nivel_educacao`, `estado_civil`, `sexo`, `dia_semana_solicitacao`, `possui_imovel`, `hora`, `tipo_contrato`, `possui_carro`); PSI > 0,25 só nos valores brutos (modelo C, só referência); nenhum VIF > 10 | Sim |
| 2.9 | A separação por data **exige `data_decisao` preenchida** (erro se faltar) | Deixar a data nula cair no teste (comportamento anterior: o contrato entrava no teste sem idade nem safra). Hoje há 0 casos; a regra protege atualizações futuras da base | Sim |

### Limitações do target
- **Maturidade e seleção:** 24 mil aprovados (≈ 20% dos que têm parcelas) ficam sem target: 20.450 com 6 ou mais parcelas planejadas mas menos de 6 vencidas (parcelas ausentes: quitação antecipada ou inadimplência?) e 3.528 com menos de 6 parcelas planejadas. A mistura de `tipo_contrato` não muda (38% / 37% / 25%), mas o efeito sobre o risco é desconhecido.
- **Deriva do target:** taxa de 25,3% no treino contra 17,5% no teste; probabilidades precisam de calibração na validação.
- **Safras recentes:** só entram contratos cuja 6ª parcela já venceu, o que pode enviesar 2024 e 2025.

## Etapa 3 — Modelagem (nb03, `src/train.py`)

Cada decisão registra o que foi sugerido, as alternativas e por que uma foi escolhida no lugar das outras.

| # | Decisão | Alternativas consideradas | Por que esta | Evidência | Validada |
|---|---|---|---|---|---|
| 3.1 | Tuning Optuna (TPE, 25 tentativas, CV 5 dobras agrupado por cliente, PR-AUC); nº de árvores vem do CV | Grade fixa; tunar na validação; mais tentativas | Validação fica intacta para escolher limiar e comparar; PR-AUC é adequada a classe minoritária. Mais tentativas = custo maior, ganho esperado de centésimos de AUC | Seção 1 | Sim |
| 3.2 | **LightGBM** como algoritmo | XGBoost; logística | LightGBM e XGBoost empatam (DeLong p=0,44/0,89; ΔAUC IC [−0,002; 0,004]; Spearman 0,95); regra pré-definida: empate → mais rápido (3,2 s vs 6,9 s). Logística fica ~0,09 de AUC abaixo | Seção 3 | Sim |
| 3.3 | **Conjunto final = A** (sem valores do contrato) | **B** (percentil dos valores no mês, sugerido por mim e que o usuário achou bom); **C** (valores brutos) | B falhou o critério 1 pré-registrado na validação (ΔAUC IC [−0,0001; 0,010]; ΔF1 IC [−0,013; 0,002]). No teste B ganha +0,009 de AUC (IC [0,003; 0,015]) mas sem ganho de F1. Critérios 2 (PSI 0,066) e 3 (Spearman 0,86) passaram. Ganho pequeno não compensa o risco de a escala dos valores mudar na submissão (PSI até 2,46 nos valores brutos). C só como referência (PSI do score 0,15, AUC 0,708) | Seção 4 | Sim (usuário aceitou o A) |
| 3.4 | DPD > 30 sem peso de classe | `n_neg/n_pos` (9,35); `√(n_neg/n_pos)` | PR-AUC praticamente igual (0,182 / 0,184 / 0,183); pesos distorcem probabilidades (Brier 0,074 vs 0,176) e o nb04 precisa delas calibradas | Seção 5 | Sim |
| 3.5 | DeLong devolve z = 0 e p = 1 quando os dois scores são idênticos | Deixar o resultado NaN (divisão 0/0) | Comparar um modelo com ele mesmo não é erro, é "sem diferença"; `tests/test_evaluate.py` confere o DeLong com o AUC do scikit-learn | Seção 3 | Sim |

Referência: AUC ≈ 0,70 (A: 0,699/0,694); logística ≈ 0,61. Poder preditivo modesto (F1 ≈ 0,48): a política deve usar calibração e faixas de risco.

## Etapa 4 — Avaliação e política (nb04, `src/policy.py`)

| # | Decisão | Alternativas consideradas | Por que esta | Evidência | Validada |
|---|---|---|---|---|---|
| 4.1 | Calibração **isotônica** ajustada na validação, escolhida por validação cruzada dentro da validação (o teste só confirma), com desempate pelo score bruto (1e-6) | Sem calibração; Platt | Score bruto superestima (média 0,22 vs 0,175 real). Brier no teste: isotônica 0,1181 < Platt 0,1208 < bruto 0,1236; ECE 0,010 / 0,035 / 0,060. A isotônica pura cria degraus que travam cortes (faixa C ficou vazia), por isso o desempate | Seção 1 | Sim |
| 4.2 | p₁ = **0,118**: maior corte com recall de "não aprovar direto" ≥ 80% (valid) | Recall-alvo maior ou menor; corte de máximo F1 único | Foco em recall pedido (política segura) sem barrar todos: 33% aprovados direto. Teste: recall 78,9%, dentro do IC | Seção 4 | Sim |
| 4.3 | p₂ = **0,250**: máximo F1 acima de p₁ | Rejeitar a partir de p₁; p₂ mais alto | Rejeitar só onde a inadimplência é 62%; 11% rejeitados, precision 62% (valid) / 62% (teste) | Seção 4 | Sim |
| 4.4 | **Decisão em 3 classes** (aprovar, ajuste, rejeitar) por p₁ e p₂; faixas A–E só como visão de gestão (A/B e C/D dividem as zonas na mediana do volume) | Manter A–E como decisões (plano original); quantis globais; cortes por inadimplência alvo | Faixas monotônicas, mas IC de Wilson sobreposto em A/B, B/C (e C/D no teste): o modelo (AUC 0,70) não distingue A–D; só E se separa. Faixas sem poder estatístico não sustentam decisões. O corte C/D cai no degrau da isotônica em 0,118 (igual a p₁) e separa clientes só pelo desempate do score bruto: C e D não são separáveis | Seção 5 | Sim |
| 4.5 | EL = PD × LGD × EAD com LGD 45/60/75% e EAD = `valor_credito`; ajuste sem redução de exposição | Reduzir exposição na faixa de ajuste | Efeito do ajuste não é medido; reduz EL em 31% (valid) / 35% (teste) só pela rejeição | Seção 6 | Sim |
| 4.6 | Alerta da submissão aceito e documentado (sem ajuste ao modelo) | Reajustar o corte para a submissão; ignorar | Nenhuma variável isolada mudou (PSI máx. 0,04); soma de +0,20 log-odds espalhada (renda, organização, ocupação); AUC adversarial 0,61. Não se separa mudança real de perfil de efeito de seleção (treino só com aprovados); mudar corte sem desfecho observado seria arbitrário | Seção 10 | Sim |
| 4.7 | **Corte único** (p₁ e p₂ do conjunto todo), com o desempenho por clientes já vistos e novos reportado à parte | Escolher p₁ só entre clientes novos; separar o treino por cliente | Não há como confirmar se o produto atende sobretudo clientes novos (65% da submissão já aparece no treino). Escolher só por novos deixaria o corte conservador demais para os demais. No teste: AUC 0,749 (já vistos) e 0,640 (novos); recall de "não aprovar direto" 82% e 76%. Se o foco for clientes novos, refazer p₁ com `outputs/tables/nb04_clientes_vistos_novos.csv` | Seção 7b | Sim |
| 4.8 | Modelo final em `models/modelo_final.joblib` com as versões das bibliotecas dentro (`versoes`) | Salvar só o modelo; reajustar a cada uso | O arquivo é *pickle* (executa código ao abrir e depende da versão das bibliotecas): as versões explicam uma falha de abertura; carregar só arquivos gerados pelo próprio projeto. `tests/test_pipeline.py` confere que reproduz `previsoes.csv` | Seção 11 | Sim |

**Alerta da submissão:** PSI do score 0,067, mas rejeição de 19,9% contra 11,0% no teste (probabilidade média 0,225 contra 0,175).

**Categoria nova na submissão:** `tipo_renda = 'Empresário'` não existe no treino; `train.preparar` avisa e trata como nulo (o aviso agora aparece no nb04). Quantidade de clientes afetados: ver a saída do nb04.

## Etapa 5 — Revisão e empacotamento (README, testes, `models/`)

| # | Decisão | Alternativas consideradas | Por que esta | Validada |
|---|---|---|---|---|
| 5.1 | `README.md` (entregas, resultados, limitações, como rodar e usar o modelo), `docs/DECISOES.md` e `docs/METRICAS.md`; deck executivo **fora do escopo por ora** | Deck no Slides | Pedido do usuário: o deck não é necessário agora | Sim |
| 5.2 | `openpyxl` fica em `requirements.txt` | Remover | O nb01 lê o dicionário de dados em Excel com `pandas.read_excel` | Sim |
| 5.3 | Testes: target, features, categorias novas, cortes e faixas, calibrador, estatísticas (`evaluate`), data de decisão, previsões e modelo final | Só testar nos notebooks | Os cálculos que decidem modelo e cortes quebram sem aviso; 14 testes | Sim |

## Limitações conhecidas
- **Clientes repetidos entre treino e teste:** 17.271 dos 43.338 clientes da base de modelagem aparecem em mais de um período (inevitável na separação por data). No teste, o AUC é 0,749 para clientes já vistos no treino e 0,640 para novos; para a submissão, 64,7% dos clientes já foram vistos. O modelo usa só variáveis de cadastro, então pode reconhecer o cliente. Por isso o desempenho para clientes novos (AUC ≈ 0,64, recall de "não aprovar direto" ≈ 76% no teste) é o mais conservador (`outputs/tables/nb04_clientes_vistos_novos.csv`).
- Viés de seleção: só contratos aprovados têm desfecho. Reject inference fora do escopo.
- Das 131.735 contratações aprovadas, 8.868 não têm parcelas (primeira parcela ainda não venceu); concentram-se em 2024 (12%) e 2025 (74%).
- 2.048 clientes da submissão não têm histórico de empréstimos e 3.769 não têm contrato com parcelas. As features do modelo são só cadastro e contrato, então isso não afeta a pontuação.
