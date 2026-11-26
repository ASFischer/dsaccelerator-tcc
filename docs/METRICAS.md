# Medidas estatísticas do projeto

Para cada medida: o que é, como funciona e **como foi usada para decidir algo**. Os números vêm dos notebooks (`outputs/tables`). Código em `src/evaluate.py`, `src/policy.py` e `src/train.py`.

## 1. Qualidade dos dados e seleção de variáveis (nb01, nb02)

### PSI — Population Stability Index (`evaluate.psi`)
- **O que é:** mede quanto a distribuição de uma variável (ou do score) mudou entre duas populações.
- **Como funciona:** divide a população de referência em 10 faixas por quantis (variáveis categóricas: uma faixa por categoria; nulos formam uma faixa própria) e soma, sobre as faixas, `(%atual − %referência) × ln(%atual / %referência)`.
- **Leitura usual:** < 0,10 estável; 0,10–0,25 atenção; > 0,25 mudança relevante.
- **Uso:** os valores brutos `valor_*` tinham PSI de até 2,46 entre treino e submissão, o que gerou a estratégia A/B/C (decisão 1.10). Para o score, PSI teste × submissão = 0,067, abaixo de 0,25. Por variável, o PSI máximo na submissão é 0,04.

### IV e WoE — Information Value / Weight of Evidence (`evaluate.iv_woe`)
- **O que é:** poder de uma variável isolada para separar bons e maus pagadores.
- **Como funciona:** por faixa, `WoE = ln(%bons / %maus)`; `IV = Σ (%bons − %maus) × WoE`. Soma 0,5 às contagens para evitar divisão por zero.
- **Leitura usual:** < 0,02 sem poder; 0,02–0,10 fraco; 0,10–0,30 médio; > 0,30 forte.
- **Uso:** 10 variáveis com IV < 0,02. O LightGBM e o XGBoost mantêm todas; só a logística usa o conjunto reduzido (decisão 2.8).

### Qui-quadrado e V de Cramér
- **O que é:** teste de associação entre duas variáveis categóricas (qui-quadrado) e sua intensidade de 0 a 1 (V de Cramér = √(χ² / (n × (k − 1)))).
- **Uso:** associação de cada variável categórica com o target na análise bivariada (nb02).

### KS e Mann-Whitney (comparação de distribuições)
- **O que é:** testes sem hipótese de normalidade: o KS compara as funções de distribuição acumuladas; o Mann-Whitney compara os postos (ranks) de dois grupos.
- **Uso:** cada variável numérica entre inadimplentes e bons pagadores (KS de duas amostras e p-valor do Mann-Whitney, nb02).

### Correlação de Spearman
- **O que é:** correlação entre os **postos** de duas variáveis (mede relação monotônica, robusta a outliers).
- **Uso:** relação de cada numérica com o target e pares de numéricas com |ρ| > 0,7 (nb02), e **concordância entre scores** de modelos diferentes (LGBM × XGB: 0,95; B × A na submissão: 0,86).

### VIF — Variance Inflation Factor
- **O que é:** quanto a variância de um coeficiente cresce por colinearidade: `VIF = 1 / (1 − R²)`, com R² da regressão da variável nas demais.
- **Uso:** calculado nos conjuntos B e C (numéricas padronizadas); nenhum VIF > 10. Relevante para a logística; árvores são pouco afetadas.

## 2. Validação do modelo

### Separação fora do tempo (*out-of-time*)
- **Como funciona:** treino até 31/12/2022, validação em 2023, teste de 2024 em diante, pela data da decisão.
- **Uso:** imita o uso real (prever o futuro). A **validação** escolhe limiar, cortes e calibração; o **teste** só confirma.

### `StratifiedGroupKFold` agrupado por cliente
- **Como funciona:** divide o treino em 5 dobras preservando a taxa de inadimplência, sem que um mesmo cliente apareça em duas dobras.
- **Uso:** tuning do Optuna e número de árvores, sem vazamento entre contratos do mesmo cliente.

### Optuna (TPE) e *early stopping*
- **Como funciona:** o TPE propõe hiperparâmetros prováveis de melhorar o PR-AUC, aprendendo com as tentativas anteriores (25 por modelo, seed fixa). O *early stopping* interrompe o treino quando o PR-AUC da dobra não melhora por 100 árvores; o número final de árvores é a média das melhores iterações × 1,1.
- **Uso:** mesmo orçamento para LGBM e XGBoost (comparação justa).

## 3. Métricas de desempenho (`evaluate.metricas`)

Os termos: TP = inadimplentes previstos como inadimplentes; FP = bons previstos como inadimplentes; FN = inadimplentes não detectados.

| Medida | Como funciona | Uso |
|---|---|---|
| **Recall** | TP / (TP + FN): fração dos inadimplentes capturados | Principal (pedido de "política segura") |
| **Precision** | TP / (TP + FP): fração de acertos entre os barrados | Custo de barrar bons clientes |
| **F1** | média harmônica de precision e recall | Principal; define o limiar de máximo F1 e o corte p₂ |
| **F2** | como o F1, mas o recall pesa 4× mais | Auxiliar: sensibilidade ao foco em recall |
| **ROC-AUC** | probabilidade de um inadimplente receber score maior que um bom (0,5 = acaso) | Poder de ordenação independente do corte |
| **Gini** | 2 × AUC − 1 | Convenção em crédito |
| **KS** | maior distância entre as acumuladas de score de maus e bons | Poder de separação em um ponto |
| **PR-AUC** (*average precision*) | área sob a curva precision × recall | Métrica do tuning: não é inflada pelos bons clientes (classe majoritária) |
| **Brier** | média de (probabilidade − resultado)² | Qualidade da probabilidade (calibração + separação) |
| **Taxa de aprovação** | fração prevista como não inadimplente | Custo comercial do corte |

- **Limiar de máximo F1** (`limiar_max_f1`): varre os cortes da curva precision × recall na validação e escolhe o de maior F1. Nunca é escolhido no teste.
- **Resultado:** AUC ≈ 0,70, F1 ≈ 0,48, recall ≈ 0,39 nesse limiar; logística: AUC ≈ 0,61.

## 4. Comparação de modelos e conjuntos de variáveis

### Teste de DeLong (`evaluate.delong`)
- **O que é:** teste de hipótese para a diferença de ROC-AUC entre dois modelos **avaliados nos mesmos dados** (pareado, usa a covariância entre as AUCs).
- **Uso:** LGBM × XGBoost: p = 0,44 (validação) e 0,89 (teste), sem diferença estatística, então vale a regra "empate → mais rápido".

### Bootstrap pareado (`evaluate.bootstrap_pareado`)
- **Como funciona:** 1.000 reamostragens com reposição do conjunto de avaliação; em cada uma calcula a diferença de AUC, recall e F1 entre dois modelos com a mesma amostra. O IC 95% são os percentis 2,5% e 97,5% das diferenças.
- **Leitura:** se o IC inclui 0, não há evidência de diferença.
- **Uso:** B × A. Na validação o ΔAUC tem IC [−0,0001; 0,010] e o ΔF1 [−0,013; 0,002], o que reprovou o B no critério pré-registrado (decisão 3.3).

### IC 95% por bootstrap das métricas (nb04)
- **Como funciona:** o mesmo método, aplicado a cada métrica de um único modelo.
- **Uso:** AUC 0,694 (IC 0,682–0,707) no teste. Também usado para checar se as métricas da política no teste ficam dentro do IC da validação.

## 5. Calibração (nb04, `policy.calibrador`, `policy.ece`)

- **Probabilidade calibrada:** entre os clientes com score 0,20, cerca de 20% devem inadimplir. É necessária porque a política usa a probabilidade, não só o ranking, e porque a taxa de inadimplência caiu de 25,3% para 17,5%.
- **Curva de confiabilidade:** compara probabilidade prevista média e taxa observada em 10 faixas de mesmo volume; a diagonal é a calibração perfeita.
- **ECE** (*expected calibration error*): média ponderada de |taxa observada − probabilidade prevista| nessas faixas.
- **Platt:** logística de uma variável sobre o logit do score. Suave, mas rígida (assume um formato).
- **Isotônica:** função em degraus não decrescente ajustada aos dados. Mais flexível; os degraus geram empates, então soma-se 1e-6 × score bruto para desempatar sem mudar a ordenação.
- **Uso:** ajustadas na validação, avaliadas no teste. Isotônica: Brier 0,1181 e ECE 0,010; Platt: 0,1208 e 0,035; bruto: 0,1236 e 0,060 (decisão 4.1).

## 6. Política de crédito (`policy.py`)

### Corte p₁ — recall-alvo
- **Como funciona:** ordena a validação do maior para o menor score e acha o maior corte em que `p ≥ corte` captura pelo menos 80% dos inadimplentes.
- **Uso:** define a fronteira do "aprovar direto" (p₁ = 0,118). Recall: 80,0% na validação e 78,9% no teste.

### Corte p₂ — máximo F1 acima de p₁
- **Como funciona:** varre os cortes ≥ p₁ e escolhe o de maior F1 de "prever inadimplente".
- **Uso:** define a fronteira do "rejeitar" (p₂ = 0,250); entre os rejeitados, 62% inadimplem.

### IC de Wilson (`policy.wilson`)
- **O que é:** intervalo de confiança para uma proporção, mais confiável que o normal simples com amostras pequenas ou proporções extremas.
- **Uso:** taxa de inadimplência por decisão e por faixa. **Sobreposição de ICs entre faixas vizinhas = não distinguíveis**: foi o que reduziu as decisões a 3 e deixou A–E só como visão de gestão (decisão 4.4).

### Bons clientes barrados por inadimplente barrado
- **Como funciona:** (FP / TP) da zona "não aprovar direto".
- **Uso:** expressa o custo comercial do recall-alvo (≈ 3,7).

### Perda esperada — EL = PD × LGD × EAD (`policy.perda_esperada`)
- **Como funciona:** PD = probabilidade calibrada de inadimplência; EAD = exposição (`valor_credito`); LGD = fração perdida em caso de inadimplência, uma **premissa** (45%, 60%, 75%) porque os dados não trazem recuperação.
- **Uso:** comparar "aprovar tudo" com a política. A política reduz a EL em 31% (validação) e 35% (teste), só pela rejeição (ajustes sem efeito medido).

## 7. Explicabilidade e deriva

### SHAP
- **Como funciona:** decompõe cada previsão em contribuições de cada variável (em log-odds), com base nos valores de Shapley da teoria dos jogos: a média ponderada do efeito da variável sobre todas as combinações das demais.
- **Uso:** importância global (|SHAP| médio: `tipo_organizacao`, `renda_anual`, `qtd_membros_familia`, `ocupacao`) e explicação de casos; também a diferença média de contribuição entre teste e submissão.

### Validação adversarial
- **Como funciona:** treina um classificador para distinguir a população A (teste) da B (submissão). AUC 0,5 = indistinguíveis; perto de 1 = muito diferentes.
- **Uso:** AUC 0,613 entre teste e submissão: diferença moderada de perfil, espalhada em várias variáveis (PSI por variável ≤ 0,04, soma de +0,20 log-odds nas contribuições).
