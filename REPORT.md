# Relatório do projeto: o que foi feito e onde

Case DS Accelerator (WeCogno): modelo de inadimplência e política de crédito. Os resultados estão no [README.md](README.md). Este arquivo lista o que foi feito, em que ordem e em qual arquivo.

## 1. Entregas

| Entrega | Onde |
|---|---|
| Previsões para os 50.000 clientes da submissão, com probabilidade e decisão | `submissao/submission_predictions.csv` |
| Modelo treinado (LightGBM) | `models/lgbm.joblib` |
| Política de crédito por faixas de risco | `src/policy.py`, `notebooks/06_credit_policy.ipynb` |
| Gráfico de importância das variáveis | `outputs/figures/feature_importance.png`, `outputs/feature_importance.csv` |
| Outros gráficos (SHAP, calibração) | `outputs/figures/` |
| README para o GitHub | `README.md` |
| Log de decisões | `.claude/DECISIONS.md` |

## 2. Fases do projeto

| Fase | O que foi feito | Onde |
|---|---|---|
| Planejamento | Leitura do enunciado e do dicionário. Plano com riscos e estrutura de pastas, revisado com a skill data-scientist | `.claude/PLAN.md`, `.claude/CONTEXT.md` |
| Preparação | Pastas, ambiente virtual e dependências | `requirements.txt`, `.gitignore` |
| Carregamento e limpeza | Datas, texto sem acento, renda, duplicatas por cliente | `src/data_loader.py` |
| Qualidade dos dados | Tamanhos, nulos, chaves, integridade entre tabelas | `notebooks/01_eda_data_quality.ipynb` |
| Target | Atraso de 30 dias ou mais nas 12 primeiras parcelas, em contratos aprovados e maduros. Teste com 60 e 90 dias | `src/target.py`, `notebooks/02_target_definition.ipynb` |
| Variáveis | Contrato atual, cadastro e histórico anterior ao corte | `src/features.py`, `src/build_dataset.py`, `notebooks/03_feature_engineering.ipynb` |
| Modelagem | Divisão por data (treino, validação, teste), regressão logística como comparação, LightGBM com parada antecipada | `src/train.py`, `notebooks/04_modeling.ipynb` |
| Avaliação | AUC com intervalo de confiança, KS, Gini, Brier, decis, calibração, PSI, SHAP, importância, teste sem sexo e idade, viés por sexo e idade | `src/evaluate.py`, `notebooks/05_evaluation_explainability.ipynb` |
| Política | Cortes escolhidos na validação e conferidos no teste e na submissão | `src/policy.py`, `notebooks/06_credit_policy.ipynb` |
| Previsões | Pontuação da submissão com a política aplicada | `src/predict.py` |
| Testes | 5 testes automáticos | `tests/test_pipeline.py` |
| Documentação | README, este relatório, log de decisões | `README.md`, `REPORT.md`, `.claude/DECISIONS.md` |

## 3. Correções pedidas depois das revisões

### Primeira revisão (`ponytail-review`)

| # | Ponto | O que foi feito | Onde |
|---|---|---|---|
| 1 | Métricas podiam estar otimistas por clientes repetidos entre treino e teste | AUC calculado também só nos clientes que o modelo não viu (0,618) e só nos que têm histórico (0,733) | `src/train.py` (`rodar`) |
| 2 | Parcela vencida e não paga sumia do histórico | Passa a contar como atraso, com atraso igual ao corte menos o vencimento | `src/features.py` (`_hist`) |
| 3 | Status final do contrato anterior vazava informação do futuro | Removidas as contagens de aprovado e cancelado; ficou só a de recusado | `src/features.py` |
| 4 | Métricas não refletiam o perfil da submissão | AUC e quadro da política só para quem tem histórico | `src/train.py`, notebook 06 |
| 5 | Faltavam testes | Testes do target, do contrato sem vencimento e do histórico sem usar o futuro | `tests/test_pipeline.py` |
| 6 | Contratos sem primeiro vencimento eram descartados | Data estimada pelas parcelas; notebook 02 mostra por que contratos ficam de fora | `src/target.py`, notebook 02 |
| 7 | Código lento e repetido | Removido o `lambda` por grupo e o `__import__`. O cache do treino nos notebooks não foi feito: cada notebook roda em kernel próprio e não ganharia nada | `src/features.py`, `src/predict.py` |

### Auditoria (`ponytail-audit`)

| # | Ponto | O que foi feito | Onde |
|---|---|---|---|
| 1 | Valores dos contratos da submissão em outra escala | Comparei 4 conjuntos de variáveis. Os valores absolutos e as razões com a renda não ajudam (AUC 0,7263 sem eles contra 0,7255 com), então foram removidos. `prazo_implicito` e `credito_sobre_bem` ficaram | `src/features.py`, README |
| 2 | Categorias com e sem acento viravam vazio em silêncio | Texto sem acento no carregador; `predict.py` falha se mais de 0,5% das linhas tiver categoria nova | `src/data_loader.py` (`normalizar`), `src/predict.py` |
| 3 | Parcelas repetidas somadas duas vezes | Cópias do mesmo pagamento contam uma vez. Parcelas com valor pago acima de 1,5 vez o previsto: de 12.136 para 5.705 | `src/target.py` (`parcelas_janela`) |
| 4 | Pipeline só rodava passando por um notebook | Criado o módulo que monta as matrizes; o notebook 03 chama esse módulo | `src/build_dataset.py` |
| 5 | Cortes escolhidos e avaliados no mesmo teste | Escolhidos na validação e conferidos no teste. Comparei o modelo final com o validado na submissão (correlação 0,97, PSI 0,015) | `src/policy.py`, notebook 06 |
| 6 | Faltavam testes de carregamento e de saída | Testes de datas, acentos e parcelas repetidas. `predict.py` confere que as 50.000 linhas são as de `base_submissao` | `tests/test_pipeline.py`, `src/predict.py` |
| 7 | Dados dos clientes podiam ir para o git | `dados/` no `.gitignore` | `.gitignore` |
| 8 | Código morto e dependências sem uso | Removidas 3 constantes, 2 pastas vazias, `seaborn` e `openpyxl` | `src/config.py`, `src/train.py`, `src/features.py`, `requirements.txt` |

### Outros pedidos

- **Nomes em inglês:** notebooks, documentos em `.claude/`, arquivos de `data/interim/` e o CSV de previsões foram renomeados. As pastas `dados/` e `submissao/` e os parquets de entrada ficaram como estão, pelos nomes do enunciado.
- **Gráfico de importância:** não existia no código; foi criado no notebook 05.

## 4. O que não foi resolvido

1. **Escala dos valores da submissão.** Remover os valores absolutos não custou AUC, mas não explica a diferença. `prazo_implicito` (a variável mais importante, 22% do ganho) continua com distribuição diferente na submissão. É preciso perguntar ao banco se a unidade é a mesma.
2. **Rejeição de 18,9% na submissão, contra 8,8% no teste.** Persiste em todas as versões de variáveis que testei. A causa provável é o histórico (96% da submissão tem histórico, contra 61% do treino), mas não confirmei. Os cortes devem ser revistos com o banco.
3. **Clientes novos.** O AUC cai para 0,618, mas não dá para separar "o modelo não conhece o cliente" de "o cliente não tem histórico".
4. **Valores finais dos contratos anteriores** (`valor_credito`, `qtd_parcelas_planejadas`) podem ter mudado depois do corte. Não medi o efeito.
5. **Sexo e idade** ficaram no modelo por decisão do projeto. Sem eles o AUC é igual. O risco regulatório está descrito no README.
6. **Ações de "aprovar com ajuste"** são sugestões, sem validação com dados.
7. **Não verificado:** quantas parcelas repetidas são cópias e quantas são pagamentos parciais; não testei o projeto em outra máquina.

## 5. Como conferir

```
python -m src.build_dataset
python -m src.train
python -m src.predict
python -m pytest tests
```

Na última execução: 5 testes passando, seis notebooks sem erro e 50.000 previsões com ids únicos.
