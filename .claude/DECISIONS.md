# Log de decisões
- Target inicial: atraso >= 30d nas 12 primeiras parcelas (comparar com 60/90d no nb02).
- Política de crédito: apenas faixas de risco. Extensão documentada: simular com hipóteses explícitas de LGD/margem + sensibilidade.
- Sexo e idade MANTIDOS no modelo por decisão do usuário. Alerta: uso de sexo em crédito tem risco regulatório/ético (LGPD, não discriminação). Fazer ablação com/sem e reportar viés por segmento (nb05).
- Modelo final: LightGBM; baseline regressão logística.
- Deps instaladas no venv do projeto; requirements.txt na raiz.

## Achados do perfil inicial (Fase 0)
- Datas vêm como string, formatos mistos (ex.: submissão tem "2025-02-24" e "24/02/2025"); `renda_anual` é string em formato BR ("90.000,00"). Parsing obrigatório em data_loader.
- Nomes reais diferem do dicionário: `indicador_ultima_solicitacao_*`, `indicador_seguro_contratado` (dicionário diz flag_*).
- historico_emprestimos: 211.944 contratos / 47.952 clientes; status: Aprovado 131.735, Cancelado 40.266, Recusado 36.359, Oferta não utilizada 3.584. id_contrato único.
- historico_parcelas: 1,66M linhas; 79.945 duplicatas em (contrato, parcela, versão); 3.655 sem data_real_pagamento (parcela existe mas não paga?) — investigar no nb02.
- Submissão: 50.400 clientes, solicitações em fev/2025; 4,1% dos clientes sem histórico (cold start). Cadastro: 50.600 clientes.
- Console do Windows mostra acentos quebrados (cp1252); os parquets estão ok em UTF-8 — usar PYTHONIOENCODING=utf-8.
- Datas: 2025-13-01 (mês 13, 51 linhas na submissão) tratado como aaaa-dd-mm; nascimento 31/02/1990 (38) vira NaT; renda_anual 'não informado' vira NaN.
- Duplicidade: cadastro (600) e submissão (400) têm id_cliente repetido; cópias com valores numéricos ruidosos no fim do arquivo (índice >= 50000). Mantida a 1ª ocorrência. Resultado: 50.000 clientes únicos em ambos. Submissão final = 50.000 linhas.

## Target (implementado em src/target.py)
- Janela: 12 primeiras parcelas; versão vigente (max versao_parcela) por (contrato, parcela); pagamentos parciais somados; parcela quitada se soma paga >= 95% do previsto. DPD = último pagamento - vencimento; parcela não quitada: DPD = data_obs - vencimento.
- data_obs = max(data_real_pagamento) = 2025-02-22. Maduro = vencimento aproximado da última parcela da janela (1º vencimento + 30d*(n-1)) + 30d <= data_obs. Aprovados sem linha em parcelas são excluídos.
- Resultado: 103.907 de 131.735 aprovados; taxa de inadimplência (>=30d) = 14,1%; >=60d = 8,2%; >=90d = 4,4%; janela 6 parcelas = 8,9%. Taxa por ano de decisão: 9% (2017) -> 21% (2019) -> 12% (2022-24): há drift, usar validação out-of-time.
- Limitação: max_dpd chega a 2.893 dias (dados sujos nas datas); não afeta o evento >=30d.
- Ponto de corte do histórico no treino = data_decisao (a data da solicitação não existe no histórico); na submissão = data_solicitacao.

## Correções da revisão (ponytail-review)
- Histórico: parcela vencida e não paga até o corte agora conta como atraso (antes sumia). Razão paga = 0 nessas parcelas.
- Histórico: removidos h_n_aprovado e h_n_cancelado (status final de hoje, vazamento). Fica h_n_recusado.
- Target: contratos sem data_primeiro_vencimento (9.023 aprovados) usam vencimento estimado pelas parcelas, em vez de sumirem. Base: 104.029 contratos, 14,1% de inadimplência. Fora da base: 8.868 aprovados sem parcelas e o restante por imaturidade.
- Métricas novas no teste OOT: AUC geral 0,725 (IC95% 0,717-0,735), KS 0,333. Só clientes que não estavam no treino (7.955 contratos): AUC 0,617. Só com histórico (96% da submissão): AUC 0,732. A queda nos clientes novos mistura "modelo não conhece o cliente" com "cliente sem histórico" (sem histórico o AUC já era 0,64); não dá para separar os dois efeitos aqui.
- Testes: tests/test_pipeline.py (target com 3 parcelas, vencimento ausente, histórico sem usar o futuro).
- Não feito: cache do treino nos notebooks (cada notebook tem kernel próprio, sem ganho).

## Correções da auditoria (ponytail-audit) — ver REPORT.md
- Escala dos valores da submissão (crédito mediano 450 mil contra ~132 mil no treino): removidos valor_credito, valor_bem, valor_parcela, credito_sobre_renda e parcela_sobre_renda. AUC igual (0,7263 contra 0,7255). Persistem prazo_implicito e credito_sobre_bem, com sinal e com desvio. Pergunta em aberto para o banco: a submissão usa a mesma unidade do histórico? A rejeição de 18,9% na submissão (8,8% no teste) persiste sem essas variáveis.
- Texto normalizado sem acento no carregador; predict.py falha se mais de 0,5% das linhas tiver categoria nova.
- Parcelas: cópias do mesmo pagamento (mesma data) contam uma vez. Parcelas com pago > 1,5x o previsto: 12.136 -> 5.705.
- src/build_dataset.py monta as matrizes; predict.py confere que as 50.000 linhas são as de base_submissao.
- Cortes 0,10 / 0,25 escolhidos na validação (aprovar 5,4%, rejeitar 42,6% de inadimplência) e conferidos no teste (6,7% e 40,9%). Modelo final x validado na submissão: correlação 0,97, PSI 0,015.
- dados/ no .gitignore. Removidos código morto, pastas vazias e as dependências seaborn e openpyxl.
- Gráfico de importância: outputs/figures/feature_importance.png (+ outputs/feature_importance.csv).

## Segmentação por faixas de score (nb 07, `src/segmentation.py`)
- 5 faixas A–E (A = menor risco) por corte na probabilidade. Cortes 0,05 / 0,10 / 0,19 / 0,25: 0,10 e 0,25 vêm da política; 0,05 e 0,19 foram escolhidos na validação (grade; faixa ≥ 3% dos contratos, taxa crescente, IC95% de Wilson sem sobreposição entre vizinhas; vence o maior menor-salto de taxa).
- Taxa real no teste fora do tempo: A 4,3% · B 8,4% · C 15,7% · D 19,0% · E 40,9% (faixas válidas; D vs C encosta no limite do IC, e D tem só 3,8% da carteira).
- Mapa para a política: A+B aprovar, C+D aprovar com ajuste, E rejeitar.
- Submissão: A 33,2% · B 33,2% · C 12,0% · D 2,7% · E 18,9%. A taxa de cada faixa vem do teste, não da submissão (deslocamento de histórico/escala segue em aberto).
- Perfil: faixas A/B têm histórico longo e sem atraso (`h_pct_atraso` baixo); D/E têm atraso em ~40% das parcelas passadas e prazos médios maiores; C tem pouco histórico. Saídas em `outputs/tables/segment_profile.csv` e `outputs/figures/segment_*.png`.
