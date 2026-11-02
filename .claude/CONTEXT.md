# Contexto — Case DS Accelerator (WeCogno)
Banco PF quer reduzir inadimplência. Entregas: modelo de P(inadimplência) por contrato, política de crédito por faixas de risco, previsões para `dados/base_submissao.parquet`.

## Bases (`dados/`) — chaves: id_cliente, id_contrato
- base_cadastral: 1 linha/cliente (demografia, renda, moradia, notas de região). Snapshot atual.
- base_submissao: contrato mais recente por cliente em análise (data_solicitacao, dia_semana, hora, tipo_contrato, valor_credito, valor_bem, valor_parcela). Sem target.
- historico_emprestimos: 1 linha/contrato (status, datas, valores, taxas, canal, produto...).
- historico_parcelas: 1 linha/parcela PAGA (previsto vs real, versao_parcela).
- dicionario_dados.xlsx: descrição das colunas.

## Pontos de atenção
Target a construir (parcelas só existem se pagas); vazamento temporal; treino só com campos disponíveis na submissão; maturidade/censura; viés de seleção (só aprovados); validação out-of-time + CV por cliente.

## Estado
Projeto concluído. Ver README.md, REPORT.md e .claude/DECISIONS.md.
