# Passo a passo: como o projeto funciona, de ponta a ponta

Este texto explica o projeto inteiro para quem não é da área: o que entra, o que sai e o que acontece no meio. Ao final de cada parte há os comandos para repetir o processo sozinho. Os detalhes técnicos de cada escolha estão em `docs/DECISOES.md`, e as medidas estatísticas em `docs/METRICAS.md`.

---

## 1. A ideia em uma frase
Uma financeira quer saber, **no momento em que o cliente pede o crédito**, qual a chance de ele atrasar o pagamento. Com essa chance, ela decide entre **aprovar**, **aprovar com ajuste** (limite menor, garantia, entrada...) ou **rejeitar**.

## 2. O caminho completo

```
 dados/ (4 tabelas)                         o que a financeira já tem
    │
    ▼
 notebook 01  qualidade dos dados           corrige formatos, acha duplicatas e erros
    │
    ▼
 notebook 02  "quem foi mau pagador?"       cria a resposta (target) e as variáveis
    │
    ▼
 notebook 03  treina o modelo               LightGBM × XGBoost × regressão logística
    │
    ▼
 notebook 04  calibra e define a política   probabilidade "honesta" + cortes p₁ e p₂
    │
    ├──► submissao/previsoes.csv            a resposta do case (50.000 clientes)
    └──► models/modelo_final.joblib         o modelo "empacotado" num arquivo
              │
              ▼
         src/scoring.py                     uma função que pontua qualquer cliente
              │
       ┌──────┴───────┐
       ▼              ▼
    API (porta 8000)  App (porta 8501)      como sistemas e pessoas usam o modelo
       └──────┬───────┘
              ▼
           Docker                           tudo sobe com um comando, em qualquer máquina
```

### 2.1 Os dados (`dados/`)
São quatro tabelas: o **cadastro** dos clientes (idade, renda, profissão...), o **histórico de empréstimos** (um contrato por linha), o **histórico de parcelas** (cada pagamento, com a data prevista e a data real) e a **base de submissão**, que traz os 50.000 pedidos novos que precisamos pontuar.

### 2.2 Qualidade (notebook 01, `src/data_loader.py`)
Os dados reais vêm sujos. "Sexta feira " e "sexta-feira" viram uma coisa só, uma renda escrita "R$ 112.500,00" vira número, e datas impossíveis viram "vazio". Nada é inventado: o que está faltando continua faltando.

### 2.3 Quem foi mau pagador (notebook 02, `src/features.py`)
Para o modelo aprender, ele precisa de exemplos com a resposta certa. A regra usada foi: **o contrato é "inadimplente" se alguma das 6 primeiras parcelas atrasou mais de 5 dias.** Só entram contratos antigos o bastante para a 6ª parcela já ter vencido.

As **variáveis** (o que o modelo olha) são só o que se sabe no momento do pedido: o tipo de contrato, o dia e a hora do pedido e o cadastro. Os valores do contrato ficaram de fora, porque mudaram de escala na base de submissão e enganariam o modelo.

### 2.4 Treinar o modelo (notebook 03, `src/train.py`)
O modelo é um **LightGBM**: um conjunto de centenas de "árvores de decisão" pequenas, e cada uma corrige um pouco o erro das anteriores. Ele foi comparado com o XGBoost (empate) e com uma regressão logística simples (pior).

A separação foi **pelo tempo**: o modelo aprende com 2022 e anos anteriores, os ajustes usam 2023 e a nota final sai de 2024 em diante. É como estudar com provas antigas e ser avaliado numa prova nova.

### 2.5 Calibrar e decidir (notebook 04, `src/policy.py`)
- **Calibração:** o modelo cru exagerava o risco (dizia 22% em média, quando o real era 17,5%). A *calibração isotônica* corrige isso, para que "12%" signifique de fato cerca de 12 em cada 100.
- **Política:** dois cortes na probabilidade:
  - abaixo de **p₁ = 0,118**: **aprovar**;
  - entre p₁ e **p₂ = 0,250**: **aprovar com ajuste**;
  - a partir de p₂: **rejeitar**.
  
  p₁ foi escolhido para que 80% dos maus pagadores não sejam aprovados direto. p₂ é o ponto de melhor equilíbrio entre acertar os maus e não recusar os bons.
- **Resultado:** a submissão ficou com 29% aprovados, 51% com ajuste e 20% rejeitados.

> **Um detalhe importante (o "degrau"):** a calibração isotônica funciona em degraus, e 38% dos clientes caem exatamente no degrau de ≈ 0,1179, colado em p₁. Para eles, o que separa "aprovar" de "ajuste" é um desempate minúsculo. Por isso a API e o app **avisam** quando um cliente está nesse degrau.

---

## 3. Do notebook ao produto

Até aqui o modelo só existe dentro de notebooks. Para alguém **usar** o modelo, foram feitas três peças.

### 3.1 O núcleo: `src/scoring.py`
É uma função só, `pontuar`, que recebe uma tabela de clientes e devolve para cada um:

| Campo | O que é |
|---|---|
| `probabilidade` | chance calibrada de inadimplência (0 a 1) |
| `score_bruto` | o número do modelo antes da calibração |
| `no_degrau_p1` | `True` se o cliente está no degrau de p₁ (ver 2.5) |
| `decisao` | aprovar / aprovar com ajuste / rejeitar |
| `faixa_gestao` | A a E, só para acompanhamento |

Ela também devolve uma lista de **avisos**: se chega uma categoria que o modelo nunca viu (por exemplo `tipo_renda = "Empresário"`), ela vira "vazio" e o aviso diz isso, em vez de esconder.

A função `explicar` diz **o que mais pesou** na nota de cada cliente (técnica SHAP). Valores positivos aumentam o risco e negativos diminuem. Eles estão na escala do score bruto, por isso não somam à probabilidade final.

Linha a linha, o que `pontuar` faz:
```python
X, avisos = _x(df)                                  # separa as 18 variáveis e trata categorias novas
bruto = a["modelo"].predict_proba(X)[:, 1]          # o LightGBM dá o score bruto
prob = a["calibrador"](bruto)                       # a isotônica transforma em probabilidade
faixa = po.faixa(prob, a["cortes_faixas"]).values   # compara com os cortes → A..E
... po.DECISAO ...                                  # A/B = aprovar, C/D = ajuste, E = rejeitar
```
Conferência: pontuando os 50.000 clientes da submissão com essa função, o resultado é **idêntico** ao `submissao/previsoes.csv`.

### 3.2 A API (`api/main.py`)
**O que é uma API:** uma "porta de entrada" para outro *programa*, e não para uma pessoa. O sistema da financeira manda os dados do cliente pela rede e recebe a resposta, sem ninguém abrir notebook.

- `GET /saude`: "está tudo bem?". Devolve qual modelo está carregado, os cortes e as métricas.
- `POST /score`: recebe de 1 a 1.000 clientes e devolve a tabela do item 3.1. Com `?explicar=true`, devolve também as contribuições SHAP.

**Proteções na entrada** (feitas pelo Pydantic, que confere os tipos):
- texto onde deveria ser número (`"idade": "trinta"`) → erro **422**;
- campo com nome errado (`"idadee"`) → erro **422**, para não virar "vazio" sem ninguém perceber;
- campo ausente → tratado como valor faltante (o modelo sabe lidar com isso).

A página **http://localhost:8000/docs** (Swagger) é gerada sozinha: lista os endpoints, traz um cliente real de exemplo e tem o botão *Try it out* para testar no navegador.

### 3.3 O app (`app/streamlit_app.py`)
**O que é o Streamlit:** uma biblioteca que transforma um script Python numa página web com botões, sliders e gráficos. É a "vitrine" para pessoas. O app tem três abas:

1. **Pontuar cliente:** você escolhe um `id_cliente` da submissão, e o formulário se preenche sozinho. Dá para mudar qualquer campo (renda, profissão, idade...) e ver a decisão mudar na hora, com o gráfico do que mais pesou.
2. **Simular política:** dois sliders movem p₁ e p₂ sobre os 22 mil contratos de 2023. A tela mostra quanto se aprova, quanto se perde e quantos maus pagadores escapam, lado a lado com os cortes do modelo. No padrão, os números batem com o notebook 04 (aprovação 33,4%, recall 80,0%, rejeição 10,8%). Passando p₁ para 0,1179, a aprovação salta para 66%: é o degrau.
3. **Submissão e monitoramento:** como as decisões se distribuem em cada base e quais variáveis mudaram de perfil (PSI > 0,25 aparece em vermelho).

O app usa `scoring.py` direto, então funciona mesmo com a API desligada.

### 3.4 O Docker (`Dockerfile`, `docker-compose.yml`)
**O que é:** uma "caixa" com tudo dentro: Python, bibliotecas nas versões certas, código, modelo e dados. Quem tem Docker roda a caixa sem instalar mais nada.
- **Imagem** é a caixa fechada, a receita pronta (`Dockerfile`).
- **Container** é a caixa em funcionamento. A mesma imagem vira dois containers: `api` (porta 8000) e `app` (porta 8501), definidos no `docker-compose.yml`.
- Detalhe: a imagem instala `libgomp1`, uma biblioteca do sistema sem a qual o LightGBM não abre no Linux.
- O `.dockerignore` deixa de fora o que não precisa ir na caixa (venv, notebooks, `data/`).

---

## 4. Como rodar sozinho

### 4.1 Preparar o ambiente (uma vez)
```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### 4.2 Testes automáticos
```powershell
venv\Scripts\python -m pytest tests
```
Saída esperada: `18 passed`. Os 4 testes da API conferem que a API reproduz `previsoes.csv`, que entrada errada dá 422, que categoria nova gera aviso e que `explicar=true` traz as contribuições.

### 4.3 API
```powershell
venv\Scripts\uvicorn api.main:app --port 8000
```
Espere a linha `Uvicorn running on http://127.0.0.1:8000`. Depois:
- no navegador, abra http://localhost:8000/docs → `POST /score` → *Try it out* → *Execute*;
- ou em outro PowerShell:
```powershell
Invoke-RestMethod http://localhost:8000/saude
$c = '[{"tipo_contrato":"Financiamento de consumo","sexo":"Feminino","idade":31,"renda_anual":90000}]'
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/score?explicar=true" -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($c))
```
Saída esperada (resumida):
```
probabilidade : 0,1677
score_bruto   : 0,2967
no_degrau_p1  : False
decisao       : aprovar com ajuste
faixa_gestao  : D
contribuicoes : @{tipo_organizacao=-0,3206; idade=0,1526; ...}
```
Para parar o servidor: `Ctrl+C`.

### 4.4 App
```powershell
venv\Scripts\streamlit run app/streamlit_app.py
```
O navegador abre em http://localhost:8501. A primeira carga leva alguns segundos (lê as bases de `dados/`).

### 4.5 Docker (Docker Desktop precisa estar aberto)
```powershell
docker compose up --build     # constrói a imagem e sobe api + app (a 1ª vez demora alguns minutos)
docker ps                     # lista os containers rodando
docker compose logs api       # mostra o que a API escreveu
docker compose down           # derruba tudo
```
Com os containers no ar, valem os mesmos endereços: http://localhost:8000/docs e http://localhost:8501.

### 4.6 Refazer o modelo do zero (opcional, ≈ 40 min)
```powershell
cd notebooks
..\venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 01_qualidade_eda.ipynb 02_target_features.ipynb 03_modelagem_lgbm_xgb.ipynb 04_avaliacao_politica.ipynb
```

---

## 5. Erros comuns

| Sintoma | Causa | Como resolver |
|---|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` (ou `streamlit`) | Bibliotecas novas não instaladas | `venv\Scripts\pip install -r requirements.txt` |
| `Error loading ASGI app. Could not import module "api.main"` | uvicorn rodado de dentro da pasta `api/` | Rode da raiz do projeto |
| `address already in use` / porta ocupada | Outro servidor (ou o Docker) já usa a porta 8000 ou 8501 | Feche o outro, rode `docker compose down` ou use `--port 8001` |
| Resposta 422 da API | Tipo errado ou nome de campo errado | Leia o `detail`: ele diz o campo (`loc`) e o motivo (`msg`) |
| `avisos` com "categorias fora do treino" | Valor que o modelo nunca viu | Não é erro: o valor virou "vazio" e o cliente foi pontuado mesmo assim |
| Acentos chegando quebrados pelo PowerShell | O PowerShell 5 não manda UTF-8 por padrão | Use `-Body ([Text.Encoding]::UTF8.GetBytes($c))`, como no exemplo |
| `docker: error during connect` | Docker Desktop fechado | Abra o Docker Desktop e espere ele ficar "running" |
| `OSError: libgomp.so.1` no container | Imagem sem a biblioteca do LightGBM | Já está resolvido no `Dockerfile`; reconstrua com `--build` |
| App sem a aba 2 (`nb04_scores_validacao.parquet` não encontrado) | Notebook 04 antigo | Reexecute o nb04 (item 4.6, só o 04) |

---

## 6. Glossário

| Termo | Significado |
|---|---|
| **Target** | A "resposta certa" que o modelo aprende a prever (aqui: atrasou mais de 5 dias?) |
| **Feature / variável** | Uma informação que o modelo olha (idade, renda...) |
| **Score bruto** | O número que o modelo dá antes da calibração |
| **Calibração** | O ajuste que faz o número virar uma probabilidade confiável |
| **p₁, p₂** | Os dois cortes da política (aprovar / ajuste / rejeitar) |
| **SHAP** | Técnica que divide a nota de um cliente em "quanto cada variável empurrou" |
| **PSI** | Mede o quanto uma variável mudou de perfil entre duas bases (> 0,25 = mudança forte) |
| **API** | Porta de entrada para outros programas usarem o modelo |
| **Endpoint** | Um "endereço" da API (`/saude`, `/score`) |
| **Payload** | Os dados enviados na chamada (a lista de clientes em JSON) |
| **JSON** | Formato de texto para trocar dados entre programas: `{"idade": 31}` |
| **FastAPI** | Biblioteca Python usada para construir a API |
| **Pydantic** | Confere se os dados recebidos têm o tipo certo |
| **uvicorn** | O "servidor" que deixa a API escutando numa porta |
| **Swagger (`/docs`)** | A página automática que documenta e testa a API |
| **422** | Código de erro HTTP para "entendi o pedido, mas os dados estão inválidos" |
| **Streamlit** | Biblioteca que transforma um script Python numa página web interativa |
| **Porta** | O "número da porta" do computador onde cada serviço atende (8000 a API, 8501 o app) |
| **Imagem × container** | A receita pronta × a receita rodando |
| **docker compose** | Sobe vários containers juntos, descritos no `docker-compose.yml` |
| **pickle / joblib** | Formato que salva objetos Python em arquivo. Só abra arquivos de quem você confia |
