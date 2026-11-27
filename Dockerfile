# Imagem única para a API e o app (o serviço escolhe o comando no docker-compose.yml).
FROM python:3.12-slim
# sem libgomp1 o LightGBM não importa
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /projeto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY models/ models/
COPY dados/ dados/
COPY outputs/ outputs/
COPY submissao/ submissao/
