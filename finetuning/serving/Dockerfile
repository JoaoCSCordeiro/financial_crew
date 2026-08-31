# Serving do classificador de sentimento fine-tuned -- imagem separada do
# Dockerfile principal do projeto (ver DECISOES-FINETUNING.md e o README
# da raiz: torch/transformers não pertencem à imagem do crew/dashboard).
#
# Contexto de build tem de ser a RAIZ do repositório, não esta pasta,
# porque precisamos de copiar o código de finetuning/ e o modelo treinado
# em finetuning/models/best/ (gerado por `python -m finetuning.run_experiments`,
# tem de existir no disco ANTES de correr este build):
#
#   docker build -f finetuning/serving/Dockerfile -t financial-crew-sentiment .
#   docker run -p 8001:8001 financial-crew-sentiment

FROM python:3.12-slim

WORKDIR /app

COPY finetuning/serving/requirements.txt ./finetuning/serving/requirements.txt
RUN pip install --no-cache-dir -r finetuning/serving/requirements.txt

COPY finetuning/__init__.py finetuning/constants.py finetuning/inference.py ./finetuning/
COPY finetuning/serving/app.py ./finetuning/serving/app.py
COPY finetuning/serving/__init__.py ./finetuning/serving/__init__.py
COPY finetuning/models/best ./finetuning/models/best

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SENTIMENT_MODEL_DIR=/app/finetuning/models/best

EXPOSE 8001

CMD ["uvicorn", "finetuning.serving.app:app", "--host", "0.0.0.0", "--port", "8001"]
