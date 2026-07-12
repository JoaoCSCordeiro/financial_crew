FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias primeiro (camada cacheada separadamente do codigo,
# para builds mais rapidos quando so o codigo muda).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Diretorios de dados persistentes (montados como volumes via docker-compose)
RUN mkdir -p /app/output /app/metrics /app/cache

EXPOSE 5000

# Por defeito arranca o dashboard. Para correr o CLI batch (main.py) em
# vez disso, sobrepoe o comando: `docker compose run cli`.
CMD ["python", "frontend/app.py"]
