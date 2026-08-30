"""
Ferramenta de Sentimento Fine-Tuned — alternativa ao julgamento do LLM.

Só é usada pelo Agente 1 (News Analyst) quando SENTIMENT_PROVIDER=finetuned
(ver agents.py/tasks.py) -- caso contrário, o comportamento original do
projeto (o próprio LLM classifica cada título) fica inalterado.

Chama, via HTTP, o serviço FastAPI de finetuning/serving/app.py (modelo
LoRA sobre distilbert-base-uncased -- ver finetuning/RESULTS.md para a
comparação medida de accuracy/latência/custo contra o LLM genérico). Esta
ferramenta NÃO importa `torch`/`transformers` -- é só um cliente HTTP fino,
para não obrigar o crew principal a instalar as dependências pesadas do
módulo de fine-tuning (ver DECISOES-FINETUNING.md).

A contagem P/N e a regra de decisão do sentimento geral (positivo/negativo/
neutro/misto) são feitas aqui em Python, deterministicamente -- exatamente
a mesma regra que a task do Agente 1 já pede ao LLM para aplicar
"literalmente" quando SENTIMENT_PROVIDER=llm (ver tasks.py). Com o
classificador fine-tuned, a contagem deixa de depender do LLM aplicar bem
a regra e passa a ser aritmética simples sobre labels já determinísticos --
consistente com a arquitetura neuro-simbólica do projeto: ferramentas
determinísticas para factos, LLM só para validar/formatar.
"""

import logging
import os
from typing import Any, Dict, List

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tools.exceptions import SentimentServiceError

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_URL = "http://localhost:8001"


def _service_url() -> str:
    return os.getenv("SENTIMENT_SERVICE_URL", DEFAULT_SERVICE_URL).rstrip("/")


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_classify_batch(titles: List[str]) -> List[Dict[str, Any]]:
    response = requests.post(
        f"{_service_url()}/classify/batch",
        json={"texts": titles},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def classify_titles(titles: List[str]) -> Dict[str, Any]:
    """
    Classifica uma lista de títulos de notícias com o modelo fine-tuned e
    aplica a mesma regra de decisão P vs N usada em tasks.py.

    Args:
        titles: títulos de notícias a classificar (a lista devolvida por
            `obter_noticias_recentes`, ver tools/news_tool.py).

    Returns:
        Dicionário com a classificação individual de cada título, a
        contagem (positivos/negativos/neutros) e o sentimento geral
        resultante da regra de decisão.

    Raises:
        SentimentServiceError: se o serviço de classificação estiver
            inacessível, exceder o timeout, ou devolver um erro HTTP,
            mesmo após retries.
    """
    if not titles:
        return {
            "classificacoes": [],
            "positivos": 0,
            "negativos": 0,
            "neutros": 0,
            "sentimento_geral": None,
        }

    try:
        results = _call_classify_batch(titles)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise SentimentServiceError(
            f"Serviço de sentimento fine-tuned inacessível em {_service_url()} "
            f"após múltiplas tentativas: {exc}. Confirma que o container de "
            "serving está a correr (ver finetuning/serving/README.md) ou usa "
            "SENTIMENT_PROVIDER=llm."
        ) from exc
    except requests.HTTPError as exc:
        raise SentimentServiceError(
            f"Serviço de sentimento fine-tuned devolveu um erro: {exc}"
        ) from exc

    classificacoes = [
        {
            "titulo": title,
            "sentimento": result["label"],
            "confianca": result["confidence"],
        }
        for title, result in zip(titles, results)
    ]

    positivos = sum(1 for c in classificacoes if c["sentimento"] == "positive")
    negativos = sum(1 for c in classificacoes if c["sentimento"] == "negative")
    neutros = sum(1 for c in classificacoes if c["sentimento"] == "neutral")

    if positivos > negativos:
        sentimento_geral = "positivo"
    elif negativos > positivos:
        sentimento_geral = "negativo"
    elif positivos == negativos and positivos >= 1:
        sentimento_geral = "misto"
    else:
        sentimento_geral = "neutro"

    return {
        "classificacoes": classificacoes,
        "positivos": positivos,
        "negativos": negativos,
        "neutros": neutros,
        "sentimento_geral": sentimento_geral,
    }
