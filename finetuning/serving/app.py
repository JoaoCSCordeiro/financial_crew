"""
Serving FastAPI do classificador de sentimento fine-tuned (LoRA sobre
distilbert-base-uncased).

Isto corre num container separado do resto do Financial Crew (ver
DECISOES-FINETUNING.md) -- o crew principal fala com este serviço por
HTTP (ver `tools/sentiment_tool.py`), nunca importa `torch`/`transformers`
diretamente, para não obrigar quem só quer correr o crew original a
instalar ~2GB de dependências de deep learning.

Localização do modelo: `SENTIMENT_MODEL_DIR` (default: `finetuning/models/best`,
o output de `python -m finetuning.run_experiments` -- ver serving/Dockerfile
para onde isto fica dentro do container).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finetuning.inference import SentimentClassifier

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "best"
MODEL_DIR = Path(os.getenv("SENTIMENT_MODEL_DIR", str(DEFAULT_MODEL_DIR)))

_state: dict = {"classifier": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_DIR.exists():
        logger.error(
            "SENTIMENT_MODEL_DIR (%s) não existe -- corre "
            "`python -m finetuning.run_experiments` primeiro, ou monta o volume "
            "certo no container.",
            MODEL_DIR,
        )
    else:
        logger.info("A carregar o classificador de %s...", MODEL_DIR)
        _state["classifier"] = SentimentClassifier(MODEL_DIR)
        logger.info("Classificador carregado.")
    yield
    _state.clear()


app = FastAPI(
    title="Financial Crew - Sentiment Classifier",
    description="Serving do modelo fine-tuned (LoRA) para classificação de sentimento financeiro.",
    version="1.0.0",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class ClassifyBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


def _get_classifier() -> SentimentClassifier:
    classifier = _state.get("classifier")
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo não carregado (SENTIMENT_MODEL_DIR={MODEL_DIR} não encontrado).",
        )
    return classifier


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _state.get("classifier") is not None else "model_not_loaded",
        "model_dir": str(MODEL_DIR),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    classifier = _get_classifier()
    start = time.perf_counter()
    result = classifier.predict(request.text)
    latency_ms = (time.perf_counter() - start) * 1000
    return ClassifyResponse(**result, latency_ms=latency_ms)


@app.post("/classify/batch", response_model=list[ClassifyResponse])
def classify_batch(request: ClassifyBatchRequest) -> list[ClassifyResponse]:
    classifier = _get_classifier()
    start = time.perf_counter()
    results = classifier.predict_batch(request.texts)
    latency_ms = (time.perf_counter() - start) * 1000
    return [
        ClassifyResponse(**r, latency_ms=latency_ms / len(results)) for r in results
    ]
