"""
Serving FastAPI do classificador de sentimento fine-tuned (LoRA sobre
distilbert-base-uncased).

Isto corre num container separado do resto do Financial Crew (ver
DECISOES-FINETUNING.md) -- o crew principal fala com este serviço por
HTTP (ver `tools/sentiment_tool.py`), nunca importa `torch`/`transformers`
diretamente, para não obrigar quem só quer correr o crew original a
instalar ~2GB de dependências de deep learning.

Localização do modelo: `SENTIMENT_MODEL_DIR` (default: `finetuning/models/best`,
o output de `python -m finetuning.registry export` -- ver serving/Dockerfile
para onde isto fica dentro do container). O `/health` devolve a versão do
Model Registry que está a servir (`MODEL_VERSION.json`, escrito pelo export).

Log de previsões: se `PREDICTION_LOG_PATH` estiver definido, cada previsão
é acrescentada a esse ficheiro em JSON Lines (texto, label, confiança,
versão do modelo). É a janela "current" que `python -m finetuning.drift
--current log:<ficheiro>` compara com a referência.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finetuning.inference import SentimentClassifier

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "best"
MODEL_DIR = Path(os.getenv("SENTIMENT_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH")

_state: dict = {"classifier": None, "model_version": None}


def _read_model_version(model_dir: Path) -> dict | None:
    """Proveniência escrita por `finetuning.registry export` (None se o adapter veio de outro lado)."""
    path = model_dir / "MODEL_VERSION.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _log_predictions(texts: list[str], results: list[dict]) -> None:
    if not PREDICTION_LOG_PATH:
        return
    version = (_state.get("model_version") or {}).get("version")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(PREDICTION_LOG_PATH, "a", encoding="utf-8") as f:
        for text, r in zip(texts, results, strict=True):
            record = {
                "ts": ts,
                "model_version": version,
                "text": text,
                "predicted_label": r["label"],
                "confidence": r["confidence"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        _state["model_version"] = _read_model_version(MODEL_DIR)
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
        "model_version": _state.get("model_version"),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    classifier = _get_classifier()
    start = time.perf_counter()
    result = classifier.predict(request.text)
    latency_ms = (time.perf_counter() - start) * 1000
    _log_predictions([request.text], [result])
    return ClassifyResponse(**result, latency_ms=latency_ms)


@app.post("/classify/batch", response_model=list[ClassifyResponse])
def classify_batch(request: ClassifyBatchRequest) -> list[ClassifyResponse]:
    classifier = _get_classifier()
    start = time.perf_counter()
    results = classifier.predict_batch(request.texts)
    latency_ms = (time.perf_counter() - start) * 1000
    _log_predictions(request.texts, results)
    return [
        ClassifyResponse(**r, latency_ms=latency_ms / len(results)) for r in results
    ]
