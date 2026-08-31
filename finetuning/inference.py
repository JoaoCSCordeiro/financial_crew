"""
Wrapper de inferência para o adapter LoRA treinado.

Usado tanto pelo script de comparação (`compare.py`) como pelo serving
FastAPI (`finetuning/serving/app.py`) -- mantém a lógica de carregar o
modelo base + adapter e classificar texto num único sítio.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetuning.constants import BASE_MODEL, ID2LABEL, LABEL2ID


class SentimentClassifier:
    """Carrega o modelo base + adapter LoRA uma vez e classifica texto."""

    def __init__(self, adapter_dir: str | Path, device: str | None = None):
        adapter_dir = str(adapter_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        """
        Classifica um único texto.

        Returns:
            {"label": "positive"|"negative"|"neutral", "confidence": float,
             "probabilities": {"negative": ..., "neutral": ..., "positive": ...}}
        """
        inputs = self.tokenizer(
            text, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        pred_id = int(torch.argmax(logits).item())
        return {
            "label": ID2LABEL[pred_id],
            "confidence": probs[pred_id],
            "probabilities": {ID2LABEL[i]: p for i, p in enumerate(probs)},
        }

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classifica vários textos num único forward pass (usado pelo serving)."""
        inputs = self.tokenizer(
            texts, truncation=True, max_length=128, padding=True, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).tolist()
        results = []
        for row in probs:
            pred_id = max(range(len(row)), key=lambda i: row[i])
            results.append(
                {
                    "label": ID2LABEL[pred_id],
                    "confidence": row[pred_id],
                    "probabilities": {ID2LABEL[i]: p for i, p in enumerate(row)},
                }
            )
        return results
