"""
Fine-tuning por LoRA de um classificador de sentimento financeiro.

Modelo base: distilbert-base-uncased (ver DECISOES-FINETUNING.md para a
justificação da escolha). Em vez de full fine-tuning (atualizar todos os
~66M parâmetros), usamos LoRA (Low-Rank Adaptation, via `peft`): injeta
pares de matrizes de baixo rank nas projeções de atenção (q_lin, k_lin,
v_lin, out_lin) e treina só essas + a cabeça de classificação
(classifier/pre_classifier, que são novas e têm de ser treinadas na
totalidade -- `modules_to_save`). Isto reduz os parâmetros treináveis em
~95%+ face a full fine-tuning, com a vantagem prática de o adapter LoRA
resultante pesar poucos MB (vs. ~260MB do modelo inteiro), o que interessa
diretamente para o passo de serving/deployment.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from finetuning.constants import BASE_MODEL, ID2LABEL, LABEL2ID
from finetuning.data import load_raw_dataset, tokenize_dataset

MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    epochs: int
    batch_size: int = 16
    weight_decay: float = 0.01
    seed: int = 42


@dataclass
class RunResult:
    config: TrainConfig
    metrics: dict
    adapter_dir: Path
    train_runtime_seconds: float


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_macro,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def run_training(config: TrainConfig) -> RunResult:
    """
    Treina um adapter LoRA sobre distilbert-base-uncased com os
    hiperparâmetros de `config`, avalia no split de teste do Financial
    PhraseBank, e guarda o adapter em `finetuning/models/<run_name>/`.

    Não faz nenhuma chamada a MLflow -- isso é responsabilidade de quem
    chama esta função (ver `run_experiments.py`), para `train.py` poder
    ser testado/reutilizado sem depender de tracking.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    raw = load_raw_dataset()
    tokenized = tokenize_dataset(raw, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        modules_to_save=["pre_classifier", "classifier"],
    )
    model = get_peft_model(model, lora_config)

    output_dir = MODELS_DIR / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=config.seed,
        logging_steps=20,
        report_to=[],
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.monotonic()
    trainer.train()
    train_runtime = time.monotonic() - start

    eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    return RunResult(
        config=config,
        metrics=eval_metrics,
        adapter_dir=adapter_dir,
        train_runtime_seconds=train_runtime,
    )


if __name__ == "__main__":
    demo_config = TrainConfig(
        run_name="manual-smoke-test",
        learning_rate=5e-4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        epochs=1,
    )
    result = run_training(demo_config)
    print("Config:", asdict(result.config))
    print("Métricas:", result.metrics)
    print("Tempo de treino (s):", result.train_runtime_seconds)
    print("Adapter guardado em:", result.adapter_dir)
