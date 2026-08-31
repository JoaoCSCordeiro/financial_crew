"""
Corre 3 experiências de fine-tuning LoRA com hiperparâmetros diferentes,
regista cada uma no MLflow (ver tracking.py) e seleciona a melhor por
`eval_f1_macro`. O adapter da melhor corrida é copiado para
`finetuning/models/best/`, de onde o serving (`finetuning/serving/app.py`)
o carrega.

As 3 configurações variam learning rate, rank do LoRA (r) e número de
épocas -- os 3 hiperparâmetros que a proposta original pedia para
comparar (ver DECISOES-FINETUNING.md para a lógica de cada escolha).

Uso:
    python -m finetuning.run_experiments
"""

from __future__ import annotations

import shutil
from dataclasses import asdict

import mlflow

from finetuning.tracking import BEST_MODEL_DIR, configure_tracking, select_best_run
from finetuning.train import TrainConfig, run_training

EXPERIMENT_CONFIGS = [
    TrainConfig(
        run_name="run-a_lr1e-3_r4_ep3",
        learning_rate=1e-3,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        epochs=3,
    ),
    TrainConfig(
        run_name="run-b_lr5e-4_r8_ep5",
        learning_rate=5e-4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        epochs=5,
    ),
    TrainConfig(
        run_name="run-c_lr2e-4_r16_ep5",
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        epochs=5,
    ),
]


def _flatten_metrics(metrics: dict) -> dict:
    """MLflow só aceita métricas numéricas -- remove entradas não numéricas."""
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def main() -> None:
    configure_tracking()

    for config in EXPERIMENT_CONFIGS:
        print(f"\n=== A treinar {config.run_name} ===")
        with mlflow.start_run(run_name=config.run_name):
            mlflow.log_params(asdict(config))
            result = run_training(config)
            mlflow.log_metrics(_flatten_metrics(result.metrics))
            mlflow.log_metric("train_runtime_seconds", result.train_runtime_seconds)
            mlflow.log_artifacts(str(result.adapter_dir), artifact_path="adapter")
            print(
                f"eval_f1_macro={result.metrics.get('eval_f1_macro'):.4f} "
                f"eval_accuracy={result.metrics.get('eval_accuracy'):.4f} "
                f"train_runtime={result.train_runtime_seconds:.1f}s"
            )

    best_run = select_best_run()
    best_run_name = best_run.data.tags.get("mlflow.runName")
    print(
        f"\n=== Melhor run: {best_run_name} (eval_f1_macro="
        f"{best_run.data.metrics.get('eval_f1_macro'):.4f}) ==="
    )

    best_adapter_source = None
    for config in EXPERIMENT_CONFIGS:
        if config.run_name == best_run_name:
            from finetuning.train import MODELS_DIR

            best_adapter_source = MODELS_DIR / config.run_name / "adapter"
            break

    if best_adapter_source is None or not best_adapter_source.exists():
        raise RuntimeError(
            f"Não encontrei o adapter local para o melhor run '{best_run_name}' "
            f"em {best_adapter_source}. Os adapters ficam guardados localmente em "
            "finetuning/models/<run_name>/adapter/ durante o treino."
        )

    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)
    shutil.copytree(best_adapter_source, BEST_MODEL_DIR)
    print(f"Adapter copiado para {BEST_MODEL_DIR} (usado pelo serving).")


if __name__ == "__main__":
    main()
