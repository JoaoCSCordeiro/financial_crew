"""
Corre 3 experiências de fine-tuning LoRA com hiperparâmetros diferentes,
regista cada uma no MLflow (ver tracking.py) e no Model Registry (ver
registry.py), onde o gate de promoção decide se a nova versão passa a
`champion`. No fim, o adapter do `champion` é exportado para
`finetuning/models/best/`, de onde o serving (`finetuning/serving/app.py`)
o carrega.

As 3 configurações variam learning rate, rank do LoRA (r) e número de
épocas -- os 3 hiperparâmetros que a proposta original pedia para
comparar (ver DECISOES-FINETUNING.md para a lógica de cada escolha).

Uso:
    python -m finetuning.run_experiments
"""

from __future__ import annotations

from dataclasses import asdict

import mlflow

from finetuning.registry import evaluate_and_promote, export_champion, register_run
from finetuning.tracking import configure_tracking, select_best_run
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
        with mlflow.start_run(run_name=config.run_name) as run:
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
        version = register_run(run.info.run_id)
        decision = evaluate_and_promote(version)
        print(f"Registry: v{version.version} -> {decision.action} ({decision.reason})")

    best_run = select_best_run()
    print(
        f"\n=== Melhor run da experiência: {best_run.data.tags.get('mlflow.runName')} "
        f"(eval_f1_macro={best_run.data.metrics.get('eval_f1_macro'):.4f}) ==="
    )
    provenance = export_champion()
    print(
        f"champion v{provenance['version']} ({provenance['run_name']}) exportado "
        "para finetuning/models/best/ (usado pelo serving)."
    )


if __name__ == "__main__":
    main()
