"""
Tracking de experiências de fine-tuning com MLflow (backend local, sem servidor).

Cada corrida de `train.py` (hiperparâmetros diferentes -- ver
`run_experiments.py`) fica registada como um MLflow run: hiperparâmetros,
métricas de avaliação e o adapter LoRA treinado como artefacto. No fim,
`select_best_run` decide qual dos runs registados é o melhor (por
`eval_f1_macro`, mais robusto que accuracy dado o desequilíbrio de classes
do Financial PhraseBank -- ver DECISOES-FINETUNING.md) e devolve o
`run_id` correspondente, para `run_experiments.py` exportar esse adapter
como o modelo "best" que o serving (`finetuning/serving/app.py`) carrega.

O tracking store é um diretório local (`finetuning/mlruns/`, fora do git
-- ver .gitignore) em vez de um servidor MLflow remoto: suficiente para
este projeto correr localmente sem infraestrutura extra, e continua a dar
a mesma UI (`mlflow ui --backend-store-uri finetuning/mlruns`) e a mesma
API de comparação de runs.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

EXPERIMENT_NAME = "financial-sentiment-lora"
FINETUNING_DIR = Path(__file__).resolve().parent
MLRUNS_DIR = FINETUNING_DIR / "mlruns"
MODELS_DIR = FINETUNING_DIR / "models"
BEST_MODEL_DIR = MODELS_DIR / "best"
SELECTION_METRIC = "eval_f1_macro"


def configure_tracking() -> None:
    """
    Aponta o MLflow para o tracking store local do projeto e garante a
    experiência.

    Usamos um backend SQLite (`mlflow.db`) em vez do antigo file store
    baseado em pastas: a partir do MLflow 3.x o file store está em modo de
    manutenção (sem novas features, ver aviso de deprecação da própria
    lib) e passar `MLFLOW_ALLOW_FILE_STORE=true` só adiava o problema.
    SQLite continua 100% local (um único ficheiro, sem servidor), só muda
    o formato de armazenamento dos runs -- os artefactos (adapters LoRA)
    continuam em ficheiros normais dentro de `mlruns/artifacts/`.
    """
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    (MLRUNS_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
    db_path = MLRUNS_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=(MLRUNS_DIR / "artifacts").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def select_best_run(metric: str = SELECTION_METRIC):
    """
    Devolve o `mlflow.entities.Run` com o melhor valor da métrica indicada,
    entre todos os runs já registados na experiência.

    Raises:
        RuntimeError: se não houver nenhum run registado ainda.
    """
    configure_tracking()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"Nenhum run encontrado na experiência '{EXPERIMENT_NAME}'. "
            "Corre `python -m finetuning.run_experiments` primeiro."
        )
    best_run_id = runs.iloc[0]["run_id"]
    return mlflow.get_run(best_run_id)


def summarize_runs() -> list[dict]:
    """Devolve params+métricas de todos os runs registados, para relatório/print."""
    configure_tracking()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{SELECTION_METRIC} DESC"],
    )
    return runs.to_dict(orient="records")


if __name__ == "__main__":
    for row in summarize_runs():
        print(
            row.get("tags.mlflow.runName"), "->", row.get(f"metrics.{SELECTION_METRIC}")
        )
