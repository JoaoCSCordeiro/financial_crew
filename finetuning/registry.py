"""
MLflow Model Registry do classificador de sentimento.

Cada run de treino (ver run_experiments.py) passa a ser registado como uma
versão do modelo `financial-sentiment`, empacotado como `pyfunc` (adapter
LoRA + código de inferência, com assinatura de input/output). A versão
que vai para produção é a que tem o alias `champion`; a decisão de
promoção está em `mlops_rules.decide_promotion` (patamar absoluto +
margem sobre o campeão atual).

Usamos aliases (`champion`, `challenger`, `previous`) em vez dos stages
antigos (Staging/Production), que o MLflow descontinuou na 2.9. O alias
`previous` guarda o campeão anterior, para `rollback` ser uma operação de
um passo.

O serving continua sem MLflow (ver DECISOES-FINETUNING.md: o container
de serving tem um requirements.txt mínimo): `export` descarrega o adapter
da versão `champion` para `finetuning/models/best/`, de onde o serving o
carrega, e escreve `MODEL_VERSION.json` ao lado para o `/health` dizer
que versão está a servir.

Uso:
    python -m finetuning.registry register-runs   # regista os runs que ainda não têm versão
    python -m finetuning.registry status
    python -m finetuning.registry export
    python -m finetuning.registry rollback
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from finetuning.constants import BASE_MODEL
from finetuning.data import DATASET_ID
from finetuning.mlops_rules import (
    KEEP_CHAMPION,
    PROMOTE,
    PromotionDecision,
    decide_promotion,
)
from finetuning.tracking import (
    BEST_MODEL_DIR,
    EXPERIMENT_NAME,
    SELECTION_METRIC,
    configure_tracking,
)

MODEL_NAME = "financial-sentiment"
MODEL_ARTIFACT_NAME = "sentiment_model"
CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous"
CHAMPION_URI = f"models:/{MODEL_NAME}@{CHAMPION}"
MODEL_VERSION_FILE = "MODEL_VERSION.json"


class SentimentPyfunc(mlflow.pyfunc.PythonModel):
    """
    Wrapper pyfunc: DataFrame com coluna `text` -> DataFrame com `label` e
    `confidence`. Permite carregar qualquer versão com
    `mlflow.pyfunc.load_model("models:/financial-sentiment@champion")`
    sem saber que por baixo é distilbert + LoRA.
    """

    batch_size = 64

    def load_context(self, context) -> None:
        from finetuning.inference import SentimentClassifier

        self.classifier = SentimentClassifier(context.artifacts["adapter"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        texts = model_input["text"].astype(str).tolist()
        results = []
        for start in range(0, len(texts), self.batch_size):
            results.extend(
                self.classifier.predict_batch(texts[start : start + self.batch_size])
            )
        return pd.DataFrame(
            {
                "label": [r["label"] for r in results],
                "confidence": [r["confidence"] for r in results],
            }
        )


def _signature():
    return infer_signature(
        pd.DataFrame({"text": ["Operating profit rose to EUR 13.1 mn."]}),
        pd.DataFrame({"label": ["positive"], "confidence": [0.97]}),
    )


def _version_f1(client: MlflowClient, version) -> float:
    return client.get_run(version.run_id).data.metrics[SELECTION_METRIC]


def _get_alias(client: MlflowClient, alias: str):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def register_run(run_id: str):
    """
    Empacota o adapter de um run de treino como pyfunc, regista-o como nova
    versão de `financial-sentiment` e devolve essa `ModelVersion`.

    O adapter vem dos artefactos do próprio run no MLflow (não da pasta
    local `models/<run_name>/`): o registry tem de depender só do que o
    tracking guardou.
    """
    configure_tracking()
    client = MlflowClient()
    run = client.get_run(run_id)

    with tempfile.TemporaryDirectory() as tmp:
        adapter_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="adapter", dst_path=tmp
        )
        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(
                name=MODEL_ARTIFACT_NAME,
                python_model=SentimentPyfunc(),
                artifacts={"adapter": adapter_dir},
                signature=_signature(),
                pip_requirements=[
                    "torch",
                    "transformers",
                    "peft",
                    "pandas",
                ],
                registered_model_name=MODEL_NAME,
            )

    version = client.get_model_version(MODEL_NAME, info.registered_model_version)
    metrics = run.data.metrics
    tags = {
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "base_model": BASE_MODEL,
        "dataset": DATASET_ID,
        SELECTION_METRIC: f"{metrics[SELECTION_METRIC]:.4f}",
        "eval_accuracy": f"{metrics.get('eval_accuracy', float('nan')):.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version.version, key, value)
    client.update_model_version(
        MODEL_NAME,
        version.version,
        description=(
            f"LoRA sobre {BASE_MODEL}, run {tags['run_name']}, "
            f"{SELECTION_METRIC}={tags[SELECTION_METRIC]} no test split do "
            "Financial PhraseBank."
        ),
    )
    return version


def evaluate_and_promote(version) -> PromotionDecision:
    """
    Aplica o gate de promoção a uma versão registada e move os aliases:
    PROMOTE -> `champion` (o anterior fica em `previous`); KEEP_CHAMPION ->
    `challenger`; REJECT -> sem alias. A decisão fica registada como tag
    da versão.
    """
    client = MlflowClient()
    champion = _get_alias(client, CHAMPION)
    champion_f1 = _version_f1(client, champion) if champion else None
    decision = decide_promotion(_version_f1(client, version), champion_f1)

    if decision.action == PROMOTE:
        if champion is not None:
            client.set_registered_model_alias(MODEL_NAME, PREVIOUS, champion.version)
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version.version)
    elif decision.action == KEEP_CHAMPION:
        client.set_registered_model_alias(MODEL_NAME, CHALLENGER, version.version)

    client.set_model_version_tag(
        MODEL_NAME, version.version, "promotion_decision", decision.action
    )
    client.set_model_version_tag(
        MODEL_NAME, version.version, "promotion_reason", decision.reason
    )
    return decision


def register_unregistered_runs() -> None:
    """Regista (por ordem cronológica) os runs de treino que ainda não têm versão."""
    configure_tracking()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    registered_runs = {
        v.run_id for v in client.search_model_versions(f"name='{MODEL_NAME}'")
    }
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time ASC"]
    )
    for run in runs:
        if run.info.run_id in registered_runs:
            continue
        version = register_run(run.info.run_id)
        decision = evaluate_and_promote(version)
        print(
            f"v{version.version} ({run.data.tags.get('mlflow.runName')}): "
            f"{decision.action} -- {decision.reason}"
        )


def export_champion(dest: Path = BEST_MODEL_DIR) -> dict:
    """
    Descarrega o adapter da versão `champion` para `dest` (a pasta que o
    serving carrega) e escreve `MODEL_VERSION.json` com a proveniência.
    """
    configure_tracking()
    client = MlflowClient()
    champion = _get_alias(client, CHAMPION)
    if champion is None:
        raise RuntimeError(
            f"'{MODEL_NAME}' não tem alias '{CHAMPION}'. "
            "Corre `python -m finetuning.registry register-runs` primeiro."
        )

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=CHAMPION_URI, dst_path=tmp)
        )
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(model_dir / "artifacts" / "adapter", dest)

    provenance = {
        "model_name": MODEL_NAME,
        "version": champion.version,
        "alias": CHAMPION,
        "run_id": champion.run_id,
        "run_name": champion.tags.get("run_name"),
        SELECTION_METRIC: champion.tags.get(SELECTION_METRIC),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / MODEL_VERSION_FILE).write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance


def rollback() -> None:
    """Repõe o campeão anterior (`previous`) como `champion`."""
    configure_tracking()
    client = MlflowClient()
    champion = _get_alias(client, CHAMPION)
    previous = _get_alias(client, PREVIOUS)
    if previous is None:
        raise RuntimeError(f"'{MODEL_NAME}' não tem alias '{PREVIOUS}'.")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, previous.version)
    if champion is not None:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS, champion.version)
    print(f"champion: v{champion.version if champion else '-'} -> v{previous.version}")


def print_status() -> None:
    configure_tracking()
    client = MlflowClient()
    versions = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version),
    )
    # search_model_versions não preenche `aliases`; o mapa alias -> versão
    # só vem completo no RegisteredModel.
    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(MODEL_NAME).aliases.items():
        aliases_by_version.setdefault(str(version), []).append(alias)
    for v in versions:
        aliases = ",".join(sorted(aliases_by_version.get(str(v.version), []))) or "-"
        print(
            f"v{v.version}  {v.tags.get('run_name', ''):<22} "
            f"{SELECTION_METRIC}={v.tags.get(SELECTION_METRIC)}  "
            f"aliases={aliases:<20} decisão={v.tags.get('promotion_decision')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command", choices=["register-runs", "status", "export", "rollback"]
    )
    args = parser.parse_args()
    if args.command == "register-runs":
        register_unregistered_runs()
    elif args.command == "status":
        print_status()
    elif args.command == "export":
        print(json.dumps(export_champion(), indent=2))
    elif args.command == "rollback":
        rollback()


if __name__ == "__main__":
    main()
