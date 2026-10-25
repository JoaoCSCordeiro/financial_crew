"""
Monitorização de drift do modelo em produção (versão `champion` do
Model Registry, ver registry.py).

Compara uma janela "current" com uma referência e regista o resultado no
MLflow (experiência `financial-sentiment-monitoring`): uma métrica por
feature, a percentagem de features com drift, o f1 na janela quando há
labels, o relatório HTML do Evidently como artefacto e a recomendação de
retreino (`mlops_rules.recommend_retrain`) como tag do run e da versão do
modelo.

Referência: metade do test split do Financial PhraseBank (dados que o
modelo nunca viu no treino, com a mesma distribuição). Janelas "current":

- `phrasebank`: a outra metade do test split. Controlo: não deve haver
  drift nem queda de f1 (se houver, os limites estão mal calibrados).
- `twitter`: tweets financeiros (zeroshot/twitter-financial-news-sentiment,
  split de validação). Shift real: tickers, URLs e frases curtas em vez de
  frases de relatórios. Tem labels, por isso mede também a queda de f1.
- `log:<ficheiro.jsonl>`: previsões registadas pelo serving
  (`PREDICTION_LOG_PATH`, ver serving/app.py). Sem labels: só drift.

As features (mlops_rules.NUMERICAL_FEATURES/CATEGORICAL_FEATURES) não
precisam de labels: forma do texto, confiança e distribuição das
previsões. Em produção as labels chegam tarde ou nunca, por isso o drift
é o primeiro sinal.

Uso:
    python -m finetuning.drift --current phrasebank
    python -m finetuning.drift --current twitter
    python -m finetuning.drift --current log:finetuning/prediction_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from datasets import load_dataset
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from finetuning.constants import BASE_MODEL, ID2LABEL
from finetuning.data import load_raw_dataset
from finetuning.mlops_rules import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TWITTER_TO_MODEL_LABEL,
    parse_value_drift,
    recommend_retrain,
    text_features,
)
from finetuning.registry import CHAMPION, CHAMPION_URI, MODEL_NAME
from finetuning.tracking import MLRUNS_DIR, SELECTION_METRIC, configure_tracking

MONITORING_EXPERIMENT = "financial-sentiment-monitoring"
TWITTER_DATASET_ID = "zeroshot/twitter-financial-news-sentiment"
SEED = 42


def _phrasebank_halves() -> tuple[pd.DataFrame, pd.DataFrame]:
    test = load_raw_dataset()["test"].to_pandas()[["text", "label"]]
    test = test.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    half = len(test) // 2
    return test.iloc[:half].reset_index(drop=True), test.iloc[half:].reset_index(
        drop=True
    )


def _twitter_window(sample: int) -> pd.DataFrame:
    df = load_dataset(TWITTER_DATASET_ID)["validation"].to_pandas()
    df["label"] = df["label"].map(TWITTER_TO_MODEL_LABEL)
    return df.sample(n=min(sample, len(df)), random_state=SEED).reset_index(drop=True)


def _log_window(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"{path} não tem previsões registadas.")
    return pd.DataFrame(rows)


def build_features(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """
    Junta as features de forma do texto às previsões do campeão. Se a
    janela já traz `predicted_label`/`confidence` (log do serving), usa as
    previsões que foram efetivamente servidas em vez de as recalcular.
    """
    texts = df["text"].astype(str).tolist()
    if "predicted_label" in df and "confidence" in df:
        preds = df[["predicted_label", "confidence"]].reset_index(drop=True)
    else:
        out = model.predict(pd.DataFrame({"text": texts}))
        preds = pd.DataFrame(
            {"predicted_label": out["label"], "confidence": out["confidence"]}
        )
    token_counts = [
        len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]
    ]
    shape = pd.DataFrame(
        [text_features(t, n) for t, n in zip(texts, token_counts, strict=True)]
    )
    features = pd.concat([shape, preds], axis=1)
    if "label" in df:
        features["label"] = df["label"].map(ID2LABEL).reset_index(drop=True)
    return features


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    definition = DataDefinition(
        numerical_columns=NUMERICAL_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )
    cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current[cols], data_definition=definition),
        Dataset.from_pandas(reference[cols], data_definition=definition),
    )
    columns = [
        c for c in (parse_value_drift(m) for m in snapshot.dict()["metrics"]) if c
    ]
    return snapshot, columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--current",
        default="twitter",
        help="phrasebank | twitter | log:<ficheiro.jsonl>",
    )
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()

    configure_tracking()
    client = MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    champion_f1 = client.get_run(champion.run_id).data.metrics[SELECTION_METRIC]
    model = mlflow.pyfunc.load_model(CHAMPION_URI)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    reference_raw, control_raw = _phrasebank_halves()
    if args.current == "phrasebank":
        current_raw = control_raw
    elif args.current == "twitter":
        current_raw = _twitter_window(args.sample)
    elif args.current.startswith("log:"):
        current_raw = _log_window(Path(args.current[4:]))
    else:
        parser.error(f"--current desconhecido: {args.current}")

    reference = build_features(reference_raw, model, tokenizer)
    current = build_features(current_raw, model, tokenizer)
    snapshot, columns = run_drift_report(reference, current)
    drift_share = sum(c.drifted for c in columns) / len(columns)

    current_f1 = f1_drop = current_acc = None
    if "label" in current:
        current_f1 = f1_score(
            current["label"], current["predicted_label"], average="macro"
        )
        current_acc = accuracy_score(current["label"], current["predicted_label"])
        f1_drop = champion_f1 - current_f1
    retrain, reasons = recommend_retrain(drift_share, f1_drop)

    if mlflow.get_experiment_by_name(MONITORING_EXPERIMENT) is None:
        # Mesmo artifact store da experiência de treino (ver tracking.py);
        # sem isto o MLflow criava ./mlartifacts na pasta de onde se corre.
        mlflow.create_experiment(
            MONITORING_EXPERIMENT,
            artifact_location=(MLRUNS_DIR / "artifacts").as_uri(),
        )
    mlflow.set_experiment(MONITORING_EXPERIMENT)
    source = args.current.split(":")[0]
    with mlflow.start_run(run_name=f"drift-{source}-v{champion.version}"):
        mlflow.log_params(
            {
                "model_version": champion.version,
                "current_source": args.current,
                "reference": "financial_phrasebank test (metade A)",
                "n_reference": len(reference),
                "n_current": len(current),
            }
        )
        metrics = {"drift_share": drift_share}
        for c in columns:
            metrics[f"drift_score_{c.column}"] = c.score
        if current_f1 is not None:
            metrics.update(
                current_f1_macro=current_f1,
                current_accuracy=current_acc,
                f1_drop_vs_champion=f1_drop,
            )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "retrain_recommended": str(retrain).lower(),
                "retrain_reasons": "; ".join(reasons) or "-",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / f"drift-{source}.html"
            snapshot.save_html(str(report_path))
            mlflow.log_artifact(str(report_path), artifact_path="evidently")
        run_id = mlflow.active_run().info.run_id

    client.set_model_version_tag(
        MODEL_NAME, champion.version, "last_drift_check", f"{source}: {run_id}"
    )
    client.set_model_version_tag(
        MODEL_NAME, champion.version, "retrain_recommended", str(retrain).lower()
    )

    print(f"\nModelo: {MODEL_NAME} v{champion.version} ({CHAMPION})")
    print(f"Referência: {len(reference)} | current ({args.current}): {len(current)}")
    for c in columns:
        flag = "DRIFT" if c.drifted else "ok"
        print(f"  {c.column:<16} {c.method:<32} {c.score:.4g}  {flag}")
    print(f"drift_share = {drift_share:.0%}")
    if current_f1 is not None:
        print(
            f"f1_macro: campeão {champion_f1:.4f} -> janela {current_f1:.4f} "
            f"(queda {f1_drop:.4f}); accuracy {current_acc:.4f}"
        )
    print(f"Retreino recomendado: {'sim' if retrain else 'não'}")
    for r in reasons:
        print(f"  - {r}")
    print(
        f"Relatório: mlflow ui --backend-store-uri sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
    )


if __name__ == "__main__":
    main()
