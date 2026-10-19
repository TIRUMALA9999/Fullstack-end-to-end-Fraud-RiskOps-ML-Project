import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Limit PyArrow parallelism to reduce memory pressure
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
FEATURES = ["amount", "log_amount", "transaction_type"]
LABEL = "is_fraud"
READ_COLUMNS = FEATURES + [LABEL, "source_step"]


def load_partition(
    root: Path,
    split: str,
    negative_rate: float = 0.05,
    seed: int = SEED,
):
    """Read train or validation data using bounded Parquet batches."""

    if split not in ("train", "validation"):
        raise ValueError(
            "Only train and validation are permitted here"
        )

    if not (0 < negative_rate <= 1):
        raise ValueError("Invalid negative sampling rate")

    manifest = json.loads(
        (root / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if manifest["feature_columns"] != FEATURES:
        raise ValueError("Feature contract mismatch")

    expected = manifest["splits"][split]
    files = sorted((root / split).glob("part-*.parquet"))

    if len(files) != expected["parts"] or not files:
        raise ValueError("Partition file count mismatch")

    rng = np.random.default_rng(seed)

    pieces = []
    rows_seen = 0
    fraud_seen = 0

    train_end = manifest["train_end_step"]
    validation_end = manifest["validation_end_step"]

    for path in files:
        parquet = pq.ParquetFile(
            path,
            pre_buffer=False,
            memory_map=False,
        )

        if not set(READ_COLUMNS).issubset(
            parquet.schema_arrow.names
        ):
            raise ValueError("Required columns missing")

        for batch in parquet.iter_batches(
            batch_size=10_000,
            columns=READ_COLUMNS,
            use_threads=False,
        ):
            frame = batch.to_pandas()

            if frame[READ_COLUMNS].isna().any().any():
                raise ValueError("Missing values found")

            if not frame[LABEL].isin([0, 1]).all():
                raise ValueError("Invalid fraud labels")

            amounts = frame["amount"].to_numpy(
                dtype="float64"
            )

            log_amounts = frame["log_amount"].to_numpy(
                dtype="float64"
            )

            if (
                not np.isfinite(amounts).all()
                or not np.isfinite(log_amounts).all()
                or (amounts < 0).any()
            ):
                raise ValueError("Invalid numerical features")

            steps = frame["source_step"]

            if split == "train":
                valid_steps = steps.between(1, train_end)
            else:
                valid_steps = steps.between(
                    train_end + 1,
                    validation_end,
                )

            if not valid_steps.all():
                raise ValueError("Chronological leakage detected")

            labels = frame[LABEL].to_numpy()

            rows_seen += len(frame)
            fraud_seen += int(labels.sum())

            if split == "train":
                # Keep every positive and sample negatives.
                keep = (
                    (labels == 1)
                    | (
                        (labels == 0)
                        & (rng.random(len(labels)) < negative_rate)
                    )
                )

                selected = frame.loc[
                    keep, FEATURES + [LABEL]
                ].copy()

                # Correct for probability-based negative sampling.
                selected["sample_weight"] = np.where(
                    selected[LABEL].to_numpy() == 1,
                    1.0,
                    1.0 / negative_rate,
                )
            else:
                # Never downsample validation.
                selected = frame.loc[
                    :, FEATURES + [LABEL]
                ].copy()

            if not selected.empty:
                pieces.append(selected)

    if (
        rows_seen != expected["rows"]
        or fraud_seen != expected["fraud_rows"]
    ):
        raise ValueError("Partition manifest mismatch")

    result = pd.concat(pieces, ignore_index=True)

    if result.empty or result[LABEL].nunique() != 2:
        raise ValueError("Partition requires both classes")

    return result


def make_model():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                ["amount", "log_amount"],
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["transaction_type"],
            ),
        ],
        sparse_threshold=0.0,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                random_state=SEED,
            ),
        ),
    ])


def evaluate_at_capacity(
    labels,
    scores,
    review_fraction=0.01,
):
    """Rank all validation transactions by risk score."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Invalid evaluation arrays")

    if not (0 < review_fraction <= 1):
        raise ValueError("Invalid review capacity")

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model scores")

    positives = int(labels.sum())

    if positives == 0 or positives == len(labels):
        raise ValueError("Both classes required")

    k = max(1, int(np.ceil(len(labels) * review_fraction)))

    ranking = np.argsort(
        -scores,
        kind="stable",
    )[:k]

    captured = int(labels[ranking].sum())

    return {
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(
            roc_auc_score(labels, scores)
        ),
        "brier_score": float(
            brier_score_loss(labels, scores)
        ),
        "validation_fraud_rate": float(
            labels.mean()
        ),
        "review_fraction": float(review_fraction),
        "reviewed_transactions": int(k),
        "fraud_captured_at_capacity": int(captured),
        "precision_at_capacity": float(captured / k),
        "recall_at_capacity": float(captured / positives),
    }


def run_training(
    input_dir: Path,
    tracking_dir: Path,
    models_dir: Path,
    negative_rate: float,
):
    manifest = json.loads(
        (input_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    print("Loading weighted training sample...", flush=True)

    train = load_partition(
        input_dir,
        "train",
        negative_rate=negative_rate,
    )

    print(
        f"Selected {len(train):,} training records",
        flush=True,
    )

    print("Loading complete validation set...", flush=True)

    validation = load_partition(
        input_dir,
        "validation",
    )

    model = make_model()

    model.fit(
        train[FEATURES],
        train[LABEL],
        classifier__sample_weight=train["sample_weight"],
    )

    scores = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    metrics = evaluate_at_capacity(
        validation[LABEL].to_numpy(),
        scores,
        review_fraction=0.01,
    )

    # -----------------------------------------------
    # MLflow database-backed tracking configuration
    # -----------------------------------------------

    tracking_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Persist tracking metadata in SQLite.
    db_path = (tracking_dir / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    # Persist model artifacts in the Docker-mounted data folder.
    artifact_dir = tracking_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_uri = artifact_dir.resolve().as_uri()

    experiment_name = "FraudGuard-PaySim"

    existing_experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if existing_experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_uri,
        )
    elif existing_experiment.artifact_location != artifact_uri:
        raise RuntimeError(
            "Existing MLflow experiment uses a different "
            "artifact location. Verify its configuration."
        )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name="logreg_weighted_baseline_v1"
    ) as run:

        mlflow.log_params({
            "algorithm": "LogisticRegression",
            "feature_version": manifest[
                "feature_pipeline_version"
            ],
            "source_sha256": manifest["source_sha256"],
            "train_end_step": manifest["train_end_step"],
            "validation_end_step": manifest[
                "validation_end_step"
            ],
            "negative_sample_rate": negative_rate,
            "train_sample_rows": len(train),
            "validation_rows": len(validation),
            "random_seed": SEED,
        })

        mlflow.log_metrics(metrics)

        output = models_dir / run.info.run_id
        output.mkdir(parents=True, exist_ok=False)

        model_path = output / "model.joblib"
        metrics_path = output / "validation_metrics.json"

        joblib.dump(model, model_path)

        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(model_path), "model")
        mlflow.log_artifact(str(metrics_path), "evaluation")

        summary = {
            "run_id": run.info.run_id,
            "model": "LogisticRegression",
            "training_sample_rows": len(train),
            "training_sample_fraud_rows": int(
                train[LABEL].sum()
            ),
            "validation_rows": len(validation),
            "metrics": metrics,
            "model_path": str(model_path),
        }

    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="FraudGuard Logistic Regression baseline"
    )

    parser.add_argument(
        "--input", type=Path, required=True
    )
    parser.add_argument(
        "--tracking", type=Path, required=True
    )
    parser.add_argument(
        "--models", type=Path, required=True
    )
    parser.add_argument(
        "--negative-rate",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    run_training(
        input_dir=args.input,
        tracking_dir=args.tracking,
        models_dir=args.models,
        negative_rate=args.negative_rate,
    )


if __name__ == "__main__":
    main()