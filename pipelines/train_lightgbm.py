
import argparse
import json
from pathlib import Path

import joblib
import mlflow

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_baseline import (
    FEATURES,
    SEED,
    evaluate_at_capacity,
    load_partition,
)


def make_lightgbm_model():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
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

    classifier = LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=100,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=2,
        verbosity=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def configure_mlflow(tracking_dir: Path):
    tracking_dir.mkdir(parents=True, exist_ok=True)

    db_path = (tracking_dir / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    artifact_dir = tracking_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_uri = artifact_dir.resolve().as_uri()

    experiment_name = "FraudGuard-PaySim"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_uri,
        )
    elif experiment.artifact_location != artifact_uri:
        raise RuntimeError(
            "MLflow experiment artifact location mismatch"
        )

    mlflow.set_experiment(experiment_name)


def run_training(
    input_dir: Path,
    tracking_dir: Path,
    models_dir: Path,
    negative_rate: float = 0.05,
):
    input_dir = Path(input_dir)
    tracking_dir = Path(tracking_dir)
    models_dir = Path(models_dir)

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

    print("Training LightGBM classifier...", flush=True)

    model = make_lightgbm_model()

    model.fit(
        train[FEATURES],
        train["is_fraud"],
        classifier__sample_weight=train["sample_weight"],
    )

    print("Evaluating complete validation set...", flush=True)

    scores = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    metrics = evaluate_at_capacity(
        validation["is_fraud"].to_numpy(),
        scores,
        review_fraction=0.01,
    )

    configure_mlflow(tracking_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(
        run_name="lightgbm_weighted_v1"
    ) as run:
        mlflow.log_params({
            "algorithm": "LightGBM",
            "source_sha256": manifest["source_sha256"],
            "feature_version": manifest[
                "feature_pipeline_version"
            ],
            "train_end_step": manifest["train_end_step"],
            "validation_end_step": manifest[
                "validation_end_step"
            ],
            "negative_sample_rate": negative_rate,
            "train_sample_rows": len(train),
            "validation_rows": len(validation),
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 100,
            "random_seed": SEED,
        })

        mlflow.log_metrics(metrics)

        output_dir = models_dir / run.info.run_id
        output_dir.mkdir(parents=True, exist_ok=False)

        model_path = output_dir / "model.joblib"
        metrics_path = (
            output_dir / "validation_metrics.json"
        )

        joblib.dump(model, model_path)

        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        summary = {
            "run_id": run.info.run_id,
            "algorithm": "LightGBM",
            "training_sample_rows": len(train),
            "training_sample_fraud_rows": int(
                train["is_fraud"].sum()
            ),
            "validation_rows": len(validation),
            "metrics": metrics,
            "model_path": str(model_path),
        }

    print(json.dumps(summary, indent=2), flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="FraudGuard LightGBM training"
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
