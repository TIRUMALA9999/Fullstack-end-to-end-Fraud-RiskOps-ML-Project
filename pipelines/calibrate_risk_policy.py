import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from train_baseline import FEATURES, load_partition
from train_lightgbm import configure_mlflow


def chronological_windows(
    frame,
    train_end,
    validation_end,
    calibration_end,
    tuning_end,
):
    if not (
        train_end
        < calibration_end
        < tuning_end
        < validation_end
    ):
        raise ValueError("Invalid chronological boundaries")

    steps = frame["source_step"]

    if not steps.between(
        train_end + 1, validation_end
    ).all():
        raise ValueError("Out-of-range validation steps")

    masks = {
        "calibration": steps <= calibration_end,
        "tuning": (
            (steps > calibration_end)
            & (steps <= tuning_end)
        ),
        "audit": steps > tuning_end,
    }

    windows = {
        name: frame.loc[mask].copy()
        for name, mask in masks.items()
    }

    for name, part in windows.items():
        if part.empty or part["is_fraud"].nunique() != 2:
            raise ValueError(
                f"{name} requires both classes"
            )

    return windows


def logit_scores(scores):
    values = np.asarray(scores, dtype=float)

    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("Invalid prediction scores")

    values = np.clip(values, 1e-7, 1 - 1e-7)

    return np.log(
        values / (1 - values)
    ).reshape(-1, 1)


def fit_calibrator(raw_scores, labels):
    labels = np.asarray(labels, dtype=int)

    if (
        len(labels) != len(raw_scores)
        or set(np.unique(labels)) != {0, 1}
    ):
        raise ValueError(
            "Calibration requires both classes"
        )

    # Sigmoid / Platt-style calibration.
    calibrator = LogisticRegression(
        C=100.0,
        max_iter=1000,
        random_state=42,
    )

    calibrator.fit(
        logit_scores(raw_scores),
        labels,
    )

    if calibrator.coef_[0, 0] <= 0:
        raise ValueError(
            "Calibration reversed score ranking"
        )

    return calibrator


def calibrate(calibrator, raw_scores):
    return calibrator.predict_proba(
        logit_scores(raw_scores)
    )[:, 1]


def select_threshold(scores, fraction=0.01):
    values = np.asarray(scores, dtype=float)

    if (
        len(values) == 0
        or not 0 < fraction <= 1
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid threshold inputs")

    k = max(
        1, int(np.ceil(len(values) * fraction))
    )

    threshold = float(
        np.partition(values, len(values) - k)[
            len(values) - k
        ]
    )

    return threshold, k


def policy_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if (
        len(labels) == 0
        or len(labels) != len(scores)
        or set(np.unique(labels)) != {0, 1}
    ):
        raise ValueError("Invalid evaluation labels")

    if (
        not np.isfinite(scores).all()
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid scores or threshold")

    flagged = scores >= threshold

    tp = int(np.count_nonzero(
        flagged & (labels == 1)
    ))
    fp = int(np.count_nonzero(
        flagged & (labels == 0)
    ))
    fn = int(np.count_nonzero(
        ~flagged & (labels == 1)
    ))
    tn = int(np.count_nonzero(
        ~flagged & (labels == 0)
    ))

    return {
        "rows": int(len(labels)),
        "fraud_rows": int(labels.sum()),
        "reviewed": int(flagged.sum()),
        "realized_review_fraction": float(
            flagged.mean()
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": float(
            tp / max(tp + fp, 1)
        ),
        "recall": float(tp / (tp + fn)),
    }


def probability_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

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
        "log_loss": float(
            log_loss(labels, scores, labels=[0, 1])
        ),
        "observed_fraud_rate": float(
            labels.mean()
        ),
        "mean_predicted_probability": float(
            scores.mean()
        ),
    }


def execute(
    input_dir,
    models_dir,
    run_id,
    tracking_dir,
    output_dir,
    report_dir,
    calibration_end=564,
    tuning_end=594,
    review_fraction=0.01,
):
    input_dir = Path(input_dir)
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)

    manifest = json.loads(
        (input_dir / "manifest.json").read_text()
    )

    print("Loading validation data...", flush=True)

    validation = load_partition(
        input_dir,
        "validation",
        include_source_step=True,
    )

    windows = chronological_windows(
        validation,
        manifest["train_end_step"],
        manifest["validation_end_step"],
        calibration_end,
        tuning_end,
    )

    model_file = models_dir / run_id / "model.joblib"

    if not model_file.is_file():
        raise FileNotFoundError(model_file)

    # Only load the model created by our trusted pipeline.
    model = joblib.load(model_file)

    raw_scores = {
        name: model.predict_proba(
            part[FEATURES]
        )[:, 1]
        for name, part in windows.items()
    }

    print("Fitting probability calibrator...", flush=True)

    calibrator = fit_calibrator(
        raw_scores["calibration"],
        windows["calibration"]["is_fraud"],
    )

    calibrated_scores = {
        name: calibrate(calibrator, scores)
        for name, scores in raw_scores.items()
    }

    print("Selecting review threshold...", flush=True)

    threshold, target_count = select_threshold(
        calibrated_scores["tuning"],
        review_fraction,
    )

    audit_labels = windows["audit"]["is_fraud"]

    audit_policy = policy_metrics(
        audit_labels,
        calibrated_scores["audit"],
        threshold,
    )

    result = {
        "model_run_id": run_id,
        "source_sha256": manifest["source_sha256"],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "boundaries": {
            "calibration_end": calibration_end,
            "tuning_end": tuning_end,
            "validation_end": manifest[
                "validation_end_step"
            ],
        },
        "window_counts": {
            name: {
                "rows": len(part),
                "fraud": int(part["is_fraud"].sum()),
            }
            for name, part in windows.items()
        },
        "target_review_fraction": review_fraction,
        "tuning_target_count": target_count,
        "tuning_actual_review_count": int(
            np.count_nonzero(
                calibrated_scores["tuning"] >= threshold
            )
        ),
        "fixed_review_threshold": threshold,
        "audit_raw": probability_metrics(
            audit_labels, raw_scores["audit"]
        ),
        "audit_calibrated": probability_metrics(
            audit_labels, calibrated_scores["audit"]
        ),
        "audit_policy": audit_policy,
        "action": "REVIEW_ONLY_SIMULATION",
    }

    output_dir.mkdir(
        parents=True, exist_ok=True
    )
    report_dir.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(
        calibrator,
        output_dir / "calibrator.joblib",
    )

    (output_dir / "review_policy.json").write_text(
        json.dumps({
            "model_run_id": run_id,
            "source_sha256": manifest["source_sha256"],
            "review_score_threshold": threshold,
            "target_review_fraction": review_fraction,
            "action": "REVIEW_ONLY",
        }, indent=2),
        encoding="utf-8",
    )

    report_file = (
        report_dir / "calibration_report.json"
    )

    report_file.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    # Log calibration results to existing MLflow tracking.
    configure_mlflow(Path(tracking_dir))

    with mlflow.start_run(
        run_name="lightgbm_calibration_policy_v1"
    ):
        mlflow.log_params({
            "base_model_run_id": run_id,
            "calibration_end_step": calibration_end,
            "policy_tune_end_step": tuning_end,
            "target_review_fraction": review_fraction,
        })

        mlflow.log_metrics({
            "audit_brier_raw": result[
                "audit_raw"
            ]["brier_score"],
            "audit_brier_calibrated": result[
                "audit_calibrated"
            ]["brier_score"],
            "audit_log_loss_raw": result[
                "audit_raw"
            ]["log_loss"],
            "audit_log_loss_calibrated": result[
                "audit_calibrated"
            ]["log_loss"],
            "audit_policy_precision": audit_policy[
                "precision"
            ],
            "audit_policy_recall": audit_policy[
                "recall"
            ],
            "audit_policy_review_fraction": audit_policy[
                "realized_review_fraction"
            ],
        })

        mlflow.log_artifact(
            str(output_dir / "calibrator.joblib"),
            artifact_path="calibration",
        )
        mlflow.log_artifact(
            str(output_dir / "review_policy.json"),
            artifact_path="policy",
        )
        mlflow.log_artifact(
            str(report_file),
            artifact_path="evaluation",
        )

    print(json.dumps(result, indent=2), flush=True)

    return result


def main():
    parser = argparse.ArgumentParser()

    for argument in (
        "input",
        "models",
        "run-id",
        "tracking",
        "output",
        "report",
    ):
        parser.add_argument(
            "--" + argument,
            required=True,
        )

    parser.add_argument(
        "--calibration-end", type=int, default=564
    )
    parser.add_argument(
        "--tuning-end", type=int, default=594
    )
    parser.add_argument(
        "--review-fraction", type=float, default=0.01
    )

    args = parser.parse_args()

    execute(
        input_dir=args.input,
        models_dir=args.models,
        run_id=args.run_id,
        tracking_dir=args.tracking,
        output_dir=args.output,
        report_dir=args.report,
        calibration_end=args.calibration_end,
        tuning_end=args.tuning_end,
        review_fraction=args.review_fraction,
    )


if __name__ == "__main__":
    main()