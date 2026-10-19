
import argparse
import hashlib
import json
import tempfile

from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from calibrate_risk_policy import calibrate
from train_baseline import FEATURES


COLUMNS = FEATURES + ["is_fraud", "source_step"]


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_holdout(source):
    """Read only the final test partition."""

    source = Path(source)

    manifest = json.loads(
        (source / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if manifest["feature_columns"] != FEATURES:
        raise ValueError("Feature contract mismatch")

    expected = manifest["splits"]["test"]
    boundary = manifest["validation_end_step"]

    files = sorted(
        (source / "test").glob("part-*.parquet")
    )

    if len(files) != expected["parts"] or not files:
        raise ValueError("Test partition count mismatch")

    frames = []

    for path in files:
        parquet = pq.ParquetFile(
            path,
            pre_buffer=False,
        )

        if not set(COLUMNS).issubset(
            parquet.schema_arrow.names
        ):
            raise ValueError("Missing model input columns")

        for batch in parquet.iter_batches(
            batch_size=10_000,
            columns=COLUMNS,
            use_threads=False,
        ):
            frame = batch.to_pandas()

            if frame.isna().any().any():
                raise ValueError("Missing feature values")

            amount = frame["amount"].to_numpy(
                dtype=float
            )
            log_amount = frame["log_amount"].to_numpy(
                dtype=float
            )

            if (
                not np.isfinite(amount).all()
                or not np.isfinite(log_amount).all()
                or (amount < 0).any()
            ):
                raise ValueError("Invalid numerical features")

            if not np.allclose(
                log_amount,
                np.log1p(amount),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("Feature transformation mismatch")

            if not frame["is_fraud"].isin([0, 1]).all():
                raise ValueError("Invalid fraud labels")

            if not frame["source_step"].gt(boundary).all():
                raise ValueError("Test boundary violation")

            if (
                frame["source_step"]
                > expected["max_step"]
            ).any():
                raise ValueError("Invalid test simulation step")

            frames.append(frame)

    result = pd.concat(frames, ignore_index=True)

    if (
        len(result) != expected["rows"]
        or int(result["is_fraud"].sum())
        != expected["fraud_rows"]
        or int(result["source_step"].min())
        != expected["min_step"]
        or int(result["source_step"].max())
        != expected["max_step"]
    ):
        raise ValueError("Test manifest reconciliation failed")

    if result["is_fraud"].nunique() != 2:
        raise ValueError("Both classes required")

    return result, manifest


def probability_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if (
        len(labels) == 0
        or len(labels) != len(scores)
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
        or len(np.unique(labels)) != 2
    ):
        raise ValueError("Invalid probability inputs")

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
        "mean_predicted_probability": float(
            scores.mean()
        ),
        "observed_fraud_rate": float(
            labels.mean()
        ),
    }


def fixed_policy_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if (
        len(labels) == 0
        or len(labels) != len(scores)
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid fixed-policy inputs")

    flagged = scores >= threshold

    tp = int(np.sum(flagged & (labels == 1)))
    fp = int(np.sum(flagged & (labels == 0)))
    fn = int(np.sum(~flagged & (labels == 1)))
    tn = int(np.sum(~flagged & (labels == 0)))

    return {
        "reviewed": tp + fp,
        "actual_review_fraction": float(flagged.mean()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def reliability_bins(labels, scores, n_bins=10):
    """Equal-count bins for a descriptive reliability diagram."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("Invalid reliability inputs")

    bins = np.array_split(
        np.argsort(scores, kind="stable"),
        min(n_bins, len(labels)),
    )

    return [
        {
            "count": len(indices),
            "mean_prediction": float(
                scores[indices].mean()
            ),
            "observed_rate": float(
                labels[indices].mean()
            ),
        }
        for indices in bins
    ]


def evaluate(
    input_dir,
    model_file,
    calibrator_file,
    policy_file,
    output_dir,
    run_id,
):
    input_dir = Path(input_dir)
    model_file = Path(model_file)
    calibrator_file = Path(calibrator_file)
    policy_file = Path(policy_file)
    output_dir = Path(output_dir)

    # Protect the completed final evaluation from overwriting.
    if output_dir.exists():
        raise FileExistsError(
            "Final holdout report already exists. "
            "Do not repeatedly tune against the test set."
        )

    policy = json.loads(
        policy_file.read_text(encoding="utf-8")
    )

    if (
        policy["model_run_id"] != run_id
        or policy["action"] != "REVIEW_ONLY"
    ):
        raise ValueError("Frozen policy mismatch")

    threshold = float(
        policy["review_score_threshold"]
    )

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid frozen threshold")

    print("Loading final holdout partition...", flush=True)

    holdout, manifest = load_holdout(input_dir)

    if (
        policy["source_sha256"]
        != manifest["source_sha256"]
    ):
        raise ValueError("Source dataset mismatch")

    # Load only trusted, locally generated artifacts.
    model = joblib.load(model_file)
    calibrator = joblib.load(calibrator_file)

    print("Scoring final holdout...", flush=True)

    raw = model.predict_proba(
        holdout[FEATURES]
    )[:, 1]

    calibrated = calibrate(calibrator, raw)
    labels = holdout["is_fraud"].to_numpy(dtype=int)

    result = {
        "evaluation": "FINAL_HOLDOUT_V1",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_run_id": run_id,
        "source_sha256": manifest["source_sha256"],
        "model_sha256": sha256_file(model_file),
        "calibrator_sha256": sha256_file(calibrator_file),
        "policy_sha256": sha256_file(policy_file),
        "test_rows": len(labels),
        "test_fraud": int(labels.sum()),
        "test_fraud_rate": float(labels.mean()),
        "test_min_step": int(
            holdout["source_step"].min()
        ),
        "test_max_step": int(
            holdout["source_step"].max()
        ),
        "frozen_review_threshold": threshold,
        "target_review_fraction": policy[
            "target_review_fraction"
        ],
        "raw": probability_metrics(labels, raw),
        "calibrated": probability_metrics(
            labels, calibrated
        ),
        "fixed_policy": fixed_policy_metrics(
            labels, calibrated, threshold
        ),
        "decision": "EVALUATION_ONLY_NOT_DEPLOYMENT_APPROVAL",
    }

    bins = reliability_bins(labels, calibrated)

    output_dir.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Publish the report only after evaluation succeeds.
    with tempfile.TemporaryDirectory(
        prefix=".holdout-staging-",
        dir=output_dir.parent,
    ) as tmp:
        staging = Path(tmp) / "report"
        staging.mkdir()

        (staging / "final_metrics.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        pd.DataFrame(bins).to_csv(
            staging / "reliability_bins.csv",
            index=False,
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect calibration",
        )

        ax.plot(
            [item["mean_prediction"] for item in bins],
            [item["observed_rate"] for item in bins],
            marker="o",
            label="Calibrated LightGBM",
        )

        ax.set_xlabel("Mean predicted fraud probability")
        ax.set_ylabel("Observed fraud rate")
        ax.set_title("Final Holdout Reliability Diagram")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            staging / "reliability_diagram.png",
            dpi=150,
        )
        plt.close(fig)

        staging.rename(output_dir)

    print(json.dumps(result, indent=2), flush=True)

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)

    parser.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        help="Acknowledge this is the final holdout evaluation.",
    )

    args = parser.parse_args()

    if not args.confirm_final_evaluation:
        parser.error(
            "Explicit final evaluation acknowledgement required"
        )

    evaluate(
        input_dir=args.input,
        model_file=args.model,
        calibrator_file=args.calibrator,
        policy_file=args.policy,
        output_dir=args.output,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
