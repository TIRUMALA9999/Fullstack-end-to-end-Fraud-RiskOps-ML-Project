
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from train_baseline import FEATURES, load_partition


REVIEW_CAPACITIES = [0.005, 0.01, 0.02, 0.05]


def evaluate_capacity(labels, scores, fraction):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("Invalid evaluation input")

    if not (0 < fraction <= 1):
        raise ValueError("Invalid review capacity")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid fraud labels")

    if not np.isfinite(scores).all():
        raise ValueError("Invalid prediction scores")

    total = len(labels)
    positives = int(labels.sum())

    if positives == 0 or positives == total:
        raise ValueError("Both classes are required")

    k = max(1, int(np.ceil(total * fraction)))

    # Stable sort gives deterministic behavior at tied scores.
    ranking = np.argsort(-scores, kind="stable")
    selected = ranking[:k]

    tp = int(labels[selected].sum())
    fp = int(k - tp)
    fn = int(positives - tp)
    tn = int(total - positives - fp)

    return {
        "review_fraction": fraction,
        "reviewed": k,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / k,
        "recall": tp / positives,
        "review_cutoff_score": float(scores[ranking[k - 1]]),
    }


def evaluate_models(data_dir, model_dir, output_dir, runs):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    # The existing loader explicitly prohibits loading 'test'.
    validation = load_partition(data_dir, "validation")

    X = validation[FEATURES]
    y = validation["is_fraud"].to_numpy(dtype=int)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": "PaySim",
        "partition": "validation",
        "rows": len(y),
        "fraud_rows": int(y.sum()),
        "models": {},
    }

    capacity_rows = []
    curves = {}
    capture_series = {}

    for name, run_id in runs.items():
        model_path = model_dir / run_id / "model.joblib"

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        print(f"Evaluating {name}...", flush=True)

        # Load only locally generated, trusted model artifacts.
        model = joblib.load(model_path)

        scores = model.predict_proba(X)[:, 1]

        if not np.isfinite(scores).all():
            raise ValueError("Non-finite model predictions")

        ap = float(average_precision_score(y, scores))
        roc_auc = float(roc_auc_score(y, scores))

        precision, recall, _ = precision_recall_curve(
            y, scores
        )

        curves[name] = {
            "precision": precision,
            "recall": recall,
            "average_precision": ap,
        }

        summary["models"][name] = {
            "run_id": run_id,
            "average_precision": ap,
            "roc_auc": roc_auc,
        }

        capture_series[name] = []

        for capacity in REVIEW_CAPACITIES:
            result = evaluate_capacity(y, scores, capacity)

            capacity_rows.append({
                "model": name,
                **result,
            })

            capture_series[name].append(result["recall"])

    # Precision-recall comparison
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, curve in curves.items():
        ax.step(
            curve["recall"],
            curve["precision"],
            where="post",
            label=(
                f"{name} "
                f"(AP={curve['average_precision']:.4f})"
            ),
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Fraud Detection: Precision-Recall")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        output_dir / "precision_recall_comparison.png",
        dpi=150,
    )
    plt.close(fig)

    # Analyst capacity comparison
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, recalls in capture_series.items():
        ax.plot(
            np.array(REVIEW_CAPACITIES) * 100,
            np.array(recalls) * 100,
            marker="o",
            label=name,
        )

    ax.set_xlabel("Transaction Review Capacity (%)")
    ax.set_ylabel("Fraud Recall (%)")
    ax.set_title("Fraud Capture vs Analyst Capacity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        output_dir / "review_capacity_comparison.png",
        dpi=150,
    )
    plt.close(fig)

    capacity_table = pd.DataFrame(capacity_rows)

    capacity_table.to_csv(
        output_dir / "review_capacity_metrics.csv",
        index=False,
    )

    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        capacity_table.to_string(index=False),
        flush=True,
    )

    print(json.dumps(summary, indent=2), flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--lightgbm-run", required=True)

    args = parser.parse_args()

    evaluate_models(
        data_dir=args.input,
        model_dir=args.models,
        output_dir=args.output,
        runs={
            "LogisticRegression": args.baseline_run,
            "LightGBM": args.lightgbm_run,
        },
    )


if __name__ == "__main__":
    main()
