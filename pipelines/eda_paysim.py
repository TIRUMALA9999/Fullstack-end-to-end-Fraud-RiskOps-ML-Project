
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REQUIRED = {
    "step", "type", "amount",
    "isFraud", "simulation_day",
}

FORBIDDEN = {
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud",
}

BAND_EDGES = [
    0, 100, 500, 1000, 5000,
    10000, 100000, float("inf"),
]

BAND_LABELS = [
    "0-100", "100-500", "500-1k",
    "1k-5k", "5k-10k", "10k-100k", "100k+",
]


def empty_stats():
    return {
        "transactions": 0,
        "fraud": 0,
        "amount_sum": 0.0,
    }


def add_group_stats(target, grouped):
    for key, row in grouped.iterrows():
        item = target[key]
        item["transactions"] += int(row["transactions"])
        item["fraud"] += int(row["fraud"])
        item["amount_sum"] += float(row["amount_sum"])


def aggregate(frame, column, target):
    grouped = frame.groupby(
        column, observed=True
    ).agg(
        transactions=("isFraud", "size"),
        fraud=("isFraud", "sum"),
        amount_sum=("amount", "sum"),
    )

    add_group_stats(target, grouped)


def stats_frame(stats, key_name, order=None):
    keys = order if order is not None else sorted(stats)
    rows = []

    for key in keys:
        item = stats.get(key, empty_stats())
        count = item["transactions"]

        rows.append({
            key_name: key,
            "transactions": count,
            "fraud_transactions": item["fraud"],
            "fraud_rate_pct": (
                100 * item["fraud"] / count
                if count else 0.0
            ),
            "mean_amount": (
                item["amount_sum"] / count
                if count else 0.0
            ),
        })

    return pd.DataFrame(rows)


def save_charts(output, total, fraud, by_type, by_day, by_band):
    charts = output / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    # 1. Class imbalance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ["Legitimate", "Fraud"],
        [total - fraud, fraud],
    )
    if fraud > 0 and total > fraud:
        ax.set_yscale("log")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of transactions (log scale if applicable)")
    fig.tight_layout()
    fig.savefig(charts / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Fraud rate by transaction type
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        by_type["type"],
        by_type["fraud_rate_pct"],
    )
    ax.set_title("Fraud Rate by Transaction Type")
    ax.set_ylabel("Fraud rate (%)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(charts / "fraud_rate_by_type.png", dpi=150)
    plt.close(fig)

    # 3. Transaction volume by simulation day
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        by_day["simulation_day"],
        by_day["transactions"],
    )
    ax.set_title("Transaction Volume by Simulation Day")
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Number of transactions")
    fig.tight_layout()
    fig.savefig(charts / "transactions_by_day.png", dpi=150)
    plt.close(fig)

    # 4. Fraud rate by simulation day
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        by_day["simulation_day"],
        by_day["fraud_rate_pct"],
    )
    ax.set_title("Fraud Rate Over Simulation Time")
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Fraud rate (%)")
    fig.tight_layout()
    fig.savefig(charts / "fraud_rate_by_day.png", dpi=150)
    plt.close(fig)

    # 5. Fraud rate by amount band
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        by_band["amount_band"],
        by_band["fraud_rate_pct"],
    )
    ax.set_title("Fraud Rate by Transaction Amount Band")
    ax.set_ylabel("Fraud rate (%)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(charts / "fraud_rate_by_amount_band.png", dpi=150)
    plt.close(fig)


def analyze_dataset(
    source: Path,
    output: Path,
    batch_size: int = 50_000,
):
    source = Path(source)
    output = Path(output)

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    manifest_file = source / "manifest.json"

    if not manifest_file.is_file():
        raise FileNotFoundError("Dataset manifest not found")

    manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )

    files = sorted(source.glob("part-*.parquet"))

    if len(files) != manifest["parts"]:
        raise ValueError("Parquet part count mismatch")

    if not files:
        raise ValueError("No Parquet files found")

    by_type = defaultdict(empty_stats)
    by_day = defaultdict(empty_stats)
    by_band = defaultdict(empty_stats)

    total = 0
    fraud = 0
    missing_cells = 0
    minimum_step = None
    maximum_step = None

    for file_number, path in enumerate(files, start=1):
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)

        if not REQUIRED.issubset(columns):
            raise ValueError("Missing required processed columns")

        if columns.intersection(FORBIDDEN):
            raise ValueError("Leaked forbidden columns detected")

        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=sorted(REQUIRED),
        ):
            frame = batch.to_pandas()

            missing = int(frame.isna().sum().sum())
            missing_cells += missing

            if missing:
                raise ValueError("Missing values detected")

            amounts = frame["amount"].to_numpy(
                dtype="float64"
            )

            if (
                not np.isfinite(amounts).all()
                or (amounts < 0).any()
            ):
                raise ValueError("Invalid amounts detected")

            if not frame["isFraud"].isin([0, 1]).all():
                raise ValueError("Invalid fraud labels")

            if (frame["step"] < 1).any():
                raise ValueError("Invalid simulation step")

            expected_day = (frame["step"] - 1) // 24

            if not (
                frame["simulation_day"] == expected_day
            ).all():
                raise ValueError("Simulation day mismatch")

            total += len(frame)
            fraud += int(frame["isFraud"].sum())

            low = int(frame["step"].min())
            high = int(frame["step"].max())

            minimum_step = (
                low if minimum_step is None
                else min(minimum_step, low)
            )
            maximum_step = (
                high if maximum_step is None
                else max(maximum_step, high)
            )

            aggregate(frame, "type", by_type)
            aggregate(frame, "simulation_day", by_day)

            frame["amount_band"] = pd.cut(
                frame["amount"],
                bins=BAND_EDGES,
                labels=BAND_LABELS,
                right=False,
            )
            aggregate(frame, "amount_band", by_band)

        if file_number % 10 == 0:
            print(
                f"Analyzed {file_number}/{len(files)} files",
                flush=True,
            )

    # Reconcile physical data against ingestion metadata.
    if (
        total != manifest["rows"]
        or fraud != manifest["fraud_rows"]
        or minimum_step != manifest["min_step"]
        or maximum_step != manifest["max_step"]
    ):
        raise ValueError("Manifest totals mismatch")

    actual_type_counts = {
        key: item["transactions"]
        for key, item in by_type.items()
    }

    if actual_type_counts != manifest["transaction_type_counts"]:
        raise ValueError("Transaction type counts mismatch")

    type_table = stats_frame(by_type, "type")
    day_table = stats_frame(by_day, "simulation_day")
    band_table = stats_frame(
        by_band,
        "amount_band",
        BAND_LABELS,
    )

    summary = {
        "dataset": "PaySim",
        "source_sha256": manifest["source_sha256"],
        "pipeline_version": manifest["pipeline_version"],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_transactions": total,
        "fraud_transactions": fraud,
        "legitimate_transactions": total - fraud,
        "fraud_rate_pct": 100 * fraud / total,
        "missing_cells": missing_cells,
        "min_step": minimum_step,
        "max_step": maximum_step,
        "unique_simulation_days": len(by_day),
        "parquet_parts": len(files),
        "analysis_batch_size": batch_size,
        "baseline_accuracy_pct": (
            100 * (total - fraud) / total
        ),
    }

    output.mkdir(parents=True, exist_ok=True)

    type_table.to_csv(
        output / "by_transaction_type.csv", index=False
    )
    day_table.to_csv(
        output / "by_simulation_day.csv", index=False
    )
    band_table.to_csv(
        output / "by_amount_band.csv", index=False
    )

    save_charts(
        output, total, fraud,
        type_table, day_table, band_table,
    )

    (output / "eda_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2), flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="FraudGuard PaySim exploratory analysis"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)

    args = parser.parse_args()

    analyze_dataset(
        args.input,
        args.output,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
