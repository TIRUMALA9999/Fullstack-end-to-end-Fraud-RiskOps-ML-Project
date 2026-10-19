
import argparse
import json
import tempfile

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FEATURE_PIPELINE_VERSION = "1.0"

SOURCE_COLUMNS = ["step", "type", "amount", "isFraud"]

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "transaction_type",
]

EXPORT_COLUMNS = FEATURE_COLUMNS + [
    "is_fraud",
    "source_step",
]

TRANSACTION_TYPES = {
    "TRANSFER", "CASH_OUT", "CASH_IN",
    "PAYMENT", "DEBIT",
}

SPLITS = ("train", "validation", "test")


def transform_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate features available before fraud classification."""

    amount = frame["amount"].to_numpy(dtype="float64")

    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("Invalid transaction amounts")

    if frame[SOURCE_COLUMNS].isna().any().any():
        raise ValueError("Missing input values")

    if not frame["type"].isin(TRANSACTION_TYPES).all():
        raise ValueError("Invalid transaction type")

    if not frame["isFraud"].isin([0, 1]).all():
        raise ValueError("Invalid fraud labels")

    return pd.DataFrame({
        "amount": amount,
        "log_amount": np.log1p(amount),
        "transaction_type": frame["type"].astype(str).to_numpy(),
        "is_fraud": frame["isFraud"].to_numpy(dtype="int8"),
        "source_step": frame["step"].to_numpy(dtype="int32"),
    })


def verify_existing_output(destination, source_manifest, train_end, val_end):
    manifest_file = destination / "manifest.json"

    if not manifest_file.is_file():
        raise FileExistsError("Existing output has no manifest")

    manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )

    expected = {
        "source_sha256": source_manifest["source_sha256"],
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "train_end_step": train_end,
        "validation_end_step": val_end,
    }

    if any(manifest.get(key) != value for key, value in expected.items()):
        raise FileExistsError(
            "Existing dataset has different configuration"
        )

    for split in SPLITS:
        files = sorted((destination / split).glob("part-*.parquet"))
        split_info = manifest["splits"][split]

        actual_rows = sum(
            pq.read_metadata(path).num_rows for path in files
        )

        if (
            len(files) != split_info["parts"]
            or actual_rows != split_info["rows"]
        ):
            raise ValueError("Existing dataset is incomplete")

    print("Existing feature dataset verified; skipping.")
    return manifest


def build_dataset(
    source: Path,
    destination: Path,
    train_end: int = 504,
    validation_end: int = 624,
    batch_size: int = 100_000,
):
    source = Path(source)
    destination = Path(destination)

    manifest_path = source / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError("Source manifest not found")

    source_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    max_step = int(source_manifest["max_step"])

    if not (1 <= train_end < validation_end < max_step):
        raise ValueError("Invalid chronological split boundaries")

    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    files = sorted(source.glob("part-*.parquet"))

    if len(files) != source_manifest["parts"]:
        raise ValueError("Source Parquet part count mismatch")

    if destination.exists():
        return verify_existing_output(
            destination, source_manifest,
            train_end, validation_end,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        name: {
            "rows": 0,
            "fraud_rows": 0,
            "parts": 0,
            "min_step": None,
            "max_step": None,
        }
        for name in SPLITS
    }

    total_rows = 0
    total_fraud = 0
    observed_min = None
    observed_max = None

    with tempfile.TemporaryDirectory(
        prefix=".feature-staging-",
        dir=destination.parent,
    ) as tmp:

        staging = Path(tmp) / "dataset"

        for name in SPLITS:
            (staging / name).mkdir(parents=True)

        for file_number, path in enumerate(files, start=1):
            parquet = pq.ParquetFile(path)

            if not set(SOURCE_COLUMNS).issubset(
                parquet.schema_arrow.names
            ):
                raise ValueError("Missing required source columns")

            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=SOURCE_COLUMNS,
            ):
                source_frame = batch.to_pandas()
                engineered = transform_features(source_frame)

                steps = engineered["source_step"].to_numpy()
                labels = engineered["is_fraud"].to_numpy()

                if (steps < 1).any() or (steps > max_step).any():
                    raise ValueError("Simulation step outside source range")

                total_rows += len(engineered)
                total_fraud += int(labels.sum())

                low = int(steps.min())
                high = int(steps.max())

                observed_min = (
                    low if observed_min is None
                    else min(observed_min, low)
                )
                observed_max = (
                    high if observed_max is None
                    else max(observed_max, high)
                )

                masks = {
                    "train": steps <= train_end,
                    "validation": (
                        (steps > train_end)
                        & (steps <= validation_end)
                    ),
                    "test": steps > validation_end,
                }

                for split_name, mask in masks.items():
                    if not mask.any():
                        continue

                    subset = engineered.loc[mask, EXPORT_COLUMNS]
                    item = stats[split_name]

                    number = item["parts"]
                    filename = f"part-{number:05d}.parquet"

                    subset.to_parquet(
                        staging / split_name / filename,
                        index=False,
                        compression="zstd",
                        engine="pyarrow",
                    )

                    count = len(subset)
                    positives = int(subset["is_fraud"].sum())

                    item["rows"] += count
                    item["fraud_rows"] += positives
                    item["parts"] += 1

                    step_min = int(subset["source_step"].min())
                    step_max = int(subset["source_step"].max())

                    item["min_step"] = (
                        step_min if item["min_step"] is None
                        else min(item["min_step"], step_min)
                    )
                    item["max_step"] = (
                        step_max if item["max_step"] is None
                        else max(item["max_step"], step_max)
                    )

            if file_number % 10 == 0:
                print(
                    f"Processed {file_number}/{len(files)} files",
                    flush=True,
                )

        if (
            total_rows != source_manifest["rows"]
            or total_fraud != source_manifest["fraud_rows"]
            or observed_min != source_manifest["min_step"]
            or observed_max != source_manifest["max_step"]
        ):
            raise ValueError("Source manifest counts mismatch")

        if sum(stats[name]["rows"] for name in SPLITS) != total_rows:
            raise ValueError("Split row counts do not reconcile")

        if sum(stats[name]["fraud_rows"] for name in SPLITS) != total_fraud:
            raise ValueError("Split fraud counts do not reconcile")

        for name in SPLITS:
            item = stats[name]

            if item["rows"] == 0:
                raise ValueError(f"{name} split is empty")

            if item["fraud_rows"] == 0:
                raise ValueError(f"{name} split has no fraud examples")

            item["fraud_rate_pct"] = (
                100 * item["fraud_rows"] / item["rows"]
            )

        manifest = {
            "dataset": "PaySim",
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "source_sha256": source_manifest["source_sha256"],
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_rows": total_rows,
            "source_fraud_rows": total_fraud,
            "train_end_step": train_end,
            "validation_end_step": validation_end,
            "feature_columns": FEATURE_COLUMNS,
            "label_column": "is_fraud",
            "audit_column": "source_step",
            "export_columns": EXPORT_COLUMNS,
            "splits": stats,
        }

        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        # Only publish a fully validated dataset.
        staging.rename(destination)

    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="FraudGuard feature engineering and split pipeline"
    )

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-end", type=int, default=504)
    parser.add_argument("--validation-end", type=int, default=624)
    parser.add_argument("--batch-size", type=int, default=100_000)

    args = parser.parse_args()

    build_dataset(
        source=args.input,
        destination=args.output,
        train_end=args.train_end,
        validation_end=args.validation_end,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
