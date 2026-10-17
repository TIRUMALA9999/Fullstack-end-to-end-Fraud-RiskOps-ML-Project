import argparse
import hashlib
import json
import tempfile

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_VERSION = "1.0"

DATASET_SOURCE = "https://www.kaggle.com/datasets/ealaxi/paysim1"

RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Only fields eligible for downstream feature engineering
# and supervised learning are retained.
KEEP_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
]

ALLOWED_TYPES = {
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "PAYMENT",
    "DEBIT",
}

DTYPES = {
    "step": "Int32",
    "type": "string",
    "amount": "float64",
    "nameOrig": "string",
    "nameDest": "string",
    "isFraud": "Int8",
}


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum without loading the file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_chunk(frame: pd.DataFrame, number: int):
    """Fail immediately when invalid records are detected."""

    if frame.isna().any().any():
        raise ValueError(
            f"Chunk {number}: missing values detected"
        )

    if not frame["step"].between(1, 744).all():
        raise ValueError(
            f"Chunk {number}: invalid simulation step"
        )

    if not frame["type"].isin(ALLOWED_TYPES).all():
        raise ValueError(
            f"Chunk {number}: unsupported transaction type"
        )

    if (
        not np.isfinite(frame["amount"]).all()
        or (frame["amount"] < 0).any()
    ):
        raise ValueError(
            f"Chunk {number}: invalid transaction amount"
        )

    if not frame["isFraud"].isin([0, 1]).all():
        raise ValueError(
            f"Chunk {number}: invalid fraud label"
        )

    for column in ("nameOrig", "nameDest"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(
                f"Chunk {number}: empty {column}"
            )


def ingest_file(
    source: Path,
    destination: Path,
    chunk_size: int = 100_000,
) -> dict:
    """Validate and convert a PaySim CSV into versioned Parquet."""

    source = Path(source)
    destination = Path(destination)

    if not source.is_file():
        raise FileNotFoundError(
            f"Source CSV not found: {source}"
        )

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    print("Calculating source checksum...", flush=True)
    source_hash = sha256_file(source)

    # A published dataset version is never silently overwritten.
    if destination.exists():
        manifest_path = destination / "manifest.json"

        if manifest_path.exists():
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

            if (
                manifest.get("source_sha256") == source_hash
                and manifest.get("pipeline_version")
                == PIPELINE_VERSION
            ):
                print(
                    "Dataset already processed. Skipping.",
                    flush=True,
                )
                return manifest

        raise FileExistsError(
            "Output version already exists with different "
            "metadata. Use a new destination version."
        )

    # Validate source headers before starting transformations.
    actual_columns = set(
        pd.read_csv(source, nrows=0).columns
    )

    missing_columns = set(RAW_COLUMNS) - actual_columns

    if missing_columns:
        raise ValueError(
            f"Missing source columns: {sorted(missing_columns)}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_rows = 0
    fraud_rows = 0
    total_parts = 0

    type_counts = Counter()

    minimum_step = None
    maximum_step = None

    # All outputs stay in a temporary directory until success.
    # A failed run cannot publish a partial dataset.
    with tempfile.TemporaryDirectory(
        prefix=".paysim-staging-",
        dir=destination.parent,
    ) as temporary_directory:

        staging = Path(temporary_directory) / "dataset"
        staging.mkdir()

        reader = pd.read_csv(
            source,
            usecols=KEEP_COLUMNS,
            dtype=DTYPES,
            chunksize=chunk_size,
            on_bad_lines="error",
        )

        for number, frame in enumerate(reader):
            validate_chunk(frame, number)

            frame = frame.loc[:, KEEP_COLUMNS].copy()

            # Day numbers are relative to the PaySim simulation.
            frame["simulation_day"] = (
                (frame["step"] - 1) // 24
            ).astype("int16")

            output_file = (
                staging / f"part-{number:05d}.parquet"
            )

            frame.to_parquet(
                output_file,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )

            count = len(frame)
            total_rows += count
            fraud_rows += int(frame["isFraud"].sum())
            total_parts += 1

            type_counts.update(
                frame["type"].value_counts().to_dict()
            )

            chunk_min = int(frame["step"].min())
            chunk_max = int(frame["step"].max())

            minimum_step = (
                chunk_min if minimum_step is None
                else min(minimum_step, chunk_min)
            )

            maximum_step = (
                chunk_max if maximum_step is None
                else max(maximum_step, chunk_max)
            )

            if total_parts % 10 == 0:
                print(
                    f"Processed {total_rows:,} rows",
                    flush=True,
                )

        if total_rows == 0:
            raise ValueError("Source dataset is empty")

        manifest = {
            "dataset": "PaySim",
            "source": DATASET_SOURCE,
            "source_file": source.name,
            "source_sha256": source_hash,
            "pipeline_version": PIPELINE_VERSION,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "rows": total_rows,
            "fraud_rows": fraud_rows,
            "legitimate_rows": total_rows - fraud_rows,
            "fraud_rate": fraud_rows / total_rows,
            "min_step": minimum_step,
            "max_step": maximum_step,
            "parts": total_parts,
            "transaction_type_counts": dict(
                sorted(type_counts.items())
            ),
            "processed_columns": (
                KEEP_COLUMNS + ["simulation_day"]
            ),
            "format": "parquet",
            "compression": "zstd",
            "chunk_size": chunk_size,
        }

        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        # Publish only after successful processing.
        staging.rename(destination)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="FraudGuard PaySim ingestion pipeline"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
    )

    args = parser.parse_args()

    manifest = ingest_file(
        source=args.input,
        destination=args.output,
        chunk_size=args.chunk_size,
    )

    print(
        json.dumps(manifest, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()