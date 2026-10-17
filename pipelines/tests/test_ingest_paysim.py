import json

import pandas as pd
import pytest

from ingest_paysim import ingest_file


def create_test_dataset(tmp_path):
    source = tmp_path / "sample.csv"

    data = pd.DataFrame({
        "step": [1, 2, 25],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [50.50, 250.00, 0.00],
        "nameOrig": ["C100", "C200", "C300"],
        "oldbalanceOrg": [100, 500, 200],
        "newbalanceOrig": [50, 250, 200],
        "nameDest": ["M100", "C400", "C500"],
        "oldbalanceDest": [0, 100, 100],
        "newbalanceDest": [0, 350, 100],
        "isFraud": [0, 1, 0],
        "isFlaggedFraud": [0, 0, 0],
    })

    data.to_csv(source, index=False)

    return source


def test_successful_ingestion(tmp_path):
    source = create_test_dataset(tmp_path)
    output = tmp_path / "processed" / "v1"

    manifest = ingest_file(
        source,
        output,
        chunk_size=2,
    )

    assert manifest["rows"] == 3
    assert manifest["fraud_rows"] == 1
    assert manifest["legitimate_rows"] == 2
    assert manifest["parts"] == 2

    assert (output / "manifest.json").exists()

    parquet_files = sorted(output.glob("*.parquet"))
    assert len(parquet_files) == 2

    processed = pd.concat(
        [pd.read_parquet(path) for path in parquet_files],
        ignore_index=True,
    )

    assert len(processed) == 3
    assert processed["simulation_day"].tolist() == [0, 0, 1]

    # Leakage-prone fields must never reach ML-ready data.
    assert "newbalanceOrig" not in processed.columns
    assert "oldbalanceOrg" not in processed.columns
    assert "oldbalanceDest" not in processed.columns
    assert "newbalanceDest" not in processed.columns
    assert "isFlaggedFraud" not in processed.columns


def test_idempotent_ingestion(tmp_path):
    source = create_test_dataset(tmp_path)
    output = tmp_path / "processed" / "v1"

    first = ingest_file(source, output, chunk_size=2)
    second = ingest_file(source, output, chunk_size=2)

    assert first == second


def test_invalid_fraud_label_rejected(tmp_path):
    source = create_test_dataset(tmp_path)
    output = tmp_path / "processed" / "v1"

    data = pd.read_csv(source)
    data.loc[0, "isFraud"] = 5
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="fraud label"):
        ingest_file(source, output, chunk_size=2)

    assert not output.exists()


def test_negative_amount_rejected(tmp_path):
    source = create_test_dataset(tmp_path)
    output = tmp_path / "processed" / "v1"

    data = pd.read_csv(source)
    data.loc[0, "amount"] = -100
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="amount"):
        ingest_file(source, output, chunk_size=2)

    assert not output.exists()


def test_missing_column_rejected(tmp_path):
    source = create_test_dataset(tmp_path)
    output = tmp_path / "processed" / "v1"

    data = pd.read_csv(source)
    data = data.drop(columns=["nameDest"])
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="Missing source columns"):
        ingest_file(source, output, chunk_size=2)

    assert not output.exists()


def test_source_change_requires_new_version(tmp_path):
    source = create_test_dataset(tmp_path)
    output = tmp_path / "processed" / "v1"

    ingest_file(source, output, chunk_size=2)

    data = pd.read_csv(source)
    data.loc[0, "amount"] = 75
    data.to_csv(source, index=False)

    with pytest.raises(FileExistsError):
        ingest_file(source, output, chunk_size=2)