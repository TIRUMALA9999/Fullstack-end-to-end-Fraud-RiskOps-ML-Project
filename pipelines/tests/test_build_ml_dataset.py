
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_ml_dataset import build_dataset


def create_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    frame = pd.DataFrame({
        "step": [1, 2, 3, 4, 5, 6],
        "type": [
            "PAYMENT", "TRANSFER", "CASH_OUT",
            "TRANSFER", "PAYMENT", "CASH_OUT",
        ],
        "amount": [10.0, 500.0, 250.0, 1000.0, 0.0, 750.0],
        "nameOrig": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "isFraud": [0, 1, 0, 1, 0, 1],
    })

    frame.to_parquet(
        source / "part-00000.parquet",
        index=False,
    )

    manifest = {
        "source_sha256": "fixture-checksum",
        "rows": 6,
        "fraud_rows": 3,
        "min_step": 1,
        "max_step": 6,
        "parts": 1,
    }

    (source / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return source, frame, manifest


def run_fixture(source, output):
    return build_dataset(
        source,
        output,
        train_end=2,
        validation_end=4,
        batch_size=2,
    )


def read_split(output, name):
    files = sorted((output / name).glob("part-*.parquet"))
    return pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )


def test_chronological_splits(tmp_path):
    source, _, _ = create_fixture(tmp_path)
    output = tmp_path / "features"

    manifest = run_fixture(source, output)

    assert manifest["source_rows"] == 6
    assert manifest["source_fraud_rows"] == 3

    for name in ("train", "validation", "test"):
        assert manifest["splits"][name]["rows"] == 2
        assert manifest["splits"][name]["fraud_rows"] == 1

    train = read_split(output, "train")
    validation = read_split(output, "validation")
    test = read_split(output, "test")

    assert train["source_step"].tolist() == [1, 2]
    assert validation["source_step"].tolist() == [3, 4]
    assert test["source_step"].tolist() == [5, 6]


def test_feature_contract_and_log_transform(tmp_path):
    source, _, _ = create_fixture(tmp_path)
    output = tmp_path / "features"

    manifest = run_fixture(source, output)
    train = read_split(output, "train")

    assert manifest["feature_columns"] == [
        "amount", "log_amount", "transaction_type"
    ]

    assert set(train.columns) == {
        "amount", "log_amount", "transaction_type",
        "is_fraud", "source_step",
    }

    assert "nameOrig" not in train.columns
    assert "step" not in manifest["feature_columns"]
    assert "is_fraud" not in manifest["feature_columns"]

    np.testing.assert_allclose(
        train["log_amount"],
        np.log1p(train["amount"]),
    )


def test_idempotent_execution(tmp_path):
    source, _, _ = create_fixture(tmp_path)
    output = tmp_path / "features"

    first = run_fixture(source, output)
    second = run_fixture(source, output)

    assert first == second


def test_invalid_split_boundaries(tmp_path):
    source, _, _ = create_fixture(tmp_path)

    with pytest.raises(ValueError, match="boundaries"):
        build_dataset(
            source,
            tmp_path / "features",
            train_end=4,
            validation_end=4,
        )


def test_manifest_mismatch(tmp_path):
    source, _, manifest = create_fixture(tmp_path)

    manifest["rows"] = 999

    (source / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest counts"):
        run_fixture(source, tmp_path / "features")

    assert not (tmp_path / "features").exists()


def test_negative_amount_rejected(tmp_path):
    source, frame, _ = create_fixture(tmp_path)

    frame.loc[0, "amount"] = -10

    frame.to_parquet(
        source / "part-00000.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="amounts"):
        run_fixture(source, tmp_path / "features")

    assert not (tmp_path / "features").exists()


def test_no_fraud_in_validation_rejected(tmp_path):
    source, frame, manifest = create_fixture(tmp_path)

    frame.loc[3, "isFraud"] = 0

    frame.to_parquet(
        source / "part-00000.parquet",
        index=False,
    )

    manifest["fraud_rows"] = 2

    (source / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no fraud examples"):
        run_fixture(source, tmp_path / "features")

    assert not (tmp_path / "features").exists()
