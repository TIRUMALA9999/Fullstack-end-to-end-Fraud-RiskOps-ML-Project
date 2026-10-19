import json

import numpy as np
import pandas as pd
import pytest

from train_baseline import (
    FEATURES,
    evaluate_at_capacity,
    load_partition,
    make_model,
)


def make_fixture(tmp_path):
    root = tmp_path / "model_ready"
    (root / "train").mkdir(parents=True)
    (root / "validation").mkdir()

    train = pd.DataFrame({
        "amount": [10.0, 300.0, 20.0, 800.0],
        "log_amount": np.log1p([10, 300, 20, 800]),
        "transaction_type": [
            "PAYMENT", "TRANSFER",
            "CASH_IN", "CASH_OUT",
        ],
        "is_fraud": [0, 1, 0, 1],
        "source_step": [1, 2, 3, 4],
    })

    validation = pd.DataFrame({
        "amount": [25.0, 750.0, 40.0, 900.0],
        "log_amount": np.log1p([25, 750, 40, 900]),
        "transaction_type": [
            "PAYMENT", "TRANSFER",
            "DEBIT", "CASH_OUT",
        ],
        "is_fraud": [0, 1, 0, 1],
        "source_step": [5, 6, 7, 8],
    })

    train.to_parquet(
        root / "train" / "part-00000.parquet",
        index=False,
    )

    validation.to_parquet(
        root / "validation" / "part-00000.parquet",
        index=False,
    )

    manifest = {
        "feature_columns": FEATURES,
        "train_end_step": 4,
        "validation_end_step": 8,
        "splits": {
            "train": {
                "rows": 4,
                "fraud_rows": 2,
                "parts": 1,
            },
            "validation": {
                "rows": 4,
                "fraud_rows": 2,
                "parts": 1,
            },
        },
    }

    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return root


def test_training_sample_keeps_fraud(tmp_path):
    root = make_fixture(tmp_path)

    train = load_partition(
        root, "train", negative_rate=1.0
    )

    assert len(train) == 4
    assert int(train["is_fraud"].sum()) == 2
    assert (train["sample_weight"] == 1).all()


def test_validation_is_not_sampled(tmp_path):
    root = make_fixture(tmp_path)

    validation = load_partition(
        root, "validation", negative_rate=0.01
    )

    assert len(validation) == 4
    assert "sample_weight" not in validation.columns


def test_test_partition_cannot_be_loaded(tmp_path):
    root = make_fixture(tmp_path)

    with pytest.raises(ValueError, match="permitted"):
        load_partition(root, "test")


def test_feature_allowlist():
    assert FEATURES == [
        "amount",
        "log_amount",
        "transaction_type",
    ]
    assert "is_fraud" not in FEATURES
    assert "source_step" not in FEATURES


def test_perfect_ranking():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = evaluate_at_capacity(
        labels,
        scores,
        review_fraction=0.5,
    )

    assert metrics["average_precision"] == 1.0
    assert metrics["precision_at_capacity"] == 1.0
    assert metrics["recall_at_capacity"] == 1.0


def test_model_pipeline_can_fit(tmp_path):
    root = make_fixture(tmp_path)

    train = load_partition(
        root, "train", negative_rate=1.0
    )

    model = make_model()

    model.fit(
        train[FEATURES],
        train["is_fraud"],
        classifier__sample_weight=train["sample_weight"],
    )

    predictions = model.predict_proba(
        train[FEATURES]
    )

    assert predictions.shape == (4, 2)
    assert np.isfinite(predictions).all()