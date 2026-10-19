
import numpy as np
import pandas as pd

from train_baseline import FEATURES, evaluate_at_capacity
from train_lightgbm import make_lightgbm_model


def sample_data():
    n = 240
    indices = np.arange(n)

    amounts = (indices + 1).astype(float) * 11.0

    frame = pd.DataFrame({
        "amount": amounts,
        "log_amount": np.log1p(amounts),
        "transaction_type": np.where(
            indices % 2 == 0,
            "TRANSFER",
            "PAYMENT",
        ),
    })

    labels = (indices % 11 == 0).astype(int)
    weights = np.ones(n, dtype=float)

    return frame, labels, weights


def test_lightgbm_pipeline_fits():
    features, labels, weights = sample_data()

    model = make_lightgbm_model()

    model.fit(
        features,
        labels,
        classifier__sample_weight=weights,
    )

    scores = model.predict_proba(features)[:, 1]

    assert scores.shape == (240,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_lightgbm_feature_contract():
    model = make_lightgbm_model()

    assert FEATURES == [
        "amount",
        "log_amount",
        "transaction_type",
    ]

    assert "is_fraud" not in FEATURES
    assert "source_step" not in FEATURES

    transformer = model.named_steps["preprocessing"]
    selected_columns = []

    for _, _, columns in transformer.transformers:
        selected_columns.extend(columns)

    assert set(selected_columns) == set(FEATURES)


def test_lightgbm_hyperparameters():
    model = make_lightgbm_model()

    classifier = model.named_steps["classifier"]

    assert classifier.n_estimators == 200
    assert classifier.learning_rate == 0.05
    assert classifier.num_leaves == 15
    assert classifier.n_jobs == 2


def test_lightgbm_supports_unseen_category():
    features, labels, weights = sample_data()

    model = make_lightgbm_model()

    model.fit(
        features,
        labels,
        classifier__sample_weight=weights,
    )

    unseen = pd.DataFrame({
        "amount": [150.0],
        "log_amount": [np.log1p(150.0)],
        "transaction_type": ["CASH_OUT"],
    })

    score = model.predict_proba(unseen)[0, 1]

    assert np.isfinite(score)
    assert 0 <= score <= 1


def test_lightgbm_evaluation_metrics():
    features, labels, weights = sample_data()

    model = make_lightgbm_model()

    model.fit(
        features,
        labels,
        classifier__sample_weight=weights,
    )

    scores = model.predict_proba(features)[:, 1]

    metrics = evaluate_at_capacity(
        labels,
        scores,
        review_fraction=0.01,
    )

    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["recall_at_capacity"] <= 1
    assert metrics["reviewed_transactions"] == 3
