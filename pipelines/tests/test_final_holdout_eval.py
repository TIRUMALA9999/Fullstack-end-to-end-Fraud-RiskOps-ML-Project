
import numpy as np
import pytest

from final_holdout_eval import (
    fixed_policy_metrics,
    probability_metrics,
    reliability_bins,
)


def test_fixed_policy_confusion_matrix():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    metrics = fixed_policy_metrics(
        labels, scores, threshold=0.75
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["reviewed"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_perfect_probabilities():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.0, 1.0, 0.0, 1.0])

    result = probability_metrics(labels, scores)

    assert result["average_precision"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["brier_score"] == 0.0


def test_reliability_bins_cover_all_records():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.7, 0.2, 0.8, 0.3, 0.9])

    bins = reliability_bins(labels, scores, n_bins=3)

    assert len(bins) == 3
    assert sum(item["count"] for item in bins) == 6


def test_invalid_threshold():
    with pytest.raises(ValueError):
        fixed_policy_metrics(
            [0, 1],
            [0.1, 0.9],
            threshold=1.5,
        )


def test_invalid_prediction_values():
    with pytest.raises(ValueError):
        probability_metrics(
            [0, 1],
            [np.nan, 0.9],
        )


def test_probability_calibration_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.9])

    metrics = probability_metrics(labels, scores)

    assert 0 <= metrics["brier_score"] <= 1
    assert metrics["log_loss"] >= 0
    assert metrics["mean_predicted_probability"] == pytest.approx(
        0.475
    )
