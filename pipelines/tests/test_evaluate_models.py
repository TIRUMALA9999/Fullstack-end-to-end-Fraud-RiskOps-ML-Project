
import numpy as np
import pytest

from evaluate_models import evaluate_capacity


def test_capacity_confusion_matrix():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    result = evaluate_capacity(
        labels, scores, fraction=0.5
    )

    assert result["reviewed"] == 2
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_perfect_fraud_capture():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.95, 0.1, 0.9, 0.2])

    result = evaluate_capacity(
        labels, scores, fraction=0.5
    )

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0


@pytest.mark.parametrize("capacity", [0, -0.1, 1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        evaluate_capacity(
            [0, 1],
            [0.1, 0.9],
            fraction=capacity,
        )


def test_invalid_scores():
    with pytest.raises(ValueError):
        evaluate_capacity(
            [0, 1],
            [np.nan, 0.9],
            fraction=0.5,
        )
