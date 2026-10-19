import numpy as np
import pandas as pd
import pytest

from calibrate_risk_policy import (
    calibrate,
    chronological_windows,
    fit_calibrator,
    logit_scores,
    policy_metrics,
    select_threshold,
)


def sample_validation():
    return pd.DataFrame({
        "source_step": [
            505, 506, 565, 566, 595, 596
        ],
        "is_fraud": [0, 1, 0, 1, 0, 1],
    })


def test_chronological_windows():
    windows = chronological_windows(
        sample_validation(),
        train_end=504,
        validation_end=624,
        calibration_end=564,
        tuning_end=594,
    )

    assert len(windows["calibration"]) == 2
    assert len(windows["tuning"]) == 2
    assert len(windows["audit"]) == 2

    assert windows["calibration"][
        "source_step"
    ].max() < windows["tuning"][
        "source_step"
    ].min()

    assert windows["tuning"][
        "source_step"
    ].max() < windows["audit"][
        "source_step"
    ].min()


def test_invalid_boundaries():
    with pytest.raises(ValueError):
        chronological_windows(
            sample_validation(),
            train_end=504,
            validation_end=624,
            calibration_end=594,
            tuning_end=564,
        )


def test_calibration_produces_valid_probabilities():
    raw = np.array([
        0.01, 0.02, 0.05,
        0.60, 0.75, 0.90,
    ])

    labels = np.array([0, 0, 0, 1, 1, 1])

    model = fit_calibrator(raw, labels)
    scores = calibrate(model, raw)

    assert len(scores) == len(raw)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert np.all(np.diff(scores) > 0)


def test_review_threshold():
    scores = np.array([0.1, 0.3, 0.5, 0.8])

    threshold, target = select_threshold(
        scores, fraction=0.25
    )

    assert target == 1
    assert threshold == 0.8


def test_fixed_threshold_confusion_matrix():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    metrics = policy_metrics(
        labels, scores, threshold=0.75
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_invalid_calibration_scores():
    with pytest.raises(ValueError):
        logit_scores(np.array([0.1, np.nan]))