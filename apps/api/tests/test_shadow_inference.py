
import numpy as np
import pytest

from fastapi import HTTPException
from pydantic import ValidationError

from shadow_inference import (
    ShadowEngine,
    ShadowPredictionRequest,
    require_shadow_key,
    verify_release,
)


class FakeModel:
    def predict_proba(self, frame):
        assert frame.columns.tolist() == [
            "amount",
            "log_amount",
            "transaction_type",
        ]

        assert frame.loc[0, "transaction_type"] == "TRANSFER"

        return np.array([[0.2, 0.8]])


class FakeCalibrator:
    def predict_proba(self, transformed):
        assert transformed.shape == (1, 1)
        return np.array([[0.9, 0.1]])


def valid_request():
    return ShadowPredictionRequest(
        source_dataset="PAYSIM",
        amount_unit="PAYSIM_SYNTHETIC_UNITS",
        amount=1500.0,
        transaction_type="TRANSFER",
        reference_id="demo-001",
    )


def test_review_prediction():
    engine = ShadowEngine(
        model=FakeModel(),
        calibrator=FakeCalibrator(),
        threshold=0.05,
    )

    result = engine.predict(valid_request())

    assert result["decision"] == "REVIEW"
    assert result["calibrated_score"] == pytest.approx(0.1)
    assert result["financial_action_taken"] is False
    assert result["deployment_mode"] == "SHADOW_ONLY"


def test_no_review_prediction():
    engine = ShadowEngine(
        model=FakeModel(),
        calibrator=FakeCalibrator(),
        threshold=0.2,
    )

    result = engine.predict(valid_request())

    assert result["decision"] == "NO_REVIEW"
    assert result["financial_action_taken"] is False


def test_rejects_usd_amount_unit():
    with pytest.raises(ValidationError):
        ShadowPredictionRequest(
            source_dataset="PAYSIM",
            amount_unit="USD",
            amount=1500.0,
            transaction_type="TRANSFER",
        )


def test_rejects_invalid_amount():
    with pytest.raises(ValidationError):
        ShadowPredictionRequest(
            source_dataset="PAYSIM",
            amount_unit="PAYSIM_SYNTHETIC_UNITS",
            amount=-100.0,
            transaction_type="TRANSFER",
        )


def test_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setenv(
        "FRAUDGUARD_SHADOW_API_KEY",
        "correct-secret",
    )

    with pytest.raises(HTTPException) as error:
        require_shadow_key("incorrect-secret")

    assert error.value.status_code == 401


def test_rejects_missing_release_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_release(tmp_path)
