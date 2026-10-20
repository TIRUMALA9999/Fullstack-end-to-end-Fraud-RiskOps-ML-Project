
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from routers.shadow_records import (
    get_shadow_summary,
    list_shadow_predictions,
)


class FakeScalarResults:
    def __init__(self, values):
        self.values = iter(values)

    def scalar(self, statement):
        return next(self.values)


class FakeListDB:
    def __init__(self, record):
        self.record = record

    def scalar(self, statement):
        return 1

    def scalars(self, statement):
        return self

    def all(self):
        return [self.record]


def sample_record():
    return SimpleNamespace(
        id=uuid4(),
        reference_id="synthetic-history-001",
        source_dataset="PAYSIM",
        amount_unit="PAYSIM_SYNTHETIC_UNITS",
        amount=Decimal("150000.00"),
        transaction_type="TRANSFER",
        calibrated_score=Decimal("0.034009247776"),
        review_threshold=Decimal("0.050098696233"),
        decision="NO_REVIEW",
        model_version="lightgbm-shadow-v1",
        model_run_id="b78db1a44df041079c5c42c1579701b2",
        deployment_mode="SHADOW_ONLY",
        created_at=datetime.now(timezone.utc),
    )


def test_prediction_history():
    db = FakeListDB(sample_record())

    result = list_shadow_predictions(
        db=db,
        limit=20,
        offset=0,
        decision=None,
    )

    assert result["total"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0]["decision"] == "NO_REVIEW"
    assert result["items"][0]["financial_action_taken"] is False


def test_history_rejects_invalid_decision():
    db = FakeListDB(sample_record())

    with pytest.raises(HTTPException) as exc:
        list_shadow_predictions(
            db=db,
            limit=20,
            offset=0,
            decision="BLOCK",
        )

    assert exc.value.status_code == 422


def test_shadow_summary():
    db = FakeScalarResults([
        10,
        3,
        Decimal("0.052"),
    ])

    result = get_shadow_summary(db=db)

    assert result["total_predictions"] == 10
    assert result["review_count"] == 3
    assert result["no_review_count"] == 7
    assert result["review_rate"] == pytest.approx(0.3)
    assert result["average_score"] == pytest.approx(0.052)


def test_empty_summary():
    db = FakeScalarResults([0, 0, None])

    result = get_shadow_summary(db=db)

    assert result["total_predictions"] == 0
    assert result["review_rate"] == 0.0
    assert result["average_score"] is None
