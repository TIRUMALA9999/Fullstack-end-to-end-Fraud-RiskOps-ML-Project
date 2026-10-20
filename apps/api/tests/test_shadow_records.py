from datetime import datetime, timezone

import pytest

from fastapi import HTTPException, Response
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

import routers.shadow_records as shadow_records

from models import AuditEvent, ShadowPrediction
from routers.shadow_records import (
    StoredShadowRequest,
    get_recorded_prediction,
    predict_and_record,
)


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def predict(self, payload):
        self.calls += 1

        return {
            "calibrated_score": 0.034009247776262415,
            "review_threshold": 0.05009869623339683,
            "decision": "NO_REVIEW",
            "model_version": "lightgbm-shadow-v1",
            "model_run_id": (
                "b78db1a44df041079c5c42c1579701b2"
            ),
        }


class FakeDB:
    def __init__(self, fail_audit=False):
        self.existing = None
        self.added = []
        self.committed = False
        self.rolled_back = False
        self.fail_audit = fail_audit

    def scalar(self, statement):
        return self.existing

    def add(self, obj):
        if (
            self.fail_audit
            and isinstance(obj, AuditEvent)
        ):
            raise SQLAlchemyError(
                "Simulated audit insert failure"
            )

        self.added.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        obj.created_at = datetime.now(timezone.utc)

    def rollback(self):
        self.rolled_back = True

    def get(self, model, identifier):
        if (
            self.existing is not None
            and self.existing.id == identifier
        ):
            return self.existing

        return None


def request(reference_id="shadow-test-001", amount=150000.0):
    return StoredShadowRequest(
        source_dataset="PAYSIM",
        amount_unit="PAYSIM_SYNTHETIC_UNITS",
        amount=amount,
        transaction_type="TRANSFER",
        reference_id=reference_id,
    )


def test_creates_prediction_and_audit_atomically(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        shadow_records, "load_release", lambda: engine
    )

    db = FakeDB()

    result = predict_and_record(
        request(), Response(), db
    )

    assert db.committed is True
    assert db.rolled_back is False
    assert engine.calls == 1

    assert len(db.added) == 2
    assert isinstance(db.added[0], ShadowPrediction)
    assert isinstance(db.added[1], AuditEvent)

    prediction = db.added[0]
    audit = db.added[1]

    assert result["decision"] == "NO_REVIEW"
    assert result["financial_action_taken"] is False
    assert audit.transaction_id is None

    assert audit.details["shadow_prediction_id"] == str(
        prediction.id
    )


def test_same_reference_is_idempotent(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        shadow_records, "load_release", lambda: engine
    )

    db = FakeDB()

    first = predict_and_record(
        request(), Response(), db
    )

    assert len(db.added) == 2

    db.existing = db.added[0]

    response = Response()

    second = predict_and_record(
        request(), response, db
    )

    assert first["id"] == second["id"]
    assert response.status_code == 200
    assert engine.calls == 1
    assert len(db.added) == 2


def test_changed_input_with_same_reference_is_rejected(
    monkeypatch,
):
    monkeypatch.setattr(
        shadow_records,
        "load_release",
        lambda: FakeEngine(),
    )

    db = FakeDB()

    predict_and_record(
        request(), Response(), db
    )

    db.existing = db.added[0]

    with pytest.raises(HTTPException) as error:
        predict_and_record(
            request(amount=250000.0),
            Response(),
            db,
        )

    assert error.value.status_code == 409


def test_audit_failure_rolls_back_prediction(monkeypatch):
    monkeypatch.setattr(
        shadow_records,
        "load_release",
        lambda: FakeEngine(),
    )

    db = FakeDB(fail_audit=True)

    with pytest.raises(HTTPException) as error:
        predict_and_record(
            request(), Response(), db
        )

    assert error.value.status_code == 503
    assert db.rolled_back is True
    assert db.committed is False


def test_persisted_request_requires_reference():
    with pytest.raises(ValidationError):
        StoredShadowRequest(
            source_dataset="PAYSIM",
            amount_unit="PAYSIM_SYNTHETIC_UNITS",
            amount=100.0,
            transaction_type="TRANSFER",
        )


def test_get_unknown_prediction_returns_404():
    import uuid

    with pytest.raises(HTTPException) as error:
        get_recorded_prediction(
            uuid.uuid4(), FakeDB()
        )

    assert error.value.status_code == 404