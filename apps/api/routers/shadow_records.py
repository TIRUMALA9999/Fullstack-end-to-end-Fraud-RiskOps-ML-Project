import logging
import uuid

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import Query
from sqlalchemy import func

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from database import get_db
from models import AuditEvent, ShadowPrediction
from shadow_inference import (
    MODEL_RUN_ID,
    ShadowPredictionRequest,
    load_release,
    require_shadow_key,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/shadow",
    tags=["Shadow Prediction Records"],
    dependencies=[Depends(require_shadow_key)],
)

DbSession = Annotated[Session, Depends(get_db)]


class StoredShadowRequest(ShadowPredictionRequest):
    # Required idempotency identifier for persisted predictions.
    reference_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"\S",
    )


class StoredShadowResponse(BaseModel):
    id: uuid.UUID
    reference_id: str
    source_dataset: str
    amount_unit: str
    amount: float
    transaction_type: str
    calibrated_score: float
    review_threshold: float
    decision: str
    model_version: str
    model_run_id: str
    deployment_mode: str
    financial_action_taken: bool
    created_at: datetime


def validated_amount(payload: StoredShadowRequest) -> Decimal:
    value = Decimal(str(payload.amount))

    # Keep values within the synthetic-data storage contract.
    if value <= 0 or value > Decimal("1000000000000"):
        raise HTTPException(
            status_code=422,
            detail="Synthetic amount must be greater than zero "
                   "and at most 1 trillion",
        )

    if value.as_tuple().exponent < -12:
        raise HTTPException(
            status_code=422,
            detail="Maximum 12 decimal places supported",
        )

    return value


def same_request(
    record: ShadowPrediction,
    payload: StoredShadowRequest,
    amount: Decimal,
) -> bool:
    return (
        record.reference_id == payload.reference_id
        and record.source_dataset == "PAYSIM"
        and record.amount_unit == "PAYSIM_SYNTHETIC_UNITS"
        and record.amount == amount
        and record.transaction_type == payload.transaction_type
        and record.model_run_id == MODEL_RUN_ID
        and record.deployment_mode == "SHADOW_ONLY"
    )


def serialize_record(record: ShadowPrediction) -> dict:
    return {
        "id": record.id,
        "reference_id": record.reference_id,
        "source_dataset": record.source_dataset,
        "amount_unit": record.amount_unit,
        "amount": float(record.amount),
        "transaction_type": record.transaction_type,
        "calibrated_score": float(record.calibrated_score),
        "review_threshold": float(record.review_threshold),
        "decision": record.decision,
        "model_version": record.model_version,
        "model_run_id": record.model_run_id,
        "deployment_mode": record.deployment_mode,
        "financial_action_taken": False,
        "created_at": record.created_at,
    }


def existing_by_reference(
    db: Session,
    reference_id: str,
):
    return db.scalar(
        select(ShadowPrediction).where(
            ShadowPrediction.reference_id == reference_id
        )
    )


@router.post(
    "/predict-and-record",
    response_model=StoredShadowResponse,
    status_code=status.HTTP_201_CREATED,
)
def predict_and_record(
    payload: StoredShadowRequest,
    response: Response,
    db: DbSession,
):
    amount = validated_amount(payload)

    if payload.reference_id != payload.reference_id.strip():
        raise HTTPException(
            status_code=422,
            detail="Reference ID must not contain surrounding whitespace",
        )

    # Idempotency check before performing inference.
    try:
        existing = existing_by_reference(
            db, payload.reference_id
        )
    except SQLAlchemyError:
        logger.exception("Shadow prediction lookup failed")
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Prediction storage unavailable",
        )

    if existing is not None:
        if not same_request(existing, payload, amount):
            raise HTTPException(
                status_code=409,
                detail="Reference ID conflicts with existing prediction",
            )

        response.status_code = 200
        return serialize_record(existing)

    # Perform real inference using the verified frozen model.
    try:
        prediction = load_release().predict(payload)
    except Exception:
        logger.exception("Shadow model inference failed")
        raise HTTPException(
            status_code=503,
            detail="Shadow inference unavailable",
        )

    record = ShadowPrediction(
        id=uuid.uuid4(),
        reference_id=payload.reference_id,
        source_dataset="PAYSIM",
        amount_unit="PAYSIM_SYNTHETIC_UNITS",
        amount=amount,
        transaction_type=payload.transaction_type,
        calibrated_score=Decimal(
            str(prediction["calibrated_score"])
        ),
        review_threshold=Decimal(
            str(prediction["review_threshold"])
        ),
        decision=prediction["decision"],
        model_version=prediction["model_version"],
        model_run_id=prediction["model_run_id"],
        deployment_mode="SHADOW_ONLY",
    )

    audit_event = AuditEvent(
        id=uuid.uuid4(),
        transaction_id=None,
        event_type="SHADOW_PREDICTION_RECORDED",
        actor_id="system:shadow_api",
        details={
            "shadow_prediction_id": str(record.id),
            "reference_id": record.reference_id,
            "model_version": record.model_version,
            "model_run_id": record.model_run_id,
            "decision": record.decision,
            "calibrated_score": str(record.calibrated_score),
            "deployment_mode": "SHADOW_ONLY",
            "source_dataset": "PAYSIM",
            "financial_action_taken": False,
        },
    )

    try:
        db.add(record)
        db.flush()

        # One commit makes the prediction and audit atomic.
        db.add(audit_event)
        db.commit()
        db.refresh(record)

    except IntegrityError:
        db.rollback()

        # Handle a concurrent retry of the same reference ID.
        try:
            existing = existing_by_reference(
                db, payload.reference_id
            )
        except SQLAlchemyError:
            logger.exception("Retry lookup failed")
            raise HTTPException(
                status_code=503,
                detail="Prediction storage unavailable",
            )

        if existing is not None:
            if not same_request(existing, payload, amount):
                raise HTTPException(
                    status_code=409,
                    detail="Reference ID conflicts with existing prediction",
                )

            response.status_code = 200
            return serialize_record(existing)

        logger.exception("Prediction insert integrity failure")
        raise HTTPException(
            status_code=503,
            detail="Prediction could not be recorded",
        )

    except SQLAlchemyError:
        db.rollback()
        logger.exception("Prediction transaction rolled back")
        raise HTTPException(
            status_code=503,
            detail="Prediction could not be recorded",
        )

    return serialize_record(record)


@router.get(
    "/predictions/{prediction_id}",
    response_model=StoredShadowResponse,
)
def get_recorded_prediction(
    prediction_id: uuid.UUID,
    db: DbSession,
):
    try:
        record = db.get(
            ShadowPrediction, prediction_id
        )
    except SQLAlchemyError:
        logger.exception("Prediction lookup failed")
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Prediction storage unavailable",
        )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Shadow prediction not found",
        )

    return serialize_record(record)


class ShadowPredictionListResponse(BaseModel):
    items: list[StoredShadowResponse]
    total: int
    limit: int
    offset: int


class ShadowSummaryResponse(BaseModel):
    total_predictions: int
    review_count: int
    no_review_count: int
    review_rate: float
    average_score: float | None


@router.get(
    "/predictions",
    response_model=ShadowPredictionListResponse,
)
def list_shadow_predictions(
    db: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    decision: str | None = Query(default=None),
):
    if decision not in (None, "REVIEW", "NO_REVIEW"):
        raise HTTPException(
            status_code=422,
            detail="Invalid decision filter",
        )

    filters = []

    if decision is not None:
        filters.append(
            ShadowPrediction.decision == decision
        )

    count_query = (
        select(func.count())
        .select_from(ShadowPrediction)
        .where(*filters)
    )

    records_query = (
        select(ShadowPrediction)
        .where(*filters)
        .order_by(
            ShadowPrediction.created_at.desc(),
            ShadowPrediction.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )

    try:
        total = int(db.scalar(count_query) or 0)
        records = db.scalars(records_query).all()
    except SQLAlchemyError:
        logger.exception("Prediction history lookup failed")
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Prediction history unavailable",
        )

    return {
        "items": [
            serialize_record(record)
            for record in records
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/summary",
    response_model=ShadowSummaryResponse,
)
def get_shadow_summary(db: DbSession):
    try:
        total = int(
            db.scalar(
                select(func.count())
                .select_from(ShadowPrediction)
            ) or 0
        )

        review_count = int(
            db.scalar(
                select(func.count())
                .select_from(ShadowPrediction)
                .where(
                    ShadowPrediction.decision == "REVIEW"
                )
            ) or 0
        )

        average_score = db.scalar(
            select(
                func.avg(
                    ShadowPrediction.calibrated_score
                )
            )
        )

    except SQLAlchemyError:
        logger.exception("Shadow analytics query failed")
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Shadow analytics unavailable",
        )

    return {
        "total_predictions": total,
        "review_count": review_count,
        "no_review_count": total - review_count,
        "review_rate": (
            review_count / total if total else 0.0
        ),
        "average_score": (
            float(average_score)
            if average_score is not None
            else None
        ),
    }
