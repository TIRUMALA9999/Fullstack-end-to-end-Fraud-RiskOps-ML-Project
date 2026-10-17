from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db
from models import AuditEvent, Transaction
from schemas import (
    TransactionCreate,
    TransactionListResponse,
    TransactionResponse,
)


router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"],
)

DbSession = Annotated[Session, Depends(get_db)]


# --------------------------------------------------
# Helper: Compare idempotent requests
# --------------------------------------------------

def same_transaction(
    existing: Transaction,
    payload: TransactionCreate,
) -> bool:
    fields = (
        "external_transaction_id",
        "amount",
        "currency",
        "transaction_type",
        "sender_account_id",
        "recipient_account_id",
        "occurred_at",
    )

    return all(
        getattr(existing, field) == getattr(payload, field)
        for field in fields
    )


# --------------------------------------------------
# POST /transactions
# --------------------------------------------------

@router.post(
    "",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_transaction(
    payload: TransactionCreate,
    response: Response,
    db: DbSession,
):
    transaction = Transaction(**payload.model_dump())

    try:
        db.add(transaction)
        db.flush()

        # Transaction and audit event are saved atomically.
        audit_event = AuditEvent(
            transaction_id=transaction.id,
            event_type="TRANSACTION_CREATED",
            actor_id="system:api",
            details={"source": "transaction_api"},
        )

        db.add(audit_event)
        db.commit()
        db.refresh(transaction)

        return transaction

    except IntegrityError:
        db.rollback()

        existing = db.scalar(
            select(Transaction).where(
                Transaction.external_transaction_id
                == payload.external_transaction_id
            )
        )

        if existing and same_transaction(existing, payload):
            # Safe retry: return the existing record.
            response.status_code = status.HTTP_200_OK
            return existing

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Transaction ID conflicts with existing data",
        )


# --------------------------------------------------
# GET /transactions
# --------------------------------------------------

@router.get(
    "",
    response_model=TransactionListResponse,
)
def list_transactions(
    db: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    total = db.scalar(
        select(func.count()).select_from(Transaction)
    )

    statement = (
        select(Transaction)
        .order_by(
            Transaction.created_at.desc(),
            Transaction.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )

    transactions = db.scalars(statement).all()

    return {
        "items": transactions,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# --------------------------------------------------
# GET /transactions/{transaction_id}
# --------------------------------------------------

@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
)
def get_transaction(
    transaction_id: UUID,
    db: DbSession,
):
    transaction = db.get(Transaction, transaction_id)

    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found",
        )

    return transaction