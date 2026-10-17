import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# --------------------------------------------------
# 1. Transactions
# --------------------------------------------------

class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    external_transaction_id: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True
    )

    amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2), nullable=False
    )

    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, default="USD"
    )

    transaction_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )

    sender_account_id: Mapped[str] = mapped_column(
        String(128), nullable=False
    )

    recipient_account_id: Mapped[str] = mapped_column(
        String(128), nullable=False
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    risk_assessments: Mapped[list["RiskAssessment"]] = relationship(
        back_populates="transaction"
    )

    __table_args__ = (
        CheckConstraint(
            "amount > 0",
            name="ck_transactions_amount_positive",
        ),
        Index("ix_transactions_occurred_at", "occurred_at"),
        Index(
            "ix_transactions_sender_time",
            "sender_account_id",
            "occurred_at",
        ),
    )


# --------------------------------------------------
# 2. Risk Assessments
# --------------------------------------------------

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    transaction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transactions.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    fraud_probability: Mapped[Decimal] = mapped_column(
        Numeric(6, 5), nullable=False
    )

    decision: Mapped[str] = mapped_column(
        String(12), nullable=False
    )

    model_version: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    transaction: Mapped["Transaction"] = relationship(
        back_populates="risk_assessments"
    )

    investigation_case: Mapped["InvestigationCase | None"] = relationship(
        back_populates="risk_assessment",
        uselist=False,
    )

    __table_args__ = (
        CheckConstraint(
            "fraud_probability >= 0 AND fraud_probability <= 1",
            name="ck_risk_probability_range",
        ),
        CheckConstraint(
            "decision IN ('APPROVE', 'REVIEW', 'BLOCK')",
            name="ck_risk_decision",
        ),
    )


# --------------------------------------------------
# 3. Investigation Cases
# --------------------------------------------------

class InvestigationCase(Base):
    __tablename__ = "investigation_cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    risk_assessment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("risk_assessments.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="OPEN",
    )

    assigned_to: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )

    resolution: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    risk_assessment: Mapped["RiskAssessment"] = relationship(
        back_populates="investigation_case"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('OPEN', 'IN_REVIEW', 'RESOLVED')",
            name="ck_case_status",
        ),
        CheckConstraint(
            "resolution IS NULL OR resolution IN "
            "('FRAUD', 'LEGITIMATE', 'INCONCLUSIVE')",
            name="ck_case_resolution",
        ),
        Index("ix_cases_status", "status"),
    )


# --------------------------------------------------
# 4. Audit Events
# --------------------------------------------------

class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    transaction_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("transactions.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    actor_id: Mapped[str] = mapped_column(
        String(128), nullable=False
    )

    details: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )