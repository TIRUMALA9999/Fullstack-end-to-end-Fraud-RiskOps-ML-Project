from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
)


TransactionType = Literal[
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "PAYMENT",
    "DEBIT",
]


class TransactionCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    external_transaction_id: str = Field(
        min_length=1,
        max_length=128,
    )

    amount: Decimal = Field(
        gt=Decimal("0"),
        max_digits=18,
        decimal_places=2,
    )

    currency: Literal["USD"] = "USD"

    transaction_type: TransactionType

    sender_account_id: str = Field(
        min_length=1,
        max_length=128,
    )

    recipient_account_id: str = Field(
        min_length=1,
        max_length=128,
    )

    occurred_at: AwareDatetime


class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    external_transaction_id: str
    amount: Decimal
    currency: str
    transaction_type: str
    sender_account_id: str
    recipient_account_id: str
    occurred_at: datetime
    created_at: datetime


class TransactionListResponse(BaseModel):
    items: list[TransactionResponse]
    total: int
    limit: int
    offset: int