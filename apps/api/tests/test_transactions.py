from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import engine, get_db
from main import app
from models import AuditEvent


@pytest.fixture
def isolated_client():
    connection = engine.connect()
    outer_transaction = connection.begin()

    def override_get_db():
        with Session(
            bind=connection,
            join_transaction_mode="create_savepoint",
        ) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client, connection
    finally:
        app.dependency_overrides.pop(get_db, None)
        outer_transaction.rollback()
        connection.close()


def sample_payload():
    return {
        "external_transaction_id": f"TEST-{uuid4().hex}",
        "amount": "245.50",
        "currency": "USD",
        "transaction_type": "TRANSFER",
        "sender_account_id": "CUST-TEST-001",
        "recipient_account_id": "CUST-TEST-002",
        "occurred_at": "2026-10-17T12:00:00Z",
    }


def test_create_transaction_and_audit(isolated_client):
    client, connection = isolated_client

    response = client.post(
        "/transactions",
        json=sample_payload(),
    )

    assert response.status_code == 201
    result = response.json()

    assert result["amount"] == "245.50"
    assert UUID(result["id"])

    audit_count = connection.scalar(
        select(func.count())
        .select_from(AuditEvent)
        .where(
            AuditEvent.transaction_id == UUID(result["id"])
        )
    )

    assert audit_count == 1


def test_idempotent_retry(isolated_client):
    client, connection = isolated_client
    payload = sample_payload()

    first = client.post("/transactions", json=payload)
    second = client.post("/transactions", json=payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]

    audit_count = connection.scalar(
        select(func.count())
        .select_from(AuditEvent)
        .where(
            AuditEvent.transaction_id
            == UUID(first.json()["id"])
        )
    )

    assert audit_count == 1


def test_conflicting_duplicate(isolated_client):
    client, _ = isolated_client
    payload = sample_payload()

    first = client.post("/transactions", json=payload)
    assert first.status_code == 201

    payload["amount"] = "999.00"

    second = client.post("/transactions", json=payload)

    assert second.status_code == 409


@pytest.mark.parametrize(
    "changes",
    [
        {"amount": "-10.00"},
        {"transaction_type": "UNKNOWN"},
        {"occurred_at": "2026-10-17T12:00:00"},
    ],
)
def test_invalid_transaction(isolated_client, changes):
    client, _ = isolated_client
    payload = sample_payload()
    payload.update(changes)

    response = client.post("/transactions", json=payload)

    assert response.status_code == 422


def test_get_transaction(isolated_client):
    client, _ = isolated_client

    created = client.post(
        "/transactions",
        json=sample_payload(),
    )

    transaction_id = created.json()["id"]

    response = client.get(
        f"/transactions/{transaction_id}"
    )

    assert response.status_code == 200
    assert response.json()["id"] == transaction_id


def test_transaction_not_found(isolated_client):
    client, _ = isolated_client

    response = client.get(f"/transactions/{uuid4()}")

    assert response.status_code == 404


def test_list_transactions(isolated_client):
    client, _ = isolated_client

    client.post("/transactions", json=sample_payload())
    client.post("/transactions", json=sample_payload())

    response = client.get(
        "/transactions?limit=1&offset=0"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) == 1
    assert data["total"] >= 2
    assert data["limit"] == 1
    assert data["offset"] == 0


def test_invalid_pagination(isolated_client):
    client, _ = isolated_client

    response = client.get("/transactions?limit=101")

    assert response.status_code == 422