import os
from typing import Any

import requests


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

REQUEST_TIMEOUT = 5


class APIClientError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code


def request_json(
    method: str,
    path: str,
    *,
    payload: dict | None = None,
    params: dict | None = None,
) -> tuple[Any, int]:

    try:
        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{path}",
            json=payload,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as error:
        raise APIClientError(
            "Cannot communicate with FastAPI backend."
        ) from error

    try:
        data = response.json()
    except ValueError as error:
        raise APIClientError(
            "Backend returned an invalid JSON response.",
            response.status_code,
        ) from error

    if not response.ok:
        detail = (
            data.get("detail", "Unknown API error")
            if isinstance(data, dict)
            else "Unexpected API response"
        )

        if isinstance(detail, list):
            detail = "; ".join(
                str(item.get("msg", "Invalid field"))
                for item in detail
            )

        raise APIClientError(
            str(detail),
            response.status_code,
        )

    return data, response.status_code


def get_health():
    return request_json("GET", "/health")[0]


def get_readiness():
    return request_json("GET", "/ready")[0]


def create_transaction(payload: dict):
    return request_json(
        "POST",
        "/transactions",
        payload=payload,
    )


def list_transactions(
    limit: int = 20,
    offset: int = 0,
):
    return request_json(
        "GET",
        "/transactions",
        params={
            "limit": limit,
            "offset": offset,
        },
    )[0]


def get_transaction(transaction_id: str):
    return request_json(
        "GET",
        f"/transactions/{transaction_id}",
    )[0]