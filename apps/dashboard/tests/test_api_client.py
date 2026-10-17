import unittest
from unittest.mock import Mock, patch

import requests

from api_client import (
    APIClientError,
    create_transaction,
    get_health,
    get_transaction,
    list_transactions,
)


def fake_response(status_code, payload):
    response = Mock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    response.json.return_value = payload
    return response


class TestAPIClient(unittest.TestCase):

    @patch("api_client.requests.request")
    def test_health_check(self, mock_request):
        mock_request.return_value = fake_response(
            200, {"status": "healthy"}
        )

        result = get_health()

        self.assertEqual(result["status"], "healthy")

    @patch("api_client.requests.request")
    def test_create_transaction(self, mock_request):
        mock_request.return_value = fake_response(
            201, {"id": "test-id"}
        )

        result, status_code = create_transaction({
            "external_transaction_id": "FG-TEST-1"
        })

        self.assertEqual(status_code, 201)
        self.assertEqual(result["id"], "test-id")

    @patch("api_client.requests.request")
    def test_idempotent_retry(self, mock_request):
        mock_request.return_value = fake_response(
            200, {"id": "existing-id"}
        )

        result, status_code = create_transaction({
            "external_transaction_id": "FG-TEST-1"
        })

        self.assertEqual(status_code, 200)
        self.assertEqual(result["id"], "existing-id")

    @patch("api_client.requests.request")
    def test_conflict_error(self, mock_request):
        mock_request.return_value = fake_response(
            409, {"detail": "Transaction ID conflict"}
        )

        with self.assertRaises(APIClientError) as caught:
            create_transaction({"external_transaction_id": "X"})

        self.assertEqual(caught.exception.status_code, 409)

    @patch("api_client.requests.request")
    def test_validation_error(self, mock_request):
        mock_request.return_value = fake_response(
            422,
            {"detail": [{"msg": "Value must be positive"}]},
        )

        with self.assertRaises(APIClientError) as caught:
            create_transaction({"amount": "-100"})

        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("positive", str(caught.exception))

    @patch("api_client.requests.request")
    def test_backend_timeout(self, mock_request):
        mock_request.side_effect = requests.Timeout()

        with self.assertRaises(APIClientError):
            get_health()

    @patch("api_client.requests.request")
    def test_history_pagination(self, mock_request):
        mock_request.return_value = fake_response(
            200, {"items": [], "total": 0}
        )

        result = list_transactions(limit=10, offset=20)

        self.assertEqual(result["total"], 0)

        kwargs = mock_request.call_args.kwargs
        self.assertEqual(
            kwargs["params"],
            {"limit": 10, "offset": 20},
        )


if __name__ == "__main__":
    unittest.main()