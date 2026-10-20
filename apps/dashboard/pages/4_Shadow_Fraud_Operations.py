import json
import os
import uuid

from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


API_URL = os.getenv(
    "FRAUDGUARD_API_URL",
    "http://api:8000",
).rstrip("/")

API_KEY = os.getenv(
    "FRAUDGUARD_SHADOW_API_KEY",
    "",
)


def api_request(method, path, payload=None, params=None):
    if not API_KEY:
        raise RuntimeError(
            "Dashboard shadow API key is not configured."
        )

    url = API_URL + path

    if params:
        url += "?" + urlencode(params)

    headers = {
        "X-Shadow-API-Key": API_KEY,
        "Accept": "application/json",
    }

    data = None

    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    except HTTPError as exc:
        try:
            error_body = json.loads(
                exc.read().decode("utf-8")
            )
            detail = error_body.get(
                "detail", "API request failed"
            )
        except (ValueError, UnicodeDecodeError):
            detail = "API request failed"

        raise RuntimeError(
            f"API returned {exc.code}: {detail}"
        ) from None

    except URLError:
        raise RuntimeError(
            "Cannot connect to the FraudGuard API."
        ) from None


def new_reference():
    return "shadow-ui-" + uuid.uuid4().hex[:16]


st.set_page_config(
    page_title="FraudGuard | Shadow Operations",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudGuard — Shadow Risk Operations")

st.caption(
    "Synthetic PaySim transaction scoring using the "
    "frozen LightGBM model. No financial actions are taken."
)

if not API_KEY:
    st.error(
        "FRAUDGUARD_SHADOW_API_KEY is missing "
        "from the Streamlit container."
    )
    st.stop()

if "shadow_next_reference" not in st.session_state:
    st.session_state["shadow_next_reference"] = (
        new_reference()
    )

if "shadow_last_result" not in st.session_state:
    st.session_state["shadow_last_result"] = None


# --------------------------------------------------
# Model information
# --------------------------------------------------

st.subheader("Active Model")

try:
    model_info = api_request(
        "GET", "/shadow/model"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model Version",
        model_info["version"],
    )
    c2.metric(
        "Deployment Mode",
        model_info["mode"],
    )
    c3.metric(
        "Review Threshold",
        f"{model_info['threshold']:.6f}",
    )

except RuntimeError as exc:
    st.error(str(exc))
    st.stop()


# --------------------------------------------------
# Transaction scoring
# --------------------------------------------------

st.divider()
st.subheader("Score a Synthetic Transaction")

with st.form("shadow_transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Amount (PaySim synthetic units)",
            min_value=0.01,
            max_value=1_000_000_000.0,
            value=150_000.0,
            step=100.0,
            format="%.2f",
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "TRANSFER",
                "CASH_OUT",
                "CASH_IN",
                "PAYMENT",
                "DEBIT",
            ],
        )

    with col2:
        reference_id = st.text_input(
            "Unique Reference ID",
            value=st.session_state[
                "shadow_next_reference"
            ],
            max_chars=100,
        )

        st.info(
            "Repeated submissions with the same reference "
            "and unchanged input return the existing record."
        )

    submitted = st.form_submit_button(
        "Score and Save Prediction",
        type="primary",
        use_container_width=True,
    )


if submitted:
    request_payload = {
        "source_dataset": "PAYSIM",
        "amount_unit": "PAYSIM_SYNTHETIC_UNITS",
        "amount": float(amount),
        "transaction_type": transaction_type,
        "reference_id": reference_id,
    }

    try:
        result = api_request(
            "POST",
            "/shadow/predict-and-record",
            payload=request_payload,
        )

        st.session_state[
            "shadow_last_result"
        ] = result

        st.session_state[
            "shadow_next_reference"
        ] = new_reference()

        st.rerun()

    except RuntimeError as exc:
        st.error(str(exc))


last_result = st.session_state[
    "shadow_last_result"
]

if last_result is not None:
    st.subheader("Most Recent Saved Prediction")

    score = float(last_result["calibrated_score"])
    threshold = float(last_result["review_threshold"])

    r1, r2, r3 = st.columns(3)

    r1.metric("Calibrated Score", f"{score:.6f}")
    r2.metric("Review Threshold", f"{threshold:.6f}")
    r3.metric("Decision", last_result["decision"])

    if last_result["decision"] == "REVIEW":
        st.warning(
            "This synthetic transaction was flagged "
            "for analyst review."
        )
    else:
        st.success(
            "This synthetic transaction was not flagged "
            "by the frozen review policy."
        )

    st.caption(
        "The score is not a validated real-world "
        "fraud probability."
    )

    with st.expander("View saved prediction details"):
        st.json(last_result)


# --------------------------------------------------
# Portfolio analytics
# --------------------------------------------------

st.divider()
st.subheader("Shadow Prediction Analytics")

try:
    summary = api_request(
        "GET",
        "/shadow/summary",
    )

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Total Predictions",
        f"{summary['total_predictions']:,}",
    )

    a2.metric(
        "Flagged for Review",
        f"{summary['review_count']:,}",
    )

    a3.metric(
        "No Review",
        f"{summary['no_review_count']:,}",
    )

    a4.metric(
        "Observed Review Rate",
        f"{summary['review_rate']:.2%}",
    )

    if summary["total_predictions"] > 0:
        chart_data = pd.DataFrame(
            {
                "Predictions": [
                    summary["review_count"],
                    summary["no_review_count"],
                ],
            },
            index=["REVIEW", "NO_REVIEW"],
        )

        st.bar_chart(chart_data)

        if summary["average_score"] is not None:
            st.caption(
                "Mean model score across saved predictions: "
                f"{summary['average_score']:.6f}"
            )

except RuntimeError as exc:
    st.error(str(exc))


# --------------------------------------------------
# Prediction history
# --------------------------------------------------

st.divider()
st.subheader("Saved Prediction History")

h1, h2, h3 = st.columns(3)

with h1:
    decision_filter = st.selectbox(
        "Decision Filter",
        ["ALL", "REVIEW", "NO_REVIEW"],
    )

with h2:
    page_size = st.selectbox(
        "Records per Page",
        [10, 20, 50, 100],
        index=1,
    )

with h3:
    page = st.number_input(
        "Page Number",
        min_value=1,
        value=1,
        step=1,
    )

params = {
    "limit": page_size,
    "offset": (page - 1) * page_size,
}

if decision_filter != "ALL":
    params["decision"] = decision_filter

try:
    history = api_request(
        "GET",
        "/shadow/predictions",
        params=params,
    )

    st.caption(
        f"Showing up to {page_size} records "
        f"out of {history['total']:,} matching predictions."
    )

    if history["items"]:
        rows = []

        for item in history["items"]:
            rows.append({
                "Reference ID": item["reference_id"],
                "Created At": item["created_at"],
                "Type": item["transaction_type"],
                "Amount": item["amount"],
                "Score": item["calibrated_score"],
                "Decision": item["decision"],
                "Model Version": item["model_version"],
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Inspect raw prediction records"
        ):
            st.json(history["items"])

    else:
        st.info(
            "No saved predictions match these filters."
        )

except RuntimeError as exc:
    st.error(str(exc))


st.divider()
st.caption(
    "FraudGuard RiskOps — Local synthetic-data shadow "
    "deployment. Never use these predictions to "
    "block real transactions."
)