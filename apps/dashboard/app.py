from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from uuid import UUID, uuid4

import streamlit as st

from api_client import (
    APIClientError,
    create_transaction,
    get_health,
    get_readiness,
    get_transaction,
    list_transactions,
)


# ==================================================
# Application configuration
# ==================================================

st.set_page_config(
    page_title="FraudGuard Teja RiskOps",
    page_icon="🛡️",
    layout="wide",
)


# ==================================================
# Sidebar navigation
# ==================================================

with st.sidebar:
    st.title("🛡️ FraudGuard")
    st.caption("Risk Operations Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "System Overview",
            "Submit Transaction",
            "Transaction History",
        ],
    )

    st.divider()
    st.caption("Environment: Local Development")
    st.caption("Version: 0.3.0")
    st.warning("Synthetic transactions only.")


st.title("FraudGuard RiskOps")
st.caption("Financial Transaction Management Platform")
st.divider()


# ==================================================
# Page 1: System overview
# ==================================================

def show_overview():
    st.header("System Overview")
    st.write("Live status from the FastAPI backend.")

    api_ok = False
    db_ok = False

    try:
        health = get_health()
        api_ok = health.get("status") == "healthy"
    except APIClientError as error:
        st.error(f"API health check failed: {error}")

    try:
        readiness = get_readiness()
        db_ok = readiness.get("database") == "connected"
    except APIClientError as error:
        st.error(f"Database readiness failed: {error}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "FastAPI Backend",
            "Healthy" if api_ok else "Unavailable",
        )

    with col2:
        st.metric(
            "PostgreSQL Database",
            "Connected" if db_ok else "Unavailable",
        )

    st.divider()
    st.subheader("Transaction Overview")

    try:
        result = list_transactions(limit=5)

        st.metric(
            "Total Stored Transactions",
            f"{result['total']:,}",
        )

        st.write("Recent Transactions")

        rows = result.get("items", [])

        if rows:
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No transactions have been submitted.")

    except APIClientError as error:
        st.error(f"Unable to retrieve transactions: {error}")

    if st.button("Refresh Overview"):
        st.rerun()


# ==================================================
# Page 2: Submit transaction
# ==================================================

def show_submit_transaction():
    st.header("Submit Financial Transaction")

    st.info(
        "This creates a synthetic transaction record. "
        "No payment is processed and no fraud score "
        "is generated at this stage."
    )

    if "external_id" not in st.session_state:
        st.session_state.external_id = (
            f"FG-UI-{uuid4().hex.upper()}"
        )

    if st.button("Generate New Transaction ID"):
        st.session_state.external_id = (
            f"FG-UI-{uuid4().hex.upper()}"
        )

    current_utc = datetime.now(timezone.utc)

    with st.form("transaction_form"):

        external_id = st.text_input(
            "External Transaction ID",
            key="external_id",
        )

        amount = st.text_input(
            "Transaction Amount (USD)",
            value="250.00",
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

        col1, col2 = st.columns(2)

        with col1:
            sender_id = st.text_input(
                "Sender Account ID",
                value="CUST-1001",
            )

        with col2:
            recipient_id = st.text_input(
                "Recipient Account ID",
                value="CUST-2001",
            )

        st.write("Transaction Timestamp (UTC)")

        date_col, time_col = st.columns(2)

        with date_col:
            txn_date = st.date_input(
                "Date",
                value=current_utc.date(),
            )

        with time_col:
            txn_time = st.time_input(
                "Time",
                value=current_utc.time().replace(
                    second=0,
                    microsecond=0,
                ),
            )

        submitted = st.form_submit_button(
            "Submit Transaction",
            type="primary",
        )

    if submitted:
        try:
            parsed_amount = Decimal(amount.strip())

            if (
                not parsed_amount.is_finite()
                or parsed_amount <= 0
            ):
                raise InvalidOperation

        except (InvalidOperation, ValueError):
            st.error(
                "Amount must be a valid positive number."
            )
            return

        occurred_at = datetime.combine(
            txn_date,
            txn_time,
            tzinfo=timezone.utc,
        ).isoformat()

        payload = {
            "external_transaction_id": external_id,
            "amount": str(parsed_amount),
            "currency": "USD",
            "transaction_type": transaction_type,
            "sender_account_id": sender_id,
            "recipient_account_id": recipient_id,
            "occurred_at": occurred_at,
        }

        try:
            result, status_code = create_transaction(
                payload
            )

            if status_code == 201:
                st.success(
                    "Transaction created successfully."
                )

            elif status_code == 200:
                st.info(
                    "This transaction already exists. "
                    "The original record was returned."
                )

            st.subheader("Saved Transaction")
            st.json(result)

        except APIClientError as error:
            if error.status_code == 409:
                st.error(
                    "Duplicate transaction ID with "
                    "conflicting data."
                )

            elif error.status_code == 422:
                st.error(
                    f"Transaction validation failed: {error}"
                )

            else:
                st.error(
                    f"Transaction submission failed: {error}"
                )


# ==================================================
# Page 3: Transaction history
# ==================================================

def show_transaction_history():
    st.header("Transaction History")

    col1, col2 = st.columns(2)

    with col1:
        page_size = st.selectbox(
            "Records per page",
            [10, 20, 50, 100],
            index=1,
        )

    with col2:
        page_number = st.number_input(
            "Page number",
            min_value=1,
            value=1,
            step=1,
        )

    offset = (int(page_number) - 1) * page_size

    try:
        result = list_transactions(
            limit=page_size,
            offset=offset,
        )

        total = result["total"]
        rows = result["items"]

        st.caption(
            f"Total records: {total} | "
            f"Current page: {page_number}"
        )

        if rows:
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No records found for this page.")

    except APIClientError as error:
        st.error(f"Unable to load history: {error}")

    st.divider()
    st.subheader("Find a Transaction by UUID")

    with st.form("lookup_form"):
        transaction_id = st.text_input(
            "Transaction UUID",
            placeholder="Paste the UUID returned by FastAPI",
        )

        search_clicked = st.form_submit_button(
            "Search Transaction"
        )

    if search_clicked:
        try:
            normalized_id = str(
                UUID(transaction_id.strip())
            )
        except ValueError:
            st.error("Enter a valid transaction UUID.")
            return

        try:
            transaction = get_transaction(normalized_id)

            st.success("Transaction found.")
            st.json(transaction)

        except APIClientError as error:
            if error.status_code == 404:
                st.warning("Transaction not found.")
            else:
                st.error(f"Lookup failed: {error}")


# ==================================================
# Page routing
# ==================================================

if page == "System Overview":
    show_overview()

elif page == "Submit Transaction":
    show_submit_transaction()

elif page == "Transaction History":
    show_transaction_history()


st.divider()
st.caption(
    "FraudGuard RiskOps | "
    "Local Development | Synthetic Financial Data"
)