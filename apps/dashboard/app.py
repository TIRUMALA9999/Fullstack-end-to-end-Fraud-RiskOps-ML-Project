import os

import requests
import streamlit as st


# -----------------------------------------------
# Configuration
# -----------------------------------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL", "http://localhost:8000"
).rstrip("/")


st.set_page_config(
    page_title="FraudGuard RiskOps",
    page_icon="🛡️",
    layout="wide",
)


# -----------------------------------------------
# API communication
# -----------------------------------------------

def fetch_api(path: str):
    try:
        response = requests.get(
            f"{API_BASE_URL}{path}",
            timeout=5,
        )
        response.raise_for_status()
        return response.json(), None

    except (requests.RequestException, ValueError) as error:
        return None, str(error)


# -----------------------------------------------
# Sidebar
# -----------------------------------------------

with st.sidebar:
    st.title("🛡️ FraudGuard")

    st.markdown("### Risk Operations Platform")
    st.divider()

    st.write("Dashboard")
    st.write("Transaction Scoring — Coming Soon")
    st.write("Fraud Investigations — Coming Soon")
    st.write("Model Monitoring — Coming Soon")

    st.divider()
    st.caption("Environment: Local Development")
    st.caption("Application Version: 0.1.0")


# -----------------------------------------------
# Main page
# -----------------------------------------------

st.title("FraudGuard RiskOps")
st.caption("Enterprise Fraud Detection & Risk Operations")

st.divider()

st.header("System Health Overview")

api_data, api_error = fetch_api("/health")
db_data, db_error = fetch_api("/ready")

api_healthy = (
    api_data is not None
    and api_data.get("status") == "healthy"
)

db_healthy = (
    db_data is not None
    and db_data.get("database") == "connected"
)


col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="FastAPI Backend",
        value="Healthy" if api_healthy else "Unhealthy",
    )

    if api_healthy:
        st.success("Backend is operational.")
    else:
        st.error("Backend health check failed.")


with col2:
    st.metric(
        label="PostgreSQL Database",
        value="Connected" if db_healthy else "Disconnected",
    )

    if db_healthy:
        st.success("Database connection is operational.")
    else:
        st.error("Database readiness check failed.")


if api_error:
    st.warning(f"API error: {api_error}")

if db_error:
    st.warning(f"Database readiness error: {db_error}")


st.divider()

st.header("Application Information")

app_info, info_error = fetch_api("/")

if app_info:
    st.json(app_info)
else:
    st.warning(
        f"Application information unavailable: {info_error}"
    )


st.divider()

if st.button("Refresh System Status"):
    st.rerun()


if api_healthy and db_healthy:
    st.success("All platform services are operational.")
else:
    st.warning("One or more platform services need attention.")


st.caption("FraudGuard RiskOps | Phase 1 - Frontend Integration")