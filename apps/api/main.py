from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database import engine
from routers.transactions import router as transactions_router


app = FastAPI(
    title="FraudGuard RiskOps API",
    description="Enterprise Fraud Detection and Risk Operations",
    version="0.2.0",
)


# Register API routers
app.include_router(transactions_router)

from shadow_inference import router as shadow_router

app.include_router(shadow_router)

from routers.shadow_records import router as shadow_records_router

app.include_router(shadow_records_router)


@app.get("/")
def root():
    return {
        "application": "FraudGuard RiskOps",
        "version": "0.2.0",
        "message": "Fraud detection backend is running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "fraudguard-api",
    }


@app.get("/ready")
def readiness_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "ready",
            "database": "connected",
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )