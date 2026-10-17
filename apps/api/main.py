import os

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError


# -------------------------------------------------
# Database configuration
# -------------------------------------------------

database_url = URL.create(
    drivername="postgresql+psycopg",
    username=os.environ.get("POSTGRES_USER", "fraudguard"),
    password=os.environ.get("POSTGRES_PASSWORD", ""),
    host=os.environ.get("POSTGRES_HOST", "db"),
    port=5432,
    database=os.environ.get("POSTGRES_DB", "fraudguard"),
)

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
)


# -------------------------------------------------
# FastAPI application
# -------------------------------------------------

app = FastAPI(
    title="FraudGuard RiskOps API",
    description="Enterprise Fraud Detection and Risk Operations",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "application": "FraudGuard RiskOps",
        "version": "0.1.0",
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