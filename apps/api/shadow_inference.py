
import hashlib
import json
import logging
import math
import os
import secrets

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
)


logger = logging.getLogger(__name__)

RELEASE_VERSION = "lightgbm-shadow-v1"
MODEL_RUN_ID = "b78db1a44df041079c5c42c1579701b2"

FEATURES = [
    "amount",
    "log_amount",
    "transaction_type",
]

TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

EXPECTED_HASHES = {
    "model.joblib":
        "44dab6f572545e0682584ef48ed4c5b183c313e7b78e44ead294ace0e1542d09",
    "calibrator.joblib":
        "a6363ab8c70799f110f781f186bb94e2ad5ba4c9a8e98402d1efc160f051b6f9",
    "review_policy.json":
        "48ab006525fcbd1805cfd7bf0bfa2928b56b29cec95b382381f5c04efc220396",
}

EXPECTED_SOURCE_SHA = (
    "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"
)

router = APIRouter(
    prefix="/shadow",
    tags=["Shadow Fraud Detection"],
)


class ShadowPredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    source_dataset: Literal["PAYSIM"]
    amount_unit: Literal["PAYSIM_SYNTHETIC_UNITS"]

    amount: Annotated[
        FiniteFloat,
        Field(ge=0),
    ]

    transaction_type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]

    reference_id: str | None = Field(
        default=None,
        max_length=100,
    )


@dataclass(frozen=True)
class ShadowEngine:
    model: object
    calibrator: object
    threshold: float

    def predict(self, request: ShadowPredictionRequest):
        amount = float(request.amount)

        frame = pd.DataFrame(
            [{
                "amount": amount,
                "log_amount": math.log1p(amount),
                "transaction_type": request.transaction_type,
            }],
            columns=FEATURES,
        )

        raw_score = float(
            self.model.predict_proba(frame)[0, 1]
        )

        if not math.isfinite(raw_score):
            raise ValueError("Non-finite model score")

        if not 0 <= raw_score <= 1:
            raise ValueError("Model score outside [0, 1]")

        clipped = np.clip(
            raw_score,
            1e-7,
            1 - 1e-7,
        )

        logit = np.log(
            clipped / (1 - clipped)
        ).reshape(1, 1)

        calibrated_score = float(
            self.calibrator.predict_proba(logit)[0, 1]
        )

        if (
            not math.isfinite(calibrated_score)
            or not 0 <= calibrated_score <= 1
        ):
            raise ValueError("Invalid calibrated score")

        decision = (
            "REVIEW"
            if calibrated_score >= self.threshold
            else "NO_REVIEW"
        )

        return {
            "reference_id": request.reference_id,
            "model_version": RELEASE_VERSION,
            "model_run_id": MODEL_RUN_ID,
            "deployment_mode": "SHADOW_ONLY",
            "calibrated_score": calibrated_score,
            "review_threshold": self.threshold,
            "decision": decision,
            "financial_action_taken": False,
            "warning": (
                "Synthetic PaySim risk score only. "
                "Not a validated real-world fraud probability."
            ),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_release(root: Path):
    manifest_file = root / "release_manifest.json"

    manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )

    if manifest.get("release_version") != RELEASE_VERSION:
        raise ValueError("Release version mismatch")

    if manifest.get("model_run_id") != MODEL_RUN_ID:
        raise ValueError("Model run mismatch")

    if manifest.get("deployment_mode") != "SHADOW_ONLY":
        raise ValueError("Unsafe release mode")

    if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA:
        raise ValueError("Source lineage mismatch")

    contract = manifest["inference_contract"]

    if contract["feature_columns"] != FEATURES:
        raise ValueError("Feature contract mismatch")

    if contract["amount_unit"] != "PAYSIM_SYNTHETIC_UNITS":
        raise ValueError("Amount unit mismatch")

    if contract["allowed_transaction_types"] != TYPES:
        raise ValueError("Transaction type mismatch")

    if contract["derived_features"].get(
        "log_amount"
    ) != "log1p(amount)":
        raise ValueError("Feature transformation mismatch")

    if manifest["review_policy"]["action"] != "REVIEW_ONLY":
        raise ValueError("Unsafe review policy")

    # Check actual files against hashes pinned in source code.
    for filename, expected in EXPECTED_HASHES.items():
        path = root / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        if manifest["artifact_sha256"][filename] != expected:
            raise ValueError(
                f"Manifest hash mismatch: {filename}"
            )

        if sha256_file(path) != expected:
            raise ValueError(
                f"Artifact integrity failure: {filename}"
            )

    policy = json.loads(
        (root / "review_policy.json").read_text(
            encoding="utf-8"
        )
    )

    threshold = float(policy["review_score_threshold"])

    if (
        policy["model_run_id"] != MODEL_RUN_ID
        or policy["source_sha256"] != EXPECTED_SOURCE_SHA
        or policy["action"] != "REVIEW_ONLY"
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or threshold != manifest["review_policy"]["threshold"]
    ):
        raise ValueError("Frozen policy validation failed")

    return threshold


@lru_cache(maxsize=1)
def load_release() -> ShadowEngine:
    release_dir = os.environ.get(
        "FRAUDGUARD_RELEASE_DIR",
        "",
    )

    if not release_dir:
        raise RuntimeError("Release path not configured")

    root = Path(release_dir)

    # Always verify before deserializing a trusted bundle.
    threshold = verify_release(root)

    model = joblib.load(root / "model.joblib")
    calibrator = joblib.load(root / "calibrator.joblib")

    logger.info(
        "Loaded shadow model release %s",
        RELEASE_VERSION,
    )

    return ShadowEngine(
        model=model,
        calibrator=calibrator,
        threshold=threshold,
    )


def require_shadow_key(
    x_shadow_api_key: Annotated[
        str | None,
        Header(alias="X-Shadow-API-Key"),
    ] = None,
):
    configured = os.environ.get(
        "FRAUDGUARD_SHADOW_API_KEY",
        "",
    )

    if not configured:
        raise HTTPException(
            status_code=503,
            detail="Shadow inference is not configured",
        )

    if (
        x_shadow_api_key is None
        or not secrets.compare_digest(
            x_shadow_api_key,
            configured,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid shadow API key",
        )


@router.get(
    "/model",
    dependencies=[Depends(require_shadow_key)],
)
def model_information():
    try:
        engine = load_release()
    except Exception:
        logger.exception("Shadow model unavailable")
        raise HTTPException(
            status_code=503,
            detail="Shadow model unavailable",
        )

    return {
        "version": RELEASE_VERSION,
        "run_id": MODEL_RUN_ID,
        "mode": "SHADOW_ONLY",
        "threshold": engine.threshold,
        "amount_unit": "PAYSIM_SYNTHETIC_UNITS",
        "features": FEATURES,
        "loaded": True,
    }


@router.post(
    "/predict",
    dependencies=[Depends(require_shadow_key)],
)
def predict_shadow(request: ShadowPredictionRequest):
    try:
        engine = load_release()
        return engine.predict(request)
    except Exception:
        logger.exception("Shadow inference failure")
        raise HTTPException(
            status_code=503,
            detail="Shadow inference unavailable",
        )
