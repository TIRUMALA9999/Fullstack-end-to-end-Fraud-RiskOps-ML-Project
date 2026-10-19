
import argparse
import hashlib
import json
import math
import shutil
import tempfile

from datetime import datetime, timezone
from pathlib import Path


FEATURES = [
    "amount",
    "log_amount",
    "transaction_type",
]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def build_release(
    model_file,
    calibrator_file,
    policy_file,
    holdout_file,
    output_dir,
    run_id,
):
    model_file = Path(model_file)
    calibrator_file = Path(calibrator_file)
    policy_file = Path(policy_file)
    holdout_file = Path(holdout_file)
    output_dir = Path(output_dir)

    if output_dir.exists():
        raise FileExistsError(
            "Release already exists. "
            "Create a new version instead of overwriting."
        )

    report = json.loads(
        holdout_file.read_text(encoding="utf-8")
    )

    policy = json.loads(
        policy_file.read_text(encoding="utf-8")
    )

    if report.get("evaluation") != "FINAL_HOLDOUT_V1":
        raise ValueError("Invalid holdout evaluation report")

    if report.get("decision") != (
        "EVALUATION_ONLY_NOT_DEPLOYMENT_APPROVAL"
    ):
        raise ValueError("Unexpected evaluation status")

    if report.get("model_run_id") != run_id:
        raise ValueError("Holdout model run mismatch")

    if policy.get("model_run_id") != run_id:
        raise ValueError("Policy model run mismatch")

    if policy.get("source_sha256") != report.get(
        "source_sha256"
    ):
        raise ValueError("Dataset lineage mismatch")

    if policy.get("action") != "REVIEW_ONLY":
        raise ValueError("Unsafe policy action")

    threshold = float(policy["review_score_threshold"])

    if not math.isfinite(threshold):
        raise ValueError("Invalid policy threshold")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold outside [0, 1]")

    if not math.isclose(
        threshold,
        float(report["frozen_review_threshold"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Frozen threshold mismatch")

    artifacts = {
        "model.joblib": (
            model_file,
            "model_sha256",
        ),
        "calibrator.joblib": (
            calibrator_file,
            "calibrator_sha256",
        ),
        "review_policy.json": (
            policy_file,
            "policy_sha256",
        ),
    }

    checksums = {}

    # Verify all source files BEFORE creating the release.
    for filename, (source, report_key) in artifacts.items():
        if not source.is_file():
            raise FileNotFoundError(source)

        actual = sha256_file(source)
        expected = report[report_key]

        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {filename}"
            )

        checksums[filename] = actual

    manifest = {
        "release_version": "lightgbm-shadow-v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_type": "LightGBM",
        "model_run_id": run_id,
        "deployment_mode": "SHADOW_ONLY",
        "source_dataset": "PaySim",
        "source_sha256": report["source_sha256"],
        "inference_contract": {
            "feature_columns": FEATURES,
            "derived_features": {
                "log_amount": "log1p(amount)",
            },
            "amount_unit": "PAYSIM_SYNTHETIC_UNITS",
            "allowed_transaction_types": TRANSACTION_TYPES,
            "label_excluded": "is_fraud",
            "source_step_excluded": True,
        },
        "review_policy": {
            "action": "REVIEW_ONLY",
            "threshold": threshold,
            "target_review_fraction": policy[
                "target_review_fraction"
            ],
        },
        "holdout_reference": {
            "report_sha256": sha256_file(
                holdout_file
            ),
            "test_rows": report["test_rows"],
            "test_fraud": report["test_fraud"],
        },
        "artifact_sha256": checksums,
    }

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".model-release-",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()

        for filename, (source, _) in artifacts.items():
            destination = staging / filename

            shutil.copy2(source, destination)

            if sha256_file(destination) != checksums[
                filename
            ]:
                raise ValueError(
                    "Copied artifact checksum mismatch"
                )

        (staging / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        if output_dir.exists():
            raise FileExistsError(output_dir)

        staging.rename(output_dir)

    print(
        json.dumps({
            "status": "RELEASE_PACKAGED",
            "release": str(output_dir),
            "deployment_mode": "SHADOW_ONLY",
            "model_run_id": run_id,
            "artifact_sha256": checksums,
        }, indent=2),
        flush=True,
    )

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Package frozen FraudGuard model"
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)

    args = parser.parse_args()

    build_release(
        model_file=args.model,
        calibrator_file=args.calibrator,
        policy_file=args.policy,
        holdout_file=args.holdout,
        output_dir=args.output,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
