
import json

import pytest

from package_release import (
    build_release,
    sha256_file,
)


def sample_artifacts(tmp_path):
    model = tmp_path / "source_model.joblib"
    calibrator = tmp_path / "source_calibrator.joblib"
    policy_file = tmp_path / "source_policy.json"
    holdout_file = tmp_path / "holdout.json"

    model.write_bytes(b"test-model")
    calibrator.write_bytes(b"test-calibrator")

    policy = {
        "model_run_id": "test-run",
        "source_sha256": "test-source",
        "action": "REVIEW_ONLY",
        "review_score_threshold": 0.05,
        "target_review_fraction": 0.01,
    }

    policy_file.write_text(json.dumps(policy))

    report = {
        "evaluation": "FINAL_HOLDOUT_V1",
        "decision": (
            "EVALUATION_ONLY_NOT_DEPLOYMENT_APPROVAL"
        ),
        "model_run_id": "test-run",
        "source_sha256": "test-source",
        "frozen_review_threshold": 0.05,
        "test_rows": 100,
        "test_fraud": 2,
        "model_sha256": sha256_file(model),
        "calibrator_sha256": sha256_file(calibrator),
        "policy_sha256": sha256_file(policy_file),
    }

    holdout_file.write_text(json.dumps(report))

    return {
        "model_file": model,
        "calibrator_file": calibrator,
        "policy_file": policy_file,
        "holdout_file": holdout_file,
        "output_dir": tmp_path / "release",
        "run_id": "test-run",
    }


def test_package_contains_expected_files(tmp_path):
    args = sample_artifacts(tmp_path)

    result = build_release(**args)

    output = args["output_dir"]

    assert (output / "model.joblib").is_file()
    assert (output / "calibrator.joblib").is_file()
    assert (output / "review_policy.json").is_file()
    assert (output / "release_manifest.json").is_file()

    assert result["deployment_mode"] == "SHADOW_ONLY"


def test_rejects_tampered_model(tmp_path):
    args = sample_artifacts(tmp_path)

    args["model_file"].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        build_release(**args)


def test_rejects_wrong_run_id(tmp_path):
    args = sample_artifacts(tmp_path)
    args["run_id"] = "incorrect-run"

    with pytest.raises(ValueError):
        build_release(**args)


def test_release_cannot_be_overwritten(tmp_path):
    args = sample_artifacts(tmp_path)

    build_release(**args)

    with pytest.raises(FileExistsError):
        build_release(**args)
