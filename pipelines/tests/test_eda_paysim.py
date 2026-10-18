
import json

import pandas as pd
import pytest

from eda_paysim import analyze_dataset


def create_fixture(tmp_path):
    source = tmp_path / "processed" / "v1"
    source.mkdir(parents=True)

    frame = pd.DataFrame({
        "step": [1, 25, 26, 49],
        "type": [
            "PAYMENT", "TRANSFER",
            "CASH_OUT", "TRANSFER",
        ],
        "amount": [10.0, 5000.0, 0.0, 200.0],
        "isFraud": [0, 1, 0, 1],
        "simulation_day": [0, 1, 1, 2],
    })

    frame.to_parquet(
        source / "part-00000.parquet",
        index=False,
    )

    manifest = {
        "source_sha256": "fixture-checksum",
        "pipeline_version": "1.0",
        "rows": 4,
        "fraud_rows": 2,
        "min_step": 1,
        "max_step": 49,
        "parts": 1,
        "transaction_type_counts": {
            "PAYMENT": 1,
            "TRANSFER": 2,
            "CASH_OUT": 1,
        },
    }

    (source / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return source, frame, manifest


def test_successful_analysis(tmp_path):
    source, _, _ = create_fixture(tmp_path)
    output = tmp_path / "reports"

    summary = analyze_dataset(
        source, output, batch_size=2
    )

    assert summary["total_transactions"] == 4
    assert summary["fraud_transactions"] == 2
    assert summary["fraud_rate_pct"] == 50.0
    assert summary["unique_simulation_days"] == 3
    assert summary["missing_cells"] == 0

    assert (output / "eda_summary.json").exists()
    assert (output / "by_transaction_type.csv").exists()
    assert (output / "by_simulation_day.csv").exists()
    assert (output / "by_amount_band.csv").exists()

    assert len(list(
        (output / "charts").glob("*.png")
    )) == 5

    by_type = pd.read_csv(
        output / "by_transaction_type.csv"
    ).set_index("type")

    assert by_type.loc[
        "TRANSFER", "fraud_transactions"
    ] == 2


def test_manifest_mismatch_rejected(tmp_path):
    source, _, manifest = create_fixture(tmp_path)
    output = tmp_path / "reports"

    manifest["rows"] = 999

    (source / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Manifest"):
        analyze_dataset(source, output, batch_size=2)


def test_invalid_label_rejected(tmp_path):
    source, frame, _ = create_fixture(tmp_path)
    output = tmp_path / "reports"

    frame.loc[0, "isFraud"] = 3
    frame.to_parquet(
        source / "part-00000.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="fraud labels"):
        analyze_dataset(source, output, batch_size=2)


def test_negative_amount_rejected(tmp_path):
    source, frame, _ = create_fixture(tmp_path)
    output = tmp_path / "reports"

    frame.loc[0, "amount"] = -100
    frame.to_parquet(
        source / "part-00000.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="Invalid amounts"):
        analyze_dataset(source, output, batch_size=2)


def test_forbidden_columns_rejected(tmp_path):
    source, frame, _ = create_fixture(tmp_path)
    output = tmp_path / "reports"

    frame["oldbalanceOrg"] = 1000
    frame.to_parquet(
        source / "part-00000.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="forbidden columns"):
        analyze_dataset(source, output, batch_size=2)


def test_invalid_simulation_day_rejected(tmp_path):
    source, frame, _ = create_fixture(tmp_path)
    output = tmp_path / "reports"

    frame.loc[0, "simulation_day"] = 20
    frame.to_parquet(
        source / "part-00000.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="Simulation day"):
        analyze_dataset(source, output, batch_size=2)


def test_missing_parquet_part_rejected(tmp_path):
    source, _, _ = create_fixture(tmp_path)
    output = tmp_path / "reports"

    (source / "part-00000.parquet").unlink()

    with pytest.raises(ValueError, match="Parquet part count"):
        analyze_dataset(source, output, batch_size=2)
