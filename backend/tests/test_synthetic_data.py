"""
Phase 3 — Synthetic Data Generator Tests

Verifies:
  - CSV file creation (3 files)
  - 10 merchant IDs (M001–M010)
  - Valid transaction amounts and timestamps
  - Correct 1-hour DetectionWindow aggregation
  - Feature contract columns
  - Three-way chronological split ordering
  - Fraud-spike labels in all three partitions
  - Spike elevation (spikes actually alter aggregate values)
  - Final-holdout physical separation
  - Database seeding (Transaction + DetectionWindow, no AnomalyDetection)
  - Reproducibility (same seed → identical output)
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend is importable
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from generate_synthetic_data import (
    DATA_DIR,
    DETECTION_WINDOWS_CSV,
    DETECTION_WINDOWS_HOLDOUT_CSV,
    MERCHANT_IDS,
    NUM_MERCHANTS,
    RANDOM_SEED,
    SIM_END,
    SIM_START,
    SPLIT_DEV_TEST_END,
    SPLIT_TRAIN_END,
    TRANSACTIONS_CSV,
    aggregate_detection_windows,
    export_csv,
    generate_transactions,
    seed_database,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_data():
    """Generate data once for all tests in this module."""
    rng = np.random.default_rng(RANDOM_SEED)
    txns_df = generate_transactions(rng)
    windows_df = aggregate_detection_windows(txns_df)
    export_csv(txns_df, windows_df)
    return txns_df, windows_df


@pytest.fixture(scope="module")
def transactions_df(generated_data):
    return generated_data[0]


@pytest.fixture(scope="module")
def windows_df(generated_data):
    return generated_data[1]


@pytest.fixture(scope="module")
def dev_csv(generated_data):
    """Load the development CSV (train + dev_test)."""
    return pd.read_csv(DETECTION_WINDOWS_CSV, parse_dates=["window_start", "window_end"])


@pytest.fixture(scope="module")
def holdout_csv(generated_data):
    """Load the final-holdout CSV."""
    return pd.read_csv(
        DETECTION_WINDOWS_HOLDOUT_CSV, parse_dates=["window_start", "window_end"]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCSVFilesCreated:
    """test_csv_files_created: all three CSV files exist."""

    def test_transactions_csv_exists(self, generated_data):
        assert TRANSACTIONS_CSV.exists(), f"{TRANSACTIONS_CSV} not found"

    def test_detection_windows_csv_exists(self, generated_data):
        assert DETECTION_WINDOWS_CSV.exists(), f"{DETECTION_WINDOWS_CSV} not found"

    def test_detection_windows_holdout_csv_exists(self, generated_data):
        assert DETECTION_WINDOWS_HOLDOUT_CSV.exists(), (
            f"{DETECTION_WINDOWS_HOLDOUT_CSV} not found"
        )


class TestMerchantCount:
    """test_merchant_count: exactly 10 merchants M001–M010."""

    def test_exactly_10_merchants_in_transactions(self, transactions_df):
        unique = sorted(transactions_df["merchant_id"].unique())
        assert unique == MERCHANT_IDS
        assert len(unique) == NUM_MERCHANTS

    def test_exactly_10_merchants_in_windows(self, windows_df):
        unique = sorted(windows_df["merchant_id"].unique())
        assert unique == MERCHANT_IDS


class TestValidTransactionAmounts:
    """test_valid_transaction_amounts: all amounts are positive."""

    def test_all_amounts_positive(self, transactions_df):
        assert (transactions_df["amount"] > 0).all(), "Some transaction amounts are not positive"

    def test_all_amounts_numeric(self, transactions_df):
        assert transactions_df["amount"].dtype in [np.float64, np.float32, float], (
            "Transaction amounts are not numeric"
        )


class TestValidTimestamps:
    """test_valid_timestamps: all timestamps within the 30-day window."""

    def test_timestamps_within_simulation_period(self, transactions_df):
        timestamps = pd.to_datetime(transactions_df["timestamp"], utc=True)
        assert (timestamps >= SIM_START).all(), "Some timestamps before simulation start"
        assert (timestamps < SIM_END).all(), "Some timestamps after simulation end"

    def test_timestamps_are_valid_datetimes(self, transactions_df):
        # Should not raise
        pd.to_datetime(transactions_df["timestamp"])


class TestDetectionWindowAggregation:
    """test_detection_window_aggregation: verify aggregation against raw transactions."""

    def test_transaction_count_matches(self, transactions_df, windows_df):
        """For a sample of windows, verify transaction_count matches raw data."""
        sample = windows_df.sample(n=min(50, len(windows_df)), random_state=42)
        for _, win in sample.iterrows():
            mask = (
                (transactions_df["merchant_id"] == win["merchant_id"])
                & (pd.to_datetime(transactions_df["timestamp"]) >= win["window_start"])
                & (pd.to_datetime(transactions_df["timestamp"]) < win["window_end"])
            )
            expected_count = mask.sum()
            assert win["transaction_count"] == expected_count, (
                f"Window {win['merchant_id']} {win['window_start']}: "
                f"expected count {expected_count}, got {win['transaction_count']}"
            )

    def test_total_amount_matches(self, transactions_df, windows_df):
        """For a sample of windows, verify total_amount matches raw data."""
        sample = windows_df.sample(n=min(50, len(windows_df)), random_state=42)
        for _, win in sample.iterrows():
            mask = (
                (transactions_df["merchant_id"] == win["merchant_id"])
                & (pd.to_datetime(transactions_df["timestamp"]) >= win["window_start"])
                & (pd.to_datetime(transactions_df["timestamp"]) < win["window_end"])
            )
            expected_total = round(float(transactions_df.loc[mask, "amount"].sum()), 2)
            assert abs(win["total_amount"] - expected_total) < 0.02, (
                f"Window {win['merchant_id']} {win['window_start']}: "
                f"expected total {expected_total}, got {win['total_amount']}"
            )

    def test_avg_transaction_amount_consistency(self, windows_df):
        """avg_transaction_amount == total_amount / transaction_count."""
        non_zero = windows_df[windows_df["transaction_count"] > 0]
        expected = non_zero["total_amount"] / non_zero["transaction_count"]
        diff = (non_zero["avg_transaction_amount"] - expected).abs()
        assert (diff < 0.02).all(), "avg_transaction_amount inconsistency detected"


class TestFeatureContractColumns:
    """test_feature_contract_columns: exact expected columns in CSVs."""

    EXPECTED_COLUMNS = [
        "merchant_id",
        "window_start",
        "window_end",
        "transaction_count",
        "total_amount",
        "avg_transaction_amount",
        "is_synthetic_fraud_spike",
        "split",
    ]

    def test_detection_windows_csv_columns(self, dev_csv):
        assert list(dev_csv.columns) == self.EXPECTED_COLUMNS

    def test_holdout_csv_columns(self, holdout_csv):
        assert list(holdout_csv.columns) == self.EXPECTED_COLUMNS


class TestThreeWayChronologicalSplit:
    """test_three_way_chronological_split: ordering, boundaries, and no overlap."""

    def test_three_distinct_split_values(self, dev_csv, holdout_csv):
        all_splits = set(dev_csv["split"].unique()) | set(holdout_csv["split"].unique())
        assert all_splits == {"train", "dev_test", "final_holdout"}

    def test_train_before_dev_test(self, dev_csv):
        train = dev_csv[dev_csv["split"] == "train"]
        dev_test = dev_csv[dev_csv["split"] == "dev_test"]
        assert train["window_start"].max() < dev_test["window_start"].min(), (
            "Train windows must all precede dev-test windows"
        )

    def test_dev_test_before_final_holdout(self, dev_csv, holdout_csv):
        dev_test = dev_csv[dev_csv["split"] == "dev_test"]
        assert dev_test["window_start"].max() < holdout_csv["window_start"].min(), (
            "Dev-test windows must all precede final-holdout windows"
        )

    def test_no_temporal_overlap(self, dev_csv, holdout_csv):
        train = dev_csv[dev_csv["split"] == "train"]
        dev_test = dev_csv[dev_csv["split"] == "dev_test"]
        holdout = holdout_csv

        # Train max < Dev-test min
        assert train["window_start"].max() < dev_test["window_start"].min()
        # Dev-test max < Holdout min
        assert dev_test["window_start"].max() < holdout["window_start"].min()

    # --- GAP 2 FIX: absolute boundary assertions ---

    def test_train_ends_before_split_boundary(self, windows_df):
        """All train window_start values must be < SPLIT_TRAIN_END."""
        train = windows_df[windows_df["split"] == "train"]
        assert train["window_start"].max() < SPLIT_TRAIN_END

    def test_dev_test_starts_at_or_after_train_boundary(self, windows_df):
        """All dev_test window_start values must be >= SPLIT_TRAIN_END."""
        dev_test = windows_df[windows_df["split"] == "dev_test"]
        assert dev_test["window_start"].min() >= SPLIT_TRAIN_END

    def test_dev_test_ends_before_holdout_boundary(self, windows_df):
        """All dev_test window_start values must be < SPLIT_DEV_TEST_END."""
        dev_test = windows_df[windows_df["split"] == "dev_test"]
        assert dev_test["window_start"].max() < SPLIT_DEV_TEST_END

    def test_holdout_starts_at_or_after_dev_test_boundary(self, windows_df):
        """All final_holdout window_start values must be >= SPLIT_DEV_TEST_END."""
        holdout = windows_df[windows_df["split"] == "final_holdout"]
        assert holdout["window_start"].min() >= SPLIT_DEV_TEST_END

    def test_all_partitions_non_empty(self, windows_df):
        """Each partition must contain at least one window."""
        for split_name in ["train", "dev_test", "final_holdout"]:
            count = len(windows_df[windows_df["split"] == split_name])
            assert count > 0, f"Partition '{split_name}' is empty"

    def test_no_undefined_split_values(self, windows_df):
        """No rows should have a split value outside the three defined values."""
        valid_splits = {"train", "dev_test", "final_holdout"}
        actual_splits = set(windows_df["split"].unique())
        undefined = actual_splits - valid_splits
        assert len(undefined) == 0, f"Undefined split values found: {undefined}"


class TestEachPartitionContainsFraudSpikes:
    """test_each_partition_contains_fraud_spikes."""

    def test_train_has_spikes_and_normal(self, windows_df):
        train = windows_df[windows_df["split"] == "train"]
        assert train["is_synthetic_fraud_spike"].any(), "Train has no spike windows"
        assert not train["is_synthetic_fraud_spike"].all(), "Train has no normal windows"

    def test_dev_test_has_spikes_and_normal(self, windows_df):
        dev = windows_df[windows_df["split"] == "dev_test"]
        assert dev["is_synthetic_fraud_spike"].any(), "Dev-test has no spike windows"
        assert not dev["is_synthetic_fraud_spike"].all(), "Dev-test has no normal windows"

    def test_final_holdout_has_spikes_and_normal(self, windows_df):
        holdout = windows_df[windows_df["split"] == "final_holdout"]
        assert holdout["is_synthetic_fraud_spike"].any(), "Final-holdout has no spike windows"
        assert not holdout["is_synthetic_fraud_spike"].all(), (
            "Final-holdout has no normal windows"
        )


class TestSpikeElevation:
    """
    test_spike_elevation: spike windows have measurably elevated
    transaction_count and/or total_amount vs normal windows.
    """

    def _assert_spike_elevated(self, partition_df: pd.DataFrame, partition_name: str):
        spikes = partition_df[partition_df["is_synthetic_fraud_spike"] == True]  # noqa: E712
        normals = partition_df[partition_df["is_synthetic_fraud_spike"] == False]  # noqa: E712

        if len(spikes) == 0 or len(normals) == 0:
            pytest.skip(f"Not enough data in {partition_name}")

        spike_avg_count = spikes["transaction_count"].mean()
        normal_avg_count = normals["transaction_count"].mean()

        spike_avg_amount = spikes["total_amount"].mean()
        normal_avg_amount = normals["total_amount"].mean()

        # At least one of count or amount should be elevated
        elevated = (
            spike_avg_count > normal_avg_count * 1.5
            or spike_avg_amount > normal_avg_amount * 1.5
        )
        assert elevated, (
            f"{partition_name}: spikes not measurably elevated. "
            f"Spike avg count={spike_avg_count:.1f} vs normal={normal_avg_count:.1f}, "
            f"Spike avg amount={spike_avg_amount:.1f} vs normal={normal_avg_amount:.1f}"
        )

    def test_train_spikes_elevated(self, windows_df):
        train = windows_df[windows_df["split"] == "train"]
        self._assert_spike_elevated(train, "train")

    def test_dev_test_spikes_elevated(self, windows_df):
        dev = windows_df[windows_df["split"] == "dev_test"]
        self._assert_spike_elevated(dev, "dev_test")

    def test_final_holdout_spikes_elevated(self, windows_df):
        holdout = windows_df[windows_df["split"] == "final_holdout"]
        self._assert_spike_elevated(holdout, "final_holdout")


class TestFinalHoldoutPhysicallySeparate:
    """test_final_holdout_is_physically_separate."""

    def test_dev_csv_has_no_holdout_rows(self, dev_csv):
        assert "final_holdout" not in dev_csv["split"].values, (
            "detection_windows.csv must not contain final_holdout rows"
        )

    def test_holdout_csv_has_only_holdout_rows(self, holdout_csv):
        assert (holdout_csv["split"] == "final_holdout").all(), (
            "detection_windows_final_holdout.csv must only contain final_holdout rows"
        )

    def test_no_row_overlap(self, dev_csv, holdout_csv):
        """No shared (merchant_id, window_start) pairs between files."""
        dev_keys = set(
            zip(dev_csv["merchant_id"], dev_csv["window_start"].astype(str))
        )
        holdout_keys = set(
            zip(holdout_csv["merchant_id"], holdout_csv["window_start"].astype(str))
        )
        overlap = dev_keys & holdout_keys
        assert len(overlap) == 0, f"Row overlap detected: {overlap}"


class TestDatabaseSeeding:
    """
    test_database_seeding: verify the REAL seed_database() function
    correctly maps and persists generated data.
    """

    @pytest.fixture(scope="class")
    def seeded_session(self, generated_data):
        """
        Call the real seed_database() with an injected in-memory session
        to verify production seeding logic.
        """
        from database import Base
        from models import AnomalyDetection, DetectionWindow, Transaction

        txns_df, windows_df = generated_data

        test_engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=test_engine)
        Session = sessionmaker(bind=test_engine)
        session = Session()

        # Call the REAL seed_database() with the injected test session
        seed_database(txns_df, windows_df, session=session)

        yield session, txns_df, windows_df, Transaction, DetectionWindow, AnomalyDetection
        session.close()

    def test_transaction_records_seeded(self, seeded_session):
        """seed_database() inserts the correct number of Transaction records."""
        session, txns_df, _, Transaction, _, _ = seeded_session
        count = session.query(Transaction).count()
        assert count == len(txns_df), f"Expected {len(txns_df)} transactions, got {count}"

    def test_detection_window_records_seeded(self, seeded_session):
        """seed_database() inserts the correct number of DetectionWindow records."""
        session, _, windows_df, _, DetectionWindow, _ = seeded_session
        count = session.query(DetectionWindow).count()
        assert count == len(windows_df), f"Expected {len(windows_df)} windows, got {count}"

    def test_no_anomaly_detection_records(self, seeded_session):
        """seed_database() must not create any AnomalyDetection records."""
        session, _, _, _, _, AnomalyDetection = seeded_session
        count = session.query(AnomalyDetection).count()
        assert count == 0, f"No AnomalyDetection records should exist, but found {count}"

    def test_seeded_merchant_ids_match(self, seeded_session):
        """Seeded Transaction records contain the expected merchant IDs."""
        session, _, _, Transaction, _, _ = seeded_session
        db_merchants = sorted(set(
            r[0] for r in session.query(Transaction.merchant_id).distinct().all()
        ))
        assert db_merchants == MERCHANT_IDS

    def test_seeded_window_split_values(self, seeded_session):
        """Seeded DetectionWindow records contain all three split values."""
        session, _, _, _, DetectionWindow, _ = seeded_session
        db_splits = sorted(set(
            r[0] for r in session.query(DetectionWindow.split).distinct().all()
        ))
        assert db_splits == ["dev_test", "final_holdout", "train"]

    def test_seeded_spike_labels_present(self, seeded_session):
        """Seeded DetectionWindow records include both spike and normal labels."""
        session, _, _, _, DetectionWindow, _ = seeded_session
        spike_count = session.query(DetectionWindow).filter(
            DetectionWindow.is_synthetic_fraud_spike == True  # noqa: E712
        ).count()
        normal_count = session.query(DetectionWindow).filter(
            DetectionWindow.is_synthetic_fraud_spike == False  # noqa: E712
        ).count()
        assert spike_count > 0, "No spike windows found in seeded data"
        assert normal_count > 0, "No normal windows found in seeded data"

    def test_seeded_window_features_mapped(self, seeded_session):
        """Spot-check that aggregate features are correctly mapped by seed_database()."""
        session, _, windows_df, _, DetectionWindow, _ = seeded_session
        # Check first 5 windows from the source data
        for _, src_row in windows_df.head(5).iterrows():
            db_row = session.query(DetectionWindow).filter(
                DetectionWindow.merchant_id == src_row["merchant_id"],
                DetectionWindow.window_start == src_row["window_start"],
            ).first()
            assert db_row is not None, (
                f"Window not found: {src_row['merchant_id']} {src_row['window_start']}"
            )
            assert db_row.transaction_count == src_row["transaction_count"]
            assert abs(db_row.total_amount - src_row["total_amount"]) < 0.02
            assert abs(db_row.avg_transaction_amount - src_row["avg_transaction_amount"]) < 0.02
            assert db_row.is_synthetic_fraud_spike == src_row["is_synthetic_fraud_spike"]
            assert db_row.split == src_row["split"]


class TestReproducibility:
    """test_reproducibility: same seed produces identical output."""

    def test_deterministic_output(self):
        """Run generator twice; verify identical DataFrames."""
        rng1 = np.random.default_rng(RANDOM_SEED)
        txns1 = generate_transactions(rng1)
        wins1 = aggregate_detection_windows(txns1)

        rng2 = np.random.default_rng(RANDOM_SEED)
        txns2 = generate_transactions(rng2)
        wins2 = aggregate_detection_windows(txns2)

        pd.testing.assert_frame_equal(txns1, txns2, check_exact=True)
        pd.testing.assert_frame_equal(wins1, wins2, check_exact=True)
