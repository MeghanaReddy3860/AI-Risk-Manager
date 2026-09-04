"""
Phase 4 — Baseline Detector Tests

Verifies:
  A. Fitting: per-merchant statistics from train only
  B. Z-score / Prediction: flagging, risk score, explanation
  C. Zero-standard-deviation: sentinel handling, no NaN/inf
  D. Database persistence: AnomalyDetection records via run_baseline_detector()
  E. Idempotency: no duplicate records on re-run
  F. Final-holdout protection: 6-layer safety enforcement
  G. Determinism: identical input → identical output
"""

import sys
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

from baseline_detector import (
    BASELINE_FEATURES,
    BASELINE_ZSCORE_THRESHOLD,
    ZERO_STD_SENTINEL_ZSCORE,
    BaselineDetector,
    run_baseline_detector,
)
from generate_synthetic_data import (
    RANDOM_SEED,
    aggregate_detection_windows,
    generate_transactions,
    seed_database,
)


# ===========================================================================
# Fixtures — Hand-crafted data for unit tests
# ===========================================================================

# M001 statistics (normal variance):
#   transaction_count:      [10, 20, 30]  → mean=20,  std=10
#   total_amount:           [100, 200, 300] → mean=200, std=100
#   avg_transaction_amount: [8, 10, 12]    → mean=10,  std=2
#
# M002 statistics (zero variance):
#   transaction_count:      [5, 5, 5]  → mean=5,  std=0
#   total_amount:           [50, 50, 50] → mean=50, std=0
#   avg_transaction_amount: [10, 10, 10] → mean=10, std=0


@pytest.fixture(scope="module")
def unit_train_df():
    """Small train DataFrame with known statistics for unit testing."""
    return pd.DataFrame(
        [
            {
                "merchant_id": "M001",
                "transaction_count": 10,
                "total_amount": 100.0,
                "avg_transaction_amount": 8.0,
                "split": "train",
                "is_synthetic_fraud_spike": False,
            },
            {
                "merchant_id": "M001",
                "transaction_count": 20,
                "total_amount": 200.0,
                "avg_transaction_amount": 10.0,
                "split": "train",
                "is_synthetic_fraud_spike": False,
            },
            {
                "merchant_id": "M001",
                "transaction_count": 30,
                "total_amount": 300.0,
                "avg_transaction_amount": 12.0,
                "split": "train",
                "is_synthetic_fraud_spike": True,
            },
            {
                "merchant_id": "M002",
                "transaction_count": 5,
                "total_amount": 50.0,
                "avg_transaction_amount": 10.0,
                "split": "train",
                "is_synthetic_fraud_spike": False,
            },
            {
                "merchant_id": "M002",
                "transaction_count": 5,
                "total_amount": 50.0,
                "avg_transaction_amount": 10.0,
                "split": "train",
                "is_synthetic_fraud_spike": False,
            },
            {
                "merchant_id": "M002",
                "transaction_count": 5,
                "total_amount": 50.0,
                "avg_transaction_amount": 10.0,
                "split": "train",
                "is_synthetic_fraud_spike": False,
            },
        ]
    )


@pytest.fixture(scope="module")
def fitted_detector(unit_train_df):
    """BaselineDetector fitted on the unit_train_df."""
    detector = BaselineDetector(zscore_threshold=2.0)
    detector.fit(unit_train_df)
    return detector


# ---------------------------------------------------------------------------
# Fixtures — Phase 3 data for integration tests (generated once per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generated_data():
    """Generate Phase 3 synthetic data once for integration tests."""
    rng = np.random.default_rng(RANDOM_SEED)
    txns_df = generate_transactions(rng)
    windows_df = aggregate_detection_windows(txns_df)
    return txns_df, windows_df


# ---------------------------------------------------------------------------
# Class-scoped DB fixtures (defined at module level to avoid deprecation)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def baseline_db(generated_data):
    """
    In-memory DB with Phase 3 data seeded and baseline detector run once.
    Used by TestBaselineDatabasePersistence and TestBaselineHoldoutProtection.
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

    seed_database(txns_df, windows_df, session=session)
    run_baseline_detector(session=session)

    yield session, DetectionWindow, AnomalyDetection
    session.close()


@pytest.fixture(scope="class")
def idempotency_db(generated_data):
    """
    In-memory DB with Phase 3 data seeded — tests manage detector runs.
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

    seed_database(txns_df, windows_df, session=session)

    yield session, DetectionWindow, AnomalyDetection
    session.close()


# ===========================================================================
# A. FITTING TESTS
# ===========================================================================


class TestBaselineFit:
    """Verify per-merchant baseline statistics from train data."""

    def test_per_merchant_mean_calculation(self, fitted_detector):
        """Mean is computed correctly for each merchant/feature."""
        m001 = fitted_detector.merchant_stats["M001"]
        assert abs(m001["transaction_count"]["mean"] - 20.0) < 1e-9
        assert abs(m001["total_amount"]["mean"] - 200.0) < 1e-9
        assert abs(m001["avg_transaction_amount"]["mean"] - 10.0) < 1e-9

        m002 = fitted_detector.merchant_stats["M002"]
        assert abs(m002["transaction_count"]["mean"] - 5.0) < 1e-9
        assert abs(m002["total_amount"]["mean"] - 50.0) < 1e-9
        assert abs(m002["avg_transaction_amount"]["mean"] - 10.0) < 1e-9

    def test_per_merchant_std_calculation(self, fitted_detector):
        """Sample std (ddof=1) is computed correctly for each merchant/feature."""
        m001 = fitted_detector.merchant_stats["M001"]
        assert abs(m001["transaction_count"]["std"] - 10.0) < 1e-9
        assert abs(m001["total_amount"]["std"] - 100.0) < 1e-9
        assert abs(m001["avg_transaction_amount"]["std"] - 2.0) < 1e-9

        # M002 has zero variance
        m002 = fitted_detector.merchant_stats["M002"]
        assert m002["transaction_count"]["std"] == 0.0
        assert m002["total_amount"]["std"] == 0.0
        assert m002["avg_transaction_amount"]["std"] == 0.0

    def test_statistics_use_train_only(self, unit_train_df):
        """fit() only accepts rows with split == 'train'."""
        # All rows are train — this should succeed
        detector = BaselineDetector()
        detector.fit(unit_train_df)
        assert detector._is_fitted

    def test_statistics_independent_between_merchants(self, fitted_detector):
        """Each merchant gets its own independent statistics."""
        m001 = fitted_detector.merchant_stats["M001"]
        m002 = fitted_detector.merchant_stats["M002"]

        # M001 and M002 have clearly different means
        assert m001["transaction_count"]["mean"] != m002["transaction_count"]["mean"]
        assert m001["total_amount"]["mean"] != m002["total_amount"]["mean"]

    def test_fraud_labels_not_used_in_fit(self, unit_train_df):
        """Changing fraud labels does not change fitted statistics."""
        # Fit with original labels (M001 row 3 is True)
        det1 = BaselineDetector()
        det1.fit(unit_train_df)

        # Fit with all labels flipped
        modified_df = unit_train_df.copy()
        modified_df["is_synthetic_fraud_spike"] = ~modified_df[
            "is_synthetic_fraud_spike"
        ]
        det2 = BaselineDetector()
        det2.fit(modified_df)

        # Statistics must be identical
        for mid in ["M001", "M002"]:
            for feat in BASELINE_FEATURES:
                assert (
                    det1.merchant_stats[mid][feat]["mean"]
                    == det2.merchant_stats[mid][feat]["mean"]
                )
                assert (
                    det1.merchant_stats[mid][feat]["std"]
                    == det2.merchant_stats[mid][feat]["std"]
                )

    def test_fit_rejects_non_train_rows(self, unit_train_df):
        """fit() raises ValueError if any row is not split='train'."""
        mixed_df = unit_train_df.copy()
        mixed_df.loc[0, "split"] = "dev_test"

        detector = BaselineDetector()
        with pytest.raises(ValueError, match="non-train split values"):
            detector.fit(mixed_df)

    def test_fit_rejects_holdout_rows(self):
        """fit() raises ValueError if final_holdout rows are present."""
        df = pd.DataFrame(
            [
                {
                    "merchant_id": "M001",
                    "transaction_count": 10,
                    "total_amount": 100.0,
                    "avg_transaction_amount": 10.0,
                    "split": "final_holdout",
                    "is_synthetic_fraud_spike": False,
                },
            ]
        )
        detector = BaselineDetector()
        with pytest.raises(ValueError, match="non-train split values"):
            detector.fit(df)


# ===========================================================================
# B. Z-SCORE / PREDICTION TESTS
# ===========================================================================


class TestBaselinePredict:
    """Verify z-score calculation, flagging, risk score, and explanation."""

    def _make_dev_test_row(self, merchant_id, tc, ta, ata, row_id=1):
        """Helper to create a single-row dev_test DataFrame."""
        return pd.DataFrame(
            [
                {
                    "id": row_id,
                    "merchant_id": merchant_id,
                    "transaction_count": tc,
                    "total_amount": ta,
                    "avg_transaction_amount": ata,
                    "split": "dev_test",
                }
            ]
        )

    def test_correct_zscore_calculation(self, fitted_detector):
        """z = (value - mean) / std, computed correctly."""
        # M001: tc=40 → z = (40-20)/10 = 2.0
        df = self._make_dev_test_row("M001", tc=40, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert len(results) == 1
        assert abs(results[0]["z_scores"]["transaction_count"] - 2.0) < 1e-9
        assert abs(results[0]["z_scores"]["total_amount"] - 0.0) < 1e-9
        assert abs(results[0]["z_scores"]["avg_transaction_amount"] - 0.0) < 1e-9

    def test_absolute_zscore_for_negative_values(self, fitted_detector):
        """Negative z-scores are handled via absolute value for flagging."""
        # M001: tc=0 → z = (0-20)/10 = -2.0, abs = 2.0 → FLAGGED
        df = self._make_dev_test_row("M001", tc=0, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["z_scores"]["transaction_count"] == -2.0
        assert results[0]["is_flagged"] is True

    def test_below_threshold_not_flagged(self, fitted_detector):
        """Window with all features |z| < threshold is not flagged."""
        # M001: tc=25 → z = 0.5, ta=250 → z = 0.5, ata=11 → z = 0.5
        df = self._make_dev_test_row("M001", tc=25, ta=250, ata=11)
        results = fitted_detector.predict(df)
        assert results[0]["is_flagged"] is False

    def test_equal_to_threshold_flagged(self, fitted_detector):
        """Window with any feature |z| == threshold IS flagged (>= not >)."""
        # M001: tc=40 → z = (40-20)/10 = 2.0 == threshold
        df = self._make_dev_test_row("M001", tc=40, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["is_flagged"] is True

    def test_above_threshold_flagged(self, fitted_detector):
        """Window with any feature |z| > threshold is flagged."""
        # M001: tc=60 → z = 4.0 > 2.0
        df = self._make_dev_test_row("M001", tc=60, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["is_flagged"] is True

    def test_any_feature_triggers_flag(self, fitted_detector):
        """Only one feature needs to exceed threshold to flag the window."""
        # Only avg_transaction_amount exceeds: ata=14 → z = (14-10)/2 = 2.0
        df = self._make_dev_test_row("M001", tc=20, ta=200, ata=14)
        results = fitted_detector.predict(df)
        assert results[0]["is_flagged"] is True
        # Verify that only ata triggered
        z = results[0]["z_scores"]
        assert abs(z["transaction_count"]) < BASELINE_ZSCORE_THRESHOLD
        assert abs(z["total_amount"]) < BASELINE_ZSCORE_THRESHOLD
        assert abs(z["avg_transaction_amount"]) >= BASELINE_ZSCORE_THRESHOLD

    def test_multiple_features_flagged(self, fitted_detector):
        """Multiple features can exceed the threshold simultaneously."""
        # tc=40 → z=2.0, ta=400 → z=2.0, ata=10 → z=0
        df = self._make_dev_test_row("M001", tc=40, ta=400, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["is_flagged"] is True
        z = results[0]["z_scores"]
        flagged_features = [
            f for f in BASELINE_FEATURES if abs(z[f]) >= BASELINE_ZSCORE_THRESHOLD
        ]
        assert len(flagged_features) == 2

    def test_normal_windows_not_flagged(self, fitted_detector):
        """A window at the mean of all features is not flagged."""
        df = self._make_dev_test_row("M001", tc=20, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["is_flagged"] is False
        assert results[0]["risk_score"] == 0.0

    def test_risk_score_at_z_zero(self, fitted_detector):
        """risk_score = 0 when all features are at the mean."""
        df = self._make_dev_test_row("M001", tc=20, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["risk_score"] == 0.0

    def test_risk_score_at_threshold(self, fitted_detector):
        """risk_score = 50 when max |z| equals the threshold."""
        # tc=40 → z=2.0, others at mean
        df = self._make_dev_test_row("M001", tc=40, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["risk_score"] == 50.0

    def test_risk_score_at_2x_threshold(self, fitted_detector):
        """risk_score = 100 when max |z| equals 2× threshold."""
        # tc=60 → z=4.0
        df = self._make_dev_test_row("M001", tc=60, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["risk_score"] == 100.0

    def test_risk_score_above_2x_threshold_capped(self, fitted_detector):
        """risk_score is capped at 100 for very high z-scores."""
        # tc=100 → z=8.0, risk = min(100, 8/2 * 50) = min(100, 200) = 100
        df = self._make_dev_test_row("M001", tc=100, ta=200, ata=10)
        results = fitted_detector.predict(df)
        assert results[0]["risk_score"] == 100.0

    def test_risk_score_always_between_0_and_100(self, fitted_detector):
        """risk_score is always in [0, 100] for various inputs."""
        test_values = [0, 10, 20, 30, 50, 80, 100, 200]
        for tc in test_values:
            df = self._make_dev_test_row("M001", tc=tc, ta=200, ata=10)
            results = fitted_detector.predict(df)
            assert 0 <= results[0]["risk_score"] <= 100

    def test_explanation_contains_all_features(self, fitted_detector):
        """Explanation mentions all three approved features."""
        df = self._make_dev_test_row("M001", tc=40, ta=200, ata=10)
        results = fitted_detector.predict(df)
        explanation = results[0]["explanation"]
        for feature in BASELINE_FEATURES:
            assert feature in explanation

    def test_explanation_contains_threshold(self, fitted_detector):
        """Explanation includes the threshold value for flagged features."""
        df = self._make_dev_test_row("M001", tc=40, ta=200, ata=10)
        results = fitted_detector.predict(df)
        explanation = results[0]["explanation"]
        assert "threshold=2.00" in explanation

    def test_explanation_identifies_flagged_features(self, fitted_detector):
        """Flagged features show 'FLAGGED', normal features show 'normal'."""
        # tc=40 → FLAGGED, ta and ata at mean → normal
        df = self._make_dev_test_row("M001", tc=40, ta=200, ata=10)
        results = fitted_detector.predict(df)
        explanation = results[0]["explanation"]
        assert "transaction_count z=2.00 (FLAGGED" in explanation
        assert "total_amount z=0.00 (normal)" in explanation
        assert "avg_transaction_amount z=0.00 (normal)" in explanation

    def test_predict_requires_fit(self):
        """predict() raises RuntimeError if called before fit()."""
        detector = BaselineDetector()
        df = pd.DataFrame(
            [
                {
                    "merchant_id": "M001",
                    "transaction_count": 10,
                    "total_amount": 100.0,
                    "avg_transaction_amount": 10.0,
                    "split": "dev_test",
                }
            ]
        )
        with pytest.raises(RuntimeError, match="must be fit"):
            detector.predict(df)


# ===========================================================================
# C. ZERO-STANDARD-DEVIATION TESTS
# ===========================================================================


class TestBaselineZeroStd:
    """Verify zero-std sentinel handling produces no NaN or infinity."""

    def test_constant_feature_equal_prediction_z_zero(self, fitted_detector):
        """When value == train mean and std == 0 → z = 0.0."""
        # M002: all features constant at mean values
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M002",
                    "transaction_count": 5,
                    "total_amount": 50.0,
                    "avg_transaction_amount": 10.0,
                    "split": "dev_test",
                }
            ]
        )
        results = fitted_detector.predict(df)
        for feature in BASELINE_FEATURES:
            assert results[0]["z_scores"][feature] == 0.0

    def test_constant_feature_different_prediction_z_sentinel(self, fitted_detector):
        """When value != train mean and std == 0 → z = ZERO_STD_SENTINEL_ZSCORE."""
        # M002: tc=6 differs from mean=5
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M002",
                    "transaction_count": 6,
                    "total_amount": 50.0,
                    "avg_transaction_amount": 10.0,
                    "split": "dev_test",
                }
            ]
        )
        results = fitted_detector.predict(df)
        assert results[0]["z_scores"]["transaction_count"] == ZERO_STD_SENTINEL_ZSCORE
        assert results[0]["z_scores"]["total_amount"] == 0.0
        assert results[0]["z_scores"]["avg_transaction_amount"] == 0.0

    def test_no_nan_or_infinity(self, fitted_detector):
        """Zero-std handling never produces NaN or infinity."""
        test_cases = [
            {"tc": 5, "ta": 50, "ata": 10},       # all at mean
            {"tc": 0, "ta": 0, "ata": 0},          # all differ from mean
            {"tc": 100, "ta": 1000, "ata": 100},   # large deviation
        ]
        for case in test_cases:
            df = pd.DataFrame(
                [
                    {
                        "id": 1,
                        "merchant_id": "M002",
                        "transaction_count": case["tc"],
                        "total_amount": case["ta"],
                        "avg_transaction_amount": case["ata"],
                        "split": "dev_test",
                    }
                ]
            )
            results = fitted_detector.predict(df)
            for feature in BASELINE_FEATURES:
                z = results[0]["z_scores"][feature]
                assert not np.isnan(z), f"NaN z-score for {feature}"
                assert not np.isinf(z), f"Infinity z-score for {feature}"

    def test_zero_std_deterministic(self, fitted_detector):
        """Zero-std handling produces identical results on repeated calls."""
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M002",
                    "transaction_count": 6,
                    "total_amount": 60.0,
                    "avg_transaction_amount": 15.0,
                    "split": "dev_test",
                }
            ]
        )
        results1 = fitted_detector.predict(df)
        results2 = fitted_detector.predict(df)
        assert results1[0]["z_scores"] == results2[0]["z_scores"]
        assert results1[0]["risk_score"] == results2[0]["risk_score"]
        assert results1[0]["is_flagged"] == results2[0]["is_flagged"]
        assert results1[0]["explanation"] == results2[0]["explanation"]


# ===========================================================================
# D. DATABASE PERSISTENCE TESTS
# ===========================================================================


class TestBaselineDatabasePersistence:
    """Verify run_baseline_detector() persists AnomalyDetection records."""

    def test_anomaly_records_inserted(self, baseline_db):
        """Baseline detector creates AnomalyDetection records."""
        session, DW, AD = baseline_db
        count = session.query(AD).filter(AD.detector_type == "baseline").count()
        assert count > 0, "No baseline AnomalyDetection records created"

    def test_correct_window_id_mapping(self, baseline_db):
        """Every baseline record references an existing DetectionWindow."""
        session, DW, AD = baseline_db
        records = (
            session.query(AD).filter(AD.detector_type == "baseline").all()
        )
        for rec in records:
            window = session.query(DW).filter(DW.id == rec.window_id).first()
            assert window is not None, (
                f"AnomalyDetection window_id={rec.window_id} not found "
                f"in DetectionWindow"
            )

    def test_only_dev_test_windows_scored(self, baseline_db):
        """Baseline records exist only for dev_test windows."""
        session, DW, AD = baseline_db
        records = (
            session.query(AD).filter(AD.detector_type == "baseline").all()
        )
        for rec in records:
            window = session.query(DW).filter(DW.id == rec.window_id).first()
            assert window.split == "dev_test", (
                f"Baseline record for non-dev_test window: "
                f"split={window.split}, window_id={rec.window_id}"
            )

    def test_detector_type_always_baseline(self, baseline_db):
        """Every Phase 4 record has detector_type='baseline'."""
        session, DW, AD = baseline_db
        records = (
            session.query(AD).filter(AD.detector_type == "baseline").all()
        )
        for rec in records:
            assert rec.detector_type == "baseline"

    def test_correct_risk_score_persisted(self, baseline_db):
        """risk_score is in [0, 100] for all persisted records."""
        session, DW, AD = baseline_db
        records = (
            session.query(AD).filter(AD.detector_type == "baseline").all()
        )
        for rec in records:
            assert 0 <= rec.risk_score <= 100, (
                f"risk_score={rec.risk_score} out of range for "
                f"window_id={rec.window_id}"
            )

    def test_correct_is_flagged_persisted(self, baseline_db):
        """is_flagged is a boolean for all persisted records."""
        session, DW, AD = baseline_db
        records = (
            session.query(AD).filter(AD.detector_type == "baseline").all()
        )
        for rec in records:
            assert isinstance(rec.is_flagged, (bool, int)), (
                f"is_flagged is not boolean for window_id={rec.window_id}"
            )

    def test_explanation_persisted(self, baseline_db):
        """explanation is a non-empty string containing 'Baseline detector'."""
        session, DW, AD = baseline_db
        records = (
            session.query(AD).filter(AD.detector_type == "baseline").all()
        )
        for rec in records:
            assert rec.explanation is not None
            assert len(rec.explanation) > 0
            assert "Baseline detector" in rec.explanation

    def test_scored_count_matches_dev_test_windows(self, baseline_db):
        """Number of baseline records equals number of dev_test windows."""
        session, DW, AD = baseline_db
        dev_test_count = (
            session.query(DW).filter(DW.split == "dev_test").count()
        )
        baseline_count = (
            session.query(AD).filter(AD.detector_type == "baseline").count()
        )
        assert baseline_count == dev_test_count, (
            f"Expected {dev_test_count} baseline records, got {baseline_count}"
        )


# ===========================================================================
# E. IDEMPOTENCY TESTS
# ===========================================================================


class TestBaselineIdempotency:
    """Verify re-running baseline detector does not duplicate records."""

    def test_first_run_creates_records(self, idempotency_db):
        """First run of baseline detector creates AnomalyDetection records."""
        session, DW, AD = idempotency_db
        run_baseline_detector(session=session)
        count = session.query(AD).filter(AD.detector_type == "baseline").count()
        assert count > 0, "First run created no baseline records"

    def test_second_run_no_duplicates(self, idempotency_db):
        """Running detector twice produces the same record count."""
        session, DW, AD = idempotency_db
        first_count = (
            session.query(AD).filter(AD.detector_type == "baseline").count()
        )
        # Run again
        run_baseline_detector(session=session)
        second_count = (
            session.query(AD).filter(AD.detector_type == "baseline").count()
        )
        assert second_count == first_count, (
            f"Duplicate records: first run={first_count}, "
            f"second run={second_count}"
        )

    def test_other_detector_types_preserved(self, idempotency_db):
        """Re-running baseline detector does not delete other detector types."""
        session, DW, AD = idempotency_db

        # Insert a fake 'ml' record for a dev_test window
        dev_test_window = (
            session.query(DW).filter(DW.split == "dev_test").first()
        )
        from models import AnomalyDetection as ADModel

        fake_ml = ADModel(
            window_id=dev_test_window.id,
            detector_type="ml",
            risk_score=42.0,
            is_flagged=True,
            explanation="Fake ML record for idempotency test",
        )
        session.add(fake_ml)
        session.commit()

        ml_count_before = (
            session.query(AD).filter(AD.detector_type == "ml").count()
        )
        assert ml_count_before >= 1

        # Re-run baseline detector
        run_baseline_detector(session=session)

        ml_count_after = (
            session.query(AD).filter(AD.detector_type == "ml").count()
        )
        assert ml_count_after == ml_count_before, (
            f"ML records were deleted: before={ml_count_before}, "
            f"after={ml_count_after}"
        )


# ===========================================================================
# F. FINAL-HOLDOUT PROTECTION TESTS
# ===========================================================================


class TestBaselineHoldoutProtection:
    """Verify the 6-layer holdout safety system."""

    def test_no_holdout_anomaly_records(self, baseline_db):
        """No AnomalyDetection records exist for final_holdout windows."""
        session, DW, AD = baseline_db
        holdout_window_ids = [
            w.id
            for w in session.query(DW).filter(DW.split == "final_holdout").all()
        ]
        if holdout_window_ids:
            holdout_anomaly_count = (
                session.query(AD)
                .filter(AD.window_id.in_(holdout_window_ids))
                .count()
            )
            assert holdout_anomaly_count == 0, (
                f"Found {holdout_anomaly_count} AnomalyDetection records "
                f"for final_holdout windows"
            )

    def test_predict_rejects_holdout_dataframe(self, fitted_detector):
        """predict() raises ValueError when given final_holdout rows."""
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M001",
                    "transaction_count": 20,
                    "total_amount": 200.0,
                    "avg_transaction_amount": 10.0,
                    "split": "final_holdout",
                }
            ]
        )
        with pytest.raises(ValueError, match="final_holdout"):
            fitted_detector.predict(df)

    def test_predict_rejects_mixed_with_holdout(self, fitted_detector):
        """predict() rejects a DataFrame containing any final_holdout row."""
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M001",
                    "transaction_count": 20,
                    "total_amount": 200.0,
                    "avg_transaction_amount": 10.0,
                    "split": "dev_test",
                },
                {
                    "id": 2,
                    "merchant_id": "M001",
                    "transaction_count": 20,
                    "total_amount": 200.0,
                    "avg_transaction_amount": 10.0,
                    "split": "final_holdout",
                },
            ]
        )
        with pytest.raises(ValueError, match="final_holdout"):
            fitted_detector.predict(df)

    def test_fit_rejects_dev_test_rows(self):
        """fit() rejects input containing dev_test rows."""
        df = pd.DataFrame(
            [
                {
                    "merchant_id": "M001",
                    "transaction_count": 10,
                    "total_amount": 100.0,
                    "avg_transaction_amount": 10.0,
                    "split": "dev_test",
                    "is_synthetic_fraud_spike": False,
                },
            ]
        )
        detector = BaselineDetector()
        with pytest.raises(ValueError, match="non-train split values"):
            detector.fit(df)

    def test_only_dev_test_eligible_for_persistence(self, baseline_db):
        """All persisted baseline records reference dev_test windows only."""
        session, DW, AD = baseline_db
        baseline_records = (
            session.query(AD).filter(AD.detector_type == "baseline").all()
        )
        for rec in baseline_records:
            window = session.query(DW).filter(DW.id == rec.window_id).first()
            assert window is not None
            assert window.split == "dev_test", (
                f"Baseline record persisted for split={window.split}"
            )


# ===========================================================================
# G. DETERMINISM TESTS
# ===========================================================================


class TestBaselineDeterminism:
    """Verify identical input produces identical output."""

    def test_same_input_same_predictions(self, unit_train_df):
        """Fitting and predicting twice produces identical results."""
        dev_test = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M001",
                    "transaction_count": 45,
                    "total_amount": 350.0,
                    "avg_transaction_amount": 11.5,
                    "split": "dev_test",
                },
                {
                    "id": 2,
                    "merchant_id": "M002",
                    "transaction_count": 5,
                    "total_amount": 50.0,
                    "avg_transaction_amount": 10.0,
                    "split": "dev_test",
                },
            ]
        )

        det1 = BaselineDetector(zscore_threshold=2.0)
        det1.fit(unit_train_df)
        results1 = det1.predict(dev_test)

        det2 = BaselineDetector(zscore_threshold=2.0)
        det2.fit(unit_train_df)
        results2 = det2.predict(dev_test)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["risk_score"] == r2["risk_score"]
            assert r1["is_flagged"] == r2["is_flagged"]
            assert r1["z_scores"] == r2["z_scores"]

    def test_same_input_same_explanations(self, unit_train_df):
        """Explanations are deterministic — character-for-character identical."""
        dev_test = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M001",
                    "transaction_count": 40,
                    "total_amount": 200.0,
                    "avg_transaction_amount": 10.0,
                    "split": "dev_test",
                },
            ]
        )

        det1 = BaselineDetector(zscore_threshold=2.0)
        det1.fit(unit_train_df)
        results1 = det1.predict(dev_test)

        det2 = BaselineDetector(zscore_threshold=2.0)
        det2.fit(unit_train_df)
        results2 = det2.predict(dev_test)

        assert results1[0]["explanation"] == results2[0]["explanation"]
