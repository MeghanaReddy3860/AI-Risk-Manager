"""
Phase 5 — ML Anomaly Detector (Isolation Forest) Tests

Verifies:
  A. Model initialization
  B. Training (fit)
  C. Prediction
  D. Risk score
  E. Explanation
  F. Database persistence
  G. Idempotency
  H. Holdout protection
  I. Determinism
  J. Feature contract / label leakage
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

from ml_anomaly_detector import (
    ML_FEATURES,
    ML_CONTAMINATION,
    ML_N_ESTIMATORS,
    ML_RANDOM_STATE,
    MLAnomalyDetector,
    run_ml_anomaly_detector,
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


@pytest.fixture(scope="module")
def unit_train_df():
    """Small train DataFrame for fast unit tests."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(50):
        rows.append(
            {
                "merchant_id": f"M{i % 5 + 1:03d}",
                "transaction_count": int(rng.integers(5, 30)),
                "total_amount": round(float(rng.uniform(50, 500)), 2),
                "avg_transaction_amount": round(float(rng.uniform(10, 50)), 2),
                "split": "train",
                "is_synthetic_fraud_spike": False,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def unit_dev_test_df():
    """Small dev_test DataFrame for fast unit tests."""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(20):
        rows.append(
            {
                "id": i + 1,
                "merchant_id": f"M{i % 5 + 1:03d}",
                "transaction_count": int(rng.integers(3, 100)),
                "total_amount": round(float(rng.uniform(20, 2000)), 2),
                "avg_transaction_amount": round(float(rng.uniform(5, 100)), 2),
                "split": "dev_test",
                "is_synthetic_fraud_spike": bool(rng.choice([True, False])),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_detector(unit_train_df):
    """MLAnomalyDetector fitted on unit_train_df."""
    detector = MLAnomalyDetector()
    detector.fit(unit_train_df)
    return detector


# ---------------------------------------------------------------------------
# Fixtures — Phase 3 data for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generated_data():
    """Generate Phase 3 synthetic data once for integration tests."""
    rng = np.random.default_rng(RANDOM_SEED)
    txns_df = generate_transactions(rng)
    windows_df = aggregate_detection_windows(txns_df)
    return txns_df, windows_df


@pytest.fixture(scope="class")
def ml_db(generated_data):
    """In-memory DB with Phase 3 data seeded and ML detector run once."""
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
    run_ml_anomaly_detector(session=session)

    yield session, DetectionWindow, AnomalyDetection
    session.close()


@pytest.fixture(scope="class")
def idempotency_db(generated_data):
    """In-memory DB with Phase 3 data seeded — tests manage detector runs."""
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
# A. MODEL INITIALIZATION
# ===========================================================================


class TestMLInitialization:
    """Verify detector initialization and configuration."""

    def test_detector_initializes(self):
        """Detector can be instantiated with defaults."""
        detector = MLAnomalyDetector()
        assert detector is not None
        assert detector._is_fitted is False

    def test_deterministic_random_state(self):
        """random_state is set to a deterministic value."""
        detector = MLAnomalyDetector()
        assert detector.random_state == ML_RANDOM_STATE
        assert isinstance(detector.random_state, int)

    def test_expected_hyperparameters(self):
        """n_estimators and contamination match config."""
        detector = MLAnomalyDetector()
        assert detector.n_estimators == ML_N_ESTIMATORS
        assert detector.contamination == ML_CONTAMINATION


# ===========================================================================
# B. TRAINING
# ===========================================================================


class TestMLTraining:
    """Verify model fitting from train data."""

    def test_model_fits_on_train(self, unit_train_df):
        """Model fits successfully on train-only data."""
        detector = MLAnomalyDetector()
        detector.fit(unit_train_df)
        assert detector._is_fitted is True
        assert detector.model is not None

    def test_fit_requires_train_partition(self):
        """fit() rejects data without split='train'."""
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
        detector = MLAnomalyDetector()
        with pytest.raises(ValueError, match="non-train split values"):
            detector.fit(df)

    def test_fit_rejects_final_holdout(self):
        """fit() rejects final_holdout rows."""
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
        detector = MLAnomalyDetector()
        with pytest.raises(ValueError, match="non-train split values"):
            detector.fit(df)

    def test_fit_uses_three_features_only(self, fitted_detector):
        """Model is fitted — implicitly uses only ML_FEATURES columns."""
        # The model object exists and has n_features_in_ == 3
        assert fitted_detector.model.n_features_in_ == len(ML_FEATURES)

    def test_fraud_labels_not_used_in_fit(self, unit_train_df):
        """Changing fraud labels does not change fitted model behavior."""
        dev_test = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M001",
                    "transaction_count": 50,
                    "total_amount": 1000.0,
                    "avg_transaction_amount": 80.0,
                    "split": "dev_test",
                },
            ]
        )

        # Fit with original labels
        det1 = MLAnomalyDetector()
        det1.fit(unit_train_df)
        results1 = det1.predict(dev_test)

        # Fit with all labels flipped
        modified = unit_train_df.copy()
        modified["is_synthetic_fraud_spike"] = True
        det2 = MLAnomalyDetector()
        det2.fit(modified)
        results2 = det2.predict(dev_test)

        assert results1[0]["risk_score"] == results2[0]["risk_score"]
        assert results1[0]["is_flagged"] == results2[0]["is_flagged"]


# ===========================================================================
# C. PREDICTION
# ===========================================================================


class TestMLPrediction:
    """Verify prediction output structure and values."""

    def test_prediction_works_after_fit(self, fitted_detector, unit_dev_test_df):
        """predict() returns results after fitting."""
        results = fitted_detector.predict(unit_dev_test_df)
        assert len(results) == len(unit_dev_test_df)

    def test_predict_before_fit_raises(self):
        """predict() raises RuntimeError if not fitted."""
        detector = MLAnomalyDetector()
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

    def test_output_contains_required_fields(self, fitted_detector, unit_dev_test_df):
        """Each prediction dict contains all required fields."""
        results = fitted_detector.predict(unit_dev_test_df)
        required = {"merchant_id", "risk_score", "is_flagged", "explanation"}
        for r in results:
            assert required.issubset(r.keys())

    def test_is_flagged_is_boolean(self, fitted_detector, unit_dev_test_df):
        """is_flagged is a boolean."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert isinstance(r["is_flagged"], bool)

    def test_anomaly_maps_to_flagged(self, fitted_detector, unit_dev_test_df):
        """IsolationForest predict()=-1 maps to is_flagged=True."""
        results = fitted_detector.predict(unit_dev_test_df)
        X = unit_dev_test_df[ML_FEATURES].values
        raw_preds = fitted_detector.model.predict(X)
        for i, r in enumerate(results):
            if raw_preds[i] == -1:
                assert r["is_flagged"] is True
            else:
                assert r["is_flagged"] is False

    def test_window_id_included_when_id_present(
        self, fitted_detector, unit_dev_test_df
    ):
        """window_id is included when 'id' column exists."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert "window_id" in r


# ===========================================================================
# D. RISK SCORE
# ===========================================================================


class TestMLRiskScore:
    """Verify risk score range and determinism."""

    def test_risk_score_always_ge_0(self, fitted_detector, unit_dev_test_df):
        """risk_score >= 0 for all predictions."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert r["risk_score"] >= 0, f"risk_score={r['risk_score']} < 0"

    def test_risk_score_always_le_100(self, fitted_detector, unit_dev_test_df):
        """risk_score <= 100 for all predictions."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert r["risk_score"] <= 100, f"risk_score={r['risk_score']} > 100"

    def test_risk_score_deterministic(self, unit_train_df, unit_dev_test_df):
        """Same input produces same risk scores."""
        det1 = MLAnomalyDetector()
        det1.fit(unit_train_df)
        r1 = det1.predict(unit_dev_test_df)

        det2 = MLAnomalyDetector()
        det2.fit(unit_train_df)
        r2 = det2.predict(unit_dev_test_df)

        for a, b in zip(r1, r2):
            assert a["risk_score"] == b["risk_score"]

    def test_risk_score_not_nan(self, fitted_detector, unit_dev_test_df):
        """No NaN risk scores."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert not np.isnan(r["risk_score"])


# ===========================================================================
# E. EXPLANATION
# ===========================================================================


class TestMLExplanation:
    """Verify explanation content and determinism."""

    def test_explanation_exists(self, fitted_detector, unit_dev_test_df):
        """explanation is a non-empty string."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert isinstance(r["explanation"], str)
            assert len(r["explanation"]) > 0

    def test_explanation_contains_isolation_forest(
        self, fitted_detector, unit_dev_test_df
    ):
        """explanation mentions 'Isolation Forest'."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert "Isolation Forest" in r["explanation"]

    def test_explanation_contains_prediction(
        self, fitted_detector, unit_dev_test_df
    ):
        """explanation contains the model prediction value."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert "prediction=" in r["explanation"]

    def test_explanation_contains_severity(
        self, fitted_detector, unit_dev_test_df
    ):
        """explanation contains severity/risk information."""
        results = fitted_detector.predict(unit_dev_test_df)
        for r in results:
            assert "severity=" in r["explanation"]

    def test_explanation_deterministic(self, unit_train_df, unit_dev_test_df):
        """Same input produces identical explanations."""
        det1 = MLAnomalyDetector()
        det1.fit(unit_train_df)
        r1 = det1.predict(unit_dev_test_df)

        det2 = MLAnomalyDetector()
        det2.fit(unit_train_df)
        r2 = det2.predict(unit_dev_test_df)

        for a, b in zip(r1, r2):
            assert a["explanation"] == b["explanation"]


# ===========================================================================
# F. DATABASE PERSISTENCE
# ===========================================================================


class TestMLDatabasePersistence:
    """Verify run_ml_anomaly_detector() persists AnomalyDetection records."""

    def test_ml_records_inserted(self, ml_db):
        """ML detector creates AnomalyDetection records."""
        session, DW, AD = ml_db
        count = session.query(AD).filter(AD.detector_type == "ml").count()
        assert count > 0, "No ML AnomalyDetection records created"

    def test_correct_window_id_mapping(self, ml_db):
        """Every ML record references an existing DetectionWindow."""
        session, DW, AD = ml_db
        records = session.query(AD).filter(AD.detector_type == "ml").all()
        for rec in records:
            window = session.query(DW).filter(DW.id == rec.window_id).first()
            assert window is not None

    def test_detector_type_always_ml(self, ml_db):
        """All Phase 5 records have detector_type='ml'."""
        session, DW, AD = ml_db
        records = session.query(AD).filter(AD.detector_type == "ml").all()
        for rec in records:
            assert rec.detector_type == "ml"

    def test_risk_score_persisted(self, ml_db):
        """risk_score is in [0, 100] for all persisted records."""
        session, DW, AD = ml_db
        records = session.query(AD).filter(AD.detector_type == "ml").all()
        for rec in records:
            assert 0 <= rec.risk_score <= 100

    def test_is_flagged_persisted(self, ml_db):
        """is_flagged is persisted for all ML records."""
        session, DW, AD = ml_db
        records = session.query(AD).filter(AD.detector_type == "ml").all()
        for rec in records:
            assert isinstance(rec.is_flagged, (bool, int))

    def test_explanation_persisted(self, ml_db):
        """explanation is a non-empty string containing 'Isolation Forest'."""
        session, DW, AD = ml_db
        records = session.query(AD).filter(AD.detector_type == "ml").all()
        for rec in records:
            assert rec.explanation is not None
            assert "Isolation Forest" in rec.explanation

    def test_only_dev_test_windows_scored(self, ml_db):
        """ML records exist only for dev_test windows."""
        session, DW, AD = ml_db
        records = session.query(AD).filter(AD.detector_type == "ml").all()
        for rec in records:
            window = session.query(DW).filter(DW.id == rec.window_id).first()
            assert window.split == "dev_test"

    def test_scored_count_matches_dev_test(self, ml_db):
        """Number of ML records equals number of dev_test windows."""
        session, DW, AD = ml_db
        dev_test_count = session.query(DW).filter(DW.split == "dev_test").count()
        ml_count = session.query(AD).filter(AD.detector_type == "ml").count()
        assert ml_count == dev_test_count


# ===========================================================================
# G. IDEMPOTENCY
# ===========================================================================


class TestMLIdempotency:
    """Verify re-running ML detector does not duplicate records."""

    def test_first_run_creates_records(self, idempotency_db):
        """First run creates ML AnomalyDetection records."""
        session, DW, AD = idempotency_db
        run_ml_anomaly_detector(session=session)
        count = session.query(AD).filter(AD.detector_type == "ml").count()
        assert count > 0

    def test_second_run_no_duplicates(self, idempotency_db):
        """Running detector twice produces the same record count."""
        session, DW, AD = idempotency_db
        first_count = session.query(AD).filter(AD.detector_type == "ml").count()
        run_ml_anomaly_detector(session=session)
        second_count = session.query(AD).filter(AD.detector_type == "ml").count()
        assert second_count == first_count

    def test_baseline_records_untouched(self, idempotency_db):
        """Running ML detector does not delete baseline records."""
        session, DW, AD = idempotency_db
        from baseline_detector import run_baseline_detector

        # Ensure baseline records exist
        run_baseline_detector(session=session)
        baseline_before = (
            session.query(AD).filter(AD.detector_type == "baseline").count()
        )
        assert baseline_before > 0

        # Re-run ML detector
        run_ml_anomaly_detector(session=session)

        baseline_after = (
            session.query(AD).filter(AD.detector_type == "baseline").count()
        )
        assert baseline_after == baseline_before


# ===========================================================================
# H. HOLDOUT PROTECTION
# ===========================================================================


class TestMLHoldoutProtection:
    """Verify the multi-layer holdout safety system."""

    def test_no_holdout_anomaly_records(self, ml_db):
        """No ML AnomalyDetection records for final_holdout windows."""
        session, DW, AD = ml_db
        holdout_ids = [
            w.id
            for w in session.query(DW).filter(DW.split == "final_holdout").all()
        ]
        if holdout_ids:
            count = (
                session.query(AD)
                .filter(AD.detector_type == "ml", AD.window_id.in_(holdout_ids))
                .count()
            )
            assert count == 0

    def test_predict_rejects_holdout(self, fitted_detector):
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

    def test_predict_rejects_mixed_holdout(self, fitted_detector):
        """predict() rejects DataFrame containing any final_holdout row."""
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

    def test_fit_rejects_holdout(self):
        """fit() rejects final_holdout rows."""
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
        detector = MLAnomalyDetector()
        with pytest.raises(ValueError, match="non-train split values"):
            detector.fit(df)


# ===========================================================================
# I. DETERMINISM
# ===========================================================================


class TestMLDeterminism:
    """Verify identical input produces identical output."""

    def test_same_training_same_predictions(self, unit_train_df, unit_dev_test_df):
        """Fitting twice with same data produces same predictions."""
        det1 = MLAnomalyDetector()
        det1.fit(unit_train_df)
        r1 = det1.predict(unit_dev_test_df)

        det2 = MLAnomalyDetector()
        det2.fit(unit_train_df)
        r2 = det2.predict(unit_dev_test_df)

        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a["is_flagged"] == b["is_flagged"]
            assert a["risk_score"] == b["risk_score"]
            assert a["explanation"] == b["explanation"]


# ===========================================================================
# J. FEATURE CONTRACT / LABEL LEAKAGE
# ===========================================================================


class TestMLFeatureContract:
    """Verify only approved features are used; no label leakage."""

    def test_model_uses_exactly_three_features(self, fitted_detector):
        """IsolationForest is trained on exactly 3 features."""
        assert fitted_detector.model.n_features_in_ == 3

    def test_merchant_id_not_ml_feature(self, unit_train_df):
        """merchant_id is metadata, not an ML feature."""
        # ML_FEATURES must not include merchant_id
        assert "merchant_id" not in ML_FEATURES

    def test_split_not_ml_feature(self):
        """split is not an ML feature."""
        assert "split" not in ML_FEATURES

    def test_fraud_label_not_ml_feature(self):
        """is_synthetic_fraud_spike is not an ML feature."""
        assert "is_synthetic_fraud_spike" not in ML_FEATURES

    def test_window_id_not_ml_feature(self):
        """window_id / id is not an ML feature."""
        assert "id" not in ML_FEATURES
        assert "window_id" not in ML_FEATURES

    def test_label_change_no_prediction_change(self, unit_train_df):
        """Changing fraud labels does not change predictions."""
        dev_test = pd.DataFrame(
            [
                {
                    "id": 1,
                    "merchant_id": "M001",
                    "transaction_count": 15,
                    "total_amount": 300.0,
                    "avg_transaction_amount": 20.0,
                    "split": "dev_test",
                },
            ]
        )

        det1 = MLAnomalyDetector()
        det1.fit(unit_train_df)
        r1 = det1.predict(dev_test)

        modified = unit_train_df.copy()
        modified["is_synthetic_fraud_spike"] = True
        det2 = MLAnomalyDetector()
        det2.fit(modified)
        r2 = det2.predict(dev_test)

        assert r1[0]["risk_score"] == r2[0]["risk_score"]
        assert r1[0]["is_flagged"] == r2[0]["is_flagged"]
        assert r1[0]["explanation"] == r2[0]["explanation"]
