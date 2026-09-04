"""
AI Risk Manager — Phase 12.1B Test Suite
=========================================

Tests for:
1. Detector execution via POST /api/pipeline/run-detectors
2. Integration with analyze-window / analysis endpoint
3. Detection retrieval via GET /api/windows/{window_id}/detections
4. Execution idempotency (stable row counts)
5. final_holdout protection
6. Empty partitions handling
7. Defense-only safety checks
"""

from datetime import datetime, timedelta, timezone
import pytest
from models import AnomalyDetection, DetectionWindow, Transaction


def seed_test_windows(db):
    """Seed synthetic train, dev_test, and final_holdout DetectionWindow records."""
    now = datetime.now(timezone.utc)

    # Seed train windows (merchant_1)
    for i in range(10):
        db.add(
            DetectionWindow(
                merchant_id="merchant_1",
                window_start=now - timedelta(hours=50 - i),
                window_end=now - timedelta(hours=49 - i),
                transaction_count=20 + i,
                total_amount=2000.0 + (i * 100),
                avg_transaction_amount=100.0,
                is_synthetic_fraud_spike=False,
                split="train",
            )
        )

    # Seed dev_test windows (merchant_1 & merchant_2)
    dev_test_window_1 = DetectionWindow(
        merchant_id="merchant_1",
        window_start=now - timedelta(hours=10),
        window_end=now - timedelta(hours=9),
        transaction_count=25,
        total_amount=2500.0,
        avg_transaction_amount=100.0,
        is_synthetic_fraud_spike=False,
        split="dev_test",
    )
    dev_test_window_2 = DetectionWindow(
        merchant_id="merchant_1",
        window_start=now - timedelta(hours=8),
        window_end=now - timedelta(hours=7),
        transaction_count=150,  # Spike
        total_amount=25000.0,
        avg_transaction_amount=166.67,
        is_synthetic_fraud_spike=True,
        split="dev_test",
    )
    db.add(dev_test_window_1)
    db.add(dev_test_window_2)

    # Seed final_holdout window (must remain protected and untouched)
    holdout_window = DetectionWindow(
        merchant_id="merchant_1",
        window_start=now - timedelta(hours=2),
        window_end=now - timedelta(hours=1),
        transaction_count=30,
        total_amount=3000.0,
        avg_transaction_amount=100.0,
        is_synthetic_fraud_spike=False,
        split="final_holdout",
    )
    db.add(holdout_window)
    db.commit()

    return dev_test_window_1.id, dev_test_window_2.id, holdout_window.id


def test_detector_execution(client, db_session):
    """TEST 1: Verify POST /api/pipeline/run-detectors populates anomaly_detections."""
    w1_id, w2_id, _ = seed_test_windows(db_session)

    res = client.post("/api/pipeline/run-detectors")
    assert res.status_code == 200
    data = res.json()

    assert "baseline" in data["detectors_run"]
    assert "ml" in data["detectors_run"]
    assert data["results"]["baseline"]["windows_scored"] == 2
    assert data["results"]["ml"]["windows_scored"] == 2

    # Check database records
    detections = db_session.query(AnomalyDetection).all()
    assert len(detections) == 4  # 2 windows x 2 detectors
    detector_types = set(d.detector_type for d in detections)
    assert detector_types == {"baseline", "ml"}


def test_analysis_integration(client, db_session):
    """TEST 2: Verify analyze-window returns real stored detector results after pipeline run."""
    w1_id, w2_id, _ = seed_test_windows(db_session)

    # Run detector pipeline first
    client.post("/api/pipeline/run-detectors")

    # Fetch stored detection result
    stored_baseline = (
        db_session.query(AnomalyDetection)
        .filter(AnomalyDetection.window_id == w2_id, AnomalyDetection.detector_type == "baseline")
        .first()
    )
    assert stored_baseline is not None

    # Call analyze-window
    res = client.post(f"/api/pipeline/analyze-window/{w2_id}?detector_type=baseline")
    assert res.status_code == 200
    dossier = res.json()

    assert dossier["detector_type"] == "baseline"
    assert dossier["risk_result"]["risk_score"] == stored_baseline.risk_score
    assert dossier["is_flagged"] == stored_baseline.is_flagged


def test_detection_retrieval(client, db_session):
    """TEST 3: Verify GET /api/windows/{window_id}/detections retrieves persisted results without computation."""
    w1_id, w2_id, _ = seed_test_windows(db_session)

    client.post("/api/pipeline/run-detectors")

    res = client.get(f"/api/windows/{w2_id}/detections")
    assert res.status_code == 200
    detections = res.json()

    assert len(detections) == 2
    detector_types = set(d["detector_type"] for d in detections)
    assert detector_types == {"baseline", "ml"}
    for d in detections:
        assert d["window_id"] == w2_id
        assert "risk_score" in d
        assert "is_flagged" in d


def test_idempotency(client, db_session):
    """TEST 4: Verify repeated execution replaces old records and prevents duplicate accumulation."""
    w1_id, w2_id, _ = seed_test_windows(db_session)

    # First run
    client.post("/api/pipeline/run-detectors")
    initial_count = db_session.query(AnomalyDetection).count()
    assert initial_count == 4

    # Second run
    client.post("/api/pipeline/run-detectors")
    second_count = db_session.query(AnomalyDetection).count()
    assert second_count == 4  # Count stays identical, old rows replaced cleanly


def test_final_holdout_protection(client, db_session):
    """TEST 5: Verify final_holdout windows are never scored or populated with detections."""
    _, _, holdout_id = seed_test_windows(db_session)

    # Confirm holdout has 0 detections initially
    initial_holdout_detections = (
        db_session.query(AnomalyDetection)
        .filter(AnomalyDetection.window_id == holdout_id)
        .count()
    )
    assert initial_holdout_detections == 0

    client.post("/api/pipeline/run-detectors")

    # Confirm holdout STILL has 0 detections
    post_holdout_detections = (
        db_session.query(AnomalyDetection)
        .filter(AnomalyDetection.window_id == holdout_id)
        .count()
    )
    assert post_holdout_detections == 0

    # Confirm GET /windows/{holdout_id}/detections returns 404 Not Found
    res = client.get(f"/api/windows/{holdout_id}/detections")
    assert res.status_code == 404


def test_empty_partitions(client, db_session):
    """TEST 6: Verify behavior when train or dev_test partitions contain no records."""
    res = client.post("/api/pipeline/run-detectors")
    assert res.status_code == 200
    data = res.json()

    # With empty database, 0 windows scored
    for dt in data["results"]:
        assert data["results"][dt]["windows_scored"] == 0
        assert data["results"][dt]["windows_flagged"] == 0


def test_defense_only_safety(client, db_session):
    """TEST 7: Verify pipeline execution strictly performs defensive detection and no punitive actions."""
    w1_id, w2_id, _ = seed_test_windows(db_session)

    res = client.post("/api/pipeline/run-detectors")
    assert res.status_code == 200

    # Ensure no punitive fields exist in database models or pipeline outputs
    detections = db_session.query(AnomalyDetection).all()
    for d in detections:
        assert hasattr(d, "risk_score")
        assert hasattr(d, "is_flagged")
        # Ensure no ban/block attributes exist
        assert not hasattr(d, "account_suspended")
        assert not hasattr(d, "merchant_blocked")


# ===========================================================================
# Phase 12.1C Read-Only Analysis Tests
# ===========================================================================

def test_analysis_missing_result_does_not_compute(client, db_session, monkeypatch):
    """TEST 12.1C-1: Verify GET /analysis on window with no stored detection returns 404 and does not compute."""
    w1_id, w2_id, _ = seed_test_windows(db_session)

    # Monkeypatch detector execution functions to fail test if invoked by GET request
    def _fail_if_called(*args, **kwargs):
        pytest.fail("Detector function was invoked during a read-only GET /analysis call!")

    monkeypatch.setattr("baseline_detector.run_baseline_detector", _fail_if_called)
    monkeypatch.setattr("ml_anomaly_detector.run_ml_anomaly_detector", _fail_if_called)

    initial_rows = db_session.query(AnomalyDetection).count()
    assert initial_rows == 0

    # Request GET /api/windows/{w1_id}/analysis
    res = client.get(f"/api/windows/{w1_id}/analysis?detector_type=baseline")
    assert res.status_code == 404
    assert "No stored detection result found" in res.json()["detail"]

    # Verify no rows were created in AnomalyDetection table
    post_rows = db_session.query(AnomalyDetection).count()
    assert post_rows == 0


def test_analysis_read_only_database_verification(client, db_session):
    """TEST 12.1C-2: Verify GET /analysis does not mutate any database state or anomaly detection count."""
    w1_id, w2_id, _ = seed_test_windows(db_session)

    # Run detector pipeline to populate detections
    client.post("/api/pipeline/run-detectors")

    rows_before = db_session.query(AnomalyDetection).count()
    assert rows_before == 4

    # Issue multiple GET analysis requests
    res1 = client.get(f"/api/windows/{w2_id}/analysis?detector_type=baseline")
    assert res1.status_code == 200
    res2 = client.get(f"/api/windows/{w2_id}/analysis?detector_type=ml")
    assert res2.status_code == 200

    rows_after = db_session.query(AnomalyDetection).count()
    assert rows_after == rows_before


def test_analysis_get_endpoint_integration(client, db_session):
    """TEST 12.1C-3: Verify GET /api/windows/{window_id}/analysis returns stored baseline and ML results."""
    w1_id, w2_id, _ = seed_test_windows(db_session)

    client.post("/api/pipeline/run-detectors")

    # Call GET /api/windows/{w2_id}/analysis
    res_base = client.get(f"/api/windows/{w2_id}/analysis?detector_type=baseline")
    assert res_base.status_code == 200
    dossier_base = res_base.json()
    assert dossier_base["detector_type"] == "baseline"
    assert dossier_base["window"]["id"] == w2_id

    res_ml = client.get(f"/api/windows/{w2_id}/analysis?detector_type=ml")
    assert res_ml.status_code == 200
    dossier_ml = res_ml.json()
    assert dossier_ml["detector_type"] == "ml"
    assert dossier_ml["window"]["id"] == w2_id


def test_analysis_audit_trail_no_side_effects(client, db_session):
    """TEST 12.2C-FIX: Verify GET and POST analysis endpoints do NOT log audit trail events."""
    from main import app

    w1_id, w2_id, holdout_id = seed_test_windows(db_session)

    # Populate detections via pipeline
    client.post("/api/pipeline/run-detectors")

    audit_manager = app.state.audit_manager
    audit_count_before = len(audit_manager)

    # 1. Issue GET /analysis request
    res_get = client.get(f"/api/windows/{w2_id}/analysis?detector_type=baseline")
    assert res_get.status_code == 200

    audit_count_after_get = len(audit_manager)
    assert audit_count_after_get == audit_count_before

    # 2. Issue POST /analyze-window request (compatibility alias)
    res_post = client.post(f"/api/pipeline/analyze-window/{w2_id}?detector_type=baseline")
    assert res_post.status_code == 200

    audit_count_after_post = len(audit_manager)
    assert audit_count_after_post == audit_count_before

    # 3. Issue GET /analysis on final_holdout window
    res_holdout = client.get(f"/api/windows/{holdout_id}/analysis")
    assert res_holdout.status_code == 404

    audit_count_after_holdout = len(audit_manager)
    assert audit_count_after_holdout == audit_count_before


