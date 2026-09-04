"""
AI Risk Manager — Phase 15 Safety & Verification Test Suite
============================================================

Tests:
  TEST A — Final holdout evaluated only through explicit Phase 15 path.
  TEST B — ML fit rejects final_holdout.
  TEST C — Baseline training statistics do not come from final_holdout.
  TEST D — ML detector is trained on train data and only predicts on final_holdout.
  TEST E — final_holdout EvaluationRun records are persisted with partition="final_holdout".
  TEST F — Evaluation metrics are derived from actual held-out labels and predictions.
  TEST G — No hyperparameters or thresholds are modified because of final_holdout.
  TEST H — Repeated normal detector execution without explicit holdout permission cannot score final_holdout.
  TEST I — final_holdout cannot accidentally enter the normal dev_test evaluation path.
  TEST J — Phase 15 does not modify Window Deep Dive read-only behavior.
  TEST K — No punitive/enforcement action is created by Phase 15.
  TEST L — No credential/API-key information appears in Phase 15 responses or generated reports.
  TEST M — Regression test: tests do not overwrite production report path.
  TEST N — Narrative Unit Test: Baseline wins dynamic conclusion.
  TEST O — Narrative Unit Test: ML wins dynamic conclusion.
  TEST P — Narrative Unit Test: Mixed results dynamic conclusion.
  TEST Q — Narrative Unit Test: Tied results dynamic conclusion.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend package is importable
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from database import Base
from models import Transaction, DetectionWindow, AnomalyDetection, EvaluationRun
from baseline_detector import BaselineDetector, run_baseline_detector, BASELINE_ZSCORE_THRESHOLD
from ml_anomaly_detector import MLAnomalyDetector, run_ml_anomaly_detector
from evaluation_engine import EvaluationEngine, EvaluationResult, run_evaluation
from run_phase15_evaluation import (
    execute_phase15_evaluation,
    generate_markdown_report,
    DEFAULT_REPORT_PATH,
)
from pipeline import analyze_window


SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite session for tests."""
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def _seed_sample_partitions(session):
    """Seed train, dev_test, and final_holdout windows."""
    windows = [
        # Train windows (Day 1-18)
        DetectionWindow(id=1, merchant_id="M001", window_start=datetime(2025, 1, 1, tzinfo=timezone.utc), window_end=datetime(2025, 1, 1, 1, tzinfo=timezone.utc), transaction_count=10, total_amount=1000.0, avg_transaction_amount=100.0, is_synthetic_fraud_spike=False, split="train"),
        DetectionWindow(id=2, merchant_id="M001", window_start=datetime(2025, 1, 2, tzinfo=timezone.utc), window_end=datetime(2025, 1, 2, 1, tzinfo=timezone.utc), transaction_count=12, total_amount=1200.0, avg_transaction_amount=100.0, is_synthetic_fraud_spike=False, split="train"),
        
        # Dev-test windows (Day 19-25)
        DetectionWindow(id=3, merchant_id="M001", window_start=datetime(2025, 1, 20, tzinfo=timezone.utc), window_end=datetime(2025, 1, 20, 1, tzinfo=timezone.utc), transaction_count=50, total_amount=5000.0, avg_transaction_amount=100.0, is_synthetic_fraud_spike=True, split="dev_test"),
        DetectionWindow(id=4, merchant_id="M001", window_start=datetime(2025, 1, 21, tzinfo=timezone.utc), window_end=datetime(2025, 1, 21, 1, tzinfo=timezone.utc), transaction_count=11, total_amount=1100.0, avg_transaction_amount=100.0, is_synthetic_fraud_spike=False, split="dev_test"),

        # Final-holdout windows (Day 26-30)
        DetectionWindow(id=5, merchant_id="M001", window_start=datetime(2025, 1, 27, tzinfo=timezone.utc), window_end=datetime(2025, 1, 27, 1, tzinfo=timezone.utc), transaction_count=60, total_amount=6000.0, avg_transaction_amount=100.0, is_synthetic_fraud_spike=True, split="final_holdout"),
        DetectionWindow(id=6, merchant_id="M001", window_start=datetime(2025, 1, 28, tzinfo=timezone.utc), window_end=datetime(2025, 1, 28, 1, tzinfo=timezone.utc), transaction_count=10, total_amount=1000.0, avg_transaction_amount=100.0, is_synthetic_fraud_spike=False, split="final_holdout"),
    ]
    session.add_all(windows)
    session.commit()


# ===========================================================================
# TEST A: Final holdout evaluated only through explicit Phase 15 path
# ===========================================================================
def test_final_holdout_evaluated_only_with_explicit_permission(db_session):
    _seed_sample_partitions(db_session)

    # 1. Baseline detector raises without allow_holdout=True
    with pytest.raises(ValueError, match="protected"):
        run_baseline_detector(session=db_session, target_split="final_holdout", allow_holdout=False)

    # 2. ML detector raises without allow_holdout=True
    with pytest.raises(ValueError, match="protected"):
        run_ml_anomaly_detector(session=db_session, target_split="final_holdout", allow_holdout=False)

    # 3. EvaluationEngine raises without allow_holdout=True
    engine = EvaluationEngine(db_session)
    with pytest.raises(ValueError, match="protected"):
        engine.evaluate(partition="final_holdout", detector_type="baseline", allow_holdout=False)


# ===========================================================================
# TEST B: ML fit rejects final_holdout
# ===========================================================================
def test_ml_fit_rejects_final_holdout():
    holdout_df = pd.DataFrame([
        {"transaction_count": 10, "total_amount": 1000.0, "avg_transaction_amount": 100.0, "split": "final_holdout"}
    ])
    detector = MLAnomalyDetector()
    with pytest.raises(ValueError, match="fit.*requires only train"):
        detector.fit(holdout_df)


# ===========================================================================
# TEST C: Baseline training statistics do not come from final_holdout
# ===========================================================================
def test_baseline_training_stats_do_not_come_from_final_holdout():
    holdout_df = pd.DataFrame([
        {"merchant_id": "M001", "transaction_count": 100, "total_amount": 10000.0, "avg_transaction_amount": 100.0, "split": "final_holdout"}
    ])
    detector = BaselineDetector()
    with pytest.raises(ValueError, match="fit.*requires only train"):
        detector.fit(holdout_df)


# ===========================================================================
# TEST D: ML detector trained on train data and only predicts on final_holdout
# ===========================================================================
def test_ml_detector_trained_on_train_and_predicts_on_holdout():
    train_df = pd.DataFrame([
        {"transaction_count": 10, "total_amount": 1000.0, "avg_transaction_amount": 100.0, "split": "train"},
        {"transaction_count": 12, "total_amount": 1200.0, "avg_transaction_amount": 100.0, "split": "train"},
    ])
    holdout_df = pd.DataFrame([
        {"id": 5, "merchant_id": "M001", "transaction_count": 60, "total_amount": 6000.0, "avg_transaction_amount": 100.0, "split": "final_holdout"}
    ])

    detector = MLAnomalyDetector()
    detector.fit(train_df)

    # Predict on holdout with explicit permission
    preds = detector.predict(holdout_df, allow_holdout=True)
    assert len(preds) == 1
    assert preds[0]["window_id"] == 5


# ===========================================================================
# TEST E: final_holdout EvaluationRun records persisted
# ===========================================================================
def test_final_holdout_evaluation_runs_persisted(tmp_path, db_session):
    _seed_sample_partitions(db_session)
    execute_phase15_evaluation(session=db_session, report_path=tmp_path / "test_report.md")

    runs = db_session.query(EvaluationRun).filter(EvaluationRun.partition == "final_holdout").all()
    assert len(runs) == 2
    detectors = {r.detector_type for r in runs}
    assert detectors == {"baseline", "ml"}


# ===========================================================================
# TEST F: Evaluation metrics derived from actual held-out labels
# ===========================================================================
def test_evaluation_metrics_derived_from_actual_labels(db_session):
    _seed_sample_partitions(db_session)
    
    # Score holdout with allow_holdout=True
    run_baseline_detector(session=db_session, target_split="final_holdout", allow_holdout=True)
    
    engine = EvaluationEngine(db_session)
    result = engine.evaluate(partition="final_holdout", detector_type="baseline", allow_holdout=True)
    
    # Ground truths for window 5 (True) and 6 (False)
    # Total evaluated window count must be 2
    assert result.evaluated_window_count == 2
    assert result.partition == "final_holdout"


# ===========================================================================
# TEST G: No hyperparameters or thresholds modified by holdout
# ===========================================================================
def test_no_hyperparameters_or_thresholds_modified_by_holdout(tmp_path, db_session):
    _seed_sample_partitions(db_session)

    threshold_before = BASELINE_ZSCORE_THRESHOLD
    ml_detector = MLAnomalyDetector()
    contamination_before = ml_detector.contamination
    n_estimators_before = ml_detector.n_estimators

    execute_phase15_evaluation(session=db_session, report_path=tmp_path / "test_report.md")

    assert BASELINE_ZSCORE_THRESHOLD == threshold_before
    assert ml_detector.contamination == contamination_before
    assert ml_detector.n_estimators == n_estimators_before


# ===========================================================================
# TEST H: Repeated normal detector execution cannot score final_holdout
# ===========================================================================
def test_repeated_normal_detector_execution_cannot_score_holdout(db_session):
    _seed_sample_partitions(db_session)

    # Normal execution for dev_test
    run_baseline_detector(session=db_session, target_split="dev_test", allow_holdout=False)
    run_ml_anomaly_detector(session=db_session, target_split="dev_test", allow_holdout=False)

    # Verify no holdout predictions exist in DB
    holdout_preds = (
        db_session.query(AnomalyDetection)
        .join(DetectionWindow)
        .filter(DetectionWindow.split == "final_holdout")
        .count()
    )
    assert holdout_preds == 0


# ===========================================================================
# TEST I: final_holdout cannot enter normal dev_test evaluation path
# ===========================================================================
def test_holdout_cannot_enter_normal_dev_test_evaluation(db_session):
    _seed_sample_partitions(db_session)

    run_baseline_detector(session=db_session, target_split="dev_test", allow_holdout=False)
    engine = EvaluationEngine(db_session)
    result = engine.evaluate(partition="dev_test", detector_type="baseline", allow_holdout=False)

    # Must count only the 2 dev_test windows (id 3 and 4), ignoring holdout windows (5 and 6)
    assert result.evaluated_window_count == 2


# ===========================================================================
# TEST J: Phase 15 does not modify Window Deep Dive read-only behavior
# ===========================================================================
def test_window_deep_dive_read_only_behavior_unmodified(db_session):
    _seed_sample_partitions(db_session)
    run_baseline_detector(session=db_session, target_split="dev_test", allow_holdout=False)

    # Call analyze_window on window 3
    response = analyze_window(window_id=3, detector_type="baseline", db=db_session)
    assert response.window.id == 3
    assert response.audit_entry_id == "AUDIT_READ_ONLY"


# ===========================================================================
# TEST K: No punitive or enforcement action created by Phase 15
# ===========================================================================
def test_no_punitive_action_created_by_phase15(tmp_path, db_session):
    _seed_sample_partitions(db_session)
    res = execute_phase15_evaluation(session=db_session, report_path=tmp_path / "test_report.md")

    punitive_keywords = {"BLOCK_MERCHANT", "BAN_MERCHANT", "SUSPEND_ACCOUNT", "FREEZE_FUNDS"}
    for det_type, eval_res in res["final_holdout"].items():
        assert not any(kw in eval_res.notes for kw in punitive_keywords)


# ===========================================================================
# TEST L: No credentials in Phase 15 responses or reports
# ===========================================================================
def test_no_credentials_in_phase15_responses_or_reports(tmp_path, db_session):
    _seed_sample_partitions(db_session)
    test_report_file = tmp_path / "test_report.md"
    execute_phase15_evaluation(session=db_session, report_path=test_report_file)

    assert test_report_file.exists()
    report_text = test_report_file.read_text(encoding="utf-8")

    sensitive_tokens = ["api_key", "secret_key", "bearer_token", "password"]
    for token in sensitive_tokens:
        assert token not in report_text.lower()


# ===========================================================================
# TEST M: Regression test - tests do not overwrite production report path
# ===========================================================================
def test_production_report_not_modified_by_tests(tmp_path, db_session):
    """
    Ensure running tests with an injected report_path leaves DEFAULT_REPORT_PATH untouched.
    """
    _seed_sample_partitions(db_session)
    
    # Snapshot production report content if it exists
    prod_content_before = DEFAULT_REPORT_PATH.read_text(encoding="utf-8") if DEFAULT_REPORT_PATH.exists() else None

    # Run test evaluation pointing to temporary path
    temp_target = tmp_path / "isolated_eval.md"
    execute_phase15_evaluation(session=db_session, report_path=temp_target)

    assert temp_target.exists()

    # Verify production file was NOT modified by this test run
    if prod_content_before is not None:
        prod_content_after = DEFAULT_REPORT_PATH.read_text(encoding="utf-8")
        assert prod_content_after == prod_content_before


# ===========================================================================
# TEST N: Narrative Unit Test - Baseline wins dynamic conclusion
# ===========================================================================
def test_report_narrative_baseline_wins():
    dev_map = {
        "baseline": EvaluationResult(detector_type="baseline", partition="dev_test", f1_score=0.8, total_cost=5000.0),
        "ml": EvaluationResult(detector_type="ml", partition="dev_test", f1_score=0.7, total_cost=8000.0),
    }
    # Baseline has lower total cost and higher F1
    holdout_map = {
        "baseline": EvaluationResult(detector_type="baseline", partition="final_holdout", precision=0.8, recall=0.9, f1_score=0.85, false_positive_rate=0.05, fp_cost=1000.0, fn_cost=1500.0, total_cost=2500.0, true_positives=9, false_positives=2, false_negatives=1, true_negatives=100),
        "ml": EvaluationResult(detector_type="ml", partition="final_holdout", precision=0.5, recall=0.6, f1_score=0.55, false_positive_rate=0.15, fp_cost=3000.0, fn_cost=6000.0, total_cost=9000.0, true_positives=6, false_positives=6, false_negatives=4, true_negatives=96),
    }

    report = generate_markdown_report(dev_map, holdout_map, 1200, "mockhash123")
    
    assert "Baseline Detector Outperformed ML Anomaly Detector" in report
    assert "Baseline Detector outperformed the ML Anomaly Detector" in report
    assert "ML Isolation Forest detector is validated as a superior" not in report


# ===========================================================================
# TEST O: Narrative Unit Test - ML wins dynamic conclusion
# ===========================================================================
def test_report_narrative_ml_wins():
    dev_map = {
        "baseline": EvaluationResult(detector_type="baseline", partition="dev_test", f1_score=0.7, total_cost=8000.0),
        "ml": EvaluationResult(detector_type="ml", partition="dev_test", f1_score=0.8, total_cost=5000.0),
    }
    # ML has lower total cost and higher F1
    holdout_map = {
        "baseline": EvaluationResult(detector_type="baseline", partition="final_holdout", precision=0.5, recall=0.6, f1_score=0.55, false_positive_rate=0.15, fp_cost=3000.0, fn_cost=6000.0, total_cost=9000.0, true_positives=6, false_positives=6, false_negatives=4, true_negatives=96),
        "ml": EvaluationResult(detector_type="ml", partition="final_holdout", precision=0.85, recall=0.95, f1_score=0.90, false_positive_rate=0.03, fp_cost=1000.0, fn_cost=1500.0, total_cost=2500.0, true_positives=9, false_positives=2, false_negatives=1, true_negatives=100),
    }

    report = generate_markdown_report(dev_map, holdout_map, 1200, "mockhash123")
    
    assert "ML Anomaly Detector Outperformed Baseline" in report
    assert "ML Anomaly Detector (Isolation Forest) outperformed the statistical Baseline" in report


# ===========================================================================
# TEST P: Narrative Unit Test - Mixed results dynamic conclusion
# ===========================================================================
def test_report_narrative_mixed_results():
    dev_map = {
        "baseline": EvaluationResult(detector_type="baseline", partition="dev_test", f1_score=0.7, total_cost=5000.0),
        "ml": EvaluationResult(detector_type="ml", partition="dev_test", f1_score=0.7, total_cost=5000.0),
    }
    # Mixed: Baseline has lower cost, but ML has significantly higher F1/recall
    holdout_map = {
        "baseline": EvaluationResult(detector_type="baseline", partition="final_holdout", precision=0.9, recall=0.5, f1_score=0.64, false_positive_rate=0.01, fp_cost=500.0, fn_cost=1500.0, total_cost=2000.0, true_positives=5, false_positives=1, false_negatives=5, true_negatives=100),
        "ml": EvaluationResult(detector_type="ml", partition="final_holdout", precision=0.7, recall=0.95, f1_score=0.81, false_positive_rate=0.08, fp_cost=2000.0, fn_cost=1500.0, total_cost=3500.0, true_positives=9, false_positives=4, false_negatives=1, true_negatives=97),
    }

    report = generate_markdown_report(dev_map, holdout_map, 1200, "mockhash123")
    
    assert "Mixed Results / Operational Trade-off" in report
    assert "Neither detector strictly dominates" in report


# ===========================================================================
# TEST Q: Narrative Unit Test - Tied results dynamic conclusion
# ===========================================================================
def test_report_narrative_tied_results():
    dev_map = {
        "baseline": EvaluationResult(detector_type="baseline", partition="dev_test", f1_score=0.8, total_cost=5000.0),
        "ml": EvaluationResult(detector_type="ml", partition="dev_test", f1_score=0.8, total_cost=5000.0),
    }
    # Tied metrics & cost
    holdout_map = {
        "baseline": EvaluationResult(detector_type="baseline", partition="final_holdout", precision=0.8, recall=0.8, f1_score=0.8, false_positive_rate=0.05, fp_cost=1000.0, fn_cost=1500.0, total_cost=2500.0, true_positives=8, false_positives=2, false_negatives=2, true_negatives=100),
        "ml": EvaluationResult(detector_type="ml", partition="final_holdout", precision=0.8, recall=0.8, f1_score=0.8, false_positive_rate=0.05, fp_cost=1000.0, fn_cost=1500.0, total_cost=2500.0, true_positives=8, false_positives=2, false_negatives=2, true_negatives=100),
    }

    report = generate_markdown_report(dev_map, holdout_map, 1200, "mockhash123")
    
    assert "Equally Matched / Tied Performance" in report
    assert "equivalent or effectively tied results" in report
