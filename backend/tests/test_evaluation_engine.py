"""
AI Risk Manager — Phase 6: Evaluation Engine Tests
====================================================

Comprehensive test suite for backend/evaluation_engine.py.

Test classes:
  TestEvaluationMetrics     — precision, recall, F1, FPR formulas
  TestConfusionMatrix       — TP/FP/FN/TN counts
  TestCostCalculation       — fp_cost, fn_cost, total_cost
  TestDetectorSeparation    — baseline/ML evaluated independently
  TestPartitionProtection   — dev_test only; final_holdout rejected
  TestMissingPredictions    — missing predictions raise ValueError
  TestDuplicatePredictions  — duplicate predictions raise ValueError
  TestPersistence           — 2 EvaluationRun records created correctly
  TestDeterminism           — identical metrics on repeated evaluation runs
  TestNoMutation            — evaluation does not mutate DB records

All tests use an isolated in-memory SQLite database.

Protected partitions are verified by querying SUPPORTED_PARTITIONS and
asserting that the engine does not secretly accept "final_holdout".
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from evaluation_engine import (
    EvaluationEngine,
    EvaluationResult,
    SUPPORTED_PARTITIONS,
    DETECTOR_TYPES,
    COST_PER_FALSE_POSITIVE,
    COST_PER_FALSE_NEGATIVE,
    _safe_precision,
    _safe_recall,
    _safe_f1,
    _safe_fpr,
    _compute_costs,
    _build_confusion_matrix,
    run_evaluation,
)
from database import Base
from models import (
    Transaction,
    DetectionWindow,
    AnomalyDetection,
    EvaluationRun,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite session for each test."""
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


def _make_window(
    session,
    *,
    id_: int,
    is_fraud: bool,
    split: str = "dev_test",
    merchant_id: str = "M001",
) -> DetectionWindow:
    """Helper: create and flush a DetectionWindow."""
    w = DetectionWindow(
        id=id_,
        merchant_id=merchant_id,
        window_start=datetime(2024, 1, 19, tzinfo=timezone.utc),
        window_end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        transaction_count=10,
        total_amount=1000.0,
        avg_transaction_amount=100.0,
        is_synthetic_fraud_spike=is_fraud,
        split=split,
    )
    session.add(w)
    session.flush()
    return w


def _make_prediction(
    session,
    *,
    window_id: int,
    detector_type: str,
    is_flagged: bool,
    risk_score: float = 50.0,
) -> AnomalyDetection:
    """Helper: create and flush an AnomalyDetection record."""
    ad = AnomalyDetection(
        window_id=window_id,
        detector_type=detector_type,
        risk_score=risk_score,
        is_flagged=is_flagged,
        explanation="test prediction",
    )
    session.add(ad)
    session.flush()
    return ad


def _seed_standard_scenario(session) -> dict:
    """
    Seed a standard 4-window scenario with both detectors.

    Windows:
        W1: fraud=True,  baseline=True  → TP (baseline)
        W2: fraud=False, baseline=True  → FP (baseline)
        W3: fraud=True,  baseline=False → FN (baseline)
        W4: fraud=False, baseline=False → TN (baseline)

        W1: fraud=True,  ml=True        → TP (ml)
        W2: fraud=False, ml=False       → TN (ml)
        W3: fraud=True,  ml=True        → TP (ml)
        W4: fraud=False, ml=False       → TN (ml)

    Returns dict with window objects.
    """
    w1 = _make_window(session, id_=1, is_fraud=True)
    w2 = _make_window(session, id_=2, is_fraud=False)
    w3 = _make_window(session, id_=3, is_fraud=True)
    w4 = _make_window(session, id_=4, is_fraud=False)

    # Baseline predictions
    _make_prediction(session, window_id=1, detector_type="baseline", is_flagged=True)   # TP
    _make_prediction(session, window_id=2, detector_type="baseline", is_flagged=True)   # FP
    _make_prediction(session, window_id=3, detector_type="baseline", is_flagged=False)  # FN
    _make_prediction(session, window_id=4, detector_type="baseline", is_flagged=False)  # TN

    # ML predictions
    _make_prediction(session, window_id=1, detector_type="ml", is_flagged=True)         # TP
    _make_prediction(session, window_id=2, detector_type="ml", is_flagged=False)        # TN
    _make_prediction(session, window_id=3, detector_type="ml", is_flagged=True)         # TP
    _make_prediction(session, window_id=4, detector_type="ml", is_flagged=False)        # TN

    session.commit()
    return {"w1": w1, "w2": w2, "w3": w3, "w4": w4}


# ===========================================================================
# TestEvaluationMetrics
# ===========================================================================


class TestEvaluationMetrics:
    """Test precision, recall, F1, and false positive rate formulas."""

    def test_precision_normal(self):
        assert _safe_precision(tp=3, fp=1) == pytest.approx(0.75)

    def test_precision_zero_denominator(self):
        """No predicted positives → precision = 0.0 (not NaN or error)."""
        assert _safe_precision(tp=0, fp=0) == 0.0

    def test_precision_perfect(self):
        assert _safe_precision(tp=5, fp=0) == 1.0

    def test_recall_normal(self):
        assert _safe_recall(tp=3, fn=1) == pytest.approx(0.75)

    def test_recall_zero_denominator(self):
        """No actual positives → recall = 0.0."""
        assert _safe_recall(tp=0, fn=0) == 0.0

    def test_recall_perfect(self):
        assert _safe_recall(tp=5, fn=0) == 1.0

    def test_f1_normal(self):
        p = _safe_precision(3, 1)   # 0.75
        r = _safe_recall(3, 1)      # 0.75
        assert _safe_f1(p, r) == pytest.approx(0.75)

    def test_f1_zero_precision_recall(self):
        """precision + recall == 0 → F1 = 0.0."""
        assert _safe_f1(0.0, 0.0) == 0.0

    def test_f1_harmonic_mean(self):
        """F1 is harmonic mean of precision and recall."""
        p, r = 0.8, 0.4
        expected = 2 * p * r / (p + r)
        assert _safe_f1(p, r) == pytest.approx(expected)

    def test_fpr_normal(self):
        assert _safe_fpr(fp=2, tn=8) == pytest.approx(0.2)

    def test_fpr_zero_denominator(self):
        """No actual negatives → FPR = 0.0."""
        assert _safe_fpr(fp=0, tn=0) == 0.0

    def test_fpr_zero_fp(self):
        assert _safe_fpr(fp=0, tn=10) == 0.0

    def test_metrics_from_engine(self, db_session):
        """Integration: engine computes metrics consistent with helpers."""
        _seed_standard_scenario(db_session)
        engine = EvaluationEngine(db_session)
        result = engine.evaluate(partition="dev_test", detector_type="baseline")

        # Baseline: TP=1, FP=1, FN=1, TN=1
        expected_p = _safe_precision(1, 1)   # 0.5
        expected_r = _safe_recall(1, 1)      # 0.5
        expected_f1 = _safe_f1(expected_p, expected_r)
        expected_fpr = _safe_fpr(1, 1)       # 0.5

        assert result.precision == pytest.approx(expected_p)
        assert result.recall == pytest.approx(expected_r)
        assert result.f1_score == pytest.approx(expected_f1)
        assert result.false_positive_rate == pytest.approx(expected_fpr)


# ===========================================================================
# TestConfusionMatrix
# ===========================================================================


class TestConfusionMatrix:
    """Test TP/FP/FN/TN counts."""

    def test_all_tp(self):
        gt = [True, True, True]
        pred = [True, True, True]
        tp, fp, fn, tn = _build_confusion_matrix(gt, pred)
        assert (tp, fp, fn, tn) == (3, 0, 0, 0)

    def test_all_tn(self):
        gt = [False, False]
        pred = [False, False]
        tp, fp, fn, tn = _build_confusion_matrix(gt, pred)
        assert (tp, fp, fn, tn) == (0, 0, 0, 2)

    def test_all_fp(self):
        gt = [False, False, False]
        pred = [True, True, True]
        tp, fp, fn, tn = _build_confusion_matrix(gt, pred)
        assert (tp, fp, fn, tn) == (0, 3, 0, 0)

    def test_all_fn(self):
        gt = [True, True]
        pred = [False, False]
        tp, fp, fn, tn = _build_confusion_matrix(gt, pred)
        assert (tp, fp, fn, tn) == (0, 0, 2, 0)

    def test_mixed_counts(self):
        gt =   [True, False, True, False]
        pred = [True, True,  False, False]
        tp, fp, fn, tn = _build_confusion_matrix(gt, pred)
        assert (tp, fp, fn, tn) == (1, 1, 1, 1)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Mismatch"):
            _build_confusion_matrix([True, False], [True])

    def test_baseline_confusion_counts(self, db_session):
        """Integration: engine confusion matrix for baseline matches expected."""
        _seed_standard_scenario(db_session)
        engine = EvaluationEngine(db_session)
        result = engine.evaluate(partition="dev_test", detector_type="baseline")

        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.false_negatives == 1
        assert result.true_negatives == 1

    def test_ml_confusion_counts(self, db_session):
        """Integration: engine confusion matrix for ML matches expected."""
        _seed_standard_scenario(db_session)
        engine = EvaluationEngine(db_session)
        result = engine.evaluate(partition="dev_test", detector_type="ml")

        # ML: TP=2 (W1 and W3), FP=0, FN=0, TN=2 (W2 and W4)
        assert result.true_positives == 2
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.true_negatives == 2


# ===========================================================================
# TestCostCalculation
# ===========================================================================


class TestCostCalculation:
    """Test cost model computation."""

    def test_zero_errors_zero_cost(self):
        fp_cost, fn_cost, total_cost = _compute_costs(fp=0, fn=0)
        assert fp_cost == 0.0
        assert fn_cost == 0.0
        assert total_cost == 0.0

    def test_fp_cost_only(self):
        fp_cost, fn_cost, total = _compute_costs(fp=3, fn=0)
        assert fp_cost == pytest.approx(3 * COST_PER_FALSE_POSITIVE)
        assert fn_cost == 0.0
        assert total == fp_cost

    def test_fn_cost_only(self):
        fp_cost, fn_cost, total = _compute_costs(fp=0, fn=2)
        assert fn_cost == pytest.approx(2 * COST_PER_FALSE_NEGATIVE)
        assert fp_cost == 0.0
        assert total == fn_cost

    def test_combined_cost(self):
        fp_cost, fn_cost, total = _compute_costs(fp=4, fn=1)
        expected_fp = 4 * COST_PER_FALSE_POSITIVE
        expected_fn = 1 * COST_PER_FALSE_NEGATIVE
        assert fp_cost == pytest.approx(expected_fp)
        assert fn_cost == pytest.approx(expected_fn)
        assert total == pytest.approx(expected_fp + expected_fn)

    def test_custom_unit_costs(self):
        fp_cost, fn_cost, total = _compute_costs(fp=1, fn=1, fp_unit_cost=100.0, fn_unit_cost=200.0)
        assert fp_cost == pytest.approx(100.0)
        assert fn_cost == pytest.approx(200.0)
        assert total == pytest.approx(300.0)

    def test_cost_from_engine(self, db_session):
        """Integration: engine-computed costs match manual calculation."""
        _seed_standard_scenario(db_session)
        engine = EvaluationEngine(db_session)
        result = engine.evaluate(partition="dev_test", detector_type="baseline")

        # Baseline: FP=1, FN=1
        expected_fp_cost = 1 * COST_PER_FALSE_POSITIVE
        expected_fn_cost = 1 * COST_PER_FALSE_NEGATIVE
        assert result.fp_cost == pytest.approx(expected_fp_cost)
        assert result.fn_cost == pytest.approx(expected_fn_cost)
        assert result.total_cost == pytest.approx(expected_fp_cost + expected_fn_cost)


# ===========================================================================
# TestDetectorSeparation
# ===========================================================================


class TestDetectorSeparation:
    """Verify baseline and ML detectors are evaluated independently."""

    def test_baseline_result_is_independent_of_ml_predictions(self, db_session):
        """Changing ML predictions must not affect baseline metrics."""
        _make_window(db_session, id_=1, is_fraud=True)
        _make_window(db_session, id_=2, is_fraud=False)

        _make_prediction(db_session, window_id=1, detector_type="baseline", is_flagged=True)
        _make_prediction(db_session, window_id=2, detector_type="baseline", is_flagged=False)

        # ML sees exactly opposite (different predictions)
        _make_prediction(db_session, window_id=1, detector_type="ml", is_flagged=False)
        _make_prediction(db_session, window_id=2, detector_type="ml", is_flagged=True)

        db_session.commit()

        engine = EvaluationEngine(db_session)
        baseline_result = engine.evaluate(partition="dev_test", detector_type="baseline")
        ml_result = engine.evaluate(partition="dev_test", detector_type="ml")

        # Baseline: TP=1, TN=1 → precision=1.0, recall=1.0, F1=1.0
        assert baseline_result.true_positives == 1
        assert baseline_result.true_negatives == 1
        assert baseline_result.false_positives == 0
        assert baseline_result.false_negatives == 0
        assert baseline_result.precision == pytest.approx(1.0)

        # ML: FN=1, FP=1 → precision=0.0, recall=0.0
        assert ml_result.false_positives == 1
        assert ml_result.false_negatives == 1
        assert ml_result.true_positives == 0
        assert ml_result.true_negatives == 0

    def test_detector_type_in_result(self, db_session):
        """EvaluationResult.detector_type must match the requested detector."""
        _seed_standard_scenario(db_session)
        engine = EvaluationEngine(db_session)

        baseline_result = engine.evaluate(partition="dev_test", detector_type="baseline")
        ml_result = engine.evaluate(partition="dev_test", detector_type="ml")

        assert baseline_result.detector_type == "baseline"
        assert ml_result.detector_type == "ml"

    def test_ml_records_do_not_affect_baseline_evaluation(self, db_session):
        """Baseline evaluation must ignore ML AnomalyDetection records."""
        _make_window(db_session, id_=1, is_fraud=True)
        _make_window(db_session, id_=2, is_fraud=False)

        # Only baseline predictions
        _make_prediction(db_session, window_id=1, detector_type="baseline", is_flagged=True)
        _make_prediction(db_session, window_id=2, detector_type="baseline", is_flagged=True)

        # ML predictions exist but must not influence baseline
        _make_prediction(db_session, window_id=1, detector_type="ml", is_flagged=False)
        _make_prediction(db_session, window_id=2, detector_type="ml", is_flagged=False)

        db_session.commit()
        engine = EvaluationEngine(db_session)
        result = engine.evaluate(partition="dev_test", detector_type="baseline")

        # Baseline: both flagged → TP=1, FP=1
        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.false_negatives == 0


# ===========================================================================
# TestPartitionProtection
# ===========================================================================


class TestPartitionProtection:
    """Verify dev_test is evaluated; final_holdout is rejected."""

    def test_final_holdout_in_supported_partitions(self):
        """final_holdout is included in SUPPORTED_PARTITIONS for Phase 15 evaluation."""
        assert "final_holdout" in SUPPORTED_PARTITIONS

    def test_dev_test_is_supported(self):
        assert "dev_test" in SUPPORTED_PARTITIONS

    def test_final_holdout_partition_raises_value_error(self, db_session):
        """Attempting to evaluate final_holdout must raise ValueError."""
        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError, match="final_holdout"):
            engine.evaluate(partition="final_holdout", detector_type="baseline")

    def test_unknown_partition_raises_value_error(self, db_session):
        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError, match="not supported"):
            engine.evaluate(partition="train", detector_type="baseline")

    def test_final_holdout_windows_not_loaded_when_evaluating_dev_test(self, db_session):
        """
        Even if final_holdout windows exist in the DB, they must not be
        loaded during a dev_test evaluation.
        """
        # dev_test window with prediction
        _make_window(db_session, id_=1, is_fraud=True, split="dev_test")
        _make_prediction(db_session, window_id=1, detector_type="baseline", is_flagged=True)

        # final_holdout window — must never be evaluated
        _make_window(db_session, id_=2, is_fraud=True, split="final_holdout")

        db_session.commit()

        engine = EvaluationEngine(db_session)
        result = engine.evaluate(partition="dev_test", detector_type="baseline")

        # Only 1 window should have been evaluated (the dev_test one)
        assert result.evaluated_window_count == 1

    def test_wrong_partition_string_rejected(self, db_session):
        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError):
            engine.evaluate(partition="holdout", detector_type="ml")


# ===========================================================================
# TestMissingPredictions
# ===========================================================================


class TestMissingPredictions:
    """Missing detector predictions must raise ValueError."""

    def test_missing_baseline_prediction_raises(self, db_session):
        _make_window(db_session, id_=1, is_fraud=True)
        _make_window(db_session, id_=2, is_fraud=False)
        # Only one of two baseline predictions present
        _make_prediction(db_session, window_id=1, detector_type="baseline", is_flagged=True)
        db_session.commit()

        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError, match="Missing.*baseline"):
            engine.evaluate(partition="dev_test", detector_type="baseline")

    def test_missing_ml_prediction_raises(self, db_session):
        _make_window(db_session, id_=1, is_fraud=False)
        db_session.commit()

        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError, match="Missing.*ml"):
            engine.evaluate(partition="dev_test", detector_type="ml")

    def test_no_predictions_at_all_raises(self, db_session):
        _make_window(db_session, id_=1, is_fraud=True)
        _make_window(db_session, id_=2, is_fraud=False)
        db_session.commit()

        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError, match="Missing"):
            engine.evaluate(partition="dev_test", detector_type="baseline")

    def test_missing_predictions_not_treated_as_negatives(self, db_session):
        """
        Missing predictions must fail, not silently be treated as is_flagged=False.
        This prevents artificially improving recall through missing data.
        """
        _make_window(db_session, id_=1, is_fraud=True)
        # No AnomalyDetection record for window_id=1
        db_session.commit()

        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError):
            engine.evaluate(partition="dev_test", detector_type="ml")


# ===========================================================================
# TestDuplicatePredictions
# ===========================================================================


class TestDuplicatePredictions:
    """Duplicate detector predictions must raise ValueError."""

    def test_duplicate_baseline_predictions_raises(self, db_session):
        _make_window(db_session, id_=1, is_fraud=True)
        # Two baseline records for the same window
        _make_prediction(db_session, window_id=1, detector_type="baseline", is_flagged=True)
        _make_prediction(db_session, window_id=1, detector_type="baseline", is_flagged=False)
        db_session.commit()

        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError, match="Duplicate.*baseline"):
            engine.evaluate(partition="dev_test", detector_type="baseline")

    def test_duplicate_ml_predictions_raises(self, db_session):
        _make_window(db_session, id_=1, is_fraud=False)
        _make_prediction(db_session, window_id=1, detector_type="ml", is_flagged=True)
        _make_prediction(db_session, window_id=1, detector_type="ml", is_flagged=True)
        db_session.commit()

        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError, match="Duplicate.*ml"):
            engine.evaluate(partition="dev_test", detector_type="ml")

    def test_duplicate_raises_not_silently_selected(self, db_session):
        """
        With two conflicting predictions (True/False), must fail, not silently
        pick one (which would corrupt evaluation integrity).
        """
        _make_window(db_session, id_=1, is_fraud=True)
        _make_prediction(db_session, window_id=1, detector_type="baseline", is_flagged=True)
        _make_prediction(db_session, window_id=1, detector_type="baseline", is_flagged=False)
        db_session.commit()

        engine = EvaluationEngine(db_session)
        with pytest.raises(ValueError):
            engine.evaluate(partition="dev_test", detector_type="baseline")


# ===========================================================================
# TestPersistence
# ===========================================================================


class TestPersistence:
    """Verify run_evaluation() creates correct EvaluationRun records."""

    def test_two_evaluation_run_records_created(self, db_session):
        """One complete evaluation must produce exactly 2 EvaluationRun records."""
        _seed_standard_scenario(db_session)
        run_evaluation(session=db_session, partition="dev_test")

        runs = db_session.query(EvaluationRun).all()
        assert len(runs) == 2

    def test_detector_types_in_records(self, db_session):
        """The two records must be for 'baseline' and 'ml'."""
        _seed_standard_scenario(db_session)
        run_evaluation(session=db_session, partition="dev_test")

        runs = db_session.query(EvaluationRun).all()
        detector_types = {r.detector_type for r in runs}
        assert detector_types == {"baseline", "ml"}

    def test_partition_field_is_dev_test(self, db_session):
        """All persisted EvaluationRun records must have partition='dev_test'."""
        _seed_standard_scenario(db_session)
        run_evaluation(session=db_session, partition="dev_test")

        runs = db_session.query(EvaluationRun).all()
        for run in runs:
            assert run.partition == "dev_test"

    def test_metrics_persisted_correctly(self, db_session):
        """Persisted metrics must match the EvaluationResult values."""
        _seed_standard_scenario(db_session)
        results = run_evaluation(session=db_session, partition="dev_test")

        for result in results:
            db_run = (
                db_session.query(EvaluationRun)
                .filter(EvaluationRun.detector_type == result.detector_type)
                .first()
            )
            assert db_run is not None
            assert db_run.precision == pytest.approx(result.precision)
            assert db_run.recall == pytest.approx(result.recall)
            assert db_run.f1_score == pytest.approx(result.f1_score)
            assert db_run.false_positive_rate == pytest.approx(result.false_positive_rate)
            assert db_run.true_positives == result.true_positives
            assert db_run.false_positives == result.false_positives
            assert db_run.false_negatives == result.false_negatives
            assert db_run.true_negatives == result.true_negatives
            assert db_run.fp_cost == pytest.approx(result.fp_cost)
            assert db_run.fn_cost == pytest.approx(result.fn_cost)
            assert db_run.total_cost == pytest.approx(result.total_cost)

    def test_notes_field_populated(self, db_session):
        """EvaluationRun.notes must be non-empty for every record."""
        _seed_standard_scenario(db_session)
        run_evaluation(session=db_session, partition="dev_test")

        runs = db_session.query(EvaluationRun).all()
        for run in runs:
            assert run.notes is not None
            assert len(run.notes) > 0

    def test_run_timestamp_is_set(self, db_session):
        """EvaluationRun.run_timestamp must be set for every record."""
        _seed_standard_scenario(db_session)
        run_evaluation(session=db_session, partition="dev_test")

        runs = db_session.query(EvaluationRun).all()
        for run in runs:
            assert run.run_timestamp is not None


# ===========================================================================
# TestDeterminism
# ===========================================================================


class TestDeterminism:
    """Same database state must produce identical metric values on repeated runs."""

    def test_repeated_evaluation_produces_same_metrics(self, db_session):
        """Run evaluation twice; all metric values must be identical."""
        _seed_standard_scenario(db_session)

        engine = EvaluationEngine(db_session)
        result1_b = engine.evaluate(partition="dev_test", detector_type="baseline")
        result1_m = engine.evaluate(partition="dev_test", detector_type="ml")
        result2_b = engine.evaluate(partition="dev_test", detector_type="baseline")
        result2_m = engine.evaluate(partition="dev_test", detector_type="ml")

        # Baseline metrics identical
        assert result1_b.precision == result2_b.precision
        assert result1_b.recall == result2_b.recall
        assert result1_b.f1_score == result2_b.f1_score
        assert result1_b.false_positive_rate == result2_b.false_positive_rate
        assert result1_b.true_positives == result2_b.true_positives
        assert result1_b.false_positives == result2_b.false_positives
        assert result1_b.false_negatives == result2_b.false_negatives
        assert result1_b.true_negatives == result2_b.true_negatives
        assert result1_b.total_cost == result2_b.total_cost

        # ML metrics identical
        assert result1_m.precision == result2_m.precision
        assert result1_m.recall == result2_m.recall
        assert result1_m.f1_score == result2_m.f1_score
        assert result1_m.total_cost == result2_m.total_cost

    def test_cost_is_deterministic(self):
        """Cost computation is a pure function — always deterministic."""
        r1 = _compute_costs(fp=5, fn=2)
        r2 = _compute_costs(fp=5, fn=2)
        assert r1 == r2

    def test_repeated_run_evaluation_appends_new_records(self, db_session):
        """
        Running evaluation twice must append new records rather than
        overwriting existing ones (auditability rule).
        """
        _seed_standard_scenario(db_session)
        run_evaluation(session=db_session, partition="dev_test")
        run_evaluation(session=db_session, partition="dev_test")

        runs = db_session.query(EvaluationRun).all()
        # 2 runs × 2 detectors = 4 records
        assert len(runs) == 4


# ===========================================================================
# TestNoMutation
# ===========================================================================


class TestNoMutation:
    """Evaluation must not modify DetectionWindow, AnomalyDetection, or Transaction."""

    def test_detection_windows_not_modified(self, db_session):
        """DetectionWindow records must be unchanged after evaluation."""
        _seed_standard_scenario(db_session)

        # Snapshot before
        windows_before = {
            w.id: {
                "is_synthetic_fraud_spike": w.is_synthetic_fraud_spike,
                "split": w.split,
                "transaction_count": w.transaction_count,
                "total_amount": w.total_amount,
            }
            for w in db_session.query(DetectionWindow).all()
        }

        run_evaluation(session=db_session, partition="dev_test")

        # Snapshot after
        windows_after = {
            w.id: {
                "is_synthetic_fraud_spike": w.is_synthetic_fraud_spike,
                "split": w.split,
                "transaction_count": w.transaction_count,
                "total_amount": w.total_amount,
            }
            for w in db_session.query(DetectionWindow).all()
        }

        assert windows_before == windows_after

    def test_anomaly_detections_not_modified(self, db_session):
        """AnomalyDetection records must be unchanged after evaluation."""
        _seed_standard_scenario(db_session)

        # Snapshot before
        preds_before = {
            (ad.window_id, ad.detector_type): {
                "is_flagged": ad.is_flagged,
                "risk_score": ad.risk_score,
            }
            for ad in db_session.query(AnomalyDetection).all()
        }

        run_evaluation(session=db_session, partition="dev_test")

        # Snapshot after
        preds_after = {
            (ad.window_id, ad.detector_type): {
                "is_flagged": ad.is_flagged,
                "risk_score": ad.risk_score,
            }
            for ad in db_session.query(AnomalyDetection).all()
        }

        assert preds_before == preds_after

    def test_transactions_not_modified(self, db_session):
        """Transaction table must not be touched by evaluation."""
        # Seed one transaction
        txn = Transaction(
            merchant_id="M001",
            amount=99.0,
            timestamp=datetime(2024, 1, 19, tzinfo=timezone.utc),
        )
        db_session.add(txn)
        _seed_standard_scenario(db_session)

        txn_count_before = db_session.query(Transaction).count()
        run_evaluation(session=db_session, partition="dev_test")
        txn_count_after = db_session.query(Transaction).count()

        assert txn_count_before == txn_count_after

    def test_evaluation_run_records_are_new_rows(self, db_session):
        """
        Evaluation creates EvaluationRun rows — it must NOT touch existing
        EvaluationRun rows from previous runs.
        """
        _seed_standard_scenario(db_session)
        run_evaluation(session=db_session, partition="dev_test")

        first_run_ids = {r.id for r in db_session.query(EvaluationRun).all()}
        run_evaluation(session=db_session, partition="dev_test")

        all_run_ids = {r.id for r in db_session.query(EvaluationRun).all()}
        second_run_ids = all_run_ids - first_run_ids

        # First run records must still exist (not deleted/overwritten)
        assert first_run_ids.issubset(all_run_ids)
        # Second run added new records
        assert len(second_run_ids) == 2
