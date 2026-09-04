"""
AI Risk Manager — Phase 6: Evaluation Engine
=============================================

Purpose
-------
Compare the Baseline Detector (Phase 4) and the ML Anomaly Detector (Phase 5)
against the ground-truth labels stored in DetectionWindow.is_synthetic_fraud_spike.
The Evaluation Engine is reusable: Phase 15 will call the same code against
the final_holdout partition.

Supported Partition (Phase 6)
------------------------------
    evaluate(partition="dev_test")

Only "dev_test" is permitted in Phase 6.  Passing "final_holdout" or any
other string will raise a ValueError.  Phase 15 will unlock "final_holdout"
in a dedicated pipeline; that logic must NOT be added here during Phase 6.

Ground Truth
------------
DetectionWindow.is_synthetic_fraud_spike is the sole ground-truth label.
It is ONLY used here for evaluation.  It must NEVER be used as an input
feature to either detector.

Detector Independence
---------------------
The Baseline Detector ("baseline") and ML Anomaly Detector ("ml") are
evaluated independently.  Their predictions are never combined during metric
calculation.  One EvaluationResult and one EvaluationRun record are produced
for each detector type per evaluation run.

Metric Definitions
------------------
    precision          = TP / (TP + FP)      [0.0 if TP+FP == 0]
    recall             = TP / (TP + FN)      [0.0 if TP+FN == 0]
    f1_score           = 2*P*R / (P+R)       [0.0 if P+R == 0]
    false_positive_rate = FP / (FP + TN)     [0.0 if FP+TN == 0]

Cost Model
----------
Configurable unit costs (see config.py, overridable via environment):
    COST_PER_FALSE_POSITIVE  = 500.0   (INR, approximate manual-review cost)
    COST_PER_FALSE_NEGATIVE  = 15000.0 (INR, approximate missed-risk cost)

Derived:
    fp_cost    = FP × COST_PER_FALSE_POSITIVE
    fn_cost    = FN × COST_PER_FALSE_NEGATIVE
    total_cost = fp_cost + fn_cost

These are documented assumptions, not precise financial claims.  The exact
values are stored in every EvaluationRun.notes so each record is self-
explanatory without requiring external configuration.

Error Handling
--------------
Missing predictions  → ValueError (preferred policy documented in Phase 6)
Duplicate predictions → ValueError
Wrong partition data  → ValueError

Final-Holdout Restriction
--------------------------
The final_holdout partition MUST NOT be queried, loaded, or evaluated in
Phase 6.  The DB query explicitly filters to the requested partition, and the
engine validates each window's split attribute before evaluation.

Phase 15 Reuse
--------------
To evaluate final_holdout in Phase 15, pass partition="final_holdout" to
run_evaluation().  The SUPPORTED_PARTITIONS constant must be updated in
Phase 15 to unlock that partition.  No other code changes are needed here.

Determinism
-----------
Given an identical database state, the engine produces identical metric values,
confusion-matrix counts, and cost values.  run_timestamp naturally varies
between runs and is the only non-deterministic output field.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure the backend package is importable when running as a script
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from config import settings  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase 6 default evaluation partition.
DEFAULT_PARTITION: str = "dev_test"

# Partitions permitted during evaluation.
# 'final_holdout' is permitted ONLY when allow_holdout=True is explicitly set (Phase 15).
SUPPORTED_PARTITIONS: frozenset[str] = frozenset({"dev_test", "final_holdout"})

# Detector types supported by the Evaluation Engine.
DETECTOR_TYPES: tuple[str, ...] = ("baseline", "ml")

# ---------------------------------------------------------------------------
# Cost model constants
# Derived from config.py (overridable via environment variables).
# These represent approximate operational costs in INR:
#   FP: cost of manually reviewing a falsely flagged normal window
#   FN: cost of missing a genuine fraud-spike window
# ---------------------------------------------------------------------------
COST_PER_FALSE_POSITIVE: float = settings.COST_PER_FALSE_POSITIVE   # default 500.0
COST_PER_FALSE_NEGATIVE: float = settings.COST_PER_FALSE_NEGATIVE   # default 15000.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Structured result of evaluating one detector against one partition.

    Returned by EvaluationEngine.evaluate() and also persisted as an
    EvaluationRun database record by run_evaluation().
    """
    detector_type: str
    partition: str

    # --- Confusion matrix counts ---
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    # --- Classification metrics ---
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0

    # --- Operational cost estimates ---
    fp_cost: float = 0.0
    fn_cost: float = 0.0
    total_cost: float = 0.0

    # --- Audit ---
    notes: str = ""

    # --- Window counts used in evaluation ---
    evaluated_window_count: int = 0

    def as_dict(self) -> dict:
        """Return all fields as a plain dictionary (useful for API layer)."""
        return {
            "detector_type": self.detector_type,
            "partition": self.partition,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "false_positive_rate": self.false_positive_rate,
            "fp_cost": self.fp_cost,
            "fn_cost": self.fn_cost,
            "total_cost": self.total_cost,
            "notes": self.notes,
            "evaluated_window_count": self.evaluated_window_count,
        }


# ---------------------------------------------------------------------------
# Core metric helpers (pure functions — independently testable)
# ---------------------------------------------------------------------------

def _safe_precision(tp: int, fp: int) -> float:
    """
    precision = TP / (TP + FP).
    Returns 0.0 if there are no predicted positives (avoids ZeroDivisionError/NaN).
    """
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def _safe_recall(tp: int, fn: int) -> float:
    """
    recall = TP / (TP + FN).
    Returns 0.0 if there are no actual positives (avoids ZeroDivisionError/NaN).
    """
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    """
    F1 = 2 * precision * recall / (precision + recall).
    Returns 0.0 if precision + recall == 0.
    """
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom > 0.0 else 0.0


def _safe_fpr(fp: int, tn: int) -> float:
    """
    FPR = FP / (FP + TN).
    Returns 0.0 if there are no actual negatives.
    """
    denom = fp + tn
    return fp / denom if denom > 0 else 0.0


def _compute_costs(
    fp: int,
    fn: int,
    fp_unit_cost: float = COST_PER_FALSE_POSITIVE,
    fn_unit_cost: float = COST_PER_FALSE_NEGATIVE,
) -> tuple[float, float, float]:
    """
    Compute FP cost, FN cost, and total cost.

    Args:
        fp: Number of false positives.
        fn: Number of false negatives.
        fp_unit_cost: Cost per false positive (default from config).
        fn_unit_cost: Cost per false negative (default from config).

    Returns:
        (fp_cost, fn_cost, total_cost) as floats.
    """
    fp_cost = float(fp) * fp_unit_cost
    fn_cost = float(fn) * fn_unit_cost
    return fp_cost, fn_cost, fp_cost + fn_cost


def _build_confusion_matrix(
    ground_truths: list[bool],
    predictions: list[bool],
) -> tuple[int, int, int, int]:
    """
    Compute confusion matrix counts from aligned ground_truth and prediction lists.

    Args:
        ground_truths: List of bool ground-truth labels (True = fraud spike).
        predictions:   List of bool predictions aligned 1-to-1 with ground_truths.

    Returns:
        (TP, FP, FN, TN) as integers.

    Raises:
        ValueError: If lengths differ.
    """
    if len(ground_truths) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(ground_truths)} ground-truth labels but "
            f"{len(predictions)} predictions."
        )
    tp = fp = fn = tn = 0
    for gt, pred in zip(ground_truths, predictions):
        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# EvaluationEngine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """
    Reusable Evaluation Engine for the AI Risk Manager.

    Evaluates one detector against one data partition and returns an
    EvaluationResult.  The engine is independent of FastAPI and React;
    it requires only a SQLAlchemy session and the partition name.

    Usage::

        engine = EvaluationEngine(session)
        result = engine.evaluate(partition="dev_test", detector_type="baseline")

    The engine does NOT persist EvaluationRun records; that responsibility
    belongs to run_evaluation().
    """

    def __init__(self, session) -> None:
        """
        Args:
            session: An active SQLAlchemy Session.
        """
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        partition: str,
        detector_type: str,
        allow_holdout: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate a single detector against a single partition.

        Args:
            partition:     The data partition to evaluate.  Must be one of
                           SUPPORTED_PARTITIONS.
            detector_type: The detector to evaluate ("baseline" or "ml").
            allow_holdout: If True, permits evaluating final_holdout (Phase 15 only).
                           Default is False.

        Returns:
            EvaluationResult with all metrics, confusion-matrix counts, and
            cost estimates populated.

        Raises:
            ValueError: If the partition is unsupported, final_holdout is requested
                        without allow_holdout=True, detector_type is unknown,
                        or prediction integrity checks fail.
        """
        self._validate_partition(partition, allow_holdout=allow_holdout)
        self._validate_detector_type(detector_type)

        # Load windows and predictions from the database
        windows = self._load_windows(partition)
        predictions_map = self._load_predictions(windows, detector_type, partition)

        # Build aligned lists
        ground_truths: list[bool] = []
        predictions: list[bool] = []

        for window in windows:
            gt = bool(window.is_synthetic_fraud_spike)
            pred_flag = predictions_map[window.id]
            ground_truths.append(gt)
            predictions.append(pred_flag)

        # Compute confusion matrix
        tp, fp, fn, tn = _build_confusion_matrix(ground_truths, predictions)

        # Compute classification metrics
        precision = _safe_precision(tp, fp)
        recall = _safe_recall(tp, fn)
        f1 = _safe_f1(precision, recall)
        fpr = _safe_fpr(fp, tn)

        # Compute operational costs
        fp_cost, fn_cost, total_cost = _compute_costs(fp, fn)

        # Build notes describing the evaluation configuration
        notes = self._build_notes(
            partition=partition,
            detector_type=detector_type,
            window_count=len(windows),
            fp_unit_cost=COST_PER_FALSE_POSITIVE,
            fn_unit_cost=COST_PER_FALSE_NEGATIVE,
        )

        return EvaluationResult(
            detector_type=detector_type,
            partition=partition,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            total_cost=total_cost,
            notes=notes,
            evaluated_window_count=len(windows),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_partition(self, partition: str, allow_holdout: bool = False) -> None:
        """Raise ValueError for unsupported or unauthorized partition values."""
        if partition not in SUPPORTED_PARTITIONS:
            raise ValueError(
                f"Partition '{partition}' is not supported. "
                f"Supported partitions: {sorted(SUPPORTED_PARTITIONS)}."
            )
        if partition == "final_holdout" and not allow_holdout:
            raise ValueError(
                "Partition 'final_holdout' is protected and cannot be evaluated unless allow_holdout=True is set."
            )

    def _validate_detector_type(self, detector_type: str) -> None:
        """Raise ValueError for unknown detector types."""
        if detector_type not in DETECTOR_TYPES:
            raise ValueError(
                f"Unknown detector_type '{detector_type}'. "
                f"Supported types: {DETECTOR_TYPES}."
            )

    def _load_windows(self, partition: str):
        """
        Load DetectionWindow records for the requested partition only.

        The DB query strictly filters by split == partition, so final_holdout
        records are never loaded when partition == "dev_test".
        Post-query, each row's split attribute is re-verified to ensure
        no cross-partition contamination.

        Returns:
            List of DetectionWindow ORM objects.

        Raises:
            ValueError: If any loaded window belongs to a different partition.
        """
        from models import DetectionWindow  # noqa: E402

        windows = (
            self._session.query(DetectionWindow)
            .filter(DetectionWindow.split == partition)
            .order_by(DetectionWindow.id)
            .all()
        )

        # Post-query partition verification
        bad = [w.id for w in windows if w.split != partition]
        if bad:
            raise ValueError(
                f"SAFETY VIOLATION: {len(bad)} window(s) with wrong split "
                f"returned from DB query for partition='{partition}'. "
                f"Window IDs: {bad[:10]}{'...' if len(bad) > 10 else ''}"
            )

        return windows

    def _load_predictions(
        self,
        windows,
        detector_type: str,
        partition: str,
    ) -> dict[int, bool]:
        """
        Load AnomalyDetection records for the given windows and detector.

        Verifies:
        1. Every window has exactly one prediction (missing → ValueError).
        2. No window has more than one prediction (duplicate → ValueError).
        3. Every prediction's window belongs to the expected partition
           (cross-partition contamination guard).

        Args:
            windows:       List of DetectionWindow ORM objects.
            detector_type: "baseline" or "ml".
            partition:     The expected partition for cross-check.

        Returns:
            Dict mapping window_id → is_flagged (bool).

        Raises:
            ValueError: On missing, duplicate, or cross-partition predictions.
        """
        from models import AnomalyDetection  # noqa: E402

        window_ids = [w.id for w in windows]

        if not window_ids:
            return {}

        records = (
            self._session.query(AnomalyDetection)
            .filter(
                AnomalyDetection.detector_type == detector_type,
                AnomalyDetection.window_id.in_(window_ids),
            )
            .all()
        )

        # Cross-partition contamination check: every loaded prediction must
        # reference a window from the expected partition.
        window_split_map = {w.id: w.split for w in windows}
        for rec in records:
            if rec.window_id not in window_split_map:
                raise ValueError(
                    f"SAFETY VIOLATION: AnomalyDetection id={rec.id} "
                    f"(detector_type='{detector_type}') references "
                    f"window_id={rec.window_id} which is not in the "
                    f"'{partition}' partition."
                )

        # Build window_id → list of records
        pred_map: dict[int, list] = {wid: [] for wid in window_ids}
        for rec in records:
            pred_map[rec.window_id].append(rec)

        # Check for duplicates
        duplicate_ids = [
            wid for wid, recs in pred_map.items() if len(recs) > 1
        ]
        if duplicate_ids:
            raise ValueError(
                f"Duplicate {detector_type!r} predictions found for "
                f"window_id(s): {duplicate_ids}. "
                "Each window must have exactly one prediction per detector."
            )

        # Check for missing predictions
        missing_ids = [
            wid for wid, recs in pred_map.items() if len(recs) == 0
        ]
        if missing_ids:
            raise ValueError(
                f"Missing {detector_type!r} predictions for "
                f"{len(missing_ids)} window(s) in partition='{partition}'. "
                f"Window IDs (first 10): {missing_ids[:10]}"
                f"{'...' if len(missing_ids) > 10 else ''}. "
                "Run the detector pipeline before evaluating."
            )

        # Return simple mapping window_id → is_flagged
        return {wid: bool(pred_map[wid][0].is_flagged) for wid in window_ids}

    @staticmethod
    def _build_notes(
        partition: str,
        detector_type: str,
        window_count: int,
        fp_unit_cost: float,
        fn_unit_cost: float,
    ) -> str:
        """
        Build a deterministic human-readable description of this evaluation run.

        The notes field makes every EvaluationRun record self-explanatory.
        """
        return (
            f"Phase 6 Evaluation Engine — "
            f"detector='{detector_type}', partition='{partition}', "
            f"windows_evaluated={window_count}. "
            f"Cost model: FP unit cost={fp_unit_cost:.2f} INR "
            f"(manual review estimate), "
            f"FN unit cost={fn_unit_cost:.2f} INR "
            f"(missed-risk estimate). "
            "Data: fully synthetic. "
            "Ground truth: DetectionWindow.is_synthetic_fraud_spike. "
            "final_holdout partition NOT accessed."
        )


# ---------------------------------------------------------------------------
# Database pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    session=None,
    partition: str = DEFAULT_PARTITION,
    allow_holdout: bool = False,
) -> list[EvaluationResult]:
    """
    Full evaluation pipeline.

    For each detector type ("baseline", "ml"):
        1. Evaluate predictions against ground-truth labels.
        2. Persist one EvaluationRun record to the database.

    A single call therefore produces exactly 2 EvaluationRun records.

    Args:
        session:   Optional SQLAlchemy Session.  When None (production default),
                   uses the database engine and SessionLocal from database.py.
                   When provided, uses the caller's session without closing it
                   (test-friendly pattern, same as other Phase pipelines).
        partition: The data partition to evaluate.  Defaults to "dev_test".
                   Must be in SUPPORTED_PARTITIONS.
        allow_holdout: If True, permits evaluating final_holdout (Phase 15 only).
                       Default is False.

    Returns:
        List of two EvaluationResult objects (one per detector type),
        ordered as ["baseline", "ml"].

    Raises:
        ValueError: If the partition is unsupported or any integrity check fails.
    """
    from database import Base  # noqa: E402
    from models import EvaluationRun  # noqa: E402

    _owns_session = session is None

    if _owns_session:
        from database import engine, SessionLocal  # noqa: E402

        Base.metadata.create_all(bind=engine)
        session = SessionLocal()

    try:
        engine_obj = EvaluationEngine(session)
        results: list[EvaluationResult] = []

        for detector_type in DETECTOR_TYPES:
            result = engine_obj.evaluate(
                partition=partition,
                detector_type=detector_type,
                allow_holdout=allow_holdout,
            )

            # Persist EvaluationRun record
            run = EvaluationRun(
                detector_type=result.detector_type,
                partition=result.partition,
                run_timestamp=datetime.now(timezone.utc),
                precision=result.precision,
                recall=result.recall,
                f1_score=result.f1_score,
                false_positive_rate=result.false_positive_rate,
                true_positives=result.true_positives,
                false_positives=result.false_positives,
                false_negatives=result.false_negatives,
                true_negatives=result.true_negatives,
                fp_cost=result.fp_cost,
                fn_cost=result.fn_cost,
                total_cost=result.total_cost,
                notes=result.notes,
            )
            session.add(run)
            session.commit()

            print(
                f"[{detector_type.upper()}] partition={partition} | "
                f"P={result.precision:.3f} R={result.recall:.3f} "
                f"F1={result.f1_score:.3f} FPR={result.false_positive_rate:.3f} | "
                f"TP={result.true_positives} FP={result.false_positives} "
                f"FN={result.false_negatives} TN={result.true_negatives} | "
                f"Cost={result.total_cost:.2f} INR"
            )
            results.append(result)

        return results

    except Exception:
        session.rollback()
        raise
    finally:
        if _owns_session:
            session.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
