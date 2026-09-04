from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship

from database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(String, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class DetectionWindow(Base):
    __tablename__ = "detection_windows"

    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(String, index=True, nullable=False)
    window_start = Column(DateTime, index=True, nullable=False)
    window_end = Column(DateTime, index=True, nullable=False)

    transaction_count = Column(Integer, nullable=False)
    total_amount = Column(Float, nullable=False)
    avg_transaction_amount = Column(Float, nullable=False)

    is_synthetic_fraud_spike = Column(Boolean, nullable=False)
    split = Column(String, index=True, nullable=False)  # 'train', 'dev_test', or 'final_holdout'

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship to anomaly detections
    anomaly_detections = relationship("AnomalyDetection", back_populates="detection_window")


class AnomalyDetection(Base):
    __tablename__ = "anomaly_detections"

    id = Column(Integer, primary_key=True, index=True)
    window_id = Column(Integer, ForeignKey("detection_windows.id"), index=True, nullable=False)
    detector_type = Column(String, index=True, nullable=False)  # e.g., 'baseline', 'ml'
    risk_score = Column(Float, nullable=False)
    is_flagged = Column(Boolean, nullable=False)
    explanation = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship back to window
    detection_window = relationship("DetectionWindow", back_populates="anomaly_detections")


class EvaluationRun(Base):
    """
    Phase 6 — Evaluation Engine result record.

    One EvaluationRun is persisted per (detector_type × partition) per
    evaluation execution.  A full dev_test evaluation therefore produces
    two records: one for 'baseline' and one for 'ml'.

    Phase 6 exclusively owns this model.  It must NOT be created or
    modified by any other phase.

    Cost columns use the configurable unit costs defined in
    evaluation_engine.py (COST_PER_FALSE_POSITIVE, COST_PER_FALSE_NEGATIVE)
    and are derived from:
        fp_cost   = false_positives  × fp_unit_cost
        fn_cost   = false_negatives  × fn_unit_cost
        total_cost = fp_cost + fn_cost

    The ``notes`` field documents the cost assumptions used in that run,
    making every record self-explanatory without requiring external
    configuration files.
    """

    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)

    # Which detector was evaluated
    detector_type = Column(String, index=True, nullable=False)  # 'baseline' | 'ml'

    # Which data partition was evaluated
    partition = Column(String, index=True, nullable=False)  # 'dev_test' (Phase 6); 'final_holdout' (Phase 15 only)

    # Wall-clock time of this evaluation run (UTC)
    run_timestamp = Column(DateTime, nullable=False)

    # --- Classification metrics ---
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    false_positive_rate = Column(Float, nullable=False)

    # --- Confusion matrix counts ---
    true_positives = Column(Integer, nullable=False)
    false_positives = Column(Integer, nullable=False)
    false_negatives = Column(Integer, nullable=False)
    true_negatives = Column(Integer, nullable=False)

    # --- Operational cost estimates ---
    fp_cost = Column(Float, nullable=False)
    fn_cost = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)

    # --- Audit ---
    # Human-readable description of evaluation configuration (cost
    # assumptions, window counts, etc.).  Always populated; never NULL.
    notes = Column(Text, nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
