"""
AI Risk Manager — Phase 11: Evaluation API Routes
=================================================

Provides REST API routes to retrieve recent detector evaluation results and
trigger synchronous evaluation runs on the dev_test partition.
Strictly protects the final_holdout partition.
"""

from __future__ import annotations

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database import get_db
from models import EvaluationRun
from evaluation_engine import run_evaluation, DEFAULT_PARTITION
from schemas import EvaluationResponse

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])

PERMITTED_EVAL_PARTITIONS = {"dev_test"}


@router.get("/latest", response_model=List[EvaluationResponse])
def get_latest_evaluations(db: Session = Depends(get_db)):
    """
    Retrieve the most recent evaluation runs for each detector type.
    """
    latest_runs = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.partition.in_(PERMITTED_EVAL_PARTITIONS))
        .order_by(EvaluationRun.id.desc())
        .limit(10)
        .all()
    )

    response = []
    for r in latest_runs:
        response.append(
            EvaluationResponse(
                id=r.id,
                detector_type=r.detector_type,
                partition=r.partition,
                run_timestamp=r.run_timestamp.isoformat(),
                precision=r.precision,
                recall=r.recall,
                f1_score=r.f1_score,
                false_positive_rate=r.false_positive_rate,
                true_positives=r.true_positives,
                false_positives=r.false_positives,
                false_negatives=r.false_negatives,
                true_negatives=r.true_negatives,
                fp_cost=r.fp_cost,
                fn_cost=r.fn_cost,
                total_cost=r.total_cost,
                notes=r.notes,
            )
        )

    return response


@router.post("/run", response_model=List[EvaluationResponse])
def trigger_evaluation(
    partition: str = Query("dev_test", description="Partition to evaluate ('dev_test' only)"),
    db: Session = Depends(get_db),
):
    """
    Synchronously run detector evaluation against the dev_test partition.
    Final-holdout evaluation is strictly prohibited.
    """
    clean_partition = partition.strip().lower()
    if clean_partition not in PERMITTED_EVAL_PARTITIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid partition '{partition}'. Only permitted evaluation partition is 'dev_test'.",
        )

    try:
        results = run_evaluation(session=db, partition=clean_partition)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Query newly stored EvaluationRun records
    latest_runs = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.partition == clean_partition)
        .order_by(EvaluationRun.id.desc())
        .limit(2)
        .all()
    )

    response = []
    for r in latest_runs:
        response.append(
            EvaluationResponse(
                id=r.id,
                detector_type=r.detector_type,
                partition=r.partition,
                run_timestamp=r.run_timestamp.isoformat(),
                precision=r.precision,
                recall=r.recall,
                f1_score=r.f1_score,
                false_positive_rate=r.false_positive_rate,
                true_positives=r.true_positives,
                false_positives=r.false_positives,
                false_negatives=r.false_negatives,
                true_negatives=r.true_negatives,
                fp_cost=r.fp_cost,
                fn_cost=r.fn_cost,
                total_cost=r.total_cost,
                notes=r.notes,
            )
        )

    return response
