"""
AI Risk Manager — Phase 11: End-to-End Risk Pipeline API Route
==============================================================

Orchestrates the unified risk management pipeline for a DetectionWindow:
  DetectionWindow -> Stored Detector Result -> Risk Scorer -> AI Explainer -> Policy Engine -> Read-Only Risk Dossier

Strictly enforces partition protection prior to pipeline execution.
Reuses existing Phase 4-10 core engines without algorithm duplication.
"""

from datetime import datetime, timezone
from typing import Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from database import get_db
from models import AnomalyDetection, DetectionWindow
from baseline_detector import run_baseline_detector, BaselineDetector
from ml_anomaly_detector import run_ml_anomaly_detector, MLAnomalyDetector
from risk_scorer import score_risk
from explanation_engine import generate_explanation, ExplanationResult
from policy_engine import evaluate_policy
from audit_trail import AuditEventType, AuditTrailManager
from schemas import (
    DetectionWindowResponse,
    DetectorRunSummaryResponse,
    DetectorSummaryDetail,
    ExplanationResultResponse,
    PolicyDecisionResponse,
    RiskDossierResponse,
    RiskScoringResultResponse,
)

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

PERMITTED_SPLITS = {"train", "dev_test"}


@router.post("/run-detectors", response_model=DetectorRunSummaryResponse)
def run_detectors_endpoint(
    detector_type: Optional[str] = Query(None, description="Detector to run ('baseline', 'ml', or None for both)"),
    db: Session = Depends(get_db),
):
    """
    Runs the Baseline and ML fraud-spike detectors over the permitted
    synthetic train/dev_test data and persists detection results.

    This is a synchronous batch operation and may take noticeable time.
    It does not process final_holdout and performs no punitive actions.
    """
    detectors_to_run = []
    if detector_type:
        clean_dt = detector_type.strip().lower()
        if clean_dt not in {"baseline", "ml"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid detector_type '{detector_type}'. Must be 'baseline' or 'ml'.",
            )
        detectors_to_run = [clean_dt]
    else:
        detectors_to_run = ["baseline", "ml"]

    if "baseline" in detectors_to_run:
        run_baseline_detector(session=db)
    if "ml" in detectors_to_run:
        run_ml_anomaly_detector(session=db)

    results_dict = {}
    for dt in detectors_to_run:
        scored = (
            db.query(AnomalyDetection)
            .filter(AnomalyDetection.detector_type == dt)
            .count()
        )
        flagged = (
            db.query(AnomalyDetection)
            .filter(
                AnomalyDetection.detector_type == dt,
                AnomalyDetection.is_flagged == True,
            )
            .count()
        )
        results_dict[dt] = DetectorSummaryDetail(
            windows_scored=scored,
            windows_flagged=flagged,
        )

    return DetectorRunSummaryResponse(
        detectors_run=detectors_to_run,
        results=results_dict,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/analyze-window/{window_id}", response_model=RiskDossierResponse)
@router.post("/analyze-window/{window_id}", response_model=RiskDossierResponse)
def analyze_window(
    window_id: int,
    detector_type: str = Query("baseline", description="Detector type ('baseline' or 'ml')"),
    use_llm: bool = Query(False, description="Whether to attempt LLM explanation generation"),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """
    Read-only risk analysis inspection for a DetectionWindow.

    Steps:
      1. Partition protection check (reject final_holdout).
      2. Fetch stored real AnomalyDetection result (strictly read-only, NO model fitting/predicting).
      3. Risk Scoring (Phase 7 pure function calculation).
      4. Explanation Generation (Phase 8 pure function explanation).
      5. Operational Policy Evaluation (Phase 9 pure function policy recommendation).
      6. Audit Trail (Strictly read-only return with AUDIT_READ_ONLY placeholder; NO audit log persistence).
    """
    clean_detector = detector_type.strip().lower()
    if clean_detector not in {"baseline", "ml"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid detector_type '{detector_type}'. Must be 'baseline' or 'ml'.",
        )

    # 1. Partition Protection Check
    window = db.query(DetectionWindow).filter(DetectionWindow.id == window_id).first()
    if not window:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DetectionWindow with ID {window_id} not found.",
        )

    if window.split not in PERMITTED_SPLITS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DetectionWindow with ID {window_id} not found.",
        )

    # Convert Window model to dictionary format expected by engines
    window_dict = {
        "window_id": window.id,
        "merchant_id": window.merchant_id,
        "transaction_count": window.transaction_count,
        "total_amount": float(window.total_amount),
        "avg_transaction_amount": float(window.avg_transaction_amount),
        "split": window.split,
    }

    # 2. Read stored AnomalyDetection result (Strictly Read-Only — NO fallback computation)
    stored = (
        db.query(AnomalyDetection)
        .filter(
            AnomalyDetection.window_id == window.id,
            AnomalyDetection.detector_type == clean_detector,
        )
        .first()
    )

    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No stored detection result found for window ID {window_id} "
                f"with detector '{clean_detector}'. Please run the detector pipeline first "
                f"(POST /api/pipeline/run-detectors)."
            ),
        )

    risk_score_val = stored.risk_score
    is_flagged_val = stored.is_flagged

    # 3. Phase 7 Risk Scorer
    risk_result = score_risk(
        risk_score=risk_score_val,
        total_amount=window.total_amount,
    )

    # 4. Phase 8 Explanation Engine
    explanation_res = generate_explanation(
        window=window_dict,
        risk_result=risk_result,
        use_llm=use_llm,
    )

    # 5. Phase 9 Policy Engine
    policy_dec = evaluate_policy(
        window=window_dict,
        risk_result=risk_result,
        explanation_result=explanation_res,
    )

    # 6. Phase 10 Audit Trail (Strictly Read-Only — NO audit logging or persistence on analysis)
    entry_id = "AUDIT_READ_ONLY"

    return RiskDossierResponse(
        window=DetectionWindowResponse(
            id=window.id,
            merchant_id=window.merchant_id,
            window_start=window.window_start.isoformat(),
            window_end=window.window_end.isoformat(),
            transaction_count=window.transaction_count,
            total_amount=window.total_amount,
            avg_transaction_amount=window.avg_transaction_amount,
            split=window.split,
            created_at=window.created_at.isoformat(),
        ),
        detector_type=clean_detector,
        is_flagged=is_flagged_val,
        risk_result=RiskScoringResultResponse(
            risk_score=risk_result.risk_score,
            risk_band=risk_result.risk_band,
            risk_multiplier=risk_result.risk_multiplier,
            estimated_exposure=risk_result.estimated_exposure,
            recommended_action=risk_result.recommended_action,
        ),
        explanation=ExplanationResultResponse(
            summary=explanation_res.summary,
            key_drivers=explanation_res.key_drivers,
            raw_text=explanation_res.raw_text,
            generated_by=explanation_res.generated_by,
        ),
        policy_decision=PolicyDecisionResponse(
            policy_id=policy_dec.policy_id,
            action_type=policy_dec.action_type,
            priority=policy_dec.priority,
            review_sla_hours=policy_dec.review_sla_hours,
            require_dual_review=policy_dec.require_dual_review,
            routing_tags=list(policy_dec.routing_tags),
            triggered_rules=list(policy_dec.triggered_rules),
            audit_metadata=dict(policy_dec.audit_metadata),
        ),
        audit_entry_id=entry_id,
    )
