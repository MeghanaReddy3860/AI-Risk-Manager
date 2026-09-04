"""
AI Risk Manager — Phase 11: Windows & Merchants API Routes
===========================================================

Provides REST API routes for listing merchants, querying detection windows,
fetching window details, and rendering merchant hourly timeseries data.
Enforces strict final-holdout protection at the API boundary.
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db
from models import DetectionWindow, AnomalyDetection
from schemas import (
    AnomalyDetectionResponse,
    DetectionWindowResponse,
    MerchantSummaryResponse,
    RiskDossierResponse,
    TimeseriesPointResponse,
)

router = APIRouter(prefix="/api", tags=["Windows & Merchants"])

PERMITTED_SPLITS = {"train", "dev_test"}


@router.get("/merchants", response_model=List[MerchantSummaryResponse])
def list_merchants(db: Session = Depends(get_db)):
    """
    List unique merchants with aggregate summary statistics across permitted splits.
    """
    # Filter out final_holdout
    query = db.query(DetectionWindow).filter(DetectionWindow.split.in_(PERMITTED_SPLITS))
    windows = query.all()

    merchant_data: dict[str, dict] = {}
    for w in windows:
        mid = w.merchant_id
        if mid not in merchant_data:
            merchant_data[mid] = {
                "merchant_id": mid,
                "total_windows": 0,
                "flagged_anomaly_count": 0,
                "total_monetary_volume": 0.0,
                "active_risk_band": "low",
            }
        merchant_data[mid]["total_windows"] += 1
        merchant_data[mid]["total_monetary_volume"] += float(w.total_amount)

        # Check flagged anomalies in database for this window
        flagged_count = (
            db.query(AnomalyDetection)
            .filter(
                AnomalyDetection.window_id == w.id,
                AnomalyDetection.is_flagged == True,
            )
            .count()
        )
        if flagged_count > 0:
            merchant_data[mid]["flagged_anomaly_count"] += flagged_count
            merchant_data[mid]["active_risk_band"] = "high"

    # Return sorted by merchant_id
    sorted_merchants = [
        MerchantSummaryResponse(**data)
        for _, data in sorted(merchant_data.items())
    ]
    return sorted_merchants


@router.get("/windows", response_model=List[DetectionWindowResponse])
def list_windows(
    merchant_id: Optional[str] = Query(None, description="Filter by merchant ID"),
    split: Optional[str] = Query(None, description="Filter by partition split ('train' or 'dev_test')"),
    is_flagged: Optional[bool] = Query(None, description="Filter by anomaly flag status"),
    limit: int = Query(50, description="Page limit (default 50, max 500)"),
    offset: int = Query(0, description="Page offset (>= 0)"),
    db: Session = Depends(get_db),
):
    """
    Query detection windows with pagination and filters.
    Strictly validates partition split parameters before database query execution.
    """
    # 1. Offset validation
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="offset must be greater than or equal to 0.",
        )

    # 2. Limit validation (> 500 rejected with 400)
    if limit > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit cannot exceed 500.",
        )
    if limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be greater than 0.",
        )

    # 3. Split validation
    if split is not None:
        split_clean = split.strip().lower()
        if split_clean not in PERMITTED_SPLITS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid split parameter '{split}'. Only permitted splits {sorted(PERMITTED_SPLITS)} are allowed.",
            )

    # 4. Construct query strictly filtering out final_holdout
    query = db.query(DetectionWindow).filter(DetectionWindow.split.in_(PERMITTED_SPLITS))

    if merchant_id:
        query = query.filter(DetectionWindow.merchant_id == merchant_id.strip())
    if split:
        query = query.filter(DetectionWindow.split == split.strip().lower())
    if is_flagged is not None:
        # Join with anomaly detections
        window_ids_flagged = (
            db.query(AnomalyDetection.window_id)
            .filter(AnomalyDetection.is_flagged == is_flagged)
            .subquery()
        )
        query = query.filter(DetectionWindow.id.in_(window_ids_flagged))

    # Deterministic ordering by window.id
    windows = query.order_by(DetectionWindow.id.asc()).offset(offset).limit(limit).all()

    response = []
    for w in windows:
        response.append(
            DetectionWindowResponse(
                id=w.id,
                merchant_id=w.merchant_id,
                window_start=w.window_start.isoformat(),
                window_end=w.window_end.isoformat(),
                transaction_count=w.transaction_count,
                total_amount=w.total_amount,
                avg_transaction_amount=w.avg_transaction_amount,
                split=w.split,
                created_at=w.created_at.isoformat(),
            )
        )

    return response


@router.get("/windows/{window_id}", response_model=DetectionWindowResponse)
def get_window(window_id: int, db: Session = Depends(get_db)):
    """
    Retrieve single DetectionWindow by ID.
    Enforces holdout protection by rejecting non-permitted partition windows with 404.
    """
    window = db.query(DetectionWindow).filter(DetectionWindow.id == window_id).first()
    if not window:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DetectionWindow with ID {window_id} not found.",
        )

    # Final-holdout partition protection check
    if window.split not in PERMITTED_SPLITS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DetectionWindow with ID {window_id} not found.",
        )

    return DetectionWindowResponse(
        id=window.id,
        merchant_id=window.merchant_id,
        window_start=window.window_start.isoformat(),
        window_end=window.window_end.isoformat(),
        transaction_count=window.transaction_count,
        total_amount=window.total_amount,
        avg_transaction_amount=window.avg_transaction_amount,
        split=window.split,
        created_at=window.created_at.isoformat(),
    )


@router.get("/windows/{window_id}/detections", response_model=List[AnomalyDetectionResponse])
def get_window_detections(window_id: int, db: Session = Depends(get_db)):
    """
    Retrieve persisted AnomalyDetection records for a specific window.
    Enforces holdout protection by rejecting non-permitted partition windows with 404.
    """
    window = db.query(DetectionWindow).filter(DetectionWindow.id == window_id).first()
    if not window or window.split not in PERMITTED_SPLITS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DetectionWindow with ID {window_id} not found.",
        )

    detections = (
        db.query(AnomalyDetection)
        .filter(AnomalyDetection.window_id == window_id)
        .order_by(AnomalyDetection.id.asc())
        .all()
    )

    response = []
    for d in detections:
        response.append(
            AnomalyDetectionResponse(
                id=d.id,
                window_id=d.window_id,
                detector_type=d.detector_type,
                risk_score=d.risk_score,
                is_flagged=d.is_flagged,
                explanation=d.explanation,
                created_at=d.created_at.isoformat(),
            )
        )
    return response


@router.get("/windows/{window_id}/analysis", response_model=RiskDossierResponse)
def get_window_analysis(
    window_id: int,
    detector_type: str = Query("baseline", description="Detector type ('baseline' or 'ml')"),
    use_llm: bool = Query(False, description="Whether to attempt LLM explanation generation"),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """
    Retrieve stored analysis dossier for a DetectionWindow (Strictly Read-Only).
    """
    from pipeline import analyze_window
    return analyze_window(
        window_id=window_id,
        detector_type=detector_type,
        use_llm=use_llm,
        request=request,
        db=db,
    )


@router.get("/merchants/{merchant_id}/timeseries", response_model=List[TimeseriesPointResponse])
def get_merchant_timeseries(
    merchant_id: str,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Return chronological hourly timeseries points for a merchant across permitted splits.
    """
    windows = (
        db.query(DetectionWindow)
        .filter(
            DetectionWindow.merchant_id == merchant_id.strip(),
            DetectionWindow.split.in_(PERMITTED_SPLITS),
        )
        .order_by(DetectionWindow.window_start.asc())
        .limit(limit)
        .all()
    )

    if not windows:
        # Check if merchant exists at all
        exists = (
            db.query(DetectionWindow)
            .filter(DetectionWindow.merchant_id == merchant_id.strip())
            .first()
        )
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Merchant '{merchant_id}' not found.",
            )

    points = []
    for w in windows:
        # Check if flagged by any detector
        is_flagged = (
            db.query(AnomalyDetection)
            .filter(
                AnomalyDetection.window_id == w.id,
                AnomalyDetection.is_flagged == True,
            )
            .first()
            is not None
        )
        points.append(
            TimeseriesPointResponse(
                timestamp=w.window_start.isoformat(),
                transaction_count=w.transaction_count,
                total_amount=w.total_amount,
                avg_transaction_amount=w.avg_transaction_amount,
                is_flagged=is_flagged,
            )
        )

    return points
