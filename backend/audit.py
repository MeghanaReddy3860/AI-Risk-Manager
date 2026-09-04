"""
AI Risk Manager — Phase 11: Audit & Analyst Action API Routes
============================================================

Provides REST API routes to retrieve cryptographically linked audit trails,
execute audit integrity verification, and record human analyst triage actions.
Uses the shared single-process AuditTrailManager instance.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from database import get_db
from models import DetectionWindow
from audit_trail import AuditEventType, AuditTrailManager
from schemas import (
    AnalystActionRequest,
    AnalystActionResponse,
    AuditReportResponse,
)

router = APIRouter(prefix="/api", tags=["Audit & Analyst Actions"])


def _get_audit_manager(request: Request) -> AuditTrailManager:
    """Retrieve shared AuditTrailManager instance from application state."""
    manager: Optional[AuditTrailManager] = getattr(request.app.state, "audit_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AuditTrailManager is not initialized.",
        )
    return manager


@router.get("/audit/window/{window_id}", response_model=AuditReportResponse)
def get_window_audit_report(
    window_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Retrieve full audit report and compliance details for a specific DetectionWindow.
    """
    # Enforce window existence and partition protection
    window = db.query(DetectionWindow).filter(DetectionWindow.id == window_id).first()
    if not window or window.split not in {"train", "dev_test"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DetectionWindow with ID {window_id} not found.",
        )

    audit_manager = _get_audit_manager(request)
    report = audit_manager.export_audit_report(window_id)
    return report


@router.get("/audit/verify")
def verify_audit_trail_integrity(request: Request):
    """
    Execute cryptographic SHA-256 integrity verification over the in-memory audit trail chain.
    """
    audit_manager = _get_audit_manager(request)
    is_valid, errors = audit_manager.verify_integrity()
    return {
        "integrity_valid": is_valid,
        "integrity_errors": errors,
        "total_records": len(audit_manager),
    }


@router.post("/analyst/action", response_model=AnalystActionResponse)
def record_analyst_action(
    action_req: AnalystActionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Record a human analyst review action/disposition into the audit trail.
    Authentication is explicitly deferred; actor identity is caller-supplied metadata.
    """
    # Enforce window existence and partition protection
    window = db.query(DetectionWindow).filter(DetectionWindow.id == action_req.window_id).first()
    if not window or window.split not in {"train", "dev_test"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DetectionWindow with ID {action_req.window_id} not found.",
        )

    audit_manager = _get_audit_manager(request)

    try:
        record = audit_manager.log_event(
            event_type=AuditEventType.ANALYST_ACTION,
            window_id=window.id,
            merchant_id=window.merchant_id,
            actor=action_req.actor,
            payload={
                "disposition": action_req.disposition,
                "notes": action_req.notes or "",
            },
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return AnalystActionResponse(
        status="success",
        entry_id=record.entry_id,
        message=f"Analyst review action '{action_req.disposition}' logged successfully.",
    )
