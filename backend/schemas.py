"""
AI Risk Manager — Phase 11: Pydantic Schemas
=============================================

Defines request and response schemas for all Phase 11 API endpoints.
Enforces type safety, payload validation, credential protection, and
defense-only guarantees.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Error Response Schema
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standardized API error response structure."""
    detail: str = Field(..., description="Human-readable error explanation.")
    status_code: int = Field(..., description="HTTP status code.")


# ---------------------------------------------------------------------------
# Merchant & Detection Window Schemas
# ---------------------------------------------------------------------------

class MerchantSummaryResponse(BaseModel):
    """Aggregated statistics for a merchant."""
    merchant_id: str = Field(..., description="Unique merchant identifier.")
    total_windows: int = Field(..., description="Total detection windows.")
    flagged_anomaly_count: int = Field(..., description="Number of flagged anomaly windows.")
    total_monetary_volume: float = Field(..., description="Total transaction volume (INR).")
    active_risk_band: str = Field(..., description="Highest current risk band ('low', 'medium', 'high', 'critical').")

    model_config = ConfigDict(from_attributes=True)


class DetectionWindowResponse(BaseModel):
    """Detection window payload with aggregate features."""
    id: int = Field(..., description="Window database ID.")
    merchant_id: str = Field(..., description="Merchant ID.")
    window_start: str = Field(..., description="Window start ISO timestamp.")
    window_end: str = Field(..., description="Window end ISO timestamp.")
    transaction_count: int = Field(..., description="Transaction count in window.")
    total_amount: float = Field(..., description="Total monetary volume (INR).")
    avg_transaction_amount: float = Field(..., description="Average transaction size (INR).")
    split: str = Field(..., description="Data partition ('train' or 'dev_test').")
    created_at: str = Field(..., description="Record creation ISO timestamp.")

    model_config = ConfigDict(from_attributes=True)


class TimeseriesPointResponse(BaseModel):
    """Hourly timeseries data point for merchant activity charts."""
    timestamp: str = Field(..., description="Window start timestamp.")
    transaction_count: int = Field(..., description="Transaction count.")
    total_amount: float = Field(..., description="Total monetary volume (INR).")
    avg_transaction_amount: float = Field(..., description="Average transaction size (INR).")
    is_flagged: bool = Field(False, description="Whether window was flagged by detector.")


# ---------------------------------------------------------------------------
# End-to-End Pipeline Dossier Schemas
# ---------------------------------------------------------------------------

class RiskScoringResultResponse(BaseModel):
    """Risk Scorer Phase 7 result payload."""
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_band: str = Field(...)
    risk_multiplier: float = Field(...)
    estimated_exposure: float = Field(..., ge=0.0)
    recommended_action: str = Field(...)


class ExplanationResultResponse(BaseModel):
    """Explanation Engine Phase 8 result payload."""
    summary: str = Field(...)
    key_drivers: List[str] = Field(default_factory=list)
    raw_text: str = Field(...)
    generated_by: str = Field(...)


class PolicyDecisionResponse(BaseModel):
    """Policy Engine Phase 9 decision payload."""
    policy_id: str = Field(...)
    action_type: str = Field(...)
    priority: str = Field(...)
    review_sla_hours: float = Field(...)
    require_dual_review: bool = Field(...)
    routing_tags: List[str] = Field(default_factory=list)
    triggered_rules: List[str] = Field(default_factory=list)
    audit_metadata: dict = Field(default_factory=dict)


class RiskDossierResponse(BaseModel):
    """Unified Risk Assessment Dossier for a detection window."""
    window: DetectionWindowResponse = Field(...)
    detector_type: str = Field(...)
    is_flagged: bool = Field(...)
    risk_result: RiskScoringResultResponse = Field(...)
    explanation: ExplanationResultResponse = Field(...)
    policy_decision: PolicyDecisionResponse = Field(...)
    audit_entry_id: str = Field(...)


# ---------------------------------------------------------------------------
# Evaluation Engine Schemas
# ---------------------------------------------------------------------------

class EvaluationResponse(BaseModel):
    """Phase 6 detector evaluation metrics."""
    id: Optional[int] = Field(None, description="Evaluation run ID if stored.")
    detector_type: str = Field(..., description="Detector type ('baseline' or 'ml').")
    partition: str = Field(..., description="Evaluated partition ('dev_test').")
    run_timestamp: str = Field(...)
    precision: float = Field(...)
    recall: float = Field(...)
    f1_score: float = Field(...)
    false_positive_rate: float = Field(...)
    true_positives: int = Field(...)
    false_positives: int = Field(...)
    false_negatives: int = Field(...)
    true_negatives: int = Field(...)
    fp_cost: float = Field(...)
    fn_cost: float = Field(...)
    total_cost: float = Field(...)
    notes: str = Field(...)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Audit & Analyst Action Schemas
# ---------------------------------------------------------------------------

class AuditRecordResponse(BaseModel):
    """Phase 10 Audit Record payload."""
    entry_id: str = Field(...)
    timestamp: str = Field(...)
    event_type: str = Field(...)
    window_id: str = Field(...)
    merchant_id: str = Field(...)
    actor: str = Field(...)
    payload: dict = Field(...)
    previous_hash: Optional[str] = Field(None)
    integrity_hash: str = Field(...)


class AuditReportResponse(BaseModel):
    """Phase 10 Compliance Audit Summary."""
    window_id: str = Field(...)
    event_count: int = Field(...)
    events: List[AuditRecordResponse] = Field(default_factory=list)
    integrity_valid: bool = Field(...)
    integrity_errors: List[str] = Field(default_factory=list)


# Defensive Review Dispositions (Strict Allowlist)
PERMITTED_ANALYST_DISPOSITIONS = {
    "escalate",
    "resolve",
    "flag_for_followup",
    "monitor",
}

# Rejected Destructive Action Words
DESTRUCTIVE_ACTION_TERMS = {
    "ban",
    "block",
    "suspend",
    "terminate",
    "freeze",
    "deactivate",
    "auto_ban",
    "auto_block",
    "auto_terminate",
    "auto_freeze",
}


class AnalystActionRequest(BaseModel):
    """Request payload for recording human analyst review action."""
    actor: str = Field(..., description="Explicit analyst/user identifier (e.g. 'ANALYST:user_01').")
    window_id: int = Field(..., description="Associated DetectionWindow ID.")
    disposition: str = Field(..., description="Permitted review disposition ('escalate', 'resolve', 'flag_for_followup', 'monitor').")
    notes: Optional[str] = Field("", max_length=2000, description="Analyst review comments (max 2000 chars).")

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("actor identifier cannot be empty.")
        return v_clean

    @field_validator("disposition")
    @classmethod
    def validate_disposition(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if v_clean in DESTRUCTIVE_ACTION_TERMS:
            raise ValueError(f"Destructive action '{v}' is strictly prohibited. Only defensive triage actions are allowed.")
        if v_clean not in PERMITTED_ANALYST_DISPOSITIONS:
            raise ValueError(f"Invalid disposition '{v}'. Must be one of {sorted(PERMITTED_ANALYST_DISPOSITIONS)}.")
        return v_clean

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return ""
        # Inspect notes for credentials
        sensitive_terms = {"api_key", "authorization", "bearer_token", "password", "secret"}
        v_lower = v.lower()
        for term in sensitive_terms:
            if term in v_lower:
                raise ValueError(f"Notes contain sensitive key pattern '{term}'. Credentials must never be logged.")
        return v


class AnalystActionResponse(BaseModel):
    """Response returned upon recording analyst review action."""
    status: str = Field("success")
    entry_id: str = Field(..., description="Logged audit record entry_id.")
    message: str = Field(...)


# ---------------------------------------------------------------------------
# Detector Run & Detection Result Schemas (Phase 12.1B)
# ---------------------------------------------------------------------------

class AnomalyDetectionResponse(BaseModel):
    """Persisted AnomalyDetection record response payload."""
    id: int = Field(..., description="Anomaly detection database ID.")
    window_id: int = Field(..., description="Associated DetectionWindow ID.")
    detector_type: str = Field(..., description="Detector type ('baseline' or 'ml').")
    risk_score: float = Field(..., description="Calculated risk score (0-100).")
    is_flagged: bool = Field(..., description="Whether window was flagged as anomalous.")
    explanation: Optional[str] = Field(None, description="Generated explanation string.")
    created_at: str = Field(..., description="Record creation ISO timestamp.")

    model_config = ConfigDict(from_attributes=True)


class DetectorSummaryDetail(BaseModel):
    """Execution statistics for a single detector type."""
    windows_scored: int = Field(..., description="Number of windows scored by this detector.")
    windows_flagged: int = Field(..., description="Number of windows flagged by this detector.")


class DetectorRunSummaryResponse(BaseModel):
    """Execution summary returned by POST /api/pipeline/run-detectors."""
    detectors_run: List[str] = Field(..., description="List of detectors executed.")
    results: dict[str, DetectorSummaryDetail] = Field(..., description="Execution summary map by detector type.")
    run_timestamp: str = Field(..., description="ISO timestamp of detector execution.")

