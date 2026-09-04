"""
AI Risk Manager — Phase 9: Policy Engine
========================================

Purpose
-------
Evaluates detector results, risk scores, estimated financial exposure, and
DetectionWindow features against configurable, defense-only operational policies.

The Policy Engine determines:
  1. **Policy ID** — Matched operational policy identifier
  2. **Action Type** — Operational defense workflow category (e.g. URGENT_TRIAGE, PRIORITY_ESCALATION)
  3. **Priority** — Human review priority (P0, P1, P2, P3)
  4. **Review SLA** — Target human review timeframe in hours (1h, 4h, 24h, 72h)
  5. **Require Dual Review** — Senior analyst secondary sign-off requirement
  6. **Routing Tags** — Categorical routing classifications
  7. **Triggered Rules** — Explicit auditable list of matching conditions

Architecture Position
---------------------
    DetectionWindow (Features)
           ↓
    Detector (Phase 4 / 5) → risk_score + is_flagged
           ↓
    Risk Scorer (Phase 7)  → risk_band + estimated_exposure + recommended_action
           ↓
    Explanation Engine (Phase 8) → ExplanationResult (Auditing reference only)
           ↓
    Policy Engine (Phase 9) → PolicyDecision

Non-Decision Role of Explanation
---------------------------------
The ``explanation_result`` parameter exists solely for pipeline interface
consistency and optional audit metadata cross-referencing. It NEVER influences
policy selection, action type, priority, SLA, dual-review requirements, or routing tags.

Defense-Only Guarantees
-----------------------
The Policy Engine strictly routes cases to human review queues. It NEVER:
  - triggers automated blocking, banning, suspension, or termination
  - executes destructive actions or bypasses security controls
  - takes irreversible actions without human review
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Ensure the backend package is importable when running as a script
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from config import settings  # noqa: E402
from risk_scorer import (  # noqa: E402
    RiskScoringResult,
    VALID_RISK_BANDS,
    _validate_risk_score,
    _validate_total_amount,
)


# ---------------------------------------------------------------------------
# Constants & Default Policy Identifiers
# ---------------------------------------------------------------------------

POL_CRITICAL_SURGE = "POL_CRITICAL_SURGE"
POL_HIGH_EXPOSURE = "POL_HIGH_EXPOSURE"
POL_MEDIUM_ANOMALY = "POL_MEDIUM_ANOMALY"
POL_ROUTINE_MONITOR = "POL_ROUTINE_MONITOR"

VALID_POLICY_IDS = (
    POL_CRITICAL_SURGE,
    POL_HIGH_EXPOSURE,
    POL_MEDIUM_ANOMALY,
    POL_ROUTINE_MONITOR,
)

# Standard defense-only action types
ACTION_URGENT_TRIAGE = "URGENT_TRIAGE"
ACTION_PRIORITY_ESCALATION = "PRIORITY_ESCALATION"
ACTION_ANALYST_QUEUE = "ANALYST_QUEUE"
ACTION_STANDARD_LOG = "STANDARD_LOG"

VALID_ACTION_TYPES = (
    ACTION_URGENT_TRIAGE,
    ACTION_PRIORITY_ESCALATION,
    ACTION_ANALYST_QUEUE,
    ACTION_STANDARD_LOG,
)

# Prohibited destructive / automated terms for defense-only enforcement
PROHIBITED_ACTION_TERMS = (
    "block",
    "ban",
    "suspend",
    "terminate",
    "freeze",
    "deactivate",
    "bypass",
    "circumvent",
    "disable",
    "auto_block",
    "auto_ban",
    "auto_suspend",
    "auto_terminate",
)


# ---------------------------------------------------------------------------
# PolicyDecision Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyDecision:
    """
    Immutable structured decision produced by the Policy Engine.

    Attributes:
        window_id:            DetectionWindow identifier.
        merchant_id:          Merchant identifier.
        policy_id:            Matched operational policy identifier.
        action_type:          Defense-only operational workflow action.
        priority:             Review priority code ('P0', 'P1', 'P2', 'P3').
        review_sla_hours:     Target human review deadline in hours.
        require_dual_review:  Whether secondary senior review is required.
        routing_tags:         Categorical routing and classification tags.
        triggered_rules:      Auditable list of conditions that evaluated to True.
        audit_metadata:       Optional cross-referencing metadata (does not influence decision).
    """
    window_id: Any
    merchant_id: str
    policy_id: str
    action_type: str
    priority: str
    review_sla_hours: float
    require_dual_review: bool
    routing_tags: list[str]
    triggered_rules: list[str]
    audit_metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# PolicyRule Dataclass for Custom Policy Extensions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyRule:
    """
    Configurable policy definition for custom policy evaluation.
    """
    policy_id: str
    action_type: str
    priority: str
    review_sla_hours: float
    require_dual_review: bool
    routing_tags: list[str]
    condition: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, list[str]]]

    def __post_init__(self):
        _validate_defense_only_action(self.action_type, self.policy_id)


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def _validate_defense_only_action(action_type: str, policy_id: str = "") -> None:
    """
    Ensure action_type and policy_id do not contain prohibited destructive / auto-enforcement terms.
    """
    raw_combined = f"{action_type} {policy_id}".lower()
    # Normalize underscores and special characters to whitespace so word boundaries match
    clean_combined = re.sub(r"[_\W]+", " ", raw_combined)

    for term in PROHIBITED_ACTION_TERMS:
        clean_term = re.sub(r"[_\W]+", " ", term.lower())
        if re.search(r"\b" + re.escape(clean_term) + r"\b", clean_combined):
            raise ValueError(
                f"Defense-only violation: Prohibited action term {term!r} in "
                f"action_type={action_type!r} / policy_id={policy_id!r}. "
                "Automated destructive actions are strictly disallowed."
            )


def _extract_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Safely extract a field from dict, object, or ORM model."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    if hasattr(obj, "get") and callable(obj.get):
        return obj.get(field_name, default)
    return default


def _validate_window_input(window: Any) -> dict[str, Any]:
    """Validate required fields in DetectionWindow input."""
    if window is None:
        raise ValueError("window cannot be None.")

    window_id = _extract_field(window, "window_id")
    if window_id is None:
        window_id = _extract_field(window, "id")
    if window_id is None:
        raise ValueError("window must contain 'window_id' or 'id'.")

    merchant_id = _extract_field(window, "merchant_id")
    if merchant_id is None or not str(merchant_id).strip():
        raise ValueError("window must contain a non-empty 'merchant_id'.")
    merchant_id = str(merchant_id).strip()

    raw_count = _extract_field(window, "transaction_count")
    if raw_count is None or isinstance(raw_count, bool) or not isinstance(raw_count, (int, float)):
        raise ValueError("window 'transaction_count' must be a numeric integer or float.")
    if math.isnan(raw_count) or math.isinf(raw_count) or raw_count < 0:
        raise ValueError(f"window 'transaction_count' must be non-negative and finite, got {raw_count}.")
    transaction_count = int(raw_count)

    raw_total = _extract_field(window, "total_amount")
    total_amount = _validate_total_amount(raw_total)

    raw_avg = _extract_field(window, "avg_transaction_amount")
    avg_transaction_amount = _validate_total_amount(raw_avg)

    return {
        "window_id": window_id,
        "merchant_id": merchant_id,
        "transaction_count": transaction_count,
        "total_amount": total_amount,
        "avg_transaction_amount": avg_transaction_amount,
    }


def _validate_risk_input(risk_result: Any) -> dict[str, Any]:
    """
    Validate risk result input.

    Note: risk_band and risk_score are treated as independently supplied inputs.
    Cross-field consistency is intentionally not enforced per Phase 9 specification.
    """
    if risk_result is None:
        raise ValueError("risk_result cannot be None.")

    if isinstance(risk_result, RiskScoringResult):
        return {
            "risk_score": risk_result.risk_score,
            "risk_band": risk_result.risk_band,
            "estimated_exposure": risk_result.estimated_exposure,
            "recommended_action": risk_result.recommended_action,
        }

    raw_score = _extract_field(risk_result, "risk_score")
    if raw_score is None:
        raise ValueError("risk_result must contain 'risk_score'.")
    risk_score = _validate_risk_score(raw_score)

    risk_band = _extract_field(risk_result, "risk_band")
    if risk_band is None:
        risk_band = _extract_field(risk_result, "risk_level")
    if risk_band is None or risk_band not in VALID_RISK_BANDS:
        raise ValueError(f"risk_result must contain valid 'risk_band' in {VALID_RISK_BANDS}.")

    raw_exposure = _extract_field(risk_result, "estimated_exposure")
    if raw_exposure is None:
        raise ValueError("risk_result must contain 'estimated_exposure'.")
    estimated_exposure = _validate_total_amount(raw_exposure)

    recommended_action = _extract_field(risk_result, "recommended_action", "")

    return {
        "risk_score": risk_score,
        "risk_band": risk_band,
        "estimated_exposure": estimated_exposure,
        "recommended_action": recommended_action,
    }


# ---------------------------------------------------------------------------
# Default Policy Evaluation Logic
# ---------------------------------------------------------------------------

def _evaluate_critical_surge_policy(
    window_data: dict[str, Any], risk_data: dict[str, Any]
) -> tuple[bool, list[str]]:
    """
    Evaluate POL_CRITICAL_SURGE conditions:
      risk_band == 'critical' OR risk_score > POLICY_CRITICAL_SCORE_THRESHOLD
    """
    matched = False
    triggered: list[str] = []

    risk_band = risk_data["risk_band"]
    risk_score = risk_data["risk_score"]
    critical_threshold = float(settings.POLICY_CRITICAL_SCORE_THRESHOLD)

    if risk_band == "critical":
        matched = True
        triggered.append("risk_band == 'critical'")

    if risk_score > critical_threshold:
        matched = True
        triggered.append(
            f"risk_score ({risk_score:.1f}) > POLICY_CRITICAL_SCORE_THRESHOLD ({critical_threshold:.1f})"
        )

    return matched, triggered


def _evaluate_high_exposure_policy(
    window_data: dict[str, Any], risk_data: dict[str, Any]
) -> tuple[bool, list[str]]:
    """
    Evaluate POL_HIGH_EXPOSURE conditions:
      risk_band == 'high' OR estimated_exposure > POLICY_HIGH_EXPOSURE_THRESHOLD
    """
    matched = False
    triggered: list[str] = []

    risk_band = risk_data["risk_band"]
    exposure = risk_data["estimated_exposure"]
    exposure_threshold = float(settings.POLICY_HIGH_EXPOSURE_THRESHOLD)

    if risk_band == "high":
        matched = True
        triggered.append("risk_band == 'high'")

    if exposure > exposure_threshold:
        matched = True
        triggered.append(
            f"estimated_exposure (₹{exposure:,.2f}) > POLICY_HIGH_EXPOSURE_THRESHOLD (₹{exposure_threshold:,.2f})"
        )

    return matched, triggered


def _evaluate_medium_anomaly_policy(
    window_data: dict[str, Any], risk_data: dict[str, Any]
) -> tuple[bool, list[str]]:
    """
    Evaluate POL_MEDIUM_ANOMALY conditions:
      risk_band == 'medium'
    """
    risk_band = risk_data["risk_band"]
    if risk_band == "medium":
        return True, ["risk_band == 'medium'"]

    return False, []


# ---------------------------------------------------------------------------
# Main Public Interface
# ---------------------------------------------------------------------------

def evaluate_policy(
    window: Any,
    risk_result: Any,
    explanation_result: Optional[Any] = None,
    custom_policies: Optional[Sequence[PolicyRule]] = None,
) -> PolicyDecision:
    """
    Evaluate operational policies for a DetectionWindow and Risk Result.

    Parameters:
        window:             DetectionWindow record, dict, or object with required fields:
                            window_id, merchant_id, transaction_count, total_amount, avg_transaction_amount.
        risk_result:        Phase 7 RiskScoringResult or dict with:
                            risk_score, risk_band (or risk_level), estimated_exposure.
        explanation_result: Optional Phase 8 ExplanationResult for interface compatibility and audit
                            cross-reference only. NEVER influences any policy decision.
        custom_policies:    Optional sequence of custom PolicyRule objects.

    Returns:
        PolicyDecision dataclass.
    """
    # 1. Validate structured inputs
    window_data = _validate_window_input(window)
    risk_data = _validate_risk_input(risk_result)

    window_id = window_data["window_id"]
    merchant_id = window_data["merchant_id"]

    # Optional audit cross-reference (does NOT affect decision)
    audit_meta: Optional[dict[str, Any]] = None
    if explanation_result is not None:
        audit_meta = {
            "has_explanation": True,
            "explanation_source": _extract_field(explanation_result, "generated_by", "unknown"),
        }

    # 2. Evaluate Custom Policies (if supplied)
    if custom_policies:
        for rule in custom_policies:
            matched, triggered = rule.condition(window_data, risk_data)
            if matched:
                return PolicyDecision(
                    window_id=window_id,
                    merchant_id=merchant_id,
                    policy_id=rule.policy_id,
                    action_type=rule.action_type,
                    priority=rule.priority,
                    review_sla_hours=rule.review_sla_hours,
                    require_dual_review=rule.require_dual_review,
                    routing_tags=list(rule.routing_tags),
                    triggered_rules=triggered,
                    audit_metadata=audit_meta,
                )

    # 3. Evaluate Built-in Policies in Deterministic Precedence Order:
    # Priority 1: POL_CRITICAL_SURGE (P0)
    matched_crit, trig_crit = _evaluate_critical_surge_policy(window_data, risk_data)
    if matched_crit:
        return PolicyDecision(
            window_id=window_id,
            merchant_id=merchant_id,
            policy_id=POL_CRITICAL_SURGE,
            action_type=ACTION_URGENT_TRIAGE,
            priority="P0",
            review_sla_hours=1.0,
            require_dual_review=True,
            routing_tags=["critical_surge", "urgent_triage"],
            triggered_rules=trig_crit,
            audit_metadata=audit_meta,
        )

    # Priority 2: POL_HIGH_EXPOSURE (P1)
    matched_high, trig_high = _evaluate_high_exposure_policy(window_data, risk_data)
    if matched_high:
        return PolicyDecision(
            window_id=window_id,
            merchant_id=merchant_id,
            policy_id=POL_HIGH_EXPOSURE,
            action_type=ACTION_PRIORITY_ESCALATION,
            priority="P1",
            review_sla_hours=4.0,
            require_dual_review=False,
            routing_tags=["high_exposure", "priority_escalation"],
            triggered_rules=trig_high,
            audit_metadata=audit_meta,
        )

    # Priority 3: POL_MEDIUM_ANOMALY (P2)
    matched_med, trig_med = _evaluate_medium_anomaly_policy(window_data, risk_data)
    if matched_med:
        return PolicyDecision(
            window_id=window_id,
            merchant_id=merchant_id,
            policy_id=POL_MEDIUM_ANOMALY,
            action_type=ACTION_ANALYST_QUEUE,
            priority="P2",
            review_sla_hours=24.0,
            require_dual_review=False,
            routing_tags=["medium_risk", "analyst_queue"],
            triggered_rules=trig_med,
            audit_metadata=audit_meta,
        )

    # Priority 4: POL_ROUTINE_MONITOR (P3 - Default Fallback)
    return PolicyDecision(
        window_id=window_id,
        merchant_id=merchant_id,
        policy_id=POL_ROUTINE_MONITOR,
        action_type=ACTION_STANDARD_LOG,
        priority="P3",
        review_sla_hours=72.0,
        require_dual_review=False,
        routing_tags=["routine_monitor", "low_risk"],
        triggered_rules=["default_routine_monitoring"],
        audit_metadata=audit_meta,
    )
