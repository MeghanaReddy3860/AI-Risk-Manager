"""
AI Risk Manager — Phase 7: Risk Scoring Engine
================================================

Purpose
-------
Translate an existing detector risk score (0–100, produced by Phase 4 or
Phase 5) into a higher-level operational risk interpretation:

    1. **Risk band** — categorical severity label (low / medium / high / critical)
    2. **Estimated financial exposure** — total_amount × risk_multiplier
    3. **Recommended defensive action** — human-review recommendation

This module is a **pure scoring layer**.  It does NOT:
  - read from or write to the database
  - modify detector predictions
  - call external APIs or LLMs
  - perform automatic blocking, banning, or irreversible actions

It consumes already-generated detector risk scores and produces a structured
risk interpretation for downstream phases (API, dashboard, audit).

Architecture Position
---------------------
    Detector (Phase 4/5) → **Risk Scoring (Phase 7)** → Explanation (Phase 8) → ...

Phase 7 never feeds results back into detectors.

Risk Band Thresholds
--------------------
Reused from ``config.py`` (environment-overridable):

    RISK_LOW_MAX    = 30   → low:      0 <= score <= 30
    RISK_MEDIUM_MAX = 60   → medium:  30 <  score <= 60
    RISK_HIGH_MAX   = 80   → high:    60 <  score <= 80
                            → critical: 80 < score <= 100

Boundary behaviour is explicit and tested.

Financial Exposure Estimate
---------------------------
Risk multipliers from ``config.py``:

    low      → 0.10
    medium   → 0.25
    high     → 0.50
    critical → 1.00

    estimated_exposure = total_amount × risk_multiplier

This is an **approximate exposure estimate**, NOT a claim that the amount
will actually be lost.  The multiplier represents the estimated fraction of
the window's total transaction value that may be at risk given the severity
level.  These values are configurable and documented.

Recommended Defensive Actions
------------------------------
    low      → "Monitor"
    medium   → "Flag for analyst review"
    high     → "Escalate for priority review"
    critical → "Immediate escalation and human review"

All recommendations are strictly **defense-only** human-review actions.
The system never automatically blocks, bans, suspends, or takes irreversible
actions.

Input Validation
----------------
    risk_score:   must be numeric, finite, 0 <= score <= 100
    total_amount: must be numeric, finite, >= 0

Invalid inputs raise ``ValueError`` rather than silently producing incorrect
results.  Values are never silently clamped.

Determinism
-----------
Given identical inputs and configuration, the scorer always produces
identical outputs.  There is no randomness, no external state, and no
side effects.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the backend package is importable when running as a script
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — reused from existing config.py
# ---------------------------------------------------------------------------

# Risk-band upper boundaries (inclusive).
# Reused from Phase 1 config; NOT duplicated.
RISK_LOW_MAX: float = float(settings.RISK_LOW_MAX)       # 30
RISK_MEDIUM_MAX: float = float(settings.RISK_MEDIUM_MAX) # 60
RISK_HIGH_MAX: float = float(settings.RISK_HIGH_MAX)     # 80

# Risk multipliers for exposure estimation (Phase 7 config).
RISK_MULTIPLIERS: dict[str, float] = {
    "low": settings.RISK_MULTIPLIER_LOW,           # 0.10
    "medium": settings.RISK_MULTIPLIER_MEDIUM,     # 0.25
    "high": settings.RISK_MULTIPLIER_HIGH,         # 0.50
    "critical": settings.RISK_MULTIPLIER_CRITICAL, # 1.00
}

# Recommended defensive actions per risk band.
# All are human-review recommendations — never automatic enforcement.
RECOMMENDED_ACTIONS: dict[str, str] = {
    "low": "Monitor",
    "medium": "Flag for analyst review",
    "high": "Escalate for priority review",
    "critical": "Immediate escalation and human review",
}

# Valid risk bands (ordered by severity).
VALID_RISK_BANDS: tuple[str, ...] = ("low", "medium", "high", "critical")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskScoringResult:
    """
    Structured output of the Risk Scoring Engine.

    Attributes:
        risk_score:         The original detector risk score (0–100).
        risk_band:          Categorical severity: low / medium / high / critical.
        risk_multiplier:    The exposure multiplier applied for this band.
        estimated_exposure: total_amount × risk_multiplier.
        recommended_action: Defense-only human-review recommendation.
    """
    risk_score: float
    risk_band: str
    risk_multiplier: float
    estimated_exposure: float
    recommended_action: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_risk_score(risk_score) -> float:
    """
    Validate and return the risk score as a float.

    Accepted: numeric values from 0 through 100 inclusive.

    Rejected (raises ValueError):
      - Non-numeric types
      - NaN
      - Positive/negative infinity
      - Values below 0
      - Values above 100

    The function never silently clamps invalid input.
    """
    # Type check: must be int or float
    if not isinstance(risk_score, (int, float)):
        raise ValueError(
            f"risk_score must be numeric (int or float), "
            f"got {type(risk_score).__name__}: {risk_score!r}"
        )

    score = float(risk_score)

    # NaN check
    if math.isnan(score):
        raise ValueError("risk_score must not be NaN.")

    # Infinity check
    if math.isinf(score):
        raise ValueError(
            f"risk_score must be finite, got {'positive' if score > 0 else 'negative'} infinity."
        )

    # Range check
    if score < 0:
        raise ValueError(
            f"risk_score must be >= 0, got {score}."
        )
    if score > 100:
        raise ValueError(
            f"risk_score must be <= 100, got {score}."
        )

    return score


def _validate_total_amount(total_amount) -> float:
    """
    Validate and return total_amount as a float.

    Accepted: numeric values >= 0 and finite.

    Rejected (raises ValueError):
      - Non-numeric types
      - NaN
      - Positive/negative infinity
      - Negative values

    The function never silently converts invalid values to zero.
    """
    if not isinstance(total_amount, (int, float)):
        raise ValueError(
            f"total_amount must be numeric (int or float), "
            f"got {type(total_amount).__name__}: {total_amount!r}"
        )

    amount = float(total_amount)

    if math.isnan(amount):
        raise ValueError("total_amount must not be NaN.")

    if math.isinf(amount):
        raise ValueError(
            f"total_amount must be finite, got {'positive' if amount > 0 else 'negative'} infinity."
        )

    if amount < 0:
        raise ValueError(
            f"total_amount must be >= 0, got {amount}."
        )

    return amount


# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------

def get_risk_band(risk_score) -> str:
    """
    Map a detector risk score (0–100) to a risk band.

    Boundary behaviour (using default config thresholds):
        0      → low
        30     → low       (score <= RISK_LOW_MAX)
        30.01  → medium
        60     → medium    (score <= RISK_MEDIUM_MAX)
        60.01  → high
        80     → high      (score <= RISK_HIGH_MAX)
        80.01  → critical
        100    → critical

    Args:
        risk_score: Numeric value 0–100 (validated internally).

    Returns:
        One of: "low", "medium", "high", "critical".

    Raises:
        ValueError: If risk_score is invalid.
    """
    score = _validate_risk_score(risk_score)

    if score <= RISK_LOW_MAX:
        return "low"
    elif score <= RISK_MEDIUM_MAX:
        return "medium"
    elif score <= RISK_HIGH_MAX:
        return "high"
    else:
        return "critical"


def estimate_exposure(total_amount, risk_band: str) -> float:
    """
    Estimate financial exposure for a detection window.

    Formula:
        estimated_exposure = total_amount × risk_multiplier

    This is an **approximate exposure estimate**, not a precise financial
    prediction.  The multiplier represents the estimated fraction of the
    window's total transaction value that may be at risk.

    Args:
        total_amount: The DetectionWindow.total_amount (validated internally).
        risk_band:    One of "low", "medium", "high", "critical".

    Returns:
        Estimated exposure as a float.

    Raises:
        ValueError: If total_amount is invalid or risk_band is unknown.
    """
    amount = _validate_total_amount(total_amount)

    if risk_band not in RISK_MULTIPLIERS:
        raise ValueError(
            f"Unknown risk_band '{risk_band}'. "
            f"Valid bands: {VALID_RISK_BANDS}."
        )

    multiplier = RISK_MULTIPLIERS[risk_band]
    return amount * multiplier


def get_recommended_action(risk_band: str) -> str:
    """
    Return the defense-only recommended action for a risk band.

    Actions:
        low      → "Monitor"
        medium   → "Flag for analyst review"
        high     → "Escalate for priority review"
        critical → "Immediate escalation and human review"

    All actions are human-review recommendations.  The system NEVER
    automatically blocks, bans, suspends, or takes irreversible actions.

    Args:
        risk_band: One of "low", "medium", "high", "critical".

    Returns:
        A human-readable defensive action string.

    Raises:
        ValueError: If risk_band is unknown.
    """
    if risk_band not in RECOMMENDED_ACTIONS:
        raise ValueError(
            f"Unknown risk_band '{risk_band}'. "
            f"Valid bands: {VALID_RISK_BANDS}."
        )

    return RECOMMENDED_ACTIONS[risk_band]


def score_risk(risk_score, total_amount) -> RiskScoringResult:
    """
    Main scoring function — produce a complete risk interpretation.

    Takes a detector risk score (0–100) and the detection window's
    total_amount, and returns a structured RiskScoringResult with:
      - risk_band
      - risk_multiplier
      - estimated_exposure
      - recommended_action

    This is a **pure function**: no database access, no side effects,
    deterministic output for identical inputs.

    Args:
        risk_score:   Detector risk score, 0–100 (from Phase 4 or Phase 5).
        total_amount: DetectionWindow.total_amount (>= 0, finite).

    Returns:
        RiskScoringResult dataclass with all fields populated.

    Raises:
        ValueError: If either input is invalid.

    Example::

        >>> result = score_risk(risk_score=65.0, total_amount=100000.0)
        >>> result.risk_band
        'high'
        >>> result.estimated_exposure
        50000.0
        >>> result.recommended_action
        'Escalate for priority review'
    """
    validated_score = _validate_risk_score(risk_score)
    validated_amount = _validate_total_amount(total_amount)

    band = get_risk_band(validated_score)
    multiplier = RISK_MULTIPLIERS[band]
    exposure = validated_amount * multiplier
    action = RECOMMENDED_ACTIONS[band]

    return RiskScoringResult(
        risk_score=validated_score,
        risk_band=band,
        risk_multiplier=multiplier,
        estimated_exposure=exposure,
        recommended_action=action,
    )
