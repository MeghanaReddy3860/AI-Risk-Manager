"""
AI Risk Manager — Phase 8: AI Explanation Engine
=================================================

Purpose
-------
Translates DetectionWindow features, detector anomaly outputs, and Phase 7
Risk Scoring results into clear, human-understandable, auditable, and
quantitatively grounded risk explanations for fraud and security analysts.

Architecture Position
---------------------
    DetectionWindow (Features)
           ↓
    Detector (Phase 4 / 5) → risk_score + is_flagged
           ↓
    Risk Scorer (Phase 7)  → risk_band + estimated_exposure + recommended_action
           ↓
    Explanation Engine (Phase 8) → ExplanationResult

Execution Modes
---------------
1. **Rule-Based Explanation (Primary & Default)**:
   - 100% deterministic, zero external dependencies, offline-capable.
   - Quantitatively grounded in approved DetectionWindow features:
     `transaction_count`, `total_amount`, `avg_transaction_amount`.
   - Compares with supplied `baseline_stats` when available.
   - Strict numerical formatting contract: multipliers `N.Nx`, percentages `N.N%`.
   - Defense-only recommendations strictly matching Phase 7.

2. **Optional LLM Explanation (OpenAI API / GPT-4o)**:
   - Invoked only when `use_llm=True` and `OPENAI_API_KEY` is present.
   - Uses a fixed, defensive system prompt.
   - Two-stage safety & grounding verification:
     a) Contextual Defense-Only Validation: checks for imperative destructive actions.
     b) Numerical Grounding Validation: verifies every extracted numeric claim
        matches supplied or rule-derived facts within tolerance.
   - Automatic fail-closed fallback to rule-based explanation on:
     - Missing / invalid API key
     - Network errors / timeouts / rate limits
     - Unsafe or imperative destructive recommendations
     - Ungrounded / fabricated numerical claims
     - Ambiguous safety classifications

Defense-Only Guarantees
-----------------------
The system strictly advises human review. It NEVER executes or recommends:
  - automated blocking, banning, suspension, or termination
  - destructive actions or circumventing security controls
  - irreversible actions without human review
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
# Constants
# ---------------------------------------------------------------------------

VALID_GENERATED_BY = ("rule_based", "llm", "rule_based_fallback")

# Approved DetectionWindow features (stable contract)
APPROVED_FEATURES = (
    "transaction_count",
    "total_amount",
    "avg_transaction_amount",
)

# Candidate action terms for defense-only validation
SENSITIVE_ACTION_TERMS = [
    "block the merchant",
    "block the account",
    "block access",
    "block transactions",
    "block",
    "ban the merchant",
    "ban the account",
    "ban",
    "terminate access",
    "terminate the account",
    "terminate",
    "suspend the merchant",
    "suspend the account",
    "suspend",
    "freeze the account",
    "freeze funds",
    "freeze",
    "close the account",
    "deactivate the account",
    "deactivate",
    "bypass security",
    "bypass controls",
    "bypass",
    "disable security",
    "evade detection",
    "evade",
    "circumvent security",
    "circumvent controls",
    "circumvent",
]

# Negation and prohibition cues (indicate safe defensive governance)
NEGATION_PROHIBITION_CUES = [
    "do not",
    "don't",
    "must not",
    "should not",
    "shall not",
    "cannot",
    "no automatic",
    "no automated",
    "not recommended",
    "avoid",
    "never",
    "without automatic",
    "without automated",
    "human review is required",
    "requires human review",
    "require human review",
    "subject to human review",
    "pending human review",
]

# Imperative / recommendation cues that trigger unsafe classification
IMPERATIVE_RECOMMENDATION_CUES = [
    "immediately",
    "now",
    "recommend",
    "recommended to",
    "should",
    "must",
    "please",
    "proceed to",
    "action required:",
    "action:",
    "execute",
    "apply",
]

# Fixed defensive system prompt for LLM
FIXED_DEFENSE_SYSTEM_PROMPT = (
    "You are an AI Risk Explanation Assistant for a defensive fraud-spike monitoring system. "
    "Your role is to provide clear, objective, and strictly defense-only explanations of flagged anomalies "
    "to human fraud analysts.\n\n"
    "CRITICAL RULES:\n"
    "1. DEFENSE-ONLY: Never recommend automatic blocking, banning, account suspension, or irreversible automated actions. "
    "Always emphasize human analyst review.\n"
    "2. NUMERICAL GROUNDING: Use ONLY the exact numbers, metrics, counts, amounts, percentages, and multipliers "
    "provided in the user prompt. NEVER invent, estimate, extrapolate, or hallucinate new numbers or statistics.\n"
    "3. FACTUAL & CONCISE: State the key anomaly drivers clearly based solely on the provided data.\n"
    "4. NO ADVERSARIAL ADVICE: Never provide guidance on evading detection or bypassing controls."
)

# Bounded OpenAI API timeout (seconds)
OPENAI_REQUEST_TIMEOUT_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# ExplanationResult Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExplanationResult:
    """
    Structured output of the AI Explanation Engine.

    Attributes:
        window_id:          Identifier of the analyzed DetectionWindow.
        merchant_id:        Merchant identifier string.
        summary:            High-level concise explanation summary.
        key_drivers:        List of identified quantitative anomaly drivers.
        risk_level:         Risk severity band ('low', 'medium', 'high', 'critical').
        estimated_exposure: Financial exposure amount in currency units.
        recommended_action: Defense-only recommended action for human analysts.
        generated_by:       Explanation origin: 'rule_based', 'llm', or 'rule_based_fallback'.
        raw_text:           Complete human-readable explanation narrative.
    """
    window_id: Any
    merchant_id: str
    summary: str
    key_drivers: list[str]
    risk_level: str
    estimated_exposure: float
    recommended_action: str
    generated_by: str
    raw_text: str

    def __post_init__(self):
        if self.generated_by not in VALID_GENERATED_BY:
            raise ValueError(
                f"Invalid generated_by: {self.generated_by!r}. Must be one of {VALID_GENERATED_BY}."
            )
        if self.risk_level not in VALID_RISK_BANDS:
            raise ValueError(
                f"Invalid risk_level: {self.risk_level!r}. Must be one of {VALID_RISK_BANDS}."
            )


# ---------------------------------------------------------------------------
# Window & Risk Input Parsing Helpers
# ---------------------------------------------------------------------------

def _extract_window_field(window: Any, field_name: str, default: Any = None) -> Any:
    """Safely extract a field from a dict, ORM model, pandas Series, or object."""
    if isinstance(window, dict):
        return window.get(field_name, default)
    if hasattr(window, field_name):
        return getattr(window, field_name)
    if hasattr(window, "get") and callable(window.get):
        return window.get(field_name, default)
    return default


def _validate_window_input(window: Any) -> dict[str, Any]:
    """
    Validate and extract required fields from the DetectionWindow input.

    Raises ValueError if required fields are missing or invalid.
    """
    if window is None:
        raise ValueError("window cannot be None.")

    window_id = _extract_window_field(window, "window_id")
    if window_id is None:
        window_id = _extract_window_field(window, "id")
    if window_id is None:
        raise ValueError("window must contain 'window_id' or 'id'.")

    merchant_id = _extract_window_field(window, "merchant_id")
    if merchant_id is None or not str(merchant_id).strip():
        raise ValueError("window must contain a non-empty 'merchant_id'.")
    merchant_id = str(merchant_id).strip()

    # Numeric features
    raw_count = _extract_window_field(window, "transaction_count")
    if raw_count is None or isinstance(raw_count, bool) or not isinstance(raw_count, (int, float)):
        raise ValueError("window 'transaction_count' must be a numeric integer or float.")
    if math.isnan(raw_count) or math.isinf(raw_count) or raw_count < 0:
        raise ValueError(f"window 'transaction_count' must be non-negative and finite, got {raw_count}.")
    transaction_count = int(raw_count)

    raw_total = _extract_window_field(window, "total_amount")
    total_amount = _validate_total_amount(raw_total)

    raw_avg = _extract_window_field(window, "avg_transaction_amount")
    avg_transaction_amount = _validate_total_amount(raw_avg)

    return {
        "window_id": window_id,
        "merchant_id": merchant_id,
        "transaction_count": transaction_count,
        "total_amount": total_amount,
        "avg_transaction_amount": avg_transaction_amount,
    }


def _validate_risk_input(risk_result: Any, total_amount: float) -> dict[str, Any]:
    """
    Validate and extract required fields from the Phase 7 risk result input.

    Accepts RiskScoringResult, dict, or object.
    Raises ValueError if invalid.
    """
    if risk_result is None:
        raise ValueError("risk_result cannot be None.")

    if isinstance(risk_result, RiskScoringResult):
        return {
            "risk_score": risk_result.risk_score,
            "risk_level": risk_result.risk_band,
            "risk_multiplier": risk_result.risk_multiplier,
            "estimated_exposure": risk_result.estimated_exposure,
            "recommended_action": risk_result.recommended_action,
        }

    raw_score = _extract_window_field(risk_result, "risk_score")
    if raw_score is None:
        raise ValueError("risk_result must contain 'risk_score'.")
    risk_score = _validate_risk_score(raw_score)

    risk_level = _extract_window_field(risk_result, "risk_band")
    if risk_level is None:
        risk_level = _extract_window_field(risk_result, "risk_level")
    if risk_level is None or risk_level not in VALID_RISK_BANDS:
        raise ValueError(f"risk_result must contain valid 'risk_band' in {VALID_RISK_BANDS}.")

    raw_exposure = _extract_window_field(risk_result, "estimated_exposure")
    if raw_exposure is None:
        raise ValueError("risk_result must contain 'estimated_exposure'.")
    estimated_exposure = _validate_total_amount(raw_exposure)

    recommended_action = _extract_window_field(risk_result, "recommended_action")
    if not recommended_action or not isinstance(recommended_action, str):
        raise ValueError("risk_result must contain non-empty string 'recommended_action'.")

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_multiplier": _extract_window_field(risk_result, "risk_multiplier", 1.0),
        "estimated_exposure": estimated_exposure,
        "recommended_action": recommended_action,
    }


# ---------------------------------------------------------------------------
# Driver Analysis & Quantitative Formatting
# ---------------------------------------------------------------------------

def _extract_baseline_stats(
    baseline_stats: Any, merchant_id: str
) -> Optional[dict[str, dict[str, float]]]:
    """
    Extract per-feature baseline mean and std for the given merchant.
    Accepts:
      - dict mapping feature -> {'mean': float, 'std': float}
      - dict mapping merchant_id -> {feature -> {'mean': float, 'std': float}}
      - BaselineDetector instance with merchant_stats
    """
    if baseline_stats is None:
        return None

    # If it's a BaselineDetector object
    if hasattr(baseline_stats, "merchant_stats"):
        return baseline_stats.merchant_stats.get(merchant_id)

    if isinstance(baseline_stats, dict):
        if merchant_id in baseline_stats and isinstance(baseline_stats[merchant_id], dict):
            return baseline_stats[merchant_id]
        if "transaction_count" in baseline_stats:
            return baseline_stats

    return None


def _format_currency(amount: float) -> str:
    """Format currency deterministically using project convention."""
    return f"₹{amount:,.2f}"


def _format_multiplier(multiplier: float) -> str:
    """Format numerical multiplier strictly as N.Nx (e.g. 4.2x)."""
    return f"{multiplier:.1f}x"


def _format_percentage(pct: float) -> str:
    """Format numerical percentage strictly as N.N% (e.g. 42.0%)."""
    return f"{pct:.1f}%"


def _analyze_drivers(
    window_data: dict[str, Any],
    stats: Optional[dict[str, dict[str, float]]],
) -> tuple[list[str], list[float]]:
    """
    Analyze anomaly drivers by comparing window metrics against baseline statistics.

    Returns:
        tuple of (driver_descriptions, allowed_derived_numbers)
    """
    drivers: list[str] = []
    allowed_numbers: list[float] = []

    count = window_data["transaction_count"]
    total = window_data["total_amount"]
    avg = window_data["avg_transaction_amount"]

    if stats is not None:
        # Feature 1: transaction_count
        count_stat = stats.get("transaction_count")
        if count_stat and "mean" in count_stat and count_stat["mean"] > 0:
            mean_count = count_stat["mean"]
            count_mult = round(count / mean_count, 1)
            count_pct = round(((count - mean_count) / mean_count) * 100.0, 1)
            allowed_numbers.extend([count_mult, count_pct, mean_count])

            if count_mult >= 1.5:
                drivers.append(
                    f"Transaction count ({count}) surged to {_format_multiplier(count_mult)} "
                    f"(+{_format_percentage(count_pct)}) of merchant baseline."
                )
            elif count_mult <= 0.5 and count > 0:
                drivers.append(
                    f"Transaction count ({count}) dropped to {_format_multiplier(count_mult)} "
                    f"({_format_percentage(count_pct)}) of merchant baseline."
                )

        # Feature 2: total_amount
        total_stat = stats.get("total_amount")
        if total_stat and "mean" in total_stat and total_stat["mean"] > 0:
            mean_total = total_stat["mean"]
            total_mult = round(total / mean_total, 1)
            total_pct = round(((total - mean_total) / mean_total) * 100.0, 1)
            allowed_numbers.extend([total_mult, total_pct, mean_total])

            if total_mult >= 1.5:
                drivers.append(
                    f"Total volume ({_format_currency(total)}) surged to {_format_multiplier(total_mult)} "
                    f"(+{_format_percentage(total_pct)}) of merchant baseline."
                )
            elif total_mult <= 0.5 and total > 0:
                drivers.append(
                    f"Total volume ({_format_currency(total)}) dropped to {_format_multiplier(total_mult)} "
                    f"({_format_percentage(total_pct)}) of merchant baseline."
                )

        # Feature 3: avg_transaction_amount
        avg_stat = stats.get("avg_transaction_amount")
        if avg_stat and "mean" in avg_stat and avg_stat["mean"] > 0:
            mean_avg = avg_stat["mean"]
            avg_mult = round(avg / mean_avg, 1)
            avg_pct = round(((avg - mean_avg) / mean_avg) * 100.0, 1)
            allowed_numbers.extend([avg_mult, avg_pct, mean_avg])

            if avg_mult >= 1.5:
                drivers.append(
                    f"Average ticket size ({_format_currency(avg)}) elevated to {_format_multiplier(avg_mult)} "
                    f"(+{_format_percentage(avg_pct)}) of merchant baseline."
                )
            elif avg_mult <= 0.5 and avg > 0:
                drivers.append(
                    f"Average ticket size ({_format_currency(avg)}) dropped to {_format_multiplier(avg_mult)} "
                    f"({_format_percentage(avg_pct)}) of merchant baseline."
                )

    # If no baseline stats were provided or no features exceeded thresholds
    if not drivers:
        drivers.append(f"Observed {count} transactions totaling {_format_currency(total)}.")
        drivers.append(f"Average transaction amount is {_format_currency(avg)}.")

    return drivers, allowed_numbers


# ---------------------------------------------------------------------------
# Deterministic Rule-Based Generator
# ---------------------------------------------------------------------------

def _generate_rule_based_explanation(
    window_data: dict[str, Any],
    risk_data: dict[str, Any],
    stats: Optional[dict[str, dict[str, float]]] = None,
    generated_by: str = "rule_based",
) -> ExplanationResult:
    """
    Generate a 100% deterministic, defense-only explanation.
    """
    window_id = window_data["window_id"]
    merchant_id = window_data["merchant_id"]
    count = window_data["transaction_count"]
    total = window_data["total_amount"]
    avg = window_data["avg_transaction_amount"]

    risk_score = risk_data["risk_score"]
    risk_level = risk_data["risk_level"]
    exposure = risk_data["estimated_exposure"]
    action = risk_data["recommended_action"]

    drivers, _ = _analyze_drivers(window_data, stats)

    # Concise Summary
    if risk_level in ("high", "critical"):
        summary = (
            f"High-severity fraud spike detected for merchant {merchant_id} "
            f"with risk score {risk_score:.1f}/100 ({risk_level.upper()} severity)."
        )
    elif risk_level == "medium":
        summary = (
            f"Moderate risk anomaly detected for merchant {merchant_id} "
            f"with risk score {risk_score:.1f}/100 (MEDIUM severity)."
        )
    else:
        summary = (
            f"Standard activity observed for merchant {merchant_id} "
            f"with risk score {risk_score:.1f}/100 (LOW severity)."
        )

    # Build multi-paragraph raw_text
    driver_lines = "\n".join(f"- {d}" for d in drivers)

    raw_text = (
        f"ANOMALY EXPLANATION REPORT (Merchant: {merchant_id}, Window: {window_id})\n"
        f"------------------------------------------------------------------------\n"
        f"Summary: {summary}\n\n"
        f"Risk Level: {risk_level.upper()} (Risk Score: {risk_score:.1f}/100)\n"
        f"Estimated Financial Exposure: {_format_currency(exposure)}\n"
        f"Recommended Action: {action}\n\n"
        f"Key Anomaly Drivers:\n"
        f"{driver_lines}\n\n"
        f"Window Metrics:\n"
        f"- Transaction Count: {count}\n"
        f"- Total Monetary Volume: {_format_currency(total)}\n"
        f"- Average Transaction Amount: {_format_currency(avg)}\n\n"
        f"Defensive Governance Notice:\n"
        f"All evaluations are defense-only and advisory. Automated blocking or banning is strictly disabled. "
        f"Please proceed with human analyst review as recommended."
    )

    return ExplanationResult(
        window_id=window_id,
        merchant_id=merchant_id,
        summary=summary,
        key_drivers=drivers,
        risk_level=risk_level,
        estimated_exposure=exposure,
        recommended_action=action,
        generated_by=generated_by,
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# Defense-Only Contextual Output Validator
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences and significant clauses."""
    # Split by standard sentence terminators and line breaks
    raw_sentences = re.split(r"[.!?\n\r;]+", text)
    return [s.strip() for s in raw_sentences if s.strip()]


def _classify_sentence_safety(sentence: str) -> str:
    """
    Classify a single sentence as 'SAFE', 'UNSAFE', or 'AMBIGUOUS'.

    Step A: Check for sensitive action terms.
    Step B: If term found, check for governing prohibition/negation cues.
    Step C: If not negated, check for imperative/recommendation structure.
    """
    lower_s = sentence.lower()

    # Step A: Check if any sensitive term exists
    matched_term = None
    for term in SENSITIVE_ACTION_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", lower_s):
            matched_term = term
            break

    if not matched_term:
        return "SAFE"

    # Step B: Check for negation / prohibition cues
    # Look in the clause or within ~6 words before the action term
    for cue in NEGATION_PROHIBITION_CUES:
        if cue in lower_s:
            # Check if cue precedes or governs the term
            cue_pos = lower_s.find(cue)
            term_pos = lower_s.find(matched_term)
            if cue_pos <= term_pos or "required before" in cue or "subject to" in cue or "pending" in cue:
                return "SAFE"

    # Step C: Check for imperative / recommendation cues
    for imp in IMPERATIVE_RECOMMENDATION_CUES:
        if imp in lower_s:
            return "UNSAFE"

    # If it starts with the action verb directly (e.g. "Block the merchant now", "Ban account")
    words = re.findall(r"\b\w+\b", lower_s)
    if words and words[0] in ("block", "ban", "terminate", "suspend", "freeze", "deactivate", "bypass", "circumvent"):
        return "UNSAFE"

    # If an active recommendation phrasing exists
    if any(phrase in lower_s for phrase in ["we recommend", "action is to", "action:", "please", "proceed to"]):
        return "UNSAFE"

    # If a sensitive action term is present without explicit prohibition, fail closed
    return "UNSAFE"


def validate_defense_only_text(text: str) -> bool:
    """
    Validate that the generated text strictly adheres to defense-only standards.

    Returns True if SAFE, False if UNSAFE or AMBIGUOUS (fail-closed).
    """
    if not text or not text.strip():
        return False

    sentences = _split_into_sentences(text)
    if not sentences:
        return False

    for sentence in sentences:
        classification = _classify_sentence_safety(sentence)
        if classification != "SAFE":
            return False

    return True


# ---------------------------------------------------------------------------
# Numerical Grounding Validator
# ---------------------------------------------------------------------------

def _extract_numeric_tokens(text: str) -> list[float]:
    """
    Extract numerical values from free text, handling currency, percentages,
    multipliers, commas, and decimals.
    """
    # Pattern captures:
    # - Currency: ₹95,000, $95,000, Rs. 95000
    # - Multipliers: 4.2x, 10.0x
    # - Percentages: 42.0%, 15%
    # - Standalone numbers: 95000, 42.5, 1,000
    tokens: list[float] = []

    # Replace common words that contain digits so they aren't parsed as numbers
    clean_text = re.sub(r"\bgpt-[345][a-z0-9]*\b", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\bphase\s*\d+\b", "", clean_text, flags=re.IGNORECASE)

    # 1. Multipliers: e.g. 4.2x
    for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)\s*x(?!\w)", clean_text, re.IGNORECASE):
        try:
            tokens.append(float(match.group(1)))
        except ValueError:
            pass

    # 2. Percentages: e.g. 42.0%
    for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)\s*%", clean_text):
        try:
            tokens.append(float(match.group(1)))
        except ValueError:
            pass

    # 3. Currency / Quantities with commas: e.g. ₹95,000.50, 10,000
    normalized = re.sub(r"[₹$€£]|Rs\.?|INR", " ", clean_text)

    # Find all decimal/integer sequences not already captured with trailing x or %
    for match in re.finditer(r"(?<![\w\.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w%x])", normalized, re.IGNORECASE):
        raw_val = match.group(1).replace(",", "")
        try:
            val = float(raw_val)
            tokens.append(val)
        except ValueError:
            pass

    return tokens


def _build_allowed_numbers(
    window_data: dict[str, Any],
    risk_data: dict[str, Any],
    derived_numbers: list[float],
) -> list[float]:
    """
    Build the set of strictly allowed numbers from input facts and approved derived stats.
    """
    allowed: list[float] = [
        float(window_data["transaction_count"]),
        float(window_data["total_amount"]),
        float(window_data["avg_transaction_amount"]),
        float(risk_data["risk_score"]),
        float(risk_data["estimated_exposure"]),
    ]

    # Include window_id if it is integer/float
    wid = window_data["window_id"]
    if isinstance(wid, (int, float)) and not isinstance(wid, bool):
        allowed.append(float(wid))

    allowed.extend(derived_numbers)

    # Also add rounded / integer variants where appropriate
    extras: list[float] = []
    for num in allowed:
        extras.append(round(num, 1))
        extras.append(round(num, 2))
        extras.append(float(int(round(num))))

    return allowed + extras


def validate_numerical_grounding(
    text: str,
    window_data: dict[str, Any],
    risk_data: dict[str, Any],
    derived_numbers: list[float],
) -> bool:
    """
    Verify that every numerical claim in the LLM text corresponds to an allowed value.

    Tolerances:
      - ±0.1 for percentages, multipliers, risk scores, and 1-decimal floats
      - ±1.0 for counts, currency amounts, and integer values against actual data facts
      - Exact match (±0.01) for scale constant 100 (e.g. '/100')
    """
    extracted_numbers = _extract_numeric_tokens(text)
    if not extracted_numbers:
        return True

    allowed_numbers = _build_allowed_numbers(window_data, risk_data, derived_numbers)

    # Scale constants that are only allowed exactly (e.g. /100 or 0)
    exact_scale_constants = [100.0, 0.0]

    for num in extracted_numbers:
        is_grounded = False

        # 1. Check exact scale constants (exact match only)
        for scale in exact_scale_constants:
            if abs(num - scale) <= 0.01:
                is_grounded = True
                break

        if is_grounded:
            continue

        # 2. Check against allowed data values
        for allowed in allowed_numbers:
            diff = abs(num - allowed)
            # Rounding tolerance for multipliers/percentages/scores
            if diff <= 0.1:
                is_grounded = True
                break
            # Tolerance for large monetary figures / integer amounts
            if diff <= 1.0 and abs(allowed) >= 10.0:
                is_grounded = True
                break

        if not is_grounded:
            return False

    return True


# ---------------------------------------------------------------------------
# LLM Integration (OpenAI API with fail-closed safety)
# ---------------------------------------------------------------------------

def _build_llm_user_prompt(
    window_data: dict[str, Any],
    risk_data: dict[str, Any],
    drivers: list[str],
) -> str:
    """
    Construct a strictly structured user prompt.
    Does NOT include arbitrary free-text or merchant-controlled strings.
    """
    merchant_id = window_data["merchant_id"]
    window_id = window_data["window_id"]
    count = window_data["transaction_count"]
    total = window_data["total_amount"]
    avg = window_data["avg_transaction_amount"]

    risk_score = risk_data["risk_score"]
    risk_level = risk_data["risk_level"]
    exposure = risk_data["estimated_exposure"]
    action = risk_data["recommended_action"]

    driver_text = "\n".join(f"- {d}" for d in drivers)

    return (
        f"Please explain this fraud-spike detection window for human analysts:\n\n"
        f"STRUCTURED FACTS:\n"
        f"- Window ID: {window_id}\n"
        f"- Merchant ID: {merchant_id}\n"
        f"- Transaction Count: {count}\n"
        f"- Total Amount: {_format_currency(total)}\n"
        f"- Average Transaction Amount: {_format_currency(avg)}\n"
        f"- Risk Score: {risk_score:.1f}/100\n"
        f"- Risk Level: {risk_level.upper()}\n"
        f"- Estimated Financial Exposure: {_format_currency(exposure)}\n"
        f"- Recommended Action: {action}\n\n"
        f"KEY QUANTITATIVE DRIVERS:\n"
        f"{driver_text}\n\n"
        f"Generate a concise, professional summary and explanation using ONLY these facts."
    )


def _call_openai_api(
    user_prompt: str,
    api_key: str,
    model: str = "gpt-4o",
    timeout: float = OPENAI_REQUEST_TIMEOUT_SECONDS,
) -> str:
    """
    Execute OpenAI chat completion with timeout and credential safety.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("OpenAI package not installed.")

    client = OpenAI(api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FIXED_DEFENSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # Maximum determinism
        max_tokens=500,
    )

    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Empty response received from OpenAI.")

    return response.choices[0].message.content.strip()


def _generate_llm_explanation(
    window_data: dict[str, Any],
    risk_data: dict[str, Any],
    stats: Optional[dict[str, dict[str, float]]] = None,
) -> ExplanationResult:
    """
    Attempt LLM generation with fail-closed validation and fallback.
    """
    # 1. Check API Key
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not api_key or not api_key.strip():
        # Fall back gracefully
        return _generate_rule_based_explanation(
            window_data, risk_data, stats, generated_by="rule_based_fallback"
        )

    # 2. Extract approved drivers and allowed numbers
    drivers, derived_numbers = _analyze_drivers(window_data, stats)
    user_prompt = _build_llm_user_prompt(window_data, risk_data, drivers)

    # 3. Call OpenAI API safely
    try:
        llm_raw_text = _call_openai_api(
            user_prompt=user_prompt,
            api_key=api_key.strip(),
            model=settings.OPENAI_MODEL or "gpt-4o",
            timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
        )
    except Exception:
        # Never log API key or let exception escape
        return _generate_rule_based_explanation(
            window_data, risk_data, stats, generated_by="rule_based_fallback"
        )

    # 4. Defense-Only Validation
    if not validate_defense_only_text(llm_raw_text):
        return _generate_rule_based_explanation(
            window_data, risk_data, stats, generated_by="rule_based_fallback"
        )

    # 5. Numerical Grounding Validation
    if not validate_numerical_grounding(llm_raw_text, window_data, risk_data, derived_numbers):
        return _generate_rule_based_explanation(
            window_data, risk_data, stats, generated_by="rule_based_fallback"
        )

    # 6. Extract concise summary from LLM output (first paragraph or summary line)
    paragraphs = [p.strip() for p in llm_raw_text.split("\n\n") if p.strip()]
    summary = paragraphs[0] if paragraphs else llm_raw_text[:200]

    return ExplanationResult(
        window_id=window_data["window_id"],
        merchant_id=window_data["merchant_id"],
        summary=summary,
        key_drivers=drivers,
        risk_level=risk_data["risk_level"],
        estimated_exposure=risk_data["estimated_exposure"],
        recommended_action=risk_data["recommended_action"],
        generated_by="llm",
        raw_text=llm_raw_text,
    )


# ---------------------------------------------------------------------------
# Main Public Interface
# ---------------------------------------------------------------------------

def generate_explanation(
    window: Any,
    risk_result: Any,
    baseline_stats: Optional[Any] = None,
    use_llm: bool = False,
) -> ExplanationResult:
    """
    Generate an AI Explanation for a DetectionWindow anomaly.

    Parameters:
        window:         DetectionWindow record, dict, or object with required fields:
                        window_id, merchant_id, transaction_count, total_amount, avg_transaction_amount.
        risk_result:    Phase 7 RiskScoringResult or dict with:
                        risk_score, risk_band (or risk_level), estimated_exposure, recommended_action.
        baseline_stats: Optional per-merchant baseline statistics from detector.
        use_llm:        If True, attempts OpenAI generation with fail-closed fallback;
                        If False (default), produces deterministic rule-based explanation.

    Returns:
        ExplanationResult dataclass.
    """
    # 1. Validate inputs
    window_data = _validate_window_input(window)
    risk_data = _validate_risk_input(risk_result, window_data["total_amount"])

    # 2. Extract baseline stats for this specific merchant
    stats = _extract_baseline_stats(baseline_stats, window_data["merchant_id"])

    # 3. Route to LLM or Rule-Based generator
    if use_llm:
        return _generate_llm_explanation(window_data, risk_data, stats)
    else:
        return _generate_rule_based_explanation(
            window_data, risk_data, stats, generated_by="rule_based"
        )
