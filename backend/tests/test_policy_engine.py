"""
AI Risk Manager — Phase 9: Policy Engine Tests
==============================================

Comprehensive test suite for backend/policy_engine.py.

Test classes:
  TestExplanationIndependence   — verifies explanation content never influences decisions
  TestConfigurationThresholds   — tests configuration usage and boundary modifications
  TestIndependentInputsAndOR    — verifies independent evaluation of risk_band and risk_score
  TestConditionAuditability     — asserts accurate reporting in triggered_rules
  TestPolicyPrecedence          — verifies deterministic resolution when multiple policies match
  TestDefaultPolicies           — verifies P0, P1, P2, P3 default policy outcomes
  TestCustomPolicies            — evaluates valid custom rules and rejection of unsafe policies
  TestInputValidation           — verifies input contract and rejection of bad data
  TestDefenseOnlyGuarantees     — ensures strictly non-destructive operations
  TestOutputContract            — validates PolicyDecision structure and frozen immutability
"""

from unittest.mock import patch
import pytest

from config import settings
from explanation_engine import ExplanationResult
from policy_engine import (
    PolicyDecision,
    PolicyRule,
    evaluate_policy,
    POL_CRITICAL_SURGE,
    POL_HIGH_EXPOSURE,
    POL_MEDIUM_ANOMALY,
    POL_ROUTINE_MONITOR,
    ACTION_URGENT_TRIAGE,
    ACTION_PRIORITY_ESCALATION,
    ACTION_ANALYST_QUEUE,
    ACTION_STANDARD_LOG,
    _validate_defense_only_action,
)
from risk_scorer import RiskScoringResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_window():
    return {
        "window_id": 101,
        "merchant_id": "merchant_001",
        "transaction_count": 42,
        "total_amount": 95000.0,
        "avg_transaction_amount": 2261.9,
    }


@pytest.fixture
def sample_risk_result():
    return RiskScoringResult(
        risk_score=75.0,
        risk_band="high",
        risk_multiplier=0.5,
        estimated_exposure=47500.0,
        recommended_action="Escalate for priority review",
    )


# ===========================================================================
# TestExplanationIndependence
# ===========================================================================

class TestExplanationIndependence:
    """Mandatory test: explanation_result MUST NOT influence any decision field."""

    def test_explanation_independence(self, sample_window, sample_risk_result):
        # 1. No explanation
        dec_none = evaluate_policy(sample_window, sample_risk_result, explanation_result=None)

        # 2. Rule-based explanation
        exp_rule_based = ExplanationResult(
            window_id=101,
            merchant_id="merchant_001",
            summary="High risk detected by rule.",
            key_drivers=["Surge 4.2x"],
            risk_level="high",
            estimated_exposure=47500.0,
            recommended_action="Escalate for priority review",
            generated_by="rule_based",
            raw_text="Detailed rule report...",
        )
        dec_rule_based = evaluate_policy(sample_window, sample_risk_result, explanation_result=exp_rule_based)

        # 3. LLM explanation
        exp_llm = ExplanationResult(
            window_id=101,
            merchant_id="merchant_001",
            summary="AI detected critical anomaly pattern.",
            key_drivers=["Volumetric surge"],
            risk_level="high",
            estimated_exposure=47500.0,
            recommended_action="Escalate for priority review",
            generated_by="llm",
            raw_text="Detailed LLM narrative...",
        )
        dec_llm = evaluate_policy(sample_window, sample_risk_result, explanation_result=exp_llm)

        # 4. Fallback explanation with drastically different text
        exp_fallback = ExplanationResult(
            window_id=101,
            merchant_id="merchant_001",
            summary="Fallback triggered due to timeout.",
            key_drivers=["Driver A", "Driver B"],
            risk_level="high",
            estimated_exposure=47500.0,
            recommended_action="Escalate for priority review",
            generated_by="rule_based_fallback",
            raw_text="Completely different fallback text...",
        )
        dec_fallback = evaluate_policy(sample_window, sample_risk_result, explanation_result=exp_fallback)

        # Assert all resulting decisions have identical decision attributes
        for dec in [dec_rule_based, dec_llm, dec_fallback]:
            assert dec.policy_id == dec_none.policy_id
            assert dec.action_type == dec_none.action_type
            assert dec.priority == dec_none.priority
            assert dec.review_sla_hours == dec_none.review_sla_hours
            assert dec.require_dual_review == dec_none.require_dual_review
            assert dec.routing_tags == dec_none.routing_tags
            assert dec.triggered_rules == dec_none.triggered_rules


# ===========================================================================
# TestConfigurationThresholds
# ===========================================================================

class TestConfigurationThresholds:
    """Test policy engine behavioral dependence on configuration constants."""

    def test_default_constants_exist_in_config(self):
        assert hasattr(settings, "POLICY_CRITICAL_SCORE_THRESHOLD")
        assert hasattr(settings, "POLICY_HIGH_EXPOSURE_THRESHOLD")
        assert hasattr(settings, "POLICY_MIN_TRANSACTION_COUNT_THRESHOLD")
        assert settings.POLICY_CRITICAL_SCORE_THRESHOLD == 80.0
        assert settings.POLICY_HIGH_EXPOSURE_THRESHOLD == 50000.0
        assert settings.POLICY_MIN_TRANSACTION_COUNT_THRESHOLD == 40

    def test_exposure_threshold_configuration_change(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 100000.0,
            "avg_transaction_amount": 10000.0,
        }
        risk = {
            "risk_score": 25.0,
            "risk_band": "low",
            "estimated_exposure": 75000.0,  # Above default 50k threshold
        }

        # 1. Under default threshold (50000), 75000 triggers POL_HIGH_EXPOSURE
        dec_default = evaluate_policy(window, risk)
        assert dec_default.policy_id == POL_HIGH_EXPOSURE

        # 2. Dynamically modify threshold to 100000
        with patch.object(settings, "POLICY_HIGH_EXPOSURE_THRESHOLD", 100000.0):
            dec_modified = evaluate_policy(window, risk)
            # 75000 is now below 100000, so it falls through to POL_ROUTINE_MONITOR
            assert dec_modified.policy_id == POL_ROUTINE_MONITOR

        # 3. Confirm restoration
        dec_restored = evaluate_policy(window, risk)
        assert dec_restored.policy_id == POL_HIGH_EXPOSURE


# ===========================================================================
# TestIndependentInputsAndOR
# ===========================================================================

class TestIndependentInputsAndOR:
    """Verify independent OR logic and acceptance of intentionally inconsistent inputs."""

    def test_inconsistent_low_band_high_score(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 1000.0,
            "avg_transaction_amount": 100.0,
        }
        # Inconsistent: risk_band="low" but risk_score=95.0
        risk = {
            "risk_score": 95.0,
            "risk_band": "low",
            "estimated_exposure": 100.0,
        }
        decision = evaluate_policy(window, risk)
        assert decision.policy_id == POL_CRITICAL_SURGE
        assert decision.priority == "P0"
        assert decision.require_dual_review is True

    def test_inconsistent_medium_band_critical_score(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 1000.0,
            "avg_transaction_amount": 100.0,
        }
        risk = {
            "risk_score": 85.0,
            "risk_band": "medium",
            "estimated_exposure": 100.0,
        }
        decision = evaluate_policy(window, risk)
        assert decision.policy_id == POL_CRITICAL_SURGE

    def test_inconsistent_critical_band_low_score(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 1000.0,
            "avg_transaction_amount": 100.0,
        }
        # Inconsistent: risk_band="critical" but risk_score=50.0
        risk = {
            "risk_score": 50.0,
            "risk_band": "critical",
            "estimated_exposure": 100.0,
        }
        decision = evaluate_policy(window, risk)
        assert decision.policy_id == POL_CRITICAL_SURGE


# ===========================================================================
# TestConditionAuditability
# ===========================================================================

class TestConditionAuditability:
    """Verify that triggered_rules reflects the exact matching condition."""

    def test_audit_only_reflects_matching_condition(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 1000.0,
            "avg_transaction_amount": 100.0,
        }
        # Low band + 95.0 score
        risk = {
            "risk_score": 95.0,
            "risk_band": "low",
            "estimated_exposure": 100.0,
        }
        decision = evaluate_policy(window, risk)
        # Must contain the score rule
        assert any("risk_score" in r and "POLICY_CRITICAL_SCORE_THRESHOLD" in r for r in decision.triggered_rules)
        # Must NOT claim risk_band == 'critical'
        assert not any("risk_band == 'critical'" in r for r in decision.triggered_rules)

    def test_audit_both_conditions_when_both_match(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 1000.0,
            "avg_transaction_amount": 100.0,
        }
        risk = {
            "risk_score": 95.0,
            "risk_band": "critical",
            "estimated_exposure": 100.0,
        }
        decision = evaluate_policy(window, risk)
        assert "risk_band == 'critical'" in decision.triggered_rules
        assert any("risk_score" in r for r in decision.triggered_rules)


# ===========================================================================
# TestPolicyPrecedence
# ===========================================================================

class TestPolicyPrecedence:
    """Verify deterministic resolution when multiple policies match."""

    def test_critical_precedence_over_high_exposure(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 200000.0,
            "avg_transaction_amount": 20000.0,
        }
        # Both critical score (95) and high exposure (100,000) match
        risk = {
            "risk_score": 95.0,
            "risk_band": "low",
            "estimated_exposure": 100000.0,
        }
        decision = evaluate_policy(window, risk)
        assert decision.policy_id == POL_CRITICAL_SURGE
        assert decision.priority == "P0"


# ===========================================================================
# TestDefaultPolicies
# ===========================================================================

class TestDefaultPolicies:
    """Verify default policy outcomes across P0, P1, P2, P3."""

    def test_p0_critical_surge(self, sample_window):
        risk = {"risk_score": 90.0, "risk_band": "critical", "estimated_exposure": 5000.0}
        dec = evaluate_policy(sample_window, risk)
        assert dec.policy_id == POL_CRITICAL_SURGE
        assert dec.action_type == ACTION_URGENT_TRIAGE
        assert dec.priority == "P0"
        assert dec.review_sla_hours == 1.0
        assert dec.require_dual_review is True

    def test_p1_high_exposure(self, sample_window):
        risk = {"risk_score": 70.0, "risk_band": "high", "estimated_exposure": 4000.0}
        dec = evaluate_policy(sample_window, risk)
        assert dec.policy_id == POL_HIGH_EXPOSURE
        assert dec.action_type == ACTION_PRIORITY_ESCALATION
        assert dec.priority == "P1"
        assert dec.review_sla_hours == 4.0
        assert dec.require_dual_review is False

    def test_p2_medium_anomaly(self):
        window = {"window_id": 1, "merchant_id": "m_1", "transaction_count": 5, "total_amount": 1000.0, "avg_transaction_amount": 200.0}
        risk = {"risk_score": 50.0, "risk_band": "medium", "estimated_exposure": 250.0}
        dec = evaluate_policy(window, risk)
        assert dec.policy_id == POL_MEDIUM_ANOMALY
        assert dec.action_type == ACTION_ANALYST_QUEUE
        assert dec.priority == "P2"
        assert dec.review_sla_hours == 24.0
        assert dec.require_dual_review is False

    def test_removed_compound_condition_does_not_promote_to_medium(self):
        """
        Prove that transaction_count=45 and risk_score=35 with risk_band='low'
        does NOT trigger POL_MEDIUM_ANOMALY and remains POL_ROUTINE_MONITOR.
        """
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 45,
            "total_amount": 1000.0,
            "avg_transaction_amount": 22.2,
        }
        risk = {
            "risk_score": 35.0,
            "risk_band": "low",
            "estimated_exposure": 100.0,
        }
        dec = evaluate_policy(window, risk)
        assert dec.policy_id == POL_ROUTINE_MONITOR
        assert dec.action_type == ACTION_STANDARD_LOG
        assert dec.priority == "P3"
        assert dec.review_sla_hours == 72.0
        assert dec.require_dual_review is False

    def test_p3_routine_monitor(self):
        window = {"window_id": 1, "merchant_id": "m_1", "transaction_count": 5, "total_amount": 1000.0, "avg_transaction_amount": 200.0}
        risk = {"risk_score": 10.0, "risk_band": "low", "estimated_exposure": 100.0}
        dec = evaluate_policy(window, risk)
        assert dec.policy_id == POL_ROUTINE_MONITOR
        assert dec.action_type == ACTION_STANDARD_LOG
        assert dec.priority == "P3"
        assert dec.review_sla_hours == 72.0
        assert dec.require_dual_review is False


# ===========================================================================
# TestCustomPolicies
# ===========================================================================

class TestCustomPolicies:
    """Test custom policy extension and defense-only rule rejection."""

    def test_safe_custom_policy_evaluated_first(self, sample_window, sample_risk_result):
        custom_rule = PolicyRule(
            policy_id="POL_SPECIAL_MERCHANT_VIP",
            action_type="SPECIAL_ANALYST_REVIEW",
            priority="P1",
            review_sla_hours=2.0,
            require_dual_review=True,
            routing_tags=["vip_merchant"],
            condition=lambda w, r: (w["merchant_id"] == "merchant_001", ["matched_vip_merchant_id"]),
        )
        decision = evaluate_policy(sample_window, sample_risk_result, custom_policies=[custom_rule])
        assert decision.policy_id == "POL_SPECIAL_MERCHANT_VIP"
        assert decision.action_type == "SPECIAL_ANALYST_REVIEW"
        assert decision.review_sla_hours == 2.0

    def test_unsafe_custom_policy_rejected(self):
        with pytest.raises(ValueError, match="Defense-only violation"):
            PolicyRule(
                policy_id="POL_AUTO_BAN",
                action_type="AUTO_BLOCK_ACCOUNT",
                priority="P0",
                review_sla_hours=0.0,
                require_dual_review=False,
                routing_tags=["auto_ban"],
                condition=lambda w, r: (True, ["always"]),
            )


# ===========================================================================
# TestInputValidation
# ===========================================================================

class TestInputValidation:
    """Test input validation for window and risk_result."""

    def test_none_window_rejected(self, sample_risk_result):
        with pytest.raises(ValueError, match="window cannot be None"):
            evaluate_policy(None, sample_risk_result)

    def test_none_risk_result_rejected(self, sample_window):
        with pytest.raises(ValueError, match="risk_result cannot be None"):
            evaluate_policy(sample_window, None)

    def test_missing_merchant_id_rejected(self, sample_risk_result):
        bad_window = {"window_id": 1, "merchant_id": "", "transaction_count": 10, "total_amount": 100.0, "avg_transaction_amount": 10.0}
        with pytest.raises(ValueError, match="merchant_id"):
            evaluate_policy(bad_window, sample_risk_result)

    def test_invalid_risk_score_rejected(self, sample_window):
        bad_risk = {"risk_score": 150.0, "risk_band": "high", "estimated_exposure": 100.0}
        with pytest.raises(ValueError, match="<= 100"):
            evaluate_policy(sample_window, bad_risk)


# ===========================================================================
# TestDefenseOnlyGuarantees
# ===========================================================================

class TestDefenseOnlyGuarantees:
    """Verify defense-only guarantees."""

    def test_no_destructive_terms_in_action_types(self):
        for action in [ACTION_URGENT_TRIAGE, ACTION_PRIORITY_ESCALATION, ACTION_ANALYST_QUEUE, ACTION_STANDARD_LOG]:
            _validate_defense_only_action(action)


# ===========================================================================
# TestOutputContract
# ===========================================================================

class TestOutputContract:
    """Verify PolicyDecision structure and frozen dataclass immutability."""

    def test_policy_decision_is_frozen(self, sample_window, sample_risk_result):
        dec = evaluate_policy(sample_window, sample_risk_result)
        with pytest.raises(Exception):
            dec.priority = "P0"
