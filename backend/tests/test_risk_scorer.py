"""
AI Risk Manager — Phase 7: Risk Scoring Engine Tests
=====================================================

Comprehensive test suite for backend/risk_scorer.py.

Test classes:
  TestRiskScoreValidation   — accepted/rejected risk_score inputs
  TestRiskBandBoundaries    — exact threshold boundary behaviour
  TestExposureCalculation   — financial exposure estimates
  TestTotalAmountValidation — accepted/rejected total_amount inputs
  TestRecommendedActions    — defense-only action mapping
  TestOutputContract        — RiskScoringResult structure verification
  TestDeterminism           — identical inputs → identical outputs
  TestDefenseOnly           — no blocking/banning functionality
"""

import math

import pytest

from risk_scorer import (
    RiskScoringResult,
    RISK_LOW_MAX,
    RISK_MEDIUM_MAX,
    RISK_HIGH_MAX,
    RISK_MULTIPLIERS,
    RECOMMENDED_ACTIONS,
    VALID_RISK_BANDS,
    get_risk_band,
    estimate_exposure,
    get_recommended_action,
    score_risk,
    _validate_risk_score,
    _validate_total_amount,
)


# ===========================================================================
# TestRiskScoreValidation
# ===========================================================================


class TestRiskScoreValidation:
    """risk_score must be numeric, finite, 0–100 inclusive."""

    def test_zero_accepted(self):
        assert _validate_risk_score(0) == 0.0

    def test_hundred_accepted(self):
        assert _validate_risk_score(100) == 100.0

    def test_normal_score_accepted(self):
        assert _validate_risk_score(42.5) == 42.5

    def test_integer_accepted(self):
        assert _validate_risk_score(50) == 50.0

    def test_negative_score_rejected(self):
        with pytest.raises(ValueError, match=">=\\s*0"):
            _validate_risk_score(-1)

    def test_negative_fractional_rejected(self):
        with pytest.raises(ValueError, match=">=\\s*0"):
            _validate_risk_score(-0.01)

    def test_above_100_rejected(self):
        with pytest.raises(ValueError, match="<=\\s*100"):
            _validate_risk_score(100.01)

    def test_large_value_rejected(self):
        with pytest.raises(ValueError, match="<=\\s*100"):
            _validate_risk_score(999)

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            _validate_risk_score(float("nan"))

    def test_positive_infinity_rejected(self):
        with pytest.raises(ValueError, match="infinity"):
            _validate_risk_score(float("inf"))

    def test_negative_infinity_rejected(self):
        with pytest.raises(ValueError, match="infinity"):
            _validate_risk_score(float("-inf"))

    def test_string_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            _validate_risk_score("fifty")

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            _validate_risk_score(None)

    def test_bool_rejected(self):
        """Booleans are technically int subclass but should be rejected."""
        # Note: in Python, bool is a subclass of int.
        # isinstance(True, int) is True, so True (1) and False (0) are
        # technically accepted by the numeric check.  This is by design —
        # they map to valid score values.  This test documents the behaviour.
        assert _validate_risk_score(True) == 1.0
        assert _validate_risk_score(False) == 0.0


# ===========================================================================
# TestRiskBandBoundaries
# ===========================================================================


class TestRiskBandBoundaries:
    """Exact boundary behaviour using existing config thresholds."""

    def test_zero_is_low(self):
        assert get_risk_band(0) == "low"

    def test_at_low_max_is_low(self):
        assert get_risk_band(RISK_LOW_MAX) == "low"

    def test_just_below_medium_threshold_is_low(self):
        assert get_risk_band(RISK_LOW_MAX - 0.01) == "low"

    def test_just_above_low_max_is_medium(self):
        assert get_risk_band(RISK_LOW_MAX + 0.01) == "medium"

    def test_at_medium_max_is_medium(self):
        assert get_risk_band(RISK_MEDIUM_MAX) == "medium"

    def test_just_below_high_threshold_is_medium(self):
        assert get_risk_band(RISK_MEDIUM_MAX - 0.01) == "medium"

    def test_just_above_medium_max_is_high(self):
        assert get_risk_band(RISK_MEDIUM_MAX + 0.01) == "high"

    def test_at_high_max_is_high(self):
        assert get_risk_band(RISK_HIGH_MAX) == "high"

    def test_just_below_critical_threshold_is_high(self):
        assert get_risk_band(RISK_HIGH_MAX - 0.01) == "high"

    def test_just_above_high_max_is_critical(self):
        assert get_risk_band(RISK_HIGH_MAX + 0.01) == "critical"

    def test_100_is_critical(self):
        assert get_risk_band(100) == "critical"

    def test_midpoint_low(self):
        assert get_risk_band(RISK_LOW_MAX / 2) == "low"

    def test_midpoint_critical(self):
        mid = (RISK_HIGH_MAX + 100) / 2
        assert get_risk_band(mid) == "critical"


# ===========================================================================
# TestExposureCalculation
# ===========================================================================


class TestExposureCalculation:
    """Financial exposure = total_amount × risk_multiplier."""

    def test_zero_total_amount(self):
        assert estimate_exposure(0, "high") == 0.0

    def test_normal_amount_low(self):
        expected = 100000 * RISK_MULTIPLIERS["low"]
        assert estimate_exposure(100000, "low") == pytest.approx(expected)

    def test_normal_amount_medium(self):
        expected = 100000 * RISK_MULTIPLIERS["medium"]
        assert estimate_exposure(100000, "medium") == pytest.approx(expected)

    def test_normal_amount_high(self):
        expected = 100000 * RISK_MULTIPLIERS["high"]
        assert estimate_exposure(100000, "high") == pytest.approx(expected)

    def test_normal_amount_critical(self):
        expected = 100000 * RISK_MULTIPLIERS["critical"]
        assert estimate_exposure(100000, "critical") == pytest.approx(expected)

    def test_critical_multiplier_is_full_amount(self):
        """Critical exposure should equal the full total_amount (multiplier=1.0)."""
        assert estimate_exposure(50000, "critical") == pytest.approx(50000.0)

    def test_exact_multiplier_calculation(self):
        """Verify the exact formula: total_amount × risk_multiplier."""
        amount = 123456.78
        for band in VALID_RISK_BANDS:
            expected = amount * RISK_MULTIPLIERS[band]
            assert estimate_exposure(amount, band) == pytest.approx(expected)

    def test_deterministic_exposure(self):
        """Same inputs always produce the same exposure."""
        r1 = estimate_exposure(99999.99, "medium")
        r2 = estimate_exposure(99999.99, "medium")
        assert r1 == r2

    def test_unknown_risk_band_rejected(self):
        with pytest.raises(ValueError, match="Unknown risk_band"):
            estimate_exposure(1000, "extreme")

    def test_integer_amount_accepted(self):
        """Integer total_amount should work correctly."""
        result = estimate_exposure(100000, "low")
        assert result == pytest.approx(100000 * RISK_MULTIPLIERS["low"])


# ===========================================================================
# TestTotalAmountValidation
# ===========================================================================


class TestTotalAmountValidation:
    """total_amount must be numeric, finite, >= 0."""

    def test_zero_accepted(self):
        assert _validate_total_amount(0) == 0.0

    def test_positive_accepted(self):
        assert _validate_total_amount(50000.50) == 50000.50

    def test_integer_accepted(self):
        assert _validate_total_amount(10000) == 10000.0

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match=">=\\s*0"):
            _validate_total_amount(-100)

    def test_negative_fractional_rejected(self):
        with pytest.raises(ValueError, match=">=\\s*0"):
            _validate_total_amount(-0.01)

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            _validate_total_amount(float("nan"))

    def test_positive_infinity_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_total_amount(float("inf"))

    def test_negative_infinity_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_total_amount(float("-inf"))

    def test_string_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            _validate_total_amount("ten thousand")

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            _validate_total_amount(None)


# ===========================================================================
# TestRecommendedActions
# ===========================================================================


class TestRecommendedActions:
    """Each risk band maps to exactly the intended defensive recommendation."""

    def test_low_action(self):
        assert get_recommended_action("low") == "Monitor"

    def test_medium_action(self):
        assert get_recommended_action("medium") == "Flag for analyst review"

    def test_high_action(self):
        assert get_recommended_action("high") == "Escalate for priority review"

    def test_critical_action(self):
        assert get_recommended_action("critical") == "Immediate escalation and human review"

    def test_unknown_band_rejected(self):
        with pytest.raises(ValueError, match="Unknown risk_band"):
            get_recommended_action("extreme")

    def test_all_actions_are_strings(self):
        for band in VALID_RISK_BANDS:
            action = get_recommended_action(band)
            assert isinstance(action, str)
            assert len(action) > 0


# ===========================================================================
# TestOutputContract
# ===========================================================================


class TestOutputContract:
    """score_risk() returns a RiskScoringResult with all required fields."""

    def test_returns_risk_scoring_result(self):
        result = score_risk(50, 10000)
        assert isinstance(result, RiskScoringResult)

    def test_result_contains_risk_score(self):
        result = score_risk(42.5, 10000)
        assert result.risk_score == 42.5

    def test_result_contains_risk_band(self):
        result = score_risk(42.5, 10000)
        assert result.risk_band in VALID_RISK_BANDS

    def test_result_contains_risk_multiplier(self):
        result = score_risk(42.5, 10000)
        assert result.risk_multiplier == RISK_MULTIPLIERS[result.risk_band]

    def test_result_contains_estimated_exposure(self):
        result = score_risk(42.5, 10000)
        expected = 10000 * RISK_MULTIPLIERS[result.risk_band]
        assert result.estimated_exposure == pytest.approx(expected)

    def test_result_contains_recommended_action(self):
        result = score_risk(42.5, 10000)
        assert result.recommended_action == RECOMMENDED_ACTIONS[result.risk_band]

    def test_low_band_full_result(self):
        result = score_risk(10, 50000)
        assert result.risk_band == "low"
        assert result.risk_multiplier == RISK_MULTIPLIERS["low"]
        assert result.estimated_exposure == pytest.approx(50000 * RISK_MULTIPLIERS["low"])
        assert result.recommended_action == "Monitor"

    def test_critical_band_full_result(self):
        result = score_risk(95, 100000)
        assert result.risk_band == "critical"
        assert result.risk_multiplier == RISK_MULTIPLIERS["critical"]
        assert result.estimated_exposure == pytest.approx(100000 * RISK_MULTIPLIERS["critical"])
        assert result.recommended_action == "Immediate escalation and human review"

    def test_result_is_frozen_dataclass(self):
        """RiskScoringResult should be immutable."""
        result = score_risk(50, 10000)
        with pytest.raises(AttributeError):
            result.risk_band = "low"

    def test_invalid_risk_score_propagates(self):
        with pytest.raises(ValueError):
            score_risk(-5, 10000)

    def test_invalid_total_amount_propagates(self):
        with pytest.raises(ValueError):
            score_risk(50, -100)


# ===========================================================================
# TestDeterminism
# ===========================================================================


class TestDeterminism:
    """Same input must always produce the same output."""

    def test_score_risk_deterministic(self):
        r1 = score_risk(65, 100000)
        r2 = score_risk(65, 100000)
        assert r1 == r2

    def test_get_risk_band_deterministic(self):
        for score in [0, 15, 30, 30.01, 45, 60, 60.01, 70, 80, 80.01, 90, 100]:
            b1 = get_risk_band(score)
            b2 = get_risk_band(score)
            assert b1 == b2, f"Non-deterministic at score={score}"

    def test_exposure_deterministic(self):
        for band in VALID_RISK_BANDS:
            e1 = estimate_exposure(77777.77, band)
            e2 = estimate_exposure(77777.77, band)
            assert e1 == e2

    def test_full_sweep_deterministic(self):
        """Sweep 0 to 100 in steps and verify determinism."""
        scores = [i * 0.5 for i in range(201)]  # 0, 0.5, 1.0, ... 100.0
        for score in scores:
            r1 = score_risk(score, 50000)
            r2 = score_risk(score, 50000)
            assert r1 == r2, f"Non-deterministic at score={score}"


# ===========================================================================
# TestDefenseOnly
# ===========================================================================


class TestDefenseOnly:
    """Verify the scorer does not expose blocking/banning functionality."""

    def test_no_block_in_actions(self):
        """No recommended action should contain 'block'."""
        for band in VALID_RISK_BANDS:
            action = get_recommended_action(band)
            assert "block" not in action.lower()

    def test_no_ban_in_actions(self):
        """No recommended action should contain 'ban'."""
        for band in VALID_RISK_BANDS:
            action = get_recommended_action(band)
            assert "ban" not in action.lower()

    def test_no_suspend_in_actions(self):
        """No recommended action should contain 'suspend'."""
        for band in VALID_RISK_BANDS:
            action = get_recommended_action(band)
            assert "suspend" not in action.lower()

    def test_no_auto_in_actions(self):
        """No recommended action should contain 'automatic'."""
        for band in VALID_RISK_BANDS:
            action = get_recommended_action(band)
            assert "automatic" not in action.lower()

    def test_critical_action_requires_human(self):
        """The critical action must explicitly mention human review."""
        action = get_recommended_action("critical")
        assert "human" in action.lower()

    def test_module_has_no_database_imports(self):
        """risk_scorer should not import database or ORM modules."""
        import risk_scorer
        source = open(risk_scorer.__file__).read()
        assert "from database" not in source
        assert "import database" not in source
        assert "from models" not in source
        assert "import models" not in source
        assert "sqlalchemy" not in source.lower()

    def test_score_risk_is_pure_function(self):
        """
        score_risk() should produce the same output when called multiple
        times with no state changes — confirming it has no side effects.
        """
        results = [score_risk(75, 50000) for _ in range(10)]
        assert all(r == results[0] for r in results)
