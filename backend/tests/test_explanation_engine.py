"""
AI Risk Manager — Phase 8: AI Explanation Engine Tests
======================================================

Comprehensive test suite for backend/explanation_engine.py.

Test classes:
  TestInputValidation          — valid/invalid window and risk inputs
  TestDeterminism              — identical inputs → identical outputs
  TestRiskBandsAndActions      — output across all risk severity levels
  TestDriverAnalysis           — frequency, volume, ticket size surge drivers
  TestFormattingContract       — strict N.Nx multiplier and N.N% percentage formatting
  TestDefenseOnlyValidator     — contextual safe vs. unsafe sentence classification
  TestNumericalGrounding       — accepted grounded numbers vs. rejected hallucinations
  TestLLMIntegrationAndFallback — mock OpenAI flows (success, missing key, timeout, error, fallback)
  TestOutputContract           — ExplanationResult structure and immutability
  TestHoldoutIsolation         — verifies no final-holdout file usage
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from explanation_engine import (
    ExplanationResult,
    generate_explanation,
    validate_defense_only_text,
    validate_numerical_grounding,
    _classify_sentence_safety,
    _extract_numeric_tokens,
    _validate_window_input,
    _validate_risk_input,
    _analyze_drivers,
    _format_multiplier,
    _format_percentage,
    VALID_GENERATED_BY,
)
from risk_scorer import RiskScoringResult, score_risk


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


@pytest.fixture
def sample_baseline_stats():
    return {
        "merchant_001": {
            "transaction_count": {"mean": 10.0, "std": 2.0},
            "total_amount": {"mean": 25000.0, "std": 5000.0},
            "avg_transaction_amount": {"mean": 2500.0, "std": 300.0},
        }
    }


# ===========================================================================
# TestInputValidation
# ===========================================================================

class TestInputValidation:
    """Validate window and risk_result inputs with strict rejection of bad data."""

    def test_none_window_rejected(self, sample_risk_result):
        with pytest.raises(ValueError, match="window cannot be None"):
            generate_explanation(None, sample_risk_result)

    def test_none_risk_result_rejected(self, sample_window):
        with pytest.raises(ValueError, match="risk_result cannot be None"):
            generate_explanation(sample_window, None)

    def test_missing_window_id_rejected(self, sample_risk_result):
        bad_window = {
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 100.0,
            "avg_transaction_amount": 10.0,
        }
        with pytest.raises(ValueError, match="window_id"):
            generate_explanation(bad_window, sample_risk_result)

    def test_missing_merchant_id_rejected(self, sample_risk_result):
        bad_window = {
            "window_id": 1,
            "merchant_id": "",
            "transaction_count": 10,
            "total_amount": 100.0,
            "avg_transaction_amount": 10.0,
        }
        with pytest.raises(ValueError, match="merchant_id"):
            generate_explanation(bad_window, sample_risk_result)

    def test_negative_transaction_count_rejected(self, sample_risk_result):
        bad_window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": -5,
            "total_amount": 100.0,
            "avg_transaction_amount": 10.0,
        }
        with pytest.raises(ValueError, match="non-negative"):
            generate_explanation(bad_window, sample_risk_result)

    def test_nan_total_amount_rejected(self, sample_risk_result):
        bad_window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": float("nan"),
            "avg_transaction_amount": 10.0,
        }
        with pytest.raises(ValueError, match="NaN"):
            generate_explanation(bad_window, sample_risk_result)

    def test_inf_total_amount_rejected(self, sample_risk_result):
        bad_window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": float("inf"),
            "avg_transaction_amount": 10.0,
        }
        with pytest.raises(ValueError, match="finite"):
            generate_explanation(bad_window, sample_risk_result)

    def test_invalid_risk_band_rejected(self, sample_window):
        bad_risk = {
            "risk_score": 50.0,
            "risk_band": "catastrophic",
            "estimated_exposure": 500.0,
            "recommended_action": "Flag for analyst review",
        }
        with pytest.raises(ValueError, match="risk_band"):
            generate_explanation(sample_window, bad_risk)

    def test_invalid_risk_score_rejected(self, sample_window):
        bad_risk = {
            "risk_score": 150.0,
            "risk_band": "high",
            "estimated_exposure": 500.0,
            "recommended_action": "Flag for analyst review",
        }
        with pytest.raises(ValueError, match="<= 100"):
            generate_explanation(sample_window, bad_risk)


# ===========================================================================
# TestDeterminism
# ===========================================================================

class TestDeterminism:
    """Rule-based explanation must produce 100% identical outputs for identical inputs."""

    def test_identical_inputs_produce_identical_outputs(self, sample_window, sample_risk_result, sample_baseline_stats):
        res1 = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=False)
        res2 = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=False)

        assert res1.summary == res2.summary
        assert res1.key_drivers == res2.key_drivers
        assert res1.risk_level == res2.risk_level
        assert res1.estimated_exposure == res2.estimated_exposure
        assert res1.recommended_action == res2.recommended_action
        assert res1.raw_text == res2.raw_text
        assert res1.generated_by == "rule_based"


# ===========================================================================
# TestRiskBandsAndActions
# ===========================================================================

class TestRiskBandsAndActions:
    """Test explanation generation across all risk bands."""

    @pytest.mark.parametrize(
        "score,band,expected_action,keyword",
        [
            (15.0, "low", "Monitor", "Standard activity"),
            (45.0, "medium", "Flag for analyst review", "Moderate risk"),
            (75.0, "high", "Escalate for priority review", "High-severity"),
            (95.0, "critical", "Immediate escalation and human review", "High-severity"),
        ],
    )
    def test_risk_band_outputs(self, sample_window, score, band, expected_action, keyword):
        risk_res = score_risk(score, sample_window["total_amount"])
        result = generate_explanation(sample_window, risk_res, use_llm=False)

        assert result.risk_level == band
        assert result.recommended_action == expected_action
        assert keyword in result.summary
        assert band.upper() in result.raw_text


# ===========================================================================
# TestDriverAnalysis
# ===========================================================================

class TestDriverAnalysis:
    """Test quantitative driver extraction for surges and shifts."""

    def test_frequency_surge_driver(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 42,
            "total_amount": 10000.0,
            "avg_transaction_amount": 238.1,
        }
        stats = {
            "m_1": {
                "transaction_count": {"mean": 10.0, "std": 2.0},
                "total_amount": {"mean": 10000.0, "std": 500.0},
                "avg_transaction_amount": {"mean": 1000.0, "std": 50.0},
            }
        }
        drivers, allowed = _analyze_drivers(window, stats["m_1"])
        assert any("4.2x" in d and "surged" in d for d in drivers)

    def test_volume_surge_driver(self):
        window = {
            "window_id": 1,
            "merchant_id": "m_1",
            "transaction_count": 10,
            "total_amount": 38000.0,
            "avg_transaction_amount": 3800.0,
        }
        stats = {
            "m_1": {
                "transaction_count": {"mean": 10.0, "std": 2.0},
                "total_amount": {"mean": 10000.0, "std": 500.0},
                "avg_transaction_amount": {"mean": 1000.0, "std": 50.0},
            }
        }
        drivers, allowed = _analyze_drivers(window, stats["m_1"])
        assert any("3.8x" in d and "Total volume" in d for d in drivers)

    def test_no_baseline_stats_fallback(self, sample_window, sample_risk_result):
        result = generate_explanation(sample_window, sample_risk_result, baseline_stats=None, use_llm=False)
        assert len(result.key_drivers) >= 1
        assert "42 transactions" in result.key_drivers[0]


# ===========================================================================
# TestFormattingContract
# ===========================================================================

class TestFormattingContract:
    """Strict test of N.Nx multiplier and N.N% percentage formatting."""

    def test_multiplier_formatting(self):
        assert _format_multiplier(4.2) == "4.2x"
        assert _format_multiplier(10.0) == "10.0x"
        assert _format_multiplier(1.0) == "1.0x"

    def test_percentage_formatting(self):
        assert _format_percentage(42.0) == "42.0%"
        assert _format_percentage(320.5) == "320.5%"
        assert _format_percentage(0.0) == "0.0%"


# ===========================================================================
# TestDefenseOnlyValidator
# ===========================================================================

class TestDefenseOnlyValidator:
    """Test two-part contextual defense-only validation."""

    # Required Unsafe cases
    def test_unsafe_imperative_block(self):
        assert validate_defense_only_text("Automatically block the account immediately.") is False

    def test_unsafe_imperative_ban(self):
        assert validate_defense_only_text("Ban the merchant now.") is False

    def test_unsafe_imperative_terminate(self):
        assert validate_defense_only_text("Terminate the account without review.") is False

    def test_unsafe_bypass_security(self):
        assert validate_defense_only_text("Bypass the payment security controls.") is False

    def test_unsafe_freeze_funds(self):
        assert validate_defense_only_text("We recommend to freeze funds immediately.") is False

    # Required Safe cases
    def test_safe_negated_block(self):
        assert validate_defense_only_text("Do not automatically block the account. Flag for analyst review.") is True

    def test_safe_no_automatic_banning(self):
        assert validate_defense_only_text("No automatic banning is recommended.") is True

    def test_safe_human_review_required(self):
        assert validate_defense_only_text("Human review is required before any account action.") is True

    def test_safe_no_candidate_terms(self):
        assert validate_defense_only_text("Transaction activity is elevated and requires analyst review.") is True

    # Ambiguous fail-closed cases
    def test_ambiguous_term_fails_closed(self):
        # A sensitive term in an ambiguous/non-negated context fails closed
        assert _classify_sentence_safety("Account action: suspend merchant profile.") == "UNSAFE"
        assert validate_defense_only_text("Account action: suspend merchant profile.") is False


# ===========================================================================
# TestNumericalGrounding
# ===========================================================================

class TestNumericalGrounding:
    """Test extraction and strict validation of numeric tokens against allowed facts."""

    def test_token_extraction(self):
        text = "Observed 42 transactions totaling ₹95,000.00 with 4.2x multiplier and 42.0% increase."
        tokens = _extract_numeric_tokens(text)
        assert 42.0 in tokens
        assert 95000.0 in tokens
        assert 4.2 in tokens
        assert 42.0 in tokens

    def test_accepted_grounded_numbers(self, sample_window, sample_risk_result):
        window_data = _validate_window_input(sample_window)
        risk_data = _validate_risk_input(sample_risk_result, window_data["total_amount"])
        derived = [4.2, 42.0]

        valid_text = (
            "Merchant merchant_001 had 42 transactions totaling ₹95,000. "
            "Risk score is 75.0 with estimated exposure of ₹47,500. Multiplier is 4.20x."
        )
        assert validate_numerical_grounding(valid_text, window_data, risk_data, derived) is True

    def test_rejected_hallucinated_percentage(self, sample_window, sample_risk_result):
        window_data = _validate_window_input(sample_window)
        risk_data = _validate_risk_input(sample_risk_result, window_data["total_amount"])
        derived = [4.2, 42.0]

        text = "Activity jumped by 88.5% which is unusual."  # 88.5% is not in allowed numbers
        assert validate_numerical_grounding(text, window_data, risk_data, derived) is False

    def test_rejected_hallucinated_risk_score(self, sample_window, sample_risk_result):
        window_data = _validate_window_input(sample_window)
        risk_data = _validate_risk_input(sample_risk_result, window_data["total_amount"])
        derived = []

        text = "The evaluated risk score is 99.0 for this window."  # True score is 75.0
        assert validate_numerical_grounding(text, window_data, risk_data, derived) is False

    def test_rejected_hallucinated_currency_amount(self, sample_window, sample_risk_result):
        window_data = _validate_window_input(sample_window)
        risk_data = _validate_risk_input(sample_risk_result, window_data["total_amount"])
        derived = []

        text = "Estimated financial exposure is ₹1,500,000.00."  # True is 47,500
        assert validate_numerical_grounding(text, window_data, risk_data, derived) is False


# ===========================================================================
# TestLLMIntegrationAndFallback
# ===========================================================================

class TestLLMIntegrationAndFallback:
    """Mock OpenAI API integration, safety filters, and fail-closed fallback."""

    def test_missing_api_key_falls_back_to_rule_based(self, sample_window, sample_risk_result, sample_baseline_stats):
        with patch.object(generate_explanation.__globals__["settings"], "OPENAI_API_KEY", ""):
            with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
                result = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=True)
                assert result.generated_by == "rule_based_fallback"
                assert "High-severity fraud spike" in result.summary

    def test_mock_successful_llm_generation(self, sample_window, sample_risk_result, sample_baseline_stats):
        valid_llm_response = (
            "High-severity anomaly detected for merchant merchant_001.\n\n"
            "Transaction count reached 42 with total volume of ₹95,000.00. "
            "Risk score is 75.0/100 and estimated exposure is ₹47,500.00. "
            "Escalate for priority review without automated blocking."
        )

        with patch.object(generate_explanation.__globals__["settings"], "OPENAI_API_KEY", "sk-mock-key"):
            with patch("explanation_engine._call_openai_api", return_value=valid_llm_response):
                result = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=True)
                assert result.generated_by == "llm"
                assert "merchant_001" in result.summary

    def test_mock_unsafe_llm_response_triggers_fallback(self, sample_window, sample_risk_result, sample_baseline_stats):
        unsafe_response = "Block the merchant account immediately and terminate access."

        with patch.object(generate_explanation.__globals__["settings"], "OPENAI_API_KEY", "sk-mock-key"):
            with patch("explanation_engine._call_openai_api", return_value=unsafe_response):
                result = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=True)
                assert result.generated_by == "rule_based_fallback"
                assert "Automated blocking or banning is strictly disabled" in result.raw_text

    def test_mock_ungrounded_llm_response_triggers_fallback(self, sample_window, sample_risk_result, sample_baseline_stats):
        hallucinated_response = "We detected 999 transactions with ₹888,888 total loss."

        with patch.object(generate_explanation.__globals__["settings"], "OPENAI_API_KEY", "sk-mock-key"):
            with patch("explanation_engine._call_openai_api", return_value=hallucinated_response):
                result = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=True)
                assert result.generated_by == "rule_based_fallback"

    def test_mock_api_timeout_triggers_fallback(self, sample_window, sample_risk_result, sample_baseline_stats):
        with patch.object(generate_explanation.__globals__["settings"], "OPENAI_API_KEY", "sk-mock-key"):
            with patch("explanation_engine._call_openai_api", side_effect=TimeoutError("Request timed out")):
                result = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=True)
                assert result.generated_by == "rule_based_fallback"

    def test_mock_api_exception_triggers_fallback(self, sample_window, sample_risk_result, sample_baseline_stats):
        with patch.object(generate_explanation.__globals__["settings"], "OPENAI_API_KEY", "sk-mock-key"):
            with patch("explanation_engine._call_openai_api", side_effect=RuntimeError("Internal Server Error")):
                result = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=True)
                assert result.generated_by == "rule_based_fallback"


# ===========================================================================
# TestOutputContract
# ===========================================================================

class TestOutputContract:
    """Verify ExplanationResult contract, frozen dataclass immutability, and field presence."""

    def test_result_structure_and_types(self, sample_window, sample_risk_result, sample_baseline_stats):
        result = generate_explanation(sample_window, sample_risk_result, sample_baseline_stats, use_llm=False)

        assert isinstance(result, ExplanationResult)
        assert result.window_id == 101
        assert result.merchant_id == "merchant_001"
        assert isinstance(result.summary, str)
        assert isinstance(result.key_drivers, list)
        assert result.risk_level == "high"
        assert result.estimated_exposure == 47500.0
        assert result.recommended_action == "Escalate for priority review"
        assert result.generated_by in VALID_GENERATED_BY
        assert isinstance(result.raw_text, str)

    def test_result_is_frozen_dataclass(self, sample_window, sample_risk_result):
        result = generate_explanation(sample_window, sample_risk_result, use_llm=False)
        with pytest.raises(Exception):  # FrozenInstanceError
            result.risk_level = "critical"


# ===========================================================================
# TestHoldoutIsolation
# ===========================================================================

class TestHoldoutIsolation:
    """Verify Phase 8 does not import, access, or mention final holdout files."""

    def test_module_has_no_holdout_references(self):
        import explanation_engine
        import inspect

        source = inspect.getsource(explanation_engine)
        assert "final_holdout" not in source
        assert "detection_windows_final_holdout.csv" not in source
