"""
AI Risk Manager — Configuration Management

Loads settings from environment variables with sensible defaults.
All thresholds and cost assumptions are configurable here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application settings loaded from environment variables."""

    # --- Application ---
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_DEBUG: bool = os.getenv("APP_DEBUG", "true").lower() == "true"
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # --- Database ---
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/risk_manager.db"
    )

    # --- AI / LLM ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # --- Risk Score Thresholds ---
    RISK_LOW_MAX: int = int(os.getenv("RISK_LOW_MAX", "30"))
    RISK_MEDIUM_MAX: int = int(os.getenv("RISK_MEDIUM_MAX", "60"))
    RISK_HIGH_MAX: int = int(os.getenv("RISK_HIGH_MAX", "80"))

    # --- Risk Multipliers (Phase 7) ---
    # Used to estimate financial exposure: total_amount × risk_multiplier.
    # These are approximate exposure fractions, NOT precise financial claims.
    RISK_MULTIPLIER_LOW: float = float(os.getenv("RISK_MULTIPLIER_LOW", "0.10"))
    RISK_MULTIPLIER_MEDIUM: float = float(os.getenv("RISK_MULTIPLIER_MEDIUM", "0.25"))
    RISK_MULTIPLIER_HIGH: float = float(os.getenv("RISK_MULTIPLIER_HIGH", "0.50"))
    RISK_MULTIPLIER_CRITICAL: float = float(os.getenv("RISK_MULTIPLIER_CRITICAL", "1.00"))

    # --- Cost Assumptions (INR) ---
    COST_PER_FALSE_POSITIVE: float = float(
        os.getenv("COST_PER_FALSE_POSITIVE", "500")
    )
    COST_PER_FALSE_NEGATIVE: float = float(
        os.getenv("COST_PER_FALSE_NEGATIVE", "15000")
    )

    # --- Detection Thresholds ---
    BASELINE_WINDOW_SIZE: int = int(os.getenv("BASELINE_WINDOW_SIZE", "7"))
    BASELINE_ZSCORE_THRESHOLD: float = float(
        os.getenv("BASELINE_ZSCORE_THRESHOLD", "2.0")
    )
    ISOLATION_FOREST_CONTAMINATION: float = float(
        os.getenv("ISOLATION_FOREST_CONTAMINATION", "0.1")
    )

    # --- Policy Engine Thresholds (Phase 9) ---
    POLICY_CRITICAL_SCORE_THRESHOLD: float = float(
        os.getenv("POLICY_CRITICAL_SCORE_THRESHOLD", "80")
    )
    POLICY_HIGH_EXPOSURE_THRESHOLD: float = float(
        os.getenv("POLICY_HIGH_EXPOSURE_THRESHOLD", "50000")
    )
    POLICY_MIN_TRANSACTION_COUNT_THRESHOLD: int = int(
        os.getenv("POLICY_MIN_TRANSACTION_COUNT_THRESHOLD", "40")
    )

    # --- Data ---
    RANDOM_SEED: int = 42
    DATA_LABEL: str = "SYNTHETIC / TEST DATA"


# Singleton instance
settings = Settings()
