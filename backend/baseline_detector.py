"""
AI Risk Manager — Phase 4: Baseline Detector

A simple, interpretable statistical baseline detector that uses per-merchant
z-scores on the three approved DetectionWindow features:

  1. transaction_count
  2. total_amount
  3. avg_transaction_amount

Design:
  - Learns per-merchant mean and standard deviation from the TRAIN partition.
  - Scores each DEV_TEST window by computing per-feature z-scores.
  - Flags a window when ANY feature's |z-score| >= BASELINE_ZSCORE_THRESHOLD.
  - Maps the maximum |z-score| to a 0–100 risk_score (anomaly severity, NOT
    a probability).
  - Generates a human-readable explanation identifying each feature's z-score
    and flagged/normal status.

Partition rules:
  - Train:         used ONLY for fitting baseline statistics
  - Dev-test:      scored for predictions
  - Final-holdout: NEVER loaded, scored, or evaluated (protected until Phase 15)

Note on BASELINE_WINDOW_SIZE:
  The config parameter BASELINE_WINDOW_SIZE (default 7) exists in config.py
  but is NOT used by this detector.  This detector uses full train-partition
  per-merchant statistics rather than a rolling/sliding window approach.
  BASELINE_WINDOW_SIZE is reserved for potential future use and must not be
  deleted or modified.

This module serves as the naive comparison point for the Phase 5 ML
Anomaly Detector (Isolation Forest).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the backend package is importable when running as a script
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from config import settings  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The three approved DetectionWindow aggregate features (stable data contract).
BASELINE_FEATURES: list[str] = [
    "transaction_count",
    "total_amount",
    "avg_transaction_amount",
]

# Default z-score threshold from config.
BASELINE_ZSCORE_THRESHOLD: float = settings.BASELINE_ZSCORE_THRESHOLD  # 2.0

# Sentinel z-score assigned when training standard deviation is zero
# and the prediction value differs from the training mean.
# This is a deterministic constant, not a tunable parameter.
ZERO_STD_SENTINEL_ZSCORE: float = 10.0


# ===========================================================================
# BaselineDetector
# ===========================================================================


class BaselineDetector:
    """
    Per-merchant z-score baseline detector.

    Learns per-merchant mean and standard deviation for each approved feature
    from the train partition.  Flags dev_test windows where any feature's
    absolute z-score meets or exceeds the configured threshold.

    Zero-standard-deviation handling:
      If the training standard deviation for a merchant/feature is zero:
        - prediction value == training mean  →  z-score = 0.0
        - prediction value != training mean  →  z-score = ZERO_STD_SENTINEL_ZSCORE (10.0)
      This prevents NaN, infinity, and division-by-zero errors.
    """

    def __init__(self, zscore_threshold: float = BASELINE_ZSCORE_THRESHOLD):
        self.zscore_threshold = zscore_threshold
        # merchant_id → {feature_name → {"mean": float, "std": float}}
        self.merchant_stats: dict[str, dict[str, dict[str, float]]] = {}
        self._is_fitted = False

    def fit(self, train_windows_df: pd.DataFrame) -> "BaselineDetector":
        """
        Compute per-merchant mean and standard deviation for each approved
        feature from the train partition.

        The ``is_synthetic_fraud_spike`` label is NOT used during fitting.
        Statistics are computed purely from the three approved feature columns.

        Args:
            train_windows_df: DataFrame containing ONLY train-partition
                DetectionWindow records.  Must have a ``split`` column with
                every value equal to ``"train"``.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If the DataFrame is missing the ``split`` column,
                or contains any rows with ``split != "train"``.
        """
        if "split" not in train_windows_df.columns:
            raise ValueError(
                "Input DataFrame must contain a 'split' column."
            )

        non_train = train_windows_df[train_windows_df["split"] != "train"]
        if len(non_train) > 0:
            bad_splits = sorted(non_train["split"].unique().tolist())
            raise ValueError(
                f"fit() requires only train-partition data. "
                f"Found non-train split values: {bad_splits}"
            )

        self.merchant_stats = {}

        for merchant_id in sorted(train_windows_df["merchant_id"].unique()):
            merchant_data = train_windows_df[
                train_windows_df["merchant_id"] == merchant_id
            ]
            stats: dict[str, dict[str, float]] = {}
            for feature in BASELINE_FEATURES:
                mean_val = float(merchant_data[feature].mean())
                std_val = float(merchant_data[feature].std(ddof=1))
                # Handle NaN std (e.g., single data point with ddof=1)
                if np.isnan(std_val):
                    std_val = 0.0
                stats[feature] = {"mean": mean_val, "std": std_val}
            self.merchant_stats[merchant_id] = stats

        self._is_fitted = True
        return self

    def predict(self, windows_df: pd.DataFrame, allow_holdout: bool = False) -> list[dict]:
        """
        Score each window using per-merchant z-scores.

        Args:
            windows_df: DataFrame of DetectionWindow records to score.
                Must NOT contain ``final_holdout`` rows unless ``allow_holdout=True``.
                If an ``id`` column is present, it is included as ``window_id``
                in the output for persistence.
            allow_holdout: If True, permits scoring final_holdout rows (Phase 15 only).
                Default is False.

        Returns:
            List of dicts, each containing:
                - ``window_id``   (int, only if ``id`` column exists)
                - ``merchant_id`` (str)
                - ``risk_score``  (float, 0–100, anomaly severity)
                - ``is_flagged``  (bool)
                - ``explanation`` (str, human-readable)
                - ``z_scores``    (dict[str, float], for testing/debugging)

        Raises:
            RuntimeError: If the detector has not been fitted.
            ValueError: If the DataFrame contains ``final_holdout`` rows and allow_holdout=False.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "BaselineDetector must be fit() before predict()."
            )

        # Layer 4: Reject final_holdout rows unless explicitly authorized for Phase 15
        if not allow_holdout and "split" in windows_df.columns:
            if (windows_df["split"] == "final_holdout").any():
                raise ValueError(
                    "predict() received final_holdout rows. "
                    "The final_holdout partition is protected and must "
                    "not be scored unless allow_holdout=True is set."
                )

        results: list[dict] = []
        has_id = "id" in windows_df.columns

        for _, row in windows_df.iterrows():
            merchant_id = row["merchant_id"]

            if merchant_id not in self.merchant_stats:
                # Unknown merchant — cannot score without baseline stats
                continue

            stats = self.merchant_stats[merchant_id]
            z_scores: dict[str, float] = {}

            for feature in BASELINE_FEATURES:
                value = float(row[feature])
                mean = stats[feature]["mean"]
                std = stats[feature]["std"]

                if std == 0.0:
                    # Zero-std handling: deterministic sentinel
                    if abs(value - mean) < 1e-9:
                        z_scores[feature] = 0.0
                    else:
                        z_scores[feature] = ZERO_STD_SENTINEL_ZSCORE
                else:
                    z_scores[feature] = (value - mean) / std

            # Flagging: any feature abs(z) >= threshold
            abs_z_scores = {f: abs(z) for f, z in z_scores.items()}
            max_abs_z = max(abs_z_scores.values()) if abs_z_scores else 0.0
            is_flagged = any(
                abs_z >= self.zscore_threshold
                for abs_z in abs_z_scores.values()
            )

            # Risk score: deterministic anomaly severity, NOT a probability
            # z = 0 → 0,  z = threshold → 50,  z = 2×threshold → 100
            risk_score = min(
                100.0,
                (max_abs_z / self.zscore_threshold) * 50.0,
            )

            # Explanation: human-readable plain English
            explanation_parts: list[str] = []
            for feature in BASELINE_FEATURES:
                z = z_scores[feature]
                abs_z = abs(z)
                if abs_z >= self.zscore_threshold:
                    explanation_parts.append(
                        f"{feature} z={z:.2f} "
                        f"(FLAGGED, threshold={self.zscore_threshold:.2f})"
                    )
                else:
                    explanation_parts.append(
                        f"{feature} z={z:.2f} (normal)"
                    )
            explanation = (
                "Baseline detector: " + "; ".join(explanation_parts) + "."
            )

            result: dict = {
                "merchant_id": merchant_id,
                "risk_score": round(risk_score, 2),
                "is_flagged": is_flagged,
                "explanation": explanation,
                "z_scores": z_scores,
            }

            if has_id:
                result["window_id"] = int(row["id"])

            results.append(result)

        return results


# ===========================================================================
# Pipeline entry point
# ===========================================================================


def run_baseline_detector(
    session=None,
    target_split: str = "dev_test",
    allow_holdout: bool = False,
) -> None:
    """
    Full Phase 4 Baseline Detector execution pipeline.

    1. Load ONLY train and target_split DetectionWindow records.
    2. Fit per-merchant z-score baseline on the train partition.
    3. Predict on the target_split partition (dev_test or final_holdout).
    4. Persist AnomalyDetection records with ``detector_type='baseline'``.

    The ``final_holdout`` partition is rejected unless ``allow_holdout=True``
    and ``target_split='final_holdout'`` are explicitly provided.

    Args:
        session: Optional SQLAlchemy Session for testing.
        target_split: The partition to predict on ("dev_test" or "final_holdout"). Default "dev_test".
        allow_holdout: If True, permits scoring final_holdout rows (Phase 15 only). Default False.
    """
    if target_split == "final_holdout" and not allow_holdout:
        raise ValueError(
            "Partition 'final_holdout' is protected and cannot be scored unless allow_holdout=True is set."
        )
    if target_split not in {"dev_test", "final_holdout"}:
        raise ValueError(
            f"Unsupported target_split '{target_split}'. Must be 'dev_test' or 'final_holdout'."
        )

    from database import Base  # noqa: E402
    from models import AnomalyDetection, DetectionWindow  # noqa: E402

    _owns_session = session is None

    if _owns_session:
        from database import engine, SessionLocal  # noqa: E402

        Base.metadata.create_all(bind=engine)
        session = SessionLocal()

    try:
        # ---------------------------------------------------------------
        # Layer 1: Load train and target_split windows
        # ---------------------------------------------------------------
        splits_to_load = list(dict.fromkeys(["train", target_split]))
        windows = (
            session.query(DetectionWindow)
            .filter(DetectionWindow.split.in_(splits_to_load))
            .all()
        )

        if not windows:
            print(f"No train/{target_split} DetectionWindow records found.")
            return

        # Convert ORM objects to DataFrame
        records = []
        for w in windows:
            records.append(
                {
                    "id": w.id,
                    "merchant_id": w.merchant_id,
                    "window_start": w.window_start,
                    "window_end": w.window_end,
                    "transaction_count": w.transaction_count,
                    "total_amount": w.total_amount,
                    "avg_transaction_amount": w.avg_transaction_amount,
                    "is_synthetic_fraud_spike": w.is_synthetic_fraud_spike,
                    "split": w.split,
                }
            )

        all_df = pd.DataFrame(records)

        # ---------------------------------------------------------------
        # Layer 2: Verify no final_holdout unless allow_holdout=True
        # ---------------------------------------------------------------
        if not allow_holdout and "final_holdout" in all_df["split"].values:
            raise RuntimeError(
                "SAFETY VIOLATION: final_holdout rows detected in loaded "
                "data. The database query should have excluded them."
            )

        train_df = all_df[all_df["split"] == "train"].copy()
        predict_df = all_df[all_df["split"] == target_split].copy()

        if train_df.empty:
            print("No train DetectionWindow records found. Cannot fit.")
            return
        if predict_df.empty:
            print(
                f"No {target_split} DetectionWindow records found. Nothing to score."
            )
            return

        # ---------------------------------------------------------------
        # Layer 3: Verify partitions are pure
        # ---------------------------------------------------------------
        train_splits = set(train_df["split"].unique())
        predict_splits = set(predict_df["split"].unique())
        if train_splits != {"train"}:
            raise RuntimeError(
                f"Training DataFrame contains non-train rows: {train_splits}"
            )
        if predict_splits != {target_split}:
            raise RuntimeError(
                f"Prediction DataFrame contains non-{target_split} rows: "
                f"{predict_splits}"
            )

        # ---------------------------------------------------------------
        # Fit on train ONLY
        # ---------------------------------------------------------------
        detector = BaselineDetector()
        detector.fit(train_df)

        # ---------------------------------------------------------------
        # Predict on target_split (Layer 4 is inside predict())
        # ---------------------------------------------------------------
        predictions = detector.predict(predict_df, allow_holdout=allow_holdout)

        # ---------------------------------------------------------------
        # Layer 5: Verify all predictions are for target_split windows
        # ---------------------------------------------------------------
        predict_window_ids = set(predict_df["id"].tolist())
        for pred in predictions:
            if pred["window_id"] not in predict_window_ids:
                raise RuntimeError(
                    f"SAFETY VIOLATION: prediction generated for "
                    f"window_id={pred['window_id']} which is not a "
                    f"{target_split} window."
                )

        # ---------------------------------------------------------------
        # Layer 6: Idempotent cleanup — delete existing baseline records
        # for target_split windows only, then insert new predictions
        # ---------------------------------------------------------------
        session.query(AnomalyDetection).filter(
            AnomalyDetection.detector_type == "baseline",
            AnomalyDetection.window_id.in_(list(predict_window_ids)),
        ).delete(synchronize_session="fetch")
        session.commit()

        # Insert new predictions in batches
        BATCH_SIZE = 500
        for i in range(0, len(predictions), BATCH_SIZE):
            batch = predictions[i : i + BATCH_SIZE]
            session.bulk_insert_mappings(
                AnomalyDetection,
                [
                    {
                        "window_id": p["window_id"],
                        "detector_type": "baseline",
                        "risk_score": p["risk_score"],
                        "is_flagged": p["is_flagged"],
                        "explanation": p["explanation"],
                    }
                    for p in batch
                ],
            )
        session.commit()

        # ---------------------------------------------------------------
        # Layer 7: Post-insert verification — no unpermitted holdout records
        # ---------------------------------------------------------------
        if not allow_holdout:
            holdout_records = (
                session.query(AnomalyDetection)
                .join(DetectionWindow)
                .filter(
                    AnomalyDetection.detector_type == "baseline",
                    AnomalyDetection.window_id.in_(list(predict_window_ids)),
                    DetectionWindow.split == "final_holdout",
                )
                .count()
            )
            if holdout_records > 0:
                raise RuntimeError(
                    f"SAFETY VIOLATION: {holdout_records} final_holdout "
                    "records found in AnomalyDetection table after dev_test pipeline execution."
                )

        # Summary
        flagged_count = sum(1 for p in predictions if p["is_flagged"])
        print(
            f"Baseline detector: scored {len(predictions)} {target_split} windows, "
            f"flagged {flagged_count}"
        )

    except Exception:
        session.rollback()
        raise
    finally:
        if _owns_session:
            session.close()


# ===========================================================================
# CLI entry point
# ===========================================================================
if __name__ == "__main__":
    run_baseline_detector()
