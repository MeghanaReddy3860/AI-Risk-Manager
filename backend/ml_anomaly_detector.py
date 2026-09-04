"""
AI Risk Manager — Phase 5: ML Anomaly Detector (Isolation Forest)

An unsupervised ML anomaly detector using scikit-learn's IsolationForest
trained on the three approved DetectionWindow features:

  1. transaction_count
  2. total_amount
  3. avg_transaction_amount

Design:
  - Fits an IsolationForest model on the TRAIN partition only.
  - Scores each DEV_TEST window using the fitted model.
  - Flags a window when IsolationForest.predict() == -1 (anomaly).
  - Maps the IsolationForest decision_function() output to a deterministic
    0–100 risk_score (anomaly severity, NOT a probability).
  - Generates a human-readable explanation.

Risk-score transformation:
  IsolationForest.decision_function(X) returns a real-valued anomaly score
  where lower (more negative) values indicate stronger anomalies and higher
  (more positive) values indicate more normal behaviour.  The transformation
  used is:

      raw = -decision_function(X)          # flip so higher = more anomalous
      clamped = clamp(raw, 0, max_raw)     # floor at 0
      risk_score = clamp(clamped / max_raw * 100, 0, 100)

  where max_raw is derived from the training data to anchor the scale.
  This preserves anomaly ordering and is deterministic.

Partition rules:
  - Train:         used ONLY for fitting the Isolation Forest model
  - Dev-test:      scored for predictions
  - Final-holdout: NEVER loaded, scored, or evaluated (protected until Phase 15)

This module is the ML comparison model against the Phase 4 statistical
baseline detector.  It does NOT use Phase 4 predictions, z-scores, or
baseline risk scores.  It does NOT use synthetic fraud labels
(is_synthetic_fraud_spike) during fitting or prediction.

Hyperparameters:
  - n_estimators:  100  (scikit-learn default)
  - contamination: from config ISOLATION_FOREST_CONTAMINATION (0.1)
  - random_state:  from config RANDOM_SEED (42)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

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
# These are the ONLY features passed to the Isolation Forest model.
# merchant_id, window_id, split, and is_synthetic_fraud_spike are NEVER
# used as ML features.
ML_FEATURES: list[str] = [
    "transaction_count",
    "total_amount",
    "avg_transaction_amount",
]

# Isolation Forest hyperparameters from existing config.
ML_N_ESTIMATORS: int = 100
ML_CONTAMINATION: float = settings.ISOLATION_FOREST_CONTAMINATION  # 0.1
ML_RANDOM_STATE: int = settings.RANDOM_SEED  # 42


# ===========================================================================
# MLAnomalyDetector
# ===========================================================================


class MLAnomalyDetector:
    """
    Unsupervised ML anomaly detector using scikit-learn IsolationForest.

    Trains on the three approved DetectionWindow features from the TRAIN
    partition only.  Produces predictions for DEV_TEST windows.

    The model is fully unsupervised: it does NOT use fraud labels
    (``is_synthetic_fraud_spike``) during fitting or prediction.
    It does NOT use Phase 4 baseline predictions, z-scores, or risk scores.
    """

    def __init__(
        self,
        n_estimators: int = ML_N_ESTIMATORS,
        contamination: float = ML_CONTAMINATION,
        random_state: int = ML_RANDOM_STATE,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model: IsolationForest | None = None
        self._is_fitted = False
        # Anchor for risk-score normalisation, learned during fit.
        self._train_max_neg_score: float = 1.0

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_features(df: pd.DataFrame) -> None:
        """Verify that all required ML features exist and are finite."""
        missing = [f for f in ML_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required ML features: {missing}"
            )
        feature_data = df[ML_FEATURES]
        if feature_data.isnull().any().any():
            raise ValueError(
                "Feature data contains NaN values. "
                "All ML features must be finite and non-null."
            )
        if not np.isfinite(feature_data.values).all():
            raise ValueError(
                "Feature data contains infinite values. "
                "All ML features must be finite."
            )

    @staticmethod
    def _reject_holdout(df: pd.DataFrame, allow_holdout: bool = False) -> None:
        """Raise ValueError if any final_holdout rows are present unless allow_holdout=True."""
        if not allow_holdout and "split" in df.columns:
            if (df["split"] == "final_holdout").any():
                raise ValueError(
                    "Received final_holdout rows. "
                    "The final_holdout partition is protected and must "
                    "not be used unless allow_holdout=True is set for Phase 15."
                )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_windows_df: pd.DataFrame) -> "MLAnomalyDetector":
        """
        Fit the Isolation Forest model on TRAIN partition data.

        Only the three approved features are used.
        ``is_synthetic_fraud_spike`` is NOT used.
        ``merchant_id`` is NOT used as an ML feature.

        Args:
            train_windows_df: DataFrame containing ONLY train-partition
                DetectionWindow records.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If the DataFrame contains non-train rows,
                is missing required features, or contains non-finite values.
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

        self._reject_holdout(train_windows_df, allow_holdout=False)
        self._validate_features(train_windows_df)

        X_train = train_windows_df[ML_FEATURES].values

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(X_train)

        # Compute normalisation anchor from training data.
        # decision_function: lower → more anomalous.
        # Negate so higher → more anomalous, then take the max across train.
        train_scores = self.model.decision_function(X_train)
        neg_scores = -train_scores
        max_neg = float(neg_scores.max())
        # Guard: if all scores are identical (degenerate model), use 1.0
        self._train_max_neg_score = max_neg if max_neg > 0 else 1.0

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, windows_df: pd.DataFrame, allow_holdout: bool = False) -> list[dict]:
        """
        Score each window using the fitted Isolation Forest model.

        Args:
            windows_df: DataFrame of DetectionWindow records to score.
                Must NOT contain ``final_holdout`` rows unless ``allow_holdout=True``.
                If an ``id`` column is present it is included as ``window_id``.
            allow_holdout: If True, permits scoring final_holdout rows (Phase 15 only).
                Default is False.

        Returns:
            List of dicts, each containing:
                - ``window_id``   (int, only if ``id`` column exists)
                - ``merchant_id`` (str)
                - ``risk_score``  (float, 0–100, anomaly severity)
                - ``is_flagged``  (bool)
                - ``explanation`` (str, human-readable)
                - ``raw_score``   (float, negated decision_function)

        Raises:
            RuntimeError: If the detector has not been fitted.
            ValueError: If the DataFrame contains ``final_holdout`` rows and allow_holdout=False,
                or has invalid features.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError(
                "MLAnomalyDetector must be fit() before predict()."
            )

        self._reject_holdout(windows_df, allow_holdout=allow_holdout)
        self._validate_features(windows_df)

        X = windows_df[ML_FEATURES].values
        has_id = "id" in windows_df.columns

        # IsolationForest outputs
        predictions = self.model.predict(X)         # -1 = anomaly, 1 = normal
        decision_scores = self.model.decision_function(X)

        results: list[dict] = []

        for i, (_, row) in enumerate(windows_df.iterrows()):
            pred = int(predictions[i])
            raw_decision = float(decision_scores[i])

            # Flagging: directly from IsolationForest convention
            is_flagged = pred == -1

            # Risk score: deterministic transformation
            # Negate decision_function so higher = more anomalous
            neg_score = -raw_decision
            # Clamp to [0, +inf), then normalise using training anchor
            clamped = max(0.0, neg_score)
            risk_score = min(100.0, (clamped / self._train_max_neg_score) * 100.0)
            risk_score = round(risk_score, 2)

            # Explanation
            explanation = (
                f"Isolation Forest detector: "
                f"anomaly prediction={pred}; "
                f"anomaly severity={risk_score:.2f}."
            )

            result: dict = {
                "merchant_id": row["merchant_id"],
                "risk_score": risk_score,
                "is_flagged": is_flagged,
                "explanation": explanation,
                "raw_score": neg_score,
            }

            if has_id:
                result["window_id"] = int(row["id"])

            results.append(result)

        return results


# ===========================================================================
# Pipeline entry point
# ===========================================================================


def run_ml_anomaly_detector(
    session=None,
    target_split: str = "dev_test",
    allow_holdout: bool = False,
) -> None:
    """
    Full Phase 5 ML anomaly detection pipeline.

    1. Load ONLY train and target_split DetectionWindow records.
    2. Fit Isolation Forest on the train partition.
    3. Predict on the target_split partition (dev_test or final_holdout).
    4. Persist AnomalyDetection records with ``detector_type='ml'``.

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
        detector = MLAnomalyDetector()
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
        # Layer 6: Idempotent cleanup — delete existing ML records
        # for target_split windows only, then insert new predictions
        # ---------------------------------------------------------------
        session.query(AnomalyDetection).filter(
            AnomalyDetection.detector_type == "ml",
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
                        "detector_type": "ml",
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
                    AnomalyDetection.detector_type == "ml",
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
            f"ML detector: scored {len(predictions)} {target_split} windows, "
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
    run_ml_anomaly_detector()
