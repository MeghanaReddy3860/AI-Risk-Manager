"""
AI Risk Manager — Phase 3: Synthetic Data Generator

Generates deterministic, reproducible synthetic merchant transaction streams
with injected fraud-spike events. Produces:

1. Raw Transaction records across 10 merchants over 30 days.
2. DetectionWindow aggregates (1-hour tumbling windows) with ground-truth labels.
3. Three-way chronological split: train / dev_test / final_holdout.

Outputs:
  - data/synthetic/transactions.csv
  - data/synthetic/detection_windows.csv          (train + dev_test only)
  - data/synthetic/detection_windows_final_holdout.csv  (final_holdout only)
  - SQLite database seeding via seed_database()

IMPORTANT:
  detection_windows_final_holdout.csv must NOT be opened, inspected, loaded,
  or evaluated against by any code until Phase 15.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
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
RANDOM_SEED: int = settings.RANDOM_SEED  # 42

NUM_MERCHANTS: int = 10
MERCHANT_IDS: list[str] = [f"M{str(i).zfill(3)}" for i in range(1, NUM_MERCHANTS + 1)]

SIMULATION_DAYS: int = 30
WINDOW_SIZE_HOURS: int = 1  # 1-hour tumbling windows

# Simulation start: a fixed UTC timestamp for determinism
SIM_START = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
SIM_END = SIM_START + timedelta(days=SIMULATION_DAYS)

# Three-way chronological split boundaries (day numbers are 1-indexed)
TRAIN_END_DAY = 18       # Train:        Day 1 – Day 18
DEV_TEST_END_DAY = 25    # Dev-test:     Day 19 – Day 25
# Final-holdout:          Day 26 – Day 30

SPLIT_TRAIN_END = SIM_START + timedelta(days=TRAIN_END_DAY)      # Day 19 00:00
SPLIT_DEV_TEST_END = SIM_START + timedelta(days=DEV_TEST_END_DAY)  # Day 26 00:00

# Output paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"

TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
DETECTION_WINDOWS_CSV = DATA_DIR / "detection_windows.csv"
DETECTION_WINDOWS_HOLDOUT_CSV = DATA_DIR / "detection_windows_final_holdout.csv"

# ---------------------------------------------------------------------------
# Merchant profiles — configurable normal transaction behaviour
# ---------------------------------------------------------------------------
# Each merchant profile defines:
#   - avg_txns_per_hour: average number of transactions per hour (Poisson λ)
#   - avg_amount: mean transaction amount (normal distribution μ)
#   - std_amount: standard deviation of transaction amount

MERCHANT_PROFILES: dict[str, dict] = {
    "M001": {"avg_txns_per_hour": 8,  "avg_amount": 250.0, "std_amount": 80.0},
    "M002": {"avg_txns_per_hour": 12, "avg_amount": 150.0, "std_amount": 50.0},
    "M003": {"avg_txns_per_hour": 5,  "avg_amount": 500.0, "std_amount": 150.0},
    "M004": {"avg_txns_per_hour": 15, "avg_amount": 100.0, "std_amount": 30.0},
    "M005": {"avg_txns_per_hour": 10, "avg_amount": 300.0, "std_amount": 100.0},
    "M006": {"avg_txns_per_hour": 7,  "avg_amount": 200.0, "std_amount": 60.0},
    "M007": {"avg_txns_per_hour": 20, "avg_amount": 75.0,  "std_amount": 25.0},
    "M008": {"avg_txns_per_hour": 6,  "avg_amount": 400.0, "std_amount": 120.0},
    "M009": {"avg_txns_per_hour": 9,  "avg_amount": 180.0, "std_amount": 55.0},
    "M010": {"avg_txns_per_hour": 11, "avg_amount": 220.0, "std_amount": 70.0},
}

# ---------------------------------------------------------------------------
# Fraud-spike schedule — deterministic, covers all three partitions
# ---------------------------------------------------------------------------
# Each spike entry:
#   merchant_id, day (1-indexed), hour, category
# Categories: "frequency", "volume", "mixed"
#
# Distribution ensures every partition has ≥ 2 spike events and contains
# all three spike categories across the full schedule.

SPIKE_SCHEDULE: list[dict] = [
    # --- Train partition (Day 1–18) — 6 spike events ---
    {"merchant_id": "M001", "day": 3,  "hour": 10, "category": "frequency"},
    {"merchant_id": "M003", "day": 5,  "hour": 14, "category": "volume"},
    {"merchant_id": "M005", "day": 8,  "hour": 9,  "category": "mixed"},
    {"merchant_id": "M007", "day": 11, "hour": 16, "category": "frequency"},
    {"merchant_id": "M002", "day": 14, "hour": 11, "category": "volume"},
    {"merchant_id": "M009", "day": 17, "hour": 13, "category": "mixed"},
    # --- Dev-test partition (Day 19–25) — 4 spike events ---
    {"merchant_id": "M004", "day": 20, "hour": 10, "category": "frequency"},
    {"merchant_id": "M006", "day": 21, "hour": 15, "category": "volume"},
    {"merchant_id": "M008", "day": 23, "hour": 12, "category": "mixed"},
    {"merchant_id": "M010", "day": 25, "hour": 9,  "category": "frequency"},
    # --- Final-holdout partition (Day 26–30) — 3 spike events ---
    {"merchant_id": "M001", "day": 27, "hour": 11, "category": "volume"},
    {"merchant_id": "M005", "day": 28, "hour": 14, "category": "mixed"},
    {"merchant_id": "M003", "day": 30, "hour": 10, "category": "frequency"},
]

# Spike multipliers — how much spikes elevate normal behaviour
SPIKE_FREQUENCY_MULTIPLIER = 5.0   # 5x normal transaction count
SPIKE_VOLUME_MULTIPLIER = 4.0      # 4x normal transaction amount
SPIKE_MIXED_FREQ_MULTIPLIER = 3.0  # 3x count for mixed
SPIKE_MIXED_VOL_MULTIPLIER = 3.0   # 3x amount for mixed


# ===========================================================================
# Generator functions
# ===========================================================================

def _get_spike_lookup(schedule: list[dict]) -> dict[tuple[str, int, int], str]:
    """Build a lookup: (merchant_id, day, hour) → spike category."""
    return {
        (s["merchant_id"], s["day"], s["hour"]): s["category"]
        for s in schedule
    }


def generate_transactions(rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate raw synthetic Transaction records for all merchants over
    the full 30-day simulation period.

    Spike windows get elevated transaction counts and/or amounts so that
    the resulting DetectionWindow aggregates are measurably different
    from normal windows.
    """
    spike_lookup = _get_spike_lookup(SPIKE_SCHEDULE)
    all_records: list[dict] = []

    for merchant_id in MERCHANT_IDS:
        profile = MERCHANT_PROFILES[merchant_id]
        avg_txns = profile["avg_txns_per_hour"]
        avg_amt = profile["avg_amount"]
        std_amt = profile["std_amount"]

        # Iterate over every hour in the 30-day simulation
        current = SIM_START
        day = 1
        while current < SIM_END:
            hour = current.hour
            # Check if this is a spike window
            spike_cat = spike_lookup.get((merchant_id, day, hour))

            if spike_cat == "frequency":
                n_txns = max(1, int(rng.poisson(avg_txns * SPIKE_FREQUENCY_MULTIPLIER)))
                amounts = np.abs(rng.normal(avg_amt, std_amt, size=n_txns))
            elif spike_cat == "volume":
                n_txns = max(1, int(rng.poisson(avg_txns)))
                amounts = np.abs(rng.normal(avg_amt * SPIKE_VOLUME_MULTIPLIER, std_amt, size=n_txns))
            elif spike_cat == "mixed":
                n_txns = max(1, int(rng.poisson(avg_txns * SPIKE_MIXED_FREQ_MULTIPLIER)))
                amounts = np.abs(rng.normal(avg_amt * SPIKE_MIXED_VOL_MULTIPLIER, std_amt, size=n_txns))
            else:
                n_txns = max(1, int(rng.poisson(avg_txns)))
                amounts = np.abs(rng.normal(avg_amt, std_amt, size=n_txns))

            # Ensure all amounts are positive (minimum 0.01)
            amounts = np.maximum(amounts, 0.01)

            # Spread transactions uniformly within the hour
            offsets = rng.uniform(0, 3600, size=n_txns)
            offsets.sort()

            for i in range(n_txns):
                ts = current + timedelta(seconds=float(offsets[i]))
                all_records.append({
                    "merchant_id": merchant_id,
                    "amount": round(float(amounts[i]), 2),
                    "timestamp": ts,
                })

            # Advance to next hour
            current += timedelta(hours=1)
            if current.hour == 0:
                day += 1

    df = pd.DataFrame(all_records)
    df.sort_values(by=["timestamp", "merchant_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def aggregate_detection_windows(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate raw transactions into 1-hour tumbling DetectionWindow records.

    For each merchant and each hour, compute:
      - transaction_count
      - total_amount
      - avg_transaction_amount

    Also assigns:
      - is_synthetic_fraud_spike (ground-truth label)
      - split (train / dev_test / final_holdout)
    """
    spike_lookup = _get_spike_lookup(SPIKE_SCHEDULE)

    windows: list[dict] = []

    for merchant_id in MERCHANT_IDS:
        merchant_txns = transactions_df[transactions_df["merchant_id"] == merchant_id]

        # Iterate over every hour in the simulation
        current = SIM_START
        day = 1
        while current < SIM_END:
            window_start = current
            window_end = current + timedelta(hours=WINDOW_SIZE_HOURS)
            hour = current.hour

            # Filter transactions in this window
            mask = (
                (merchant_txns["timestamp"] >= window_start)
                & (merchant_txns["timestamp"] < window_end)
            )
            window_txns = merchant_txns[mask]

            transaction_count = len(window_txns)
            total_amount = float(window_txns["amount"].sum()) if transaction_count > 0 else 0.0
            avg_transaction_amount = (
                total_amount / transaction_count if transaction_count > 0 else 0.0
            )

            # Ground-truth label
            is_spike = (merchant_id, day, hour) in spike_lookup

            # Chronological split assignment
            if window_start < SPLIT_TRAIN_END:
                split = "train"
            elif window_start < SPLIT_DEV_TEST_END:
                split = "dev_test"
            else:
                split = "final_holdout"

            windows.append({
                "merchant_id": merchant_id,
                "window_start": window_start,
                "window_end": window_end,
                "transaction_count": transaction_count,
                "total_amount": round(total_amount, 2),
                "avg_transaction_amount": round(avg_transaction_amount, 2),
                "is_synthetic_fraud_spike": is_spike,
                "split": split,
            })

            # Advance to next hour
            current += timedelta(hours=1)
            if current.hour == 0:
                day += 1

    df = pd.DataFrame(windows)
    df.sort_values(by=["window_start", "merchant_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def export_csv(
    transactions_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> None:
    """
    Export data to three CSV files:
      1. transactions.csv — all raw transactions
      2. detection_windows.csv — train + dev_test windows only
      3. detection_windows_final_holdout.csv — final_holdout windows only
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. All transactions
    transactions_df.to_csv(TRANSACTIONS_CSV, index=False)

    # 2. Development dataset (train + dev_test)
    dev_df = windows_df[windows_df["split"].isin(["train", "dev_test"])].copy()
    dev_df.to_csv(DETECTION_WINDOWS_CSV, index=False)

    # 3. Final holdout (physically separate)
    holdout_df = windows_df[windows_df["split"] == "final_holdout"].copy()
    holdout_df.to_csv(DETECTION_WINDOWS_HOLDOUT_CSV, index=False)

    print(f"Exported {len(transactions_df):,} transactions to {TRANSACTIONS_CSV}")
    print(f"Exported {len(dev_df):,} detection windows (train + dev_test) to {DETECTION_WINDOWS_CSV}")
    print(f"Exported {len(holdout_df):,} detection windows (final_holdout) to {DETECTION_WINDOWS_HOLDOUT_CSV}")


def seed_database(
    transactions_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    session=None,
) -> None:
    """
    Seed the Phase 2 SQLite database with generated Transaction and
    DetectionWindow records.

    Does NOT create AnomalyDetection records — those belong to Phase 4/5.
    Does NOT create EvaluationRun records — those belong to Phase 6.

    Args:
        transactions_df: DataFrame of raw Transaction records.
        windows_df: DataFrame of aggregated DetectionWindow records.
        session: Optional SQLAlchemy Session for testing. When None
                 (default), uses the production database engine and
                 SessionLocal — preserving existing production behavior.
    """
    from database import Base  # noqa: E402
    from models import Transaction, DetectionWindow  # noqa: E402

    _owns_session = session is None

    if _owns_session:
        from database import engine, SessionLocal  # noqa: E402

        # Ensure the database directory exists (SQLite needs it)
        from config import settings as _settings
        db_url = _settings.DATABASE_URL
        if db_url.startswith("sqlite:///"):
            db_path = Path(db_url.replace("sqlite:///", ""))
            if not db_path.is_absolute():
                # Resolve relative to backend dir (where the script runs)
                db_path = _BACKEND_DIR / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        session = SessionLocal()

    try:
        # Clear existing data for idempotent seeding
        session.query(DetectionWindow).delete()
        session.query(Transaction).delete()
        session.commit()

        # Insert transactions in batches
        BATCH_SIZE = 1000
        txn_records = transactions_df.to_dict("records")
        for i in range(0, len(txn_records), BATCH_SIZE):
            batch = txn_records[i:i + BATCH_SIZE]
            session.bulk_insert_mappings(Transaction, [
                {
                    "merchant_id": r["merchant_id"],
                    "amount": r["amount"],
                    "timestamp": r["timestamp"],
                }
                for r in batch
            ])
        session.commit()

        # Insert detection windows in batches
        win_records = windows_df.to_dict("records")
        for i in range(0, len(win_records), BATCH_SIZE):
            batch = win_records[i:i + BATCH_SIZE]
            session.bulk_insert_mappings(DetectionWindow, [
                {
                    "merchant_id": r["merchant_id"],
                    "window_start": r["window_start"],
                    "window_end": r["window_end"],
                    "transaction_count": r["transaction_count"],
                    "total_amount": r["total_amount"],
                    "avg_transaction_amount": r["avg_transaction_amount"],
                    "is_synthetic_fraud_spike": r["is_synthetic_fraud_spike"],
                    "split": r["split"],
                }
                for r in batch
            ])
        session.commit()

        print(f"Seeded database: {len(txn_records):,} transactions, "
              f"{len(win_records):,} detection windows")

    except Exception:
        session.rollback()
        raise
    finally:
        if _owns_session:
            session.close()


def generate_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full Phase 3 data generation pipeline:
      1. Generate transactions
      2. Aggregate into detection windows
      3. Export CSVs
      4. Seed database

    Returns (transactions_df, windows_df) for testing convenience.
    """
    print(f"Generating synthetic data with RANDOM_SEED={RANDOM_SEED}...")
    rng = np.random.default_rng(RANDOM_SEED)

    # Step 1: Generate raw transactions
    print("Step 1/4: Generating transactions...")
    transactions_df = generate_transactions(rng)

    # Step 2: Aggregate into detection windows
    print("Step 2/4: Aggregating detection windows...")
    windows_df = aggregate_detection_windows(transactions_df)

    # Step 3: Export CSVs
    print("Step 3/4: Exporting CSV files...")
    export_csv(transactions_df, windows_df)

    # Step 4: Seed database
    print("Step 4/4: Seeding database...")
    seed_database(transactions_df, windows_df)

    # Summary
    train_count = len(windows_df[windows_df["split"] == "train"])
    dev_test_count = len(windows_df[windows_df["split"] == "dev_test"])
    holdout_count = len(windows_df[windows_df["split"] == "final_holdout"])
    spike_count = len(windows_df[windows_df["is_synthetic_fraud_spike"]])

    print(f"\n{'='*60}")
    print(f"Phase 3 Data Generation Complete")
    print(f"{'='*60}")
    print(f"Merchants:        {NUM_MERCHANTS}")
    print(f"Simulation:       {SIMULATION_DAYS} days ({SIM_START.date()} to {SIM_END.date()})")
    print(f"Total windows:    {len(windows_df):,}")
    print(f"  Train:          {train_count:,} (Day 1-18)")
    print(f"  Dev-test:       {dev_test_count:,} (Day 19-25)")
    print(f"  Final-holdout:  {holdout_count:,} (Day 26-30)")
    print(f"Spike windows:    {spike_count}")
    print(f"Total txns:       {len(transactions_df):,}")
    print(f"{'='*60}")

    return transactions_df, windows_df


# ===========================================================================
# CLI entry point
# ===========================================================================
if __name__ == "__main__":
    generate_all()
