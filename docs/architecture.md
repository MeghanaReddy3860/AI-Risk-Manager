# System Architecture

## Overview
The AI Risk Manager is a defense-only, fraud-spike detection system designed to monitor merchant transaction streams. It detects anomalous surges in suspicious activity, explains the reasons behind the spikes, estimates potential financial exposure, and recommends defensive actions. The system operates exclusively on synthetic data.

## System Components

### 1. Frontend (React + Vite)
- **Technologies:** React, Vite, TypeScript, Tailwind CSS, Recharts
- **Responsibilities:**
  - Provide a dashboard interface for security analysts to monitor transaction streams.
  - Visualize fraud spikes, data anomalies, and risk scores using Recharts.
  - Display AI-generated explanations and recommended defensive actions.
  - Present data in an auditable and easy-to-understand format.

### 2. Backend (FastAPI)
- **Technologies:** Python, FastAPI
- **Responsibilities:**
  - Serve as the central communication hub between the frontend, the database, and the ML engine.
  - Expose RESTful API endpoints for the dashboard to retrieve transaction data, risk scores, and alerts.
  - Orchestrate the fraud-spike detection workflow by passing data to the ML model and persisting results.
  - Enforce system constraints and ensure auditability of all actions.

### 3. Machine Learning Engine (scikit-learn)
- **Technologies:** scikit-learn (Isolation Forest), pandas, NumPy
- **Responsibilities:**
  - Ingest transaction data to establish a baseline of normal activity.
  - Run the Isolation Forest algorithm to detect anomalous surges and fraud spikes in the transaction stream.
  - Generate explanations for why a specific spike was flagged.
  - Calculate risk scores and estimate financial exposure.

### 4. Database (SQLite)
- **Technologies:** SQLite (PostgreSQL-ready)
- **Responsibilities:**
  - Store synthetic transaction data.
  - Persist detection results, risk scores, and generated explanations.
  - Maintain an audit trail of detected anomalies and recommended actions for review.

#### Database Models & Stable Feature Contract
The database architecture uses exactly three core models for Phase 2:
1. **Transaction:** Raw synthetic transaction data (`id`, `merchant_id`, `amount`, `timestamp`, `created_at`).
2. **DetectionWindow:** Aggregated activity over a defined time window + ground truth (`is_synthetic_fraud_spike`).
3. **AnomalyDetection:** Detector output for that window (`id`, `window_id`, `detector_type`, `risk_score`, `is_flagged`, `explanation`, `created_at`). AnomalyDetection is responsible for storing the results produced by the Baseline Detector and ML Anomaly Detector for individual detection windows. Aggregate evaluation metrics and their persistence are handled entirely by the Phase 6 Evaluation Engine.

**DetectionWindow Aggregate Features (Stable Data Contract):**
These exact features form a stable data contract across Phases 3–6 and Phase 15. They must not be silently renamed or redefined.
1. `transaction_count`:
   - *Exact Field Name:* `transaction_count`
   - *Calculation:* Count of transactions for a specific `merchant_id` within the time window.
   - *Purpose:* Detects frequency-based surges (e.g., card testing or bot attacks).
   - *Source fields:* Count of `Transaction.id`.
   - *Time-window:* 1-hour tumbling window.
   - *Resulting Unit/Type:* Integer count.
2. `total_amount`:
   - *Exact Field Name:* `total_amount`
   - *Calculation:* Sum of `amount` for a specific `merchant_id` within the time window.
   - *Purpose:* Detects volume-based surges (e.g., large cash-outs).
   - *Source fields:* Sum of `Transaction.amount`.
   - *Time-window:* 1-hour tumbling window.
   - *Resulting Unit/Type:* Float.
3. `avg_transaction_amount`:
   - *Exact Field Name:* `avg_transaction_amount`
   - *Calculation:* `total_amount / transaction_count` for the specific merchant and DetectionWindow.
   - *Purpose:* Detect unusual changes in the average transaction value within a time window. This provides a value-based signal in addition to overall transaction count and total monetary volume.
   - *Source fields:* `Transaction.amount`
   - *Time-window:* 1-hour tumbling window.
   - *Resulting Unit/Type:* Float.

**Phase 6 Evaluation Ownership:**
Phase 6 exclusively owns the Evaluation Engine, evaluation metrics (Precision, Recall, F1, FPR, Cost), and the `EvaluationRun` persistence schema. Phase 2 must NOT implement or define `EvaluationRun`.

#### Database Configuration (Phase 2)
Phase 2 is responsible for:
- SQLite connection configuration
- SQLAlchemy engine/session configuration
- environment-variable handling
- development database settings
- keeping the configuration PostgreSQL-ready

Database connection and environment configuration are established in Phase 2; later backend API phases consume this shared database configuration.

### 5. Data Layer (Synthetic Data)
- **Characteristics:** Completely synthetic datasets generated specifically for defensive risk detection evaluation.
- **Responsibilities:**
  - Simulate realistic merchant transaction streams without processing any real financial data.
  - Serve as the safe testing ground for evaluating the ML model's accuracy and the system's overall effectiveness.

## Data Flow & Workflow

### 1. Data Ingestion
Synthetic Transaction records are generated and stored.

### 2. DetectionWindow Aggregation
Raw Transaction records are grouped by merchant and the defined 1-hour tumbling time window.

For each DetectionWindow, calculate:
- `transaction_count`
- `total_amount`
- `avg_transaction_amount`

The DetectionWindow also contains:
`is_synthetic_fraud_spike`
as the ground-truth label for whether that aggregated window contains a known synthetic fraud-spike event.

### 3. Monitoring & Detection
The Baseline Detector and ML Anomaly Detector operate on DetectionWindow records/features rather than treating individual Transaction records as the primary detection unit.

The Baseline Detector uses the documented aggregate features.
The ML Anomaly Detector uses the DetectionWindow feature representation and Isolation Forest.

### 4. Analysis & Scoring
When a DetectionWindow is flagged, the system calculates:
- risk score
- financial exposure estimate
- explanation
- recommended defensive action

These results correspond to the DetectionWindow being analyzed.

### 5. Persistence & Audit
The detector outputs are stored as AnomalyDetection records associated with the relevant DetectionWindow.

Maintain the audit trail.

### 6. Visualization
The React frontend retrieves the DetectionWindow information and associated AnomalyDetection results through the FastAPI backend and visualizes the fraud spikes, metrics, alerts, and explanations.

## Evaluation & Metrics

### 6. Evaluation Engine
- **Technologies:** scikit-learn (metrics module), pandas
- **Responsibilities:**
  - Compute and report model performance using held-out, labeled synthetic data.
  - Quantify not just detection accuracy but the operational cost of errors.
  - Provide this evaluation as a reproducible, auditable process — not a one-time manual check.

### Ground-Truth Labeling Strategy
Since Isolation Forest is unsupervised, it does not learn from labels — but evaluation requires them.
- The fraud-spike ground-truth concept belongs completely to the `DetectionWindow`.
- The synthetic data generator must inject known, labeled fraud-spike events as a flag column `is_synthetic_fraud_spike: true/false` on the aggregated `DetectionWindow`.
- This label represents whether the aggregated window contains a known synthetic fraud-spike event.
- These labels are used only for evaluation, never fed into the model during training/inference — this preserves the unsupervised design while still enabling honest scoring.

### Train / Dev-Test / Final-Holdout Split
Synthetic transaction data is split into **three** chronological partitions (no random shuffling) to prevent data leakage and ensure honest final evaluation:

| Partition | Day Range | % of 30-day span | Purpose |
|---|---|---|---|
| **Train** | Day 1 – Day 18 | 60% | Fit the Isolation Forest "normal" baseline. |
| **Dev-test** | Day 19 – Day 25 | ~23% | Repeated use during Phases 4–14 for baseline-vs-ML comparison, threshold tuning, model iteration, and development decisions. |
| **Final-holdout** | Day 26 – Day 30 | ~17% | Untouched until Phase 15. Used only for the final held-out evaluation. |

- Chronological (not random) splitting simulates realistic deployment — the model must detect spikes in future unseen data, not interpolate within shuffled historical data.
- All three partitions contain both normal activity windows and injected fraud-spike windows.
- The split is deterministic and reproducible using `RANDOM_SEED = 42`.

**Physical file separation:**
- `data/synthetic/detection_windows.csv` — Contains only `train` and `dev_test` rows. This is the working dataset for Phases 3–14.
- `data/synthetic/detection_windows_final_holdout.csv` — Contains only `final_holdout` rows. This file must NOT be opened, inspected, loaded, or evaluated against by any code until Phase 15.

**Usage rules per partition:**
| Partition | Permitted Uses | Prohibited Uses |
|---|---|---|
| **Train** | Isolation Forest fitting; establishing "normal" baseline | — |
| **Dev-test** | Baseline-vs-ML comparison, threshold tuning, model iteration, feature evaluation, development decisions (Phases 4–14) | Final evaluation (Phase 15) |
| **Final-holdout** | Final evaluation using the Phase 6 Evaluation Engine (Phase 15 only) | Training, threshold selection, feature selection, tuning, baseline-vs-ML comparison, architecture decisions, model-version selection — all prohibited during Phases 4–14 |

Do not weaken or reinterpret this requirement in later phases. The purpose of this three-way split is to guarantee that Phase 15 provides an honest estimate of final performance without data leakage from Phases 4–14.

### Metrics Reported
For every evaluation run, the system computes and logs:
| Metric | Purpose |
|---|---|
| **Precision** | Of all flagged spikes, what fraction were real fraud spikes (per ground-truth labels)? |
| **Recall** | Of all actual fraud spikes in the dev-test (or final-holdout) set, what fraction did the model catch? |
| **F1 Score** | Harmonic mean of precision/recall, for a single-number comparison across model versions. |
| **False Positive Rate** | Fraction of normal activity incorrectly flagged as a spike. |
| **False Positive Cost** | Estimated real-world cost of false positives — e.g., FP_count × avg_analyst_review_time_cost, or FP_count × avg_merchant_friction_cost. This must be reported alongside accuracy metrics, not instead of them. |

### Where Evaluation Runs
- Evaluation is implemented as a standalone, repeatable script/module (e.g., `backend/evaluation/run_eval.py`), separate from the live detection pipeline.
- It is run whenever the model is retrained or the synthetic dataset changes, and results are persisted to the database (or a versioned report file) for audit purposes.
- The dashboard may optionally surface the latest evaluation metrics to analysts for transparency, but evaluation itself is a backend/ML concern, not a live frontend computation.

### Baseline Comparison
The Evaluation Engine reports metrics for BOTH the Baseline Detector (Phase 4) and the ML Anomaly Detector (Phase 5), covering at minimum:
- Precision
- Recall
- F1 Score
- False Positive Rate
- False Positive Cost

**Phase 4 — Baseline Detector:** A simple, interpretable statistical comparison model (e.g., rolling mean plus N standard deviations, applied to appropriate time-window features). It is intentionally simple, serves as a naive comparison point, and must NOT use the ML model's predictions. It must be evaluated using the same held-out evaluation framework.

**Phase 5 — ML Anomaly Detector:** Uses the Isolation Forest approach. It is evaluated against the same test conditions as the baseline detector. The purpose is to determine whether the ML approach provides measurable value beyond the simpler baseline.

The ML detector's value must be demonstrated through honest measured comparison with the baseline, not assumed simply because it uses machine learning. It is not required to outperform the baseline on every metric. Evaluation reports honestly where the ML detector improves, matches, or performs worse. Evaluation methodology must not change after seeing results in order to favor either detector.

### Final Held-Out Evaluation
Phase 6 builds the reusable evaluation infrastructure (metric computation, baseline-vs-ML comparison, false-positive cost calculation, reproducible reports, and persistence).

Phase 15 is the final execution of this already-built evaluation infrastructure against the final-holdout partition loaded from `data/synthetic/detection_windows_final_holdout.csv`. Phase 15 is NOT a second implementation of the evaluation engine. Phase 15 must use the reusable Evaluation Engine created in Phase 6.

The final-holdout partition is created and physically separated during Phase 3 (not derived later from the dev-test set). It must:
- remain completely unseen during Phases 4–14
- not be used for model tuning
- not be used for threshold selection
- not be used for feature selection
- not be used to guide architecture or design decisions
- not be used to repeatedly compare model versions during development
- not be loaded or inspected by any Phase 4–14 code

This three-way physical separation — established at data generation time — prevents data leakage and provides a more honest estimate of final performance.

### Defense-Only Enforcement Note
To ensure the system remains strictly defense-only and non-disqualifying:
- The system only flags and explains — it never auto-blocks, auto-bans, or takes irreversible action on merchants/accounts without human review.
- Detection thresholds and model internals are not exposed via any public-facing API, to avoid enabling adversarial evasion.

## Security & Constraints
- **Defense-Only:** The system is explicitly designed for defensive purposes. No offensive capabilities are implemented.
- **Synthetic Data Strictly:** No real merchant data or real payments are ever processed. All data used is clearly labeled as synthetic.
- **Auditability:** All metrics and anomaly explanations are honestly computed and persistently logged for review.
