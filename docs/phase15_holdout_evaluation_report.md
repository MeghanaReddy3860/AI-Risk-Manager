# Phase 15 — Final Held-Out Evaluation Report

> **SYNTHETIC / TEST DATA ONLY** — This project uses entirely synthetic data for defensive risk detection evaluation. It does not process real merchant data or real payments.
>
> **DEFENSE-ONLY SYSTEM** — No automated blocking, merchant banning, payment freezing, or punitive action is performed.

---

## 1. Purpose

This document presents the formal evaluation results for **Phase 15 — Final Controlled Held-Out Evaluation** of the AI Risk Manager. The evaluation compares the statistical **Baseline Detector** (Phase 4) against the **ML Anomaly Detector (Isolation Forest)** (Phase 5) across both the development benchmark dataset (`dev_test`) and the previously unseen, protected held-out dataset (`final_holdout`).

---

## 2. Dataset Partition Definition

The synthetic dataset consists of 30 days of 1-hour tumbling DetectionWindow aggregates across 10 merchants, deterministically generated with `RANDOM_SEED = 42`.

| Partition | Time Range | Span | Purpose | Status in Phase 15 |
|---|---|---|---|---|
| **Train** | Day 1 – Day 18 | 60% | Detector fitting (mean/std & Isolation Forest) | Used ONLY for fitting |
| **Dev-test** | Day 19 – Day 25 | ~23% | Benchmark comparison & threshold iteration (Phases 4–14) | Evaluated |
| **Final-holdout** | Day 26 – Day 30 | ~17% | **Protected held-out evaluation** (Phase 15 only) | **Evaluated (1,200 windows)** |

---

## 3. Holdout Protection & Data Leakage Verification

- **Fitting Separation**: Both detectors were fitted **EXCLUSIVELY** on the `train` partition (Day 1–Day 18).
- **Zero Exposure**: `final_holdout` data was **NEVER** accessed, loaded, or inspected during model fitting, feature calculation, hyperparameter selection, or threshold tuning.
- **Controlled Access**: Access to `final_holdout` for prediction was explicitly gated via `allow_holdout=True` in the Phase 15 orchestrator.
- **Data Leakage Check**: **PASSED**. Detector parameters and z-score thresholds on `final_holdout` are 100% identical to those used on `dev_test`.

---

## 4. Evaluation Results Summary

### Side-by-Side Comparison: Dev-Test vs. Final-Holdout

| Detector | Metric | Dev-test (Day 19–25) | Final-holdout (Day 26–30) | Delta |
|---|---|---|---|---|
| **Baseline Detector** | **Precision** | 0.0268 (2.7%) | 0.0303 (3.0%) | +0.3% |
| | **Recall** | 1.0000 (100.0%) | 1.0000 (100.0%) | +0.0% |
| | **F1 Score** | 0.0523 | 0.0588 | +0.0065 |
| | **False Positive Rate (FPR)** | 0.0865 (8.7%) | 0.0802 (8.0%) | -0.6% |
| | **Total Operational Cost** | ₹72,500.00 | ₹48,000.00 | — |
| **ML Isolation Forest** | **Precision** | 0.0204 (2.0%) | 0.0236 (2.4%) | +0.3% |
| | **Recall** | 1.0000 (100.0%) | 1.0000 (100.0%) | +0.0% |
| | **F1 Score** | 0.0400 | 0.0462 | +0.0062 |
| | **False Positive Rate (FPR)** | 0.1146 (11.5%) | 0.1036 (10.4%) | -1.1% |
| | **Total Operational Cost** | ₹96,000.00 | ₹62,000.00 | — |

---

## 5. Detailed Final-Holdout Confusion Matrices & Costs

### Baseline Detector (`final_holdout`)
- **True Positives (TP)**: 3
- **False Positives (FP)**: 96
- **False Negatives (FN)**: 0
- **True Negatives (TN)**: 1101
- **FP Cost (₹500 / review)**: ₹48,000.00
- **FN Cost (₹15,000 / missed spike)**: ₹0.00
- **Total Operational Cost**: ₹48,000.00

### ML Isolation Forest Detector (`final_holdout`)
- **True Positives (TP)**: 3
- **False Positives (FP)**: 124
- **False Negatives (FN)**: 0
- **True Negatives (TN)**: 1073
- **FP Cost (₹500 / review)**: ₹62,000.00
- **FN Cost (₹15,000 / missed spike)**: ₹0.00
- **Total Operational Cost**: ₹62,000.00

---

## 6. Performance Interpretation & Analysis

1. **Statistical Baseline Efficacy**:
   - The statistical Baseline Detector achieved a lower total operational cost (₹48,000.00 vs ₹62,000.00).
   - Simple per-merchant feature z-scores provided cleaner separation for this synthetic transaction stream with fewer false positive reviews.
2. **Cost-Sensitivity**:
   - Because the financial penalty for a missed fraud spike (False Negative = ₹15,000) far outweighs manual review costs (False Positive = ₹500), high recall is critical to minimizing financial exposure.

---

## 7. Audit & Defense-Only Verification

- **Evaluation Execution Timestamp**: `2026-08-31 12:41:21 UTC`
- **Audit Cryptographic Hash**: `469aa8610764d1f5606c61d3bbe390878e62778db62314cc25bf8c7cc944d5c6`
- **Defense-Only Compliance**:
  - No automated punitive actions (blocking, banning, rate-limiting, or funds freezing) were configured or triggered.
  - Results are strictly persisted to `EvaluationRun` for analyst review and governance auditing.

---

## 8. Final Conclusion: Baseline Detector Outperformed ML Anomaly Detector

Based on the final held-out evaluation metrics, the statistical Baseline Detector outperformed the ML Anomaly Detector on this dataset, achieving lower total operational cost (₹48,000.00 vs ₹62,000.00) and stronger relevant detection metrics.
