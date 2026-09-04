# AI Risk Manager — Fraud-Spike Detector

> **SYNTHETIC / TEST DATA ONLY** — This project uses entirely synthetic data for defensive risk detection evaluation. It does not process real merchant data or real payments.

## Overview

An AI-powered fraud-spike detection system that monitors merchant transaction streams, detects anomalous surges in suspicious activity, explains why spikes occur, estimates financial exposure, and recommends defensive actions.

**This system is strictly defense-only.**

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, TypeScript, Tailwind CSS, Recharts |
| Backend | Python, FastAPI |
| ML | scikit-learn (Isolation Forest), pandas, NumPy |
| Database | SQLite (PostgreSQL-ready) |
| Testing | pytest, API tests |

## Project Structure

```
ai-risk-manager/
├── backend/          # FastAPI backend + ML engine
├── frontend/         # React dashboard
├── data/synthetic/   # Generated synthetic datasets
└── docs/             # Documentation
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Run Tests

```bash
cd backend
pytest -v
```

## Development Phases

- [x] Phase 0 — Requirements & Architecture
- [x] Phase 1 — Project Scaffold
- [x] Phase 2 — Database
- [x] Phase 3 — Synthetic Data Generator
- [x] Phase 4 — Baseline Detector
- [x] Phase 5 — ML Anomaly Detector
- [x] Phase 6 — Evaluation Engine
- [x] Phase 7 — Risk Scoring
- [x] Phase 8 — AI Explanation
- [x] Phase 9 — Policy Engine
- [x] Phase 10 — Audit Trail
- [x] Phase 11 — Backend API
- [x] Phase 12 — Frontend Dashboard (Stream Monitor, Deep Dive, Benchmarks, Audit Trail)
- [x] Phase 13 — End-to-End System Testing & Integration
- [x] Phase 14 — Security, Defense-Only & Read-Only Audit
- [x] Phase 15 — Final Controlled Held-Out Evaluation (final_holdout partition protected, evaluated strictly in Phase 15)
- [x] Phase 16 — Internship Demo Submission Readiness (412 backend tests passing, frontend built & linted, defense-only verified)

## License

This project is for educational / internship demonstration purposes.

## Important

- All data is synthetic and clearly labeled.
- No real financial transactions are processed.
- No offensive fraud capabilities are implemented.
- All metrics are honestly computed, never fabricated.
