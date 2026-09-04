"""
AI Risk Manager — Phase 11: Backend API Test Suite
===================================================

Comprehensive integration tests for all Phase 11 FastAPI REST endpoints:
  - Health Endpoint
  - Merchant APIs
  - Detection Window APIs
  - Timeseries API
  - End-to-End Pipeline API
  - Evaluation APIs
  - Audit Trail APIs
  - Analyst Action API
  - Security & Credential Protection
  - Defense-Only & Final-Holdout Protection
"""

import datetime
import pytest
from main import app
from models import DetectionWindow, AnomalyDetection
from audit_trail import AuditTrailManager


@pytest.fixture(autouse=True)
def setup_test_app_state():
    """Ensure shared AuditTrailManager is attached for tests."""
    app.state.audit_manager = AuditTrailManager()


@pytest.fixture
def sample_window(db_session):
    """Seed a sample DetectionWindow in the test database with detection results."""
    now = datetime.datetime.now(datetime.timezone.utc)
    win = DetectionWindow(
        merchant_id="merchant_001",
        window_start=now - datetime.timedelta(hours=1),
        window_end=now,
        transaction_count=25,
        total_amount=5000.0,
        avg_transaction_amount=200.0,
        is_synthetic_fraud_spike=False,
        split="dev_test",
    )
    db_session.add(win)
    db_session.commit()
    db_session.refresh(win)

    # Seed stored detection results for sample_window
    det_base = AnomalyDetection(
        window_id=win.id,
        detector_type="baseline",
        risk_score=45.0,
        is_flagged=False,
        explanation="Baseline detector: normal activity.",
    )
    det_ml = AnomalyDetection(
        window_id=win.id,
        detector_type="ml",
        risk_score=52.0,
        is_flagged=False,
        explanation="Isolation Forest detector: normal activity.",
    )
    db_session.add(det_base)
    db_session.add(det_ml)
    db_session.commit()
    return win


@pytest.fixture
def holdout_window(db_session):
    """Seed a final_holdout DetectionWindow in the test database."""
    now = datetime.datetime.now(datetime.timezone.utc)
    win = DetectionWindow(
        merchant_id="merchant_001",
        window_start=now - datetime.timedelta(hours=1),
        window_end=now,
        transaction_count=100,
        total_amount=50000.0,
        avg_transaction_amount=500.0,
        is_synthetic_fraud_spike=True,
        split="final_holdout",
    )
    db_session.add(win)
    db_session.commit()
    db_session.refresh(win)
    return win


# ===========================================================================
# 1. Health & Merchant APIs
# ===========================================================================

def test_health_check_endpoint(client):
    """Verify GET /api/health returns 200 OK with expected status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "SYNTHETIC" in data["data_mode"]


def test_list_merchants_endpoint(client, sample_window):
    """Verify GET /api/merchants returns array of merchant summaries."""
    response = client.get("/api/merchants")
    assert response.status_code == 200
    merchants = response.json()
    assert isinstance(merchants, list)
    assert len(merchants) >= 1
    m = merchants[0]
    assert m["merchant_id"] == "merchant_001"
    assert m["total_windows"] == 1


# ===========================================================================
# 2. Window APIs & Holdout Protection
# ===========================================================================

def test_list_windows_default_pagination(client, sample_window):
    """Verify GET /api/windows default limit=50 and offset=0."""
    response = client.get("/api/windows")
    assert response.status_code == 200
    windows = response.json()
    assert isinstance(windows, list)
    assert len(windows) >= 1


def test_list_windows_limit_exceeded_rejected(client):
    """Verify limit > 500 is rejected with HTTP 400 Bad Request."""
    response = client.get("/api/windows?limit=501")
    assert response.status_code == 400
    assert "limit cannot exceed 500" in response.json()["detail"]


def test_list_windows_negative_offset_rejected(client):
    """Verify negative offset is rejected with HTTP 400 Bad Request."""
    response = client.get("/api/windows?offset=-5")
    assert response.status_code == 400
    assert "offset must be greater than or equal to 0" in response.json()["detail"]


def test_list_windows_invalid_split_rejected(client):
    """Verify invalid split (e.g. 'final_holdout' or 'invalid') returns HTTP 400."""
    for bad_split in ["final_holdout", "holdout", "production", "unknown_split"]:
        response = client.get(f"/api/windows?split={bad_split}")
        assert response.status_code == 400
        assert "Invalid split parameter" in response.json()["detail"]


def test_list_windows_valid_split_filtering(client, sample_window):
    """Verify valid split filters ('train', 'dev_test') return matching windows."""
    response = client.get("/api/windows?split=dev_test")
    assert response.status_code == 200
    windows = response.json()
    assert len(windows) >= 1
    assert windows[0]["split"] == "dev_test"


def test_get_single_window_success(client, sample_window):
    """Verify GET /api/windows/{id} returns single window details."""
    response = client.get(f"/api/windows/{sample_window.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == sample_window.id
    assert data["merchant_id"] == "merchant_001"


def test_get_single_window_not_found(client):
    """Verify requesting non-existent window_id returns HTTP 404."""
    response = client.get("/api/windows/999999")
    assert response.status_code == 404


def test_get_single_window_final_holdout_protected(client, holdout_window):
    """Verify requesting a final_holdout window ID returns HTTP 404."""
    response = client.get(f"/api/windows/{holdout_window.id}")
    assert response.status_code == 404


def test_get_merchant_timeseries_valid(client, sample_window):
    """Verify timeseries endpoint returns chronological points for a merchant."""
    response = client.get(f"/api/merchants/{sample_window.merchant_id}/timeseries")
    assert response.status_code == 200
    points = response.json()
    assert isinstance(points, list)
    assert len(points) >= 1
    assert "timestamp" in points[0]


def test_get_merchant_timeseries_unknown(client):
    """Verify unknown merchant_id returns HTTP 404."""
    response = client.get("/api/merchants/unknown_merchant_999/timeseries")
    assert response.status_code == 404


# ===========================================================================
# 3. Pipeline API & Holdout Protection
# ===========================================================================

def test_pipeline_analyze_window_success(client, sample_window):
    """Verify POST /api/pipeline/analyze-window/{id} runs pipeline on valid window."""
    response = client.post(f"/api/pipeline/analyze-window/{sample_window.id}?detector_type=baseline")
    assert response.status_code == 200
    dossier = response.json()
    assert dossier["window"]["id"] == sample_window.id
    assert dossier["detector_type"] == "baseline"
    assert "risk_score" in dossier["risk_result"]
    assert "summary" in dossier["explanation"]
    assert "policy_id" in dossier["policy_decision"]
    assert dossier["audit_entry_id"] is not None


def test_use_llm_parameter_does_not_affect_policy_decision(client, sample_window):
    """Verify toggling use_llm=false vs use_llm=true does NOT alter policy decisions."""
    r_rule = client.post(f"/api/pipeline/analyze-window/{sample_window.id}?detector_type=baseline&use_llm=false")
    r_llm = client.post(f"/api/pipeline/analyze-window/{sample_window.id}?detector_type=baseline&use_llm=true")

    assert r_rule.status_code == 200
    assert r_llm.status_code == 200

    d_rule = r_rule.json()["policy_decision"]
    d_llm = r_llm.json()["policy_decision"]

    assert d_rule["policy_id"] == d_llm["policy_id"]
    assert d_rule["action_type"] == d_llm["action_type"]
    assert d_rule["priority"] == d_llm["priority"]
    assert d_rule["review_sla_hours"] == d_llm["review_sla_hours"]
    assert d_rule["require_dual_review"] == d_llm["require_dual_review"]
    assert d_rule["routing_tags"] == d_llm["routing_tags"]
    assert d_rule["triggered_rules"] == d_llm["triggered_rules"]


def test_pipeline_analyze_window_holdout_protected(client, holdout_window):
    """Verify pipeline rejects final_holdout window with HTTP 404 before execution."""
    response = client.post(f"/api/pipeline/analyze-window/{holdout_window.id}")
    assert response.status_code == 404


def test_pipeline_analyze_window_invalid_detector(client):
    """Verify invalid detector_type returns HTTP 400."""
    response = client.post("/api/pipeline/analyze-window/1?detector_type=invalid_det")
    assert response.status_code == 400


def test_custom_policy_injection_not_exposed(client, sample_window):
    """Verify API endpoints do not accept client-injected custom policies or rule definitions."""
    # Attempting to post unapproved custom_policy payload to analyst action
    payload = {
        "actor": "ANALYST:attacker",
        "window_id": sample_window.id,
        "disposition": "escalate",
        "custom_policy": {"policy_id": "POL_INJECTED", "action_type": "AUTO_BLOCK"},
    }
    response = client.post("/api/analyst/action", json=payload)
    # Extra field is ignored or validated, but disposition remains escalate and no AUTO_BLOCK is stored
    assert response.status_code in (200, 400, 422)
    if response.status_code == 200:
        # Verify recorded audit record does not contain custom AUTO_BLOCK
        audit_rep = client.get(f"/api/audit/window/{sample_window.id}").json()
        for ev in audit_rep["events"]:
            assert ev["payload"].get("action_type") != "AUTO_BLOCK"


# ===========================================================================
# 4. Evaluation APIs & Holdout Protection
# ===========================================================================

def test_get_latest_evaluations(client):
    """Verify GET /api/evaluation/latest returns array."""
    response = client.get("/api/evaluation/latest")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_trigger_evaluation_invalid_partition(client):
    """Verify triggering evaluation with 'final_holdout' returns HTTP 400."""
    for bad_part in ["final_holdout", "holdout", "invalid"]:
        response = client.post(f"/api/evaluation/run?partition={bad_part}")
        assert response.status_code == 400
        assert "Invalid partition" in response.json()["detail"]


# ===========================================================================
# 5. Audit & Analyst Action APIs (Defense-Only & Credential Checks)
# ===========================================================================

def test_verify_audit_trail_integrity_endpoint(client):
    """Verify GET /api/audit/verify returns integrity status."""
    response = client.get("/api/audit/verify")
    assert response.status_code == 200
    data = response.json()
    assert "integrity_valid" in data
    assert "integrity_errors" in data
    assert data["integrity_valid"] is True


def test_analyst_action_valid(client, sample_window):
    """Verify POST /api/analyst/action logs permitted defensive review action."""
    payload = {
        "actor": "ANALYST:analyst_01",
        "window_id": sample_window.id,
        "disposition": "escalate",
        "notes": "Escalating for dual analyst review per P1 policy.",
    }
    response = client.post("/api/analyst/action", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "entry_id" in res

    # Verify audit event preserves exact caller-supplied actor
    report = client.get(f"/api/audit/window/{sample_window.id}").json()
    action_events = [e for e in report["events"] if e["event_type"] == "ANALYST_ACTION"]
    assert len(action_events) >= 1
    assert action_events[-1]["actor"] == "ANALYST:analyst_01"
    payload = {
        "actor": "ANALYST:test_user",
        "window_id": sample_window.id,
        "disposition": "escalate",
        "notes": "Escalating for dual analyst review per P1 policy.",
    }
    response = client.post("/api/analyst/action", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "entry_id" in res


def test_analyst_action_destructive_rejected(client):
    """Verify destructive action terms (BAN, BLOCK, SUSPEND, TERMINATE) return HTTP 400."""
    for bad_action in ["ban", "block", "suspend", "terminate", "auto_block"]:
        payload = {
            "actor": "ANALYST:test_user",
            "window_id": 1,
            "disposition": bad_action,
            "notes": "Attempting automated enforcement",
        }
        response = client.post("/api/analyst/action", json=payload)
        assert response.status_code == 400
        assert "prohibited" in response.json()["detail"] or "Invalid disposition" in response.json()["detail"]


def test_analyst_action_credential_in_notes_rejected(client):
    """Verify notes containing sensitive key patterns (e.g. api_key) return HTTP 400."""
    payload = {
        "actor": "ANALYST:test_user",
        "window_id": 1,
        "disposition": "monitor",
        "notes": "Here is the api_key: secret_12345",
    }
    response = client.post("/api/analyst/action", json=payload)
    assert response.status_code == 400
    assert "Credentials must never be logged" in response.json()["detail"]


def test_no_credential_leakage_in_responses(client):
    """Verify responses never leak OPENAI_API_KEY or secret authorization tokens."""
    routes_to_test = [
        "/api/health",
        "/api/merchants",
        "/api/windows",
        "/api/evaluation/latest",
        "/api/audit/verify",
    ]
    for route in routes_to_test:
        res = client.get(route)
        res_str = str(res.json())
        assert "OPENAI_API_KEY" not in res_str
        assert "sk_live" not in res_str
        assert "sk_test" not in res_str
        assert "password" not in res_str
