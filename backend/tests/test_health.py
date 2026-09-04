"""
Tests for the health check endpoint.

Validates that:
1. The endpoint returns HTTP 200.
2. The response contains expected fields.
3. The status is 'healthy'.
4. The synthetic data label is present.
"""


def test_health_check_returns_200(client):
    """Health endpoint should return 200 OK."""
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_check_status_is_healthy(client):
    """Health endpoint should report 'healthy' status."""
    response = client.get("/api/health")
    data = response.json()
    assert data["status"] == "healthy"


def test_health_check_has_required_fields(client):
    """Health endpoint should include all expected fields."""
    response = client.get("/api/health")
    data = response.json()

    required_fields = ["status", "service", "version", "timestamp", "data_mode"]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_health_check_service_name(client):
    """Health endpoint should return correct service name."""
    response = client.get("/api/health")
    data = response.json()
    assert data["service"] == "ai-risk-manager"


def test_health_check_synthetic_data_label(client):
    """Health endpoint must clearly indicate synthetic data usage."""
    response = client.get("/api/health")
    data = response.json()
    assert "SYNTHETIC" in data["data_mode"]
