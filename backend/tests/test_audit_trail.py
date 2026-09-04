"""
AI Risk Manager — Phase 10: Audit Trail Engine Tests
====================================================

Comprehensive test suite for backend/audit_trail.py.

Covers all 28 required test areas:
  1. Event type definitions
  2. First event creation
  3. Multiple event creation
  4. Hash chaining
  5. Deterministic hashing
  6. Dictionary-order-independent hashing
  7. Immutable AuditRecord
  8. Protected nested payload
  9. Window filtering
  10. Merchant filtering
  11. Date filtering
  12. Empty result behavior
  13. Integrity verification success
  14. Payload tamper detection
  15. Hash tamper detection
  16. Previous-hash tamper detection
  17. Field tamper detection
  18. Middle-record deletion detection
  19. Invalid event type
  20. Invalid identifiers
  21. Non-serializable payload
  22. Credential leakage prevention
  23. Export report structure
  24. Export report JSON serializability
  25. Defense-only behavior
  26. Analyst action logging
  27. No final-holdout dependency
  28. Append-only behavior
"""

import copy
import datetime
import inspect
import json
import pytest

from audit_trail import (
    AuditEventType,
    AuditRecord,
    AuditTrailManager,
    VALID_EVENT_TYPES,
    compute_integrity_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def audit_manager():
    return AuditTrailManager()


# ===========================================================================
# Tests 1-7: Core Event Creation, Chaining & Immutability
# ===========================================================================

def test_1_event_type_definitions():
    """Verify AuditEventType contains exactly the approved lifecycle events."""
    expected = {
        "WINDOW_CREATED",
        "DETECTION_FLAGGED",
        "RISK_EVALUATED",
        "EXPLANATION_GENERATED",
        "POLICY_DECISION",
        "ANALYST_ACTION",
    }
    assert set(VALID_EVENT_TYPES) == expected
    for event_name in expected:
        assert getattr(AuditEventType, event_name).value == event_name


def test_2_first_event_creation(audit_manager):
    """First logged event must have previous_hash=None and valid integrity_hash."""
    record = audit_manager.log_event(
        event_type=AuditEventType.WINDOW_CREATED,
        window_id=101,
        merchant_id="merchant_001",
        actor="SYSTEM:DATA_INGEST",
        payload={"transaction_count": 42, "total_amount": 95000.0},
    )
    assert record.entry_id is not None
    assert record.previous_hash is None
    assert isinstance(record.integrity_hash, str)
    assert len(record.integrity_hash) == 64  # SHA-256 hex length
    assert record.event_type == "WINDOW_CREATED"
    assert record.window_id == "101"
    assert record.merchant_id == "merchant_001"
    assert record.actor == "SYSTEM:DATA_INGEST"


def test_3_multiple_event_creation(audit_manager):
    """Logging multiple events creates a sequential record list."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "m_1", "SYSTEM:INGEST", {})
    audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 101, "m_1", "SYSTEM:DETECTOR", {"risk_score": 75.0})
    audit_manager.log_event(AuditEventType.POLICY_DECISION, 101, "m_1", "SYSTEM:POLICY", {"policy_id": "POL_HIGH_EXPOSURE"})

    assert len(audit_manager) == 3


def test_4_hash_chaining(audit_manager):
    """Each subsequent record must point to the preceding record's integrity_hash."""
    r1 = audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "m_1", "ACTOR_1", {"step": 1})
    r2 = audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 101, "m_1", "ACTOR_2", {"step": 2})
    r3 = audit_manager.log_event(AuditEventType.RISK_EVALUATED, 101, "m_1", "ACTOR_3", {"step": 3})

    assert r1.previous_hash is None
    assert r2.previous_hash == r1.integrity_hash
    assert r3.previous_hash == r2.integrity_hash


def test_5_deterministic_hashing():
    """Identical event fields and timestamps produce byte-identical hashes."""
    ts = "2026-08-29T18:00:00.000000Z"
    h1 = compute_integrity_hash(ts, "DETECTION_FLAGGED", "101", "m_1", {"score": 80.0}, "prev_hash_123")
    h2 = compute_integrity_hash(ts, "DETECTION_FLAGGED", "101", "m_1", {"score": 80.0}, "prev_hash_123")
    assert h1 == h2


def test_6_dictionary_order_independent_hashing():
    """Different dictionary key insertion orders produce identical canonical hashes."""
    ts = "2026-08-29T18:00:00.000000Z"
    payload_a = {"alpha": 1, "beta": 2, "nested": {"x": 10, "y": 20}}
    payload_b = {"beta": 2, "nested": {"y": 20, "x": 10}, "alpha": 1}

    h1 = compute_integrity_hash(ts, "POLICY_DECISION", "101", "m_1", payload_a, None)
    h2 = compute_integrity_hash(ts, "POLICY_DECISION", "101", "m_1", payload_b, None)
    assert h1 == h2


def test_7_immutable_audit_record(audit_manager):
    """AuditRecord is frozen and prevents attribute mutation for all fields."""
    record = audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "m_1", "ACTOR", {"key": "val"})

    with pytest.raises(Exception):
        record.actor = "NEW_ACTOR"

    with pytest.raises(Exception):
        record.payload = {"new": "payload"}

    with pytest.raises(Exception):
        record.integrity_hash = "new_hash"

    with pytest.raises(Exception):
        record.previous_hash = "new_prev_hash"

    with pytest.raises(Exception):
        record.event_type = "NEW_EVENT"

    with pytest.raises(Exception):
        record.window_id = 999

    with pytest.raises(Exception):
        record.merchant_id = "m_forged"

    with pytest.raises(Exception):
        record.timestamp = "2026-01-01T00:00:00Z"

    with pytest.raises(Exception):
        record.entry_id = "new_id"


# ===========================================================================
# Tests 8-12: Payload Protection & Query Filtering
# ===========================================================================

def test_8_protected_nested_payload(audit_manager):
    """Mutating the payload dictionary after logging must not mutate stored record."""
    input_payload = {"risk_score": 50.0, "details": {"tag": "initial"}}
    record = audit_manager.log_event(AuditEventType.RISK_EVALUATED, 101, "m_1", "ACTOR", input_payload)

    # Mutate original input dict
    input_payload["risk_score"] = 999.0
    input_payload["details"]["tag"] = "corrupted"

    # Query stored record and verify it remained untouched
    stored = audit_manager.get_trail_for_window(101)[0]
    assert stored.payload["risk_score"] == 50.0
    assert stored.payload["details"]["tag"] == "initial"


def test_9_window_filtering(audit_manager):
    """get_trail_for_window returns only events for the requested window."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "m_1", "ACTOR", {"w": 101})
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 102, "m_2", "ACTOR", {"w": 102})
    audit_manager.log_event(AuditEventType.POLICY_DECISION, 101, "m_1", "ACTOR", {"w": 101})

    trail_101 = audit_manager.get_trail_for_window(101)
    trail_102 = audit_manager.get_trail_for_window(102)

    assert len(trail_101) == 2
    assert len(trail_102) == 1
    assert all(r.window_id == "101" for r in trail_101)
    assert all(r.window_id == "102" for r in trail_102)


def test_10_merchant_filtering(audit_manager):
    """get_trail_for_merchant returns all events across windows for a merchant."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "merchant_A", "ACTOR", {})
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 102, "merchant_B", "ACTOR", {})
    audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 103, "merchant_A", "ACTOR", {})

    trail_a = audit_manager.get_trail_for_merchant("merchant_A")
    assert len(trail_a) == 2
    assert all(r.merchant_id == "merchant_A" for r in trail_a)


def test_11_date_filtering(audit_manager):
    """get_trail_for_merchant respects start_time and end_time datetime boundaries."""
    t1 = datetime.datetime(2026, 8, 1, 10, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.datetime(2026, 8, 1, 12, 0, tzinfo=datetime.timezone.utc)
    t3 = datetime.datetime(2026, 8, 1, 14, 0, tzinfo=datetime.timezone.utc)

    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 1, "m_1", "ACTOR", {}, timestamp=t1)
    audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 2, "m_1", "ACTOR", {}, timestamp=t2)
    audit_manager.log_event(AuditEventType.RISK_EVALUATED, 3, "m_1", "ACTOR", {}, timestamp=t3)

    # Filter between 11:00 and 13:00 -> should only match event 2 (12:00)
    filter_start = datetime.datetime(2026, 8, 1, 11, 0, tzinfo=datetime.timezone.utc)
    filter_end = datetime.datetime(2026, 8, 1, 13, 0, tzinfo=datetime.timezone.utc)
    filtered = audit_manager.get_trail_for_merchant("m_1", start_time=filter_start, end_time=filter_end)

    assert len(filtered) == 1
    assert filtered[0].window_id == "2"


def test_12_empty_result_behavior(audit_manager):
    """Querying unknown window or merchant returns an empty list without error."""
    assert audit_manager.get_trail_for_window("non_existent_window") == []
    assert audit_manager.get_trail_for_merchant("non_existent_merchant") == []


# ===========================================================================
# Tests 13-18: Integrity Verification & Tamper Detection
# ===========================================================================

def test_13_integrity_verification_success(audit_manager):
    """Untampered audit trail verifies successfully."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 1, "m_1", "A1", {"data": 1})
    audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 1, "m_1", "A2", {"data": 2})
    audit_manager.log_event(AuditEventType.POLICY_DECISION, 1, "m_1", "A3", {"data": 3})

    is_valid, errors = audit_manager.verify_integrity()
    assert is_valid is True
    assert errors == []


def test_14_payload_tamper_detection(audit_manager):
    """Tampering with an internal record payload breaks integrity verification."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 1, "m_1", "A1", {"risk_score": 10.0})
    audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 1, "m_1", "A2", {"risk_score": 50.0})

    # Simulate internal storage tampering
    orig = audit_manager._records[0]
    tampered_payload = {"risk_score": 99.9}
    tampered_record = AuditRecord(
        entry_id=orig.entry_id,
        timestamp=orig.timestamp,
        event_type=orig.event_type,
        window_id=orig.window_id,
        merchant_id=orig.merchant_id,
        actor=orig.actor,
        payload=tampered_payload,
        previous_hash=orig.previous_hash,
        integrity_hash=orig.integrity_hash,
    )
    audit_manager._records[0] = tampered_record

    is_valid, errors = audit_manager.verify_integrity()
    assert is_valid is False
    assert any("Hash mismatch" in err for err in errors)


def test_15_hash_tamper_detection(audit_manager):
    """Tampering with stored integrity_hash breaks integrity verification."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 1, "m_1", "A1", {})
    orig = audit_manager._records[0]
    tampered_record = AuditRecord(
        entry_id=orig.entry_id,
        timestamp=orig.timestamp,
        event_type=orig.event_type,
        window_id=orig.window_id,
        merchant_id=orig.merchant_id,
        actor=orig.actor,
        payload=orig.payload,
        previous_hash=orig.previous_hash,
        integrity_hash="0000000000000000000000000000000000000000000000000000000000000000",
    )
    audit_manager._records[0] = tampered_record

    is_valid, errors = audit_manager.verify_integrity()
    assert is_valid is False
    assert any("Hash mismatch" in err for err in errors)


def test_16_previous_hash_tamper_detection(audit_manager):
    """Tampering with previous_hash pointer breaks chain continuity."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 1, "m_1", "A1", {})
    audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 1, "m_1", "A2", {})

    orig = audit_manager._records[1]
    tampered_record = AuditRecord(
        entry_id=orig.entry_id,
        timestamp=orig.timestamp,
        event_type=orig.event_type,
        window_id=orig.window_id,
        merchant_id=orig.merchant_id,
        actor=orig.actor,
        payload=orig.payload,
        previous_hash="fake_previous_hash",
        integrity_hash=orig.integrity_hash,
    )
    audit_manager._records[1] = tampered_record

    is_valid, errors = audit_manager.verify_integrity()
    assert is_valid is False
    assert any("Broken hash chain" in err for err in errors)


def test_17_field_tamper_detection(audit_manager):
    """Tampering with merchant_id, window_id, event_type, or timestamp is detected."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 1, "merchant_001", "A1", {})
    orig = audit_manager._records[0]

    tampered_record = AuditRecord(
        entry_id=orig.entry_id,
        timestamp=orig.timestamp,
        event_type=orig.event_type,
        window_id=orig.window_id,
        merchant_id="merchant_FORGED",  # Altered field
        actor=orig.actor,
        payload=orig.payload,
        previous_hash=orig.previous_hash,
        integrity_hash=orig.integrity_hash,
    )
    audit_manager._records[0] = tampered_record

    is_valid, errors = audit_manager.verify_integrity()
    assert is_valid is False
    assert any("Hash mismatch" in err for err in errors)


def test_18_middle_record_deletion_detection(audit_manager):
    """Simulating deletion of a middle record is detected as a broken hash chain."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 1, "m_1", "A1", {})
    audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 1, "m_1", "A2", {})
    audit_manager.log_event(AuditEventType.POLICY_DECISION, 1, "m_1", "A3", {})

    # Delete record index 1
    del audit_manager._records[1]

    is_valid, errors = audit_manager.verify_integrity()
    assert is_valid is False
    assert any("Broken hash chain" in err for err in errors)


# ===========================================================================
# Tests 19-24: Input Validation, Credential Safety & Reports
# ===========================================================================

def test_19_invalid_event_type(audit_manager):
    """Invalid event type string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid event_type"):
        audit_manager.log_event("INVALID_ACTION", 101, "m_1", "ACTOR", {})


def test_20_invalid_identifiers(audit_manager):
    """Empty or None identifiers raise ValueError."""
    with pytest.raises(ValueError, match="window_id"):
        audit_manager.log_event(AuditEventType.WINDOW_CREATED, "", "m_1", "ACTOR", {})

    with pytest.raises(ValueError, match="merchant_id"):
        audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "", "ACTOR", {})

    with pytest.raises(ValueError, match="actor"):
        audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "m_1", "", {})


def test_21_non_serializable_payload(audit_manager):
    """Non-JSON-serializable payload raises ValueError."""
    class CustomObject:
        pass

    with pytest.raises(ValueError, match="JSON-serializable"):
        audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "m_1", "ACTOR", {"obj": CustomObject()})


def test_22_credential_leakage_prevention(audit_manager):
    """Payloads containing sensitive credential keys are rejected and values never logged."""
    sensitive_keys = [
        "api_key",
        "apiKey",
        "authorization",
        "bearer_token",
        "password",
        "secret",
        "private_key",
        "session_token",
    ]
    for key in sensitive_keys:
        with pytest.raises(ValueError, match="Credential security violation"):
            audit_manager.log_event(
                AuditEventType.RISK_EVALUATED,
                101,
                "m_1",
                "ACTOR",
                {key: "SUPER_SECRET_VALUE_12345"},
            )

    # Verify no records were saved
    assert len(audit_manager) == 0


def test_23_export_report_structure(audit_manager):
    """export_audit_report returns structured audit summary."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "m_1", "ACTOR_1", {"count": 10})
    audit_manager.log_event(AuditEventType.POLICY_DECISION, 101, "m_1", "ACTOR_2", {"policy": "POL_ROUTINE"})

    report = audit_manager.export_audit_report(101)
    assert report["window_id"] == "101"
    assert report["event_count"] == 2
    assert len(report["events"]) == 2
    assert report["integrity_valid"] is True
    assert report["integrity_errors"] == []


def test_24_export_report_json_serializability(audit_manager):
    """export_audit_report is 100% JSON-serializable."""
    audit_manager.log_event(AuditEventType.WINDOW_CREATED, 101, "m_1", "ACTOR", {"k": "v"})
    report = audit_manager.export_audit_report(101)
    serialized = json.dumps(report)
    assert isinstance(serialized, str)


# ===========================================================================
# Tests 25-28: Governance, Analyst Actions, Holdout & Append-Only
# ===========================================================================

def test_25_defense_only_behavior(audit_manager):
    """Audit trail only records defense-oriented actions and triage workflows."""
    record = audit_manager.log_event(
        event_type=AuditEventType.POLICY_DECISION,
        window_id=101,
        merchant_id="m_1",
        actor="SYSTEM:POLICY_ENGINE",
        payload={
            "policy_id": "POL_CRITICAL_SURGE",
            "action_type": "URGENT_TRIAGE",
            "priority": "P0",
            "review_sla_hours": 1.0,
            "require_dual_review": True,
        },
    )
    assert record.payload["action_type"] == "URGENT_TRIAGE"


def test_26_analyst_action_logging(audit_manager):
    """ANALYST_ACTION event type records human review outcomes."""
    record = audit_manager.log_event(
        event_type=AuditEventType.ANALYST_ACTION,
        window_id=101,
        merchant_id="m_1",
        actor="ANALYST:analyst_01",
        payload={
            "disposition": "verified_normal_flash_sale",
            "notes": "Merchant confirmed holiday sale promotional volume surge.",
            "dual_review_signoff": True,
        },
    )
    assert record.event_type == "ANALYST_ACTION"
    assert record.actor == "ANALYST:analyst_01"


def test_27_no_final_holdout_dependency():
    """Verify audit_trail module contains zero references to final holdout."""
    import audit_trail
    source = inspect.getsource(audit_trail)
    assert "final_holdout" not in source
    assert "detection_windows_final_holdout.csv" not in source


def test_28_append_only_behavior(audit_manager):
    """
    Verify append-only properties:
      1. No update/delete/edit/remove methods exist on manager.
      2. Logging a new event strictly appends to storage.
      3. Existing records remain unchanged and chronological order is preserved.
    """
    assert not hasattr(audit_manager, "update_event")
    assert not hasattr(audit_manager, "delete_event")
    assert not hasattr(audit_manager, "edit_event")
    assert not hasattr(audit_manager, "remove_event")

    r1 = audit_manager.log_event(AuditEventType.WINDOW_CREATED, 1, "m_1", "A1", {"seq": 1})
    r2 = audit_manager.log_event(AuditEventType.DETECTION_FLAGGED, 1, "m_1", "A2", {"seq": 2})

    # Query trail and verify order
    trail = audit_manager.get_trail_for_window(1)
    assert len(trail) == 2
    assert trail[0].entry_id == r1.entry_id
    assert trail[1].entry_id == r2.entry_id

    # Append a third event
    r3 = audit_manager.log_event(AuditEventType.POLICY_DECISION, 1, "m_1", "A3", {"seq": 3})
    trail_after = audit_manager.get_trail_for_window(1)
    assert len(trail_after) == 3
    assert trail_after[0].entry_id == r1.entry_id
    assert trail_after[1].entry_id == r2.entry_id
    assert trail_after[2].entry_id == r3.entry_id
