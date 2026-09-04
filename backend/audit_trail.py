"""
AI Risk Manager — Phase 10: Audit Trail Engine
==============================================

Purpose
-------
Provides an immutable, append-only, tamper-evident audit logging engine that
records every critical event across the risk management lifecycle (window creation,
anomaly detection, risk evaluation, explanation generation, policy decisions,
and analyst review actions).

Key Guarantees
--------------
1. **Append-Only Storage**: No update, edit, or delete operations exist.
2. **SHA-256 Hash Chaining**: Every record cryptographically links to the previous
   record's integrity hash.
3. **Tamper & Deletion Detection**: Any modification to historical records or payload
   data breaks the hash chain and is immediately flagged by ``verify_integrity()``.
4. **Canonical Hashing**: Uses deterministic JSON serialization (sorted keys, compact
   separators, UTF-8 encoding) to ensure payload hashing is independent of dictionary
   insertion order.
5. **Credential Protection**: Recursively inspects payloads for sensitive authorization
   keys (API keys, tokens, passwords) and rejects them before logging.
6. **Defense-Only Governance**: Designed strictly for human review observability and
   compliance reporting; never triggers automated destructive actions.
"""

from __future__ import annotations

import copy
import datetime
import enum
import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Audit Event Types Enum
# ---------------------------------------------------------------------------

class AuditEventType(str, enum.Enum):
    """
    Standard lifecycle event types for the AI Risk Manager audit trail.
    """
    WINDOW_CREATED = "WINDOW_CREATED"
    DETECTION_FLAGGED = "DETECTION_FLAGGED"
    RISK_EVALUATED = "RISK_EVALUATED"
    EXPLANATION_GENERATED = "EXPLANATION_GENERATED"
    POLICY_DECISION = "POLICY_DECISION"
    ANALYST_ACTION = "ANALYST_ACTION"


VALID_EVENT_TYPES = tuple(e.value for e in AuditEventType)


# ---------------------------------------------------------------------------
# Credential Protection Constants
# ---------------------------------------------------------------------------

SENSITIVE_KEY_PATTERNS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "bearer_token",
    "access_token",
    "refresh_token",
    "password",
    "passwd",
    "secret",
    "private_key",
    "session_token",
    "cookie",
}


# ---------------------------------------------------------------------------
# AuditRecord Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditRecord:
    """
    Immutable audit log entry.

    Attributes:
        entry_id:        Unique event identifier (UUID string).
        timestamp:       Timezone-aware UTC ISO timestamp string.
        event_type:      AuditEventType string.
        window_id:       Associated DetectionWindow identifier.
        merchant_id:     Associated merchant identifier.
        actor:           System component or analyst identifier.
        payload:         Deep-copied structured payload data.
        previous_hash:   SHA-256 hash of preceding record (None for first record).
        integrity_hash:  SHA-256 hash computed over canonical event fields.
    """
    entry_id: str
    timestamp: str
    event_type: str
    window_id: Any
    merchant_id: str
    actor: str
    payload: dict[str, Any]
    previous_hash: Optional[str]
    integrity_hash: str

    def __post_init__(self):
        # Ensure deep-copy of payload to protect against mutation of input dict
        if isinstance(self.payload, dict):
            object.__setattr__(self, "payload", copy.deepcopy(self.payload))


# ---------------------------------------------------------------------------
# Validation & Serialization Helpers
# ---------------------------------------------------------------------------

def _validate_event_type(event_type: Union[AuditEventType, str]) -> str:
    """Validate and return the event type string."""
    if isinstance(event_type, AuditEventType):
        return event_type.value
    if isinstance(event_type, str) and event_type in VALID_EVENT_TYPES:
        return event_type
    raise ValueError(
        f"Invalid event_type: {event_type!r}. Must be one of {VALID_EVENT_TYPES}."
    )


def _validate_identifier(value: Any, field_name: str) -> str:
    """Ensure identifiers (window_id, merchant_id, actor) are non-empty."""
    if value is None:
        raise ValueError(f"{field_name} cannot be None.")
    val_str = str(value).strip()
    if not val_str:
        raise ValueError(f"{field_name} cannot be empty.")
    return val_str


def _check_sensitive_keys(data: Any, path: str = "") -> None:
    """
    Recursively inspect payload for credential-bearing or secret keys.
    Raises ValueError if a sensitive key is found without exposing secret values.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            k_clean = str(k).lower().strip()
            if k_clean in SENSITIVE_KEY_PATTERNS:
                raise ValueError(
                    f"Credential security violation: Sensitive field detected in payload at '{path + str(k)}'."
                )
            _check_sensitive_keys(v, path=f"{path}{k}.")
    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            _check_sensitive_keys(item, path=f"{path}[{idx}].")


def _validate_payload(payload: Any) -> dict[str, Any]:
    """
    Validate that payload is a dictionary, JSON-serializable, and credential-safe.
    """
    if payload is None:
        raise ValueError("payload cannot be None. Pass an empty dict {} if no data.")
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}.")

    # Check for sensitive credentials
    _check_sensitive_keys(payload)

    # Verify JSON serializability
    try:
        json.dumps(payload, ensure_ascii=False)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"payload must be JSON-serializable: {e}")

    return copy.deepcopy(payload)


def _get_utc_timestamp_str(ts: Optional[datetime.datetime] = None) -> str:
    """Generate a standardized canonical UTC ISO timestamp."""
    if ts is None:
        ts = datetime.datetime.now(datetime.timezone.utc)
    elif ts.tzinfo is None:
        # Assume UTC if naive, but localize explicitly
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    else:
        ts = ts.astimezone(datetime.timezone.utc)

    # Format strictly as ISO-8601 UTC with 'Z' suffix
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def compute_integrity_hash(
    timestamp: str,
    event_type: str,
    window_id: Any,
    merchant_id: str,
    payload: dict[str, Any],
    previous_hash: Optional[str],
) -> str:
    """
    Compute canonical SHA-256 hash over approved audit fields.

    Exact canonical fields:
      - timestamp
      - event_type
      - window_id
      - merchant_id
      - payload
      - previous_hash
    """
    canonical_dict = {
        "event_type": str(event_type),
        "merchant_id": str(merchant_id),
        "payload": payload,
        "previous_hash": previous_hash,
        "timestamp": str(timestamp),
        "window_id": str(window_id),
    }

    # Deterministic JSON encoding: sorted keys, compact separators, UTF-8
    canonical_bytes = json.dumps(
        canonical_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(canonical_bytes).hexdigest()


# ---------------------------------------------------------------------------
# AuditTrailManager
# ---------------------------------------------------------------------------

class AuditTrailManager:
    """
    Append-only, tamper-evident audit trail store and verification engine.
    """

    def __init__(self):
        # Internal sequential storage of AuditRecord instances
        self._records: list[AuditRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def log_event(
        self,
        event_type: Union[AuditEventType, str],
        window_id: Any,
        merchant_id: str,
        actor: str,
        payload: dict[str, Any],
        timestamp: Optional[datetime.datetime] = None,
    ) -> AuditRecord:
        """
        Record a new verified, cryptographically chained audit event.

        Parameters:
            event_type:  AuditEventType enum or string.
            window_id:   Identifier for DetectionWindow.
            merchant_id: Target merchant string.
            actor:       Identifier of system component or analyst.
            payload:     JSON-serializable, credential-safe event details.
            timestamp:   Optional explicit datetime (defaults to now UTC).

        Returns:
            Immutable AuditRecord.
        """
        # 1. Validate inputs
        clean_event_type = _validate_event_type(event_type)
        clean_window_id = _validate_identifier(window_id, "window_id")
        clean_merchant_id = _validate_identifier(merchant_id, "merchant_id")
        clean_actor = _validate_identifier(actor, "actor")
        clean_payload = _validate_payload(payload)

        # 2. Establish timestamp and hash chain
        ts_str = _get_utc_timestamp_str(timestamp)
        entry_id = str(uuid.uuid4())

        previous_hash = (
            self._records[-1].integrity_hash if self._records else None
        )

        # 3. Compute SHA-256 integrity hash
        integrity_hash = compute_integrity_hash(
            timestamp=ts_str,
            event_type=clean_event_type,
            window_id=clean_window_id,
            merchant_id=clean_merchant_id,
            payload=clean_payload,
            previous_hash=previous_hash,
        )

        # 4. Construct immutable AuditRecord and append
        record = AuditRecord(
            entry_id=entry_id,
            timestamp=ts_str,
            event_type=clean_event_type,
            window_id=clean_window_id,
            merchant_id=clean_merchant_id,
            actor=clean_actor,
            payload=clean_payload,
            previous_hash=previous_hash,
            integrity_hash=integrity_hash,
        )

        self._records.append(record)
        return copy.deepcopy(record)

    def get_trail_for_window(self, window_id: Any) -> list[AuditRecord]:
        """
        Return chronological audit trail for a specific window_id.
        Returns defensive copies to prevent caller mutation of internal records.
        """
        target_str = str(window_id).strip()
        matched = [
            copy.deepcopy(r)
            for r in self._records
            if str(r.window_id) == target_str
        ]
        return matched

    def get_trail_for_merchant(
        self,
        merchant_id: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> list[AuditRecord]:
        """
        Return chronological audit trail for a specific merchant, with optional
        inclusive datetime filtering.
        """
        target_str = str(merchant_id).strip()
        matched: list[AuditRecord] = []

        # Parse filter timestamps to UTC aware
        utc_start = (
            start_time.astimezone(datetime.timezone.utc)
            if start_time and start_time.tzinfo
            else (start_time.replace(tzinfo=datetime.timezone.utc) if start_time else None)
        )
        utc_end = (
            end_time.astimezone(datetime.timezone.utc)
            if end_time and end_time.tzinfo
            else (end_time.replace(tzinfo=datetime.timezone.utc) if end_time else None)
        )

        for r in self._records:
            if str(r.merchant_id) != target_str:
                continue

            record_dt = datetime.datetime.fromisoformat(r.timestamp.replace("Z", "+00:00"))

            if utc_start and record_dt < utc_start:
                continue
            if utc_end and record_dt > utc_end:
                continue

            matched.append(copy.deepcopy(r))

        return matched

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """
        Validate the complete cryptographic audit chain.

        Verifies:
          1. First record has previous_hash == None.
          2. Sequential records match preceding record's integrity_hash.
          3. Independent hash recomputation matches stored integrity_hash.

        Returns:
            (is_valid: bool, error_messages: list[str])
        """
        errors: list[str] = []

        if not self._records:
            return True, []

        for idx, record in enumerate(self._records):
            # 1. Chain continuity verification
            if idx == 0:
                if record.previous_hash is not None:
                    errors.append(
                        f"Record 0 (id={record.entry_id}): First record must have previous_hash=None, "
                        f"found {record.previous_hash!r}."
                    )
            else:
                expected_prev_hash = self._records[idx - 1].integrity_hash
                if record.previous_hash != expected_prev_hash:
                    errors.append(
                        f"Record {idx} (id={record.entry_id}): Broken hash chain. "
                        f"previous_hash={record.previous_hash!r} does not match "
                        f"preceding integrity_hash={expected_prev_hash!r}."
                    )

            # 2. Hash integrity recomputation
            recomputed_hash = compute_integrity_hash(
                timestamp=record.timestamp,
                event_type=record.event_type,
                window_id=record.window_id,
                merchant_id=record.merchant_id,
                payload=record.payload,
                previous_hash=record.previous_hash,
            )

            if recomputed_hash != record.integrity_hash:
                errors.append(
                    f"Record {idx} (id={record.entry_id}): Hash mismatch. "
                    f"Stored integrity_hash={record.integrity_hash!r} != "
                    f"Recomputed hash={recomputed_hash!r} (tampering detected)."
                )

        is_valid = len(errors) == 0
        return is_valid, errors

    def export_audit_report(self, window_id: Any) -> dict[str, Any]:
        """
        Export a structured compliance and governance audit summary for a window.
        """
        window_events = self.get_trail_for_window(window_id)
        is_valid, errors = self.verify_integrity()

        # Format events into JSON serializable dicts
        serialized_events: list[dict[str, Any]] = []
        for e in window_events:
            serialized_events.append({
                "entry_id": e.entry_id,
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "window_id": e.window_id,
                "merchant_id": e.merchant_id,
                "actor": e.actor,
                "payload": e.payload,
                "previous_hash": e.previous_hash,
                "integrity_hash": e.integrity_hash,
            })

        return {
            "window_id": str(window_id),
            "event_count": len(serialized_events),
            "events": serialized_events,
            "integrity_valid": is_valid,
            "integrity_errors": errors,
        }
