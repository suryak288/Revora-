"""In-memory audit store for bounded simulated executions.

This is an in-memory demo audit store and does not survive server restart.
"""

from threading import Lock

from app.execution.models import AuditEvent

_audit_events: list[AuditEvent] = []
_seen_executions: set[str] = set()
_lock = Lock()


def record_audit_event(event: AuditEvent) -> bool:
    """Record an audit event. Returns False if execution_id already seen."""
    with _lock:
        if event.execution_id in _seen_executions:
            return False
            
        _audit_events.append(event)
        _seen_executions.add(event.execution_id)
        return True


def get_recent_audit_events(limit: int = 50) -> list[AuditEvent]:
    """Return the most recent audit events, newest first."""
    with _lock:
        # Reverse the list so newest are first, then slice
        return list(reversed(_audit_events))[:limit]


def clear_audit_events_for_testing() -> None:
    """Clear the audit store, useful for tests."""
    with _lock:
        _audit_events.clear()
        _seen_executions.clear()
