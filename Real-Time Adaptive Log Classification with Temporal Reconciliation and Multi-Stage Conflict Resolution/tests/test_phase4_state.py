from datetime import datetime, timezone

from fastapi.testclient import TestClient

from training import server
from training.schemas import ClassificationSignal, FinalDecision, LogEvent
from training.state_store import DuplicateEventError, SQLiteStateStore


def event(event_id: str = "evt-004", timestamp: str = "2026-08-15T10:30:00Z") -> LogEvent:
    return LogEvent(id=event_id, timestamp=timestamp, source="ModernCRM", message="Backup completed successfully.")


def signal() -> ClassificationSignal:
    return ClassificationSignal(stage="regex", label="System Notification", confidence=1.0, matched=True, reason="test")


def decision() -> FinalDecision:
    return FinalDecision(
        label="System Notification", confidence=1.0, conflict_detected=False,
        selected_stage="regex", resolution_reason="test",
    )


def test_store_persists_event_signals_and_decision(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    original = event()

    store.record_event(original, [signal()], decision())

    reopened = SQLiteStateStore(tmp_path / "state.sqlite3")
    assert reopened.has_event(original) is True
    assert reopened.event_count() == 1


def test_store_rejects_the_same_id_and_timestamp(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    original = event()
    store.record_event(original, [signal()], decision())

    try:
        store.record_event(original, [signal()], decision())
    except DuplicateEventError:
        pass
    else:
        raise AssertionError("Expected a duplicate event error")
    assert store.event_count() == 1


def test_same_id_with_a_different_timestamp_is_a_distinct_event(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    first = event(timestamp="2026-08-15T10:30:00Z")
    later = event(timestamp="2026-08-15T10:31:00Z")

    store.record_event(first, [signal()], decision())
    store.record_event(later, [signal()], decision())

    assert store.event_count() == 2


def test_events_endpoint_returns_409_and_does_not_reclassify_a_duplicate(monkeypatch) -> None:
    api_client = TestClient(server.app)
    calls = []

    def fake_signals(*_):
        calls.append("called")
        return [signal()]

    monkeypatch.setattr(server, "generate_signals", fake_signals)
    payload = {
        "id": "evt-api-duplicate",
        "timestamp": "2026-08-15T10:30:00Z",
        "source": "ModernCRM",
        "message": "Backup completed successfully.",
    }

    first = api_client.post("/events", json=payload)
    duplicate = api_client.post("/events", json=payload)

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "Duplicate event: this id and timestamp have already been processed."
    assert calls == ["called"]
