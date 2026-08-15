from training.schemas import ClassificationSignal, FinalDecision, LogEvent
from training.state_store import SQLiteStateStore


def event(timestamp: str) -> LogEvent:
    return LogEvent(id="log-temporal", timestamp=timestamp, source="ModernCRM", message="A message")


def signal() -> ClassificationSignal:
    return ClassificationSignal(stage="regex", label="System Notification", confidence=1.0, matched=True, reason="test")


def decision(label: str) -> FinalDecision:
    return FinalDecision(label=label, confidence=0.9, conflict_detected=False, selected_stage="regex", resolution_reason="test")


def test_late_event_is_recorded_without_replacing_newer_log_state(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    newer = event("2026-08-15T11:00:00Z")
    older = event("2026-08-15T10:00:00Z")

    store.record_event(newer, [signal()], decision("Newer Label"))
    reconciliation, _ = store.record_event(older, [signal()], decision("Older Label"))

    state = store.get_current_state("log-temporal")
    assert reconciliation.was_late is True
    assert reconciliation.state_updated is False
    assert reconciliation.effective_event_timestamp.isoformat() == "2026-08-15T11:00:00+00:00"
    assert reconciliation.replayed_event_count == 2
    assert state["label"] == "Newer Label"
    assert state["effective_event_timestamp"] == "2026-08-15T11:00:00Z"


def test_chronological_and_out_of_order_arrival_produce_the_same_final_state(tmp_path) -> None:
    ordered_store = SQLiteStateStore(tmp_path / "ordered.sqlite3")
    unordered_store = SQLiteStateStore(tmp_path / "unordered.sqlite3")
    older = event("2026-08-15T10:00:00Z")
    newer = event("2026-08-15T11:00:00Z")

    ordered_store.record_event(older, [signal()], decision("Older Label"))
    ordered_store.record_event(newer, [signal()], decision("Newer Label"))
    unordered_store.record_event(newer, [signal()], decision("Newer Label"))
    unordered_store.record_event(older, [signal()], decision("Older Label"))

    assert ordered_store.get_current_state("log-temporal") == unordered_store.get_current_state("log-temporal")


def test_timestamp_is_canonicalised_to_utc_for_temporal_ordering(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    utc_event = event("2026-08-15T10:30:00Z")
    same_in_india = event("2026-08-15T16:00:00+05:30")

    store.record_event(utc_event, [signal()], decision("UTC Label"))
    assert store.has_event(same_in_india) is True
