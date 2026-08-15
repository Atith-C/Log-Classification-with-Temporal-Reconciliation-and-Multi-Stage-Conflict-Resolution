from fastapi.testclient import TestClient

from training import server


client = TestClient(server.app)


def valid_event() -> dict[str, str]:
    return {
        "id": "evt-001",
        "timestamp": "2026-08-15T10:30:00Z",
        "source": "ModernCRM",
        "message": "Backup completed successfully.",
    }


def test_events_accepts_and_normalises_a_valid_event(monkeypatch) -> None:
    monkeypatch.setattr(server, "generate_signals", lambda *_: [])
    response = client.post("/events", json=valid_event())

    assert response.status_code == 200
    payload = response.json()
    assert payload["audit"]["event_id"] == "evt-001"
    assert payload["audit"]["decision_timestamp"] == "2026-08-15T10:30:00Z"
    del payload["audit"]
    assert payload == {
        "status": "classified",
        "event": {
            "id": "evt-001",
            "timestamp": "2026-08-15T10:30:00Z",
            "source": "ModernCRM",
            "message": "Backup completed successfully.",
        },
        "signals": [],
        "final_decision": {
            "label": "Unclassified",
            "confidence": 0.0,
            "conflict_detected": False,
            "selected_stage": None,
            "resolution_reason": "No matched classification signal was available.",
        },
        "reconciliation": {
            "was_late": False,
            "state_updated": True,
            "effective_event_timestamp": "2026-08-15T10:30:00Z",
            "replayed_event_count": 1,
            "reason": "State was reconciled from chronological event history.",
        },
        "detail": "Signals generated and resolved deterministically.",
    }


def test_events_rejects_missing_required_field_with_400() -> None:
    event = valid_event()
    del event["message"]

    response = client.post("/events", json=event)

    assert response.status_code == 400
    assert response.json()["detail"][0]["loc"] == ["body", "message"]


def test_events_rejects_malformed_timestamp_with_400() -> None:
    event = valid_event()
    event["timestamp"] = "not-a-timestamp"

    response = client.post("/events", json=event)

    assert response.status_code == 400
    assert response.json()["detail"][0]["loc"] == ["body", "timestamp"]


def test_events_rejects_blank_values_and_unexpected_fields_with_400() -> None:
    event = valid_event()
    event["source"] = "   "
    event["unexpected"] = "not allowed"

    response = client.post("/events", json=event)

    assert response.status_code == 400
    errors = response.json()["detail"]
    assert {tuple(error["loc"]) for error in errors} == {
        ("body", "source"),
        ("body", "unexpected"),
    }
