from fastapi.testclient import TestClient

from training import server
from training.schemas import ClassificationSignal


def test_audit_is_persisted_and_replay_returns_identical_record(monkeypatch) -> None:
    client = TestClient(server.app)
    stable_signal = ClassificationSignal(stage="regex", label="User Action", confidence=1.0, matched=True, reason="test")
    monkeypatch.setattr(server, "generate_signals", lambda *_: [stable_signal])
    payload = {"id": "evt-replay", "timestamp": "2026-08-15T10:30:00Z", "source": "ModernCRM", "message": "User User1 logged in."}

    processed = client.post("/events", json=payload)
    replayed = client.post("/replay", json={"event_ids": ["evt-replay"]})

    assert processed.status_code == 200
    assert replayed.status_code == 200
    assert replayed.json()["audit_records"] == [processed.json()["audit"]]
    assert replayed.json()["reconstructed_state"]["evt-replay"]["final_decision"] == processed.json()["final_decision"]


def test_replay_rebuilds_late_event_audits_and_current_state(monkeypatch) -> None:
    client = TestClient(server.app)
    stable_signal = ClassificationSignal(stage="regex", label="System Notification", confidence=1.0, matched=True, reason="test")
    monkeypatch.setattr(server, "generate_signals", lambda *_: [stable_signal])
    newer = {"id": "evt-late", "timestamp": "2026-08-15T11:00:00Z", "source": "ModernCRM", "message": "Newer"}
    older = {"id": "evt-late", "timestamp": "2026-08-15T10:00:00Z", "source": "ModernCRM", "message": "Older"}

    assert client.post("/events", json=newer).status_code == 200
    late = client.post("/events", json=older)
    replayed = client.post("/replay", json={"event_ids": ["evt-late"]})

    assert late.status_code == 200
    assert replayed.status_code == 200
    assert replayed.json()["audit_records"][1] == late.json()["audit"]
    assert replayed.json()["reconstructed_state"]["evt-late"]["effective_event_timestamp"] == "2026-08-15T11:00:00Z"


def test_replay_rejects_unknown_event_ids() -> None:
    client = TestClient(server.app)
    response = client.post("/replay", json={"event_ids": ["missing"]})

    assert response.status_code == 404
