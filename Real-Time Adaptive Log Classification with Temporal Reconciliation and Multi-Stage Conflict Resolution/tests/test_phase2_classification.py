from training import classify, processor_bert
from training.processor_llm import classify_signal_with_llm
from training.processor_regex import classify_signal_with_regex
from training.schemas import ClassificationSignal
from training import server
from fastapi.testclient import TestClient


class FakeEmbeddingModel:
    def encode(self, messages):
        assert messages == ["A test message"]
        return [[0.1, 0.2]]


class FakeClassifier:
    classes_ = ["Error", "Security Alert"]

    def predict_proba(self, embeddings):
        assert embeddings == [[0.1, 0.2]]
        return [[0.25, 0.75]]


class LowConfidenceClassifier(FakeClassifier):
    def predict_proba(self, embeddings):
        return [[0.49, 0.48]]


def signal(stage: str, label: str, confidence: float, matched: bool) -> ClassificationSignal:
    return ClassificationSignal(stage=stage, label=label, confidence=confidence, matched=matched, reason="test")


def test_regex_signal_returns_label_confidence_and_provenance() -> None:
    result = classify_signal_with_regex("Backup completed successfully.")

    assert result.stage == "regex"
    assert result.label == "System Notification"
    assert result.confidence == 1.0
    assert result.matched is True
    assert "Matched predefined pattern" in result.reason


def test_regex_signal_returns_a_stable_unclassified_result() -> None:
    result = classify_signal_with_regex("No matching rule here")

    assert result.label == "Unclassified"
    assert result.confidence == 0.0
    assert result.matched is False


def test_bert_signal_uses_best_probability(monkeypatch) -> None:
    monkeypatch.setattr(processor_bert, "_embedding_model", FakeEmbeddingModel())
    monkeypatch.setattr(processor_bert, "_classification_model", FakeClassifier())

    result = processor_bert.classify_signal_with_bert("A test message")

    assert result.label == "Security Alert"
    assert result.confidence == 0.75
    assert result.matched is True


def test_bert_signal_marks_low_confidence_prediction_unclassified(monkeypatch) -> None:
    monkeypatch.setattr(processor_bert, "_embedding_model", FakeEmbeddingModel())
    monkeypatch.setattr(processor_bert, "_classification_model", LowConfidenceClassifier())

    result = processor_bert.classify_signal_with_bert("A test message")

    assert result.label == "Unclassified"
    assert result.confidence == 0.49
    assert result.matched is False


def test_local_llm_mock_is_deterministic_and_never_needs_an_api_key() -> None:
    first = classify_signal_with_llm("This feature is no longer supported.")
    second = classify_signal_with_llm("This feature is no longer supported.")

    assert first == second
    assert first.label == "Deprecation Warning"
    assert first.confidence == 0.95
    assert first.reason == "Local mock response rule: unsupported-warning"


def test_pipeline_calls_llm_when_regex_and_bert_disagree(monkeypatch) -> None:
    monkeypatch.setattr(classify, "classify_signal_with_regex", lambda _: signal("regex", "User Action", 1.0, True))
    monkeypatch.setattr(classify, "classify_signal_with_bert", lambda _: signal("bert", "Security Alert", 0.95, True))
    monkeypatch.setattr(classify, "classify_signal_with_llm", lambda _: signal("llm", "Workflow Error", 0.92, True))

    results = classify.classify_event("ModernCRM", "A test message")

    assert [result.stage for result in results] == ["regex", "bert", "llm"]


def test_pipeline_skips_llm_when_regex_and_high_confidence_bert_agree(monkeypatch) -> None:
    monkeypatch.setattr(classify, "classify_signal_with_regex", lambda _: signal("regex", "User Action", 1.0, True))
    monkeypatch.setattr(classify, "classify_signal_with_bert", lambda _: signal("bert", "User Action", 0.90, True))
    monkeypatch.setattr(classify, "classify_signal_with_llm", lambda _: (_ for _ in ()).throw(AssertionError("LLM should not run")))

    results = classify.classify_event("ModernCRM", "A test message")

    assert [result.stage for result in results] == ["regex", "bert"]


def test_events_endpoint_exposes_structured_signals(monkeypatch) -> None:
    api_client = TestClient(server.app)
    monkeypatch.setattr(server, "generate_signals", lambda *_: [signal("regex", "User Action", 1.0, True)])

    response = api_client.post("/events", json={
        "id": "evt-phase2",
        "timestamp": "2026-08-15T10:30:00Z",
        "source": "ModernCRM",
        "message": "User User1 logged in.",
    })

    assert response.status_code == 200
    assert response.json()["signals"] == [{
        "stage": "regex", "label": "User Action", "confidence": 1.0,
        "matched": True, "reason": "test",
    }]
    assert response.json()["final_decision"]["label"] == "User Action"
