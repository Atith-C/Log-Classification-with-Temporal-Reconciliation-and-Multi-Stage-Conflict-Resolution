from datetime import datetime, timezone

from training.resolver import resolve_signals
from training.schemas import ClassificationSignal


EVENT_TIME = datetime(2026, 8, 15, 10, 30, tzinfo=timezone.utc)


def signal(stage: str, label: str, confidence: float, matched: bool = True) -> ClassificationSignal:
    return ClassificationSignal(stage=stage, label=label, confidence=confidence, matched=matched, reason="test")


def test_resolver_keeps_an_agreed_label_and_combines_its_confidence() -> None:
    result = resolve_signals([
        signal("regex", "User Action", 1.0),
        signal("bert", "User Action", 0.8),
    ], EVENT_TIME)

    assert result.label == "User Action"
    assert result.conflict_detected is False
    assert result.selected_stage == "regex"
    assert result.confidence == (1.0 + (0.8 * 0.9)) / 1.9


def test_resolver_uses_weighted_score_when_signals_conflict() -> None:
    result = resolve_signals([
        signal("regex", "User Action", 0.90),
        signal("bert", "Security Alert", 0.95),
        signal("llm", "Security Alert", 0.90),
    ], EVENT_TIME)

    assert result.label == "Security Alert"
    assert result.conflict_detected is True
    assert result.selected_stage == "bert"
    assert "weighted score" in result.resolution_reason


def test_resolver_breaks_equal_scores_with_highest_individual_confidence() -> None:
    result = resolve_signals([
        signal("regex", "User Action", 0.90),
        signal("bert", "Security Alert", 1.0),
    ], EVENT_TIME)

    assert result.label == "Security Alert"
    assert result.selected_stage == "bert"


def test_resolver_ignores_unmatched_and_unclassified_signals() -> None:
    result = resolve_signals([
        signal("regex", "Unclassified", 0.0, False),
        signal("bert", "Unclassified", 0.49, False),
        signal("llm", "Unclassified", 0.0, False),
    ], EVENT_TIME)

    assert result.label == "Unclassified"
    assert result.confidence == 0.0
    assert result.selected_stage is None
    assert result.conflict_detected is False


def test_resolver_is_repeatably_deterministic() -> None:
    signals = [signal("regex", "User Action", 0.9), signal("bert", "Security Alert", 1.0)]

    assert resolve_signals(signals, EVENT_TIME) == resolve_signals(signals, EVENT_TIME)
