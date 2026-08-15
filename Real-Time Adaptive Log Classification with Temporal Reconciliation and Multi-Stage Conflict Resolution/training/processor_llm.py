# processor_llm.py

"""Deterministic local stand-in for the required mocked LLaMA classifier."""

import json
from pathlib import Path

try:
    from .schemas import ClassificationSignal
except ImportError:
    from schemas import ClassificationSignal


MOCK_RESPONSE_PATH = Path(__file__).resolve().parent / "resources" / "llm_mock_responses.json"


def classify_signal_with_llm(log_message: str) -> ClassificationSignal:
    """Return the first matching local mock response in configured order."""
    with MOCK_RESPONSE_PATH.open(encoding="utf-8") as response_file:
        responses = json.load(response_file)
    normalized_message = log_message.casefold()
    for rule in responses["rules"]:
        if all(keyword.casefold() in normalized_message for keyword in rule["keywords"]):
            return ClassificationSignal(
                stage="llm", label=rule["label"], confidence=float(rule["confidence"]),
                matched=True, reason=f"Local mock response rule: {rule['id']}",
            )
    default = responses["default"]
    return ClassificationSignal(
        stage="llm", label=default["label"], confidence=float(default["confidence"]),
        matched=False, reason="No local mock response rule matched.",
    )


def classify_with_llm(log_message: str) -> str:
    """Legacy label-only adapter retained for the CSV compatibility route."""
    return classify_signal_with_llm(log_message).label


if __name__ == "__main__":
    tests = [
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025",
        "System reboot initiated by user 12345."
    ]
    for log in tests:
        print(log, "->", classify_with_llm(log))
