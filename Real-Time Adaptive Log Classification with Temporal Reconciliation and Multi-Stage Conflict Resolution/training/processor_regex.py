# processor_regex.py

import re

try:
    from .schemas import ClassificationSignal
except ImportError:
    from schemas import ClassificationSignal


UNCLASSIFIED = "Unclassified"

def classify_signal_with_regex(log_message: str) -> ClassificationSignal:
    """Return a deterministic regex classification signal for one log message."""
    regex_patterns = {
        r"User User\d+ logged (in|out)\.?": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully\.?": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully\.?": "System Notification",
        r"System reboot initiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"
    }

    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_message, flags=re.IGNORECASE):
            return ClassificationSignal(stage="regex", label=label, confidence=1.0, matched=True, reason=f"Matched predefined pattern: {pattern}")

    return ClassificationSignal(stage="regex", label=UNCLASSIFIED, confidence=0.0, matched=False, reason="No predefined regex pattern matched.")


def classify_with_regex(log_message: str) -> str | None:
    """Legacy label-only adapter retained for the CSV compatibility route."""
    signal = classify_signal_with_regex(log_message)
    return signal.label if signal.matched else None


if __name__ == "__main__":
    tests = [
        "Backup completed successfully.",
        "Account with ID 1234 created by User1.",
        "System reboot initiated by user Admin123",
        "Hey Bro, chill ya!"
    ]
    for log in tests:
        print(log, "->", classify_with_regex(log))
