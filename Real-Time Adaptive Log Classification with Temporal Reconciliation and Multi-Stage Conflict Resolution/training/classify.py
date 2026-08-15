"""Synchronous deterministic classification-signal pipeline."""

import os
from typing import Sequence

import pandas as pd

try:
    from .processor_bert import classify_signal_with_bert
    from .processor_llm import classify_signal_with_llm
    from .processor_regex import classify_signal_with_regex
    from .schemas import ClassificationSignal
except ImportError:
    from processor_bert import classify_signal_with_bert
    from processor_llm import classify_signal_with_llm
    from processor_regex import classify_signal_with_regex
    from schemas import ClassificationSignal


LLM_AMBIGUITY_THRESHOLD = 0.70


def classify_event(source: str, log_message: str) -> list[ClassificationSignal]:
    """Produce all needed signals for one event in a single process."""
    del source
    regex_signal = classify_signal_with_regex(log_message)
    bert_signal = classify_signal_with_bert(log_message)
    signals = [regex_signal, bert_signal]
    labels_disagree = regex_signal.matched and bert_signal.matched and regex_signal.label != bert_signal.label
    if not regex_signal.matched or not bert_signal.matched or bert_signal.confidence < LLM_AMBIGUITY_THRESHOLD or labels_disagree:
        signals.append(classify_signal_with_llm(log_message))
    return signals


def _classify_log_obsolete(source, log_msg):
    """
    Apply classification in this order:
    1. If source == 'LegacyCRM' → try LLM
    2. Else → try regex
    3. If regex fails → try BERT
    4. If still unclassified → fallback to LLM
    """
    if source == "LegacyCRM":
        label = classify_with_llm(log_msg)
    else:
        label = classify_with_regex(log_msg)
        if not label:
            label = classify_with_bert(log_msg)
            if label == "Unclassified":
                label = classify_with_llm(log_msg)
    return label


def classify_log(source: str, log_message: str) -> str:
    """Legacy CSV adapter: return the first matched stage label."""
    for signal in classify_event(source, log_message):
        if signal.matched:
            return signal.label
    return "Unclassified"


def classify(logs: Sequence[tuple[str, str]]) -> list[str]:
    """Legacy batch CSV adapter."""
    return [classify_log(source, log_message) for source, log_message in logs]


def classify_csv(input_file):
    """
    Classify all logs in a CSV file.
    Input CSV must contain 'source' and 'log_message' columns.
    Output is saved to resources/output.csv
    """
    df = pd.read_csv(input_file)

    # Perform classification
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

    # Ensure resources folder exists
    output_dir = os.path.join(os.path.dirname(__file__), "resources")
    os.makedirs(output_dir, exist_ok=True)

    # Save the modified file
    output_file = os.path.join(output_dir, "output.csv")
    df.to_csv(output_file, index=False)

    return output_file


if __name__ == '__main__':
    # Example run on synthetic dataset
    output_file = classify_csv("dataset/synthetic_logs.csv")
    print(f"Classification complete. Results saved to {output_file}")

    # Uncomment to test manually
    # logs = [
    #     ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
    #     ("BillingSystem", "User 12345 logged in."),
    #     ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
    #     ("AnalyticsEngine", "Backup completed successfully."),
    #     ("ModernHR", "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE 200 len: 1583 time: 0.1878400"),
    #     ("ModernHR", "Admin access escalation detected for user 9429"),
    #     ("LegacyCRM", "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
    #     ("LegacyCRM", "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module."),
    #     ("LegacyCRM", "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."),
    #     ("LegacyCRM", "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025")
    # ]
    # labels = classify(logs)
    # for log, label in zip(logs, labels):
    #     print(log[0], "->", label)
