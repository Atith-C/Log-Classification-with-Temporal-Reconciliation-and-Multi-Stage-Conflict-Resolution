"""Deterministic multi-stage conflict resolution for classification signals."""

from collections import defaultdict
from datetime import datetime
from typing import Iterable

try:
    from .schemas import ClassificationSignal, FinalDecision
except ImportError:
    from schemas import ClassificationSignal, FinalDecision


# These stable weights are configuration, not runtime-learned values.
STAGE_WEIGHTS = {"regex": 1.00, "bert": 0.90, "llm": 0.85}
STAGE_PRIORITY = {"regex": 0, "bert": 1, "llm": 2}
UNCLASSIFIED = "Unclassified"


def resolve_signals(signals: Iterable[ClassificationSignal], event_timestamp: datetime) -> FinalDecision:
    """Choose one final classification using a stable confidence-weighted policy.

    Only matched, non-``Unclassified`` signals vote. A label's score is the sum
    of each vote's confidence multiplied by the configured stage weight. Ties
    use highest individual confidence, then fixed stage priority, then lexical
    label order. ``event_timestamp`` is deliberately accepted as part of the
    deterministic decision context; Phase 5 uses it to reconcile competing
    state transitions from different events.
    """
    del event_timestamp
    valid_signals = [signal for signal in signals if signal.matched and signal.label != UNCLASSIFIED]
    if not valid_signals:
        return FinalDecision(
            label=UNCLASSIFIED,
            confidence=0.0,
            conflict_detected=False,
            selected_stage=None,
            resolution_reason="No matched classification signal was available.",
        )

    labels = {signal.label for signal in valid_signals}
    grouped: dict[str, list[ClassificationSignal]] = defaultdict(list)
    for signal in valid_signals:
        grouped[signal.label].append(signal)

    def ranking(label: str) -> tuple[float, float, int, str]:
        votes = grouped[label]
        weighted_score = sum(vote.confidence * STAGE_WEIGHTS[vote.stage] for vote in votes)
        highest_confidence = max(vote.confidence for vote in votes)
        best_priority = min(STAGE_PRIORITY[vote.stage] for vote in votes)
        return (-weighted_score, -highest_confidence, best_priority, label)

    winning_label = min(grouped, key=ranking)
    winning_votes = grouped[winning_label]
    selected_vote = min(
        winning_votes,
        key=lambda vote: (-vote.confidence, STAGE_PRIORITY[vote.stage]),
    )
    total_weight = sum(STAGE_WEIGHTS[vote.stage] for vote in winning_votes)
    final_confidence = sum(vote.confidence * STAGE_WEIGHTS[vote.stage] for vote in winning_votes) / total_weight
    conflict_detected = len(labels) > 1
    score = -ranking(winning_label)[0]
    reason = (
        f"Selected {winning_label!r} with weighted score {score:.6f}; "
        "ties are resolved by individual confidence, fixed stage priority, then label order."
    )
    return FinalDecision(
        label=winning_label,
        confidence=final_confidence,
        conflict_detected=conflict_detected,
        selected_stage=selected_vote.stage,
        resolution_reason=reason,
    )
