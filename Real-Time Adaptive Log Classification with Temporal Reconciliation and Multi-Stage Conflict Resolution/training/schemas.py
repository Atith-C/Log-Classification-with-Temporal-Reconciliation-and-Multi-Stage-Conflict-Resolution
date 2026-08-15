"""HTTP request and response models for the event-processing API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LogEvent(BaseModel):
    """The immutable event contract accepted by ``POST /events``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(..., min_length=1, max_length=128, description="Stable event identifier")
    timestamp: datetime = Field(..., description="ISO-8601 event timestamp")
    source: str = Field(..., min_length=1, max_length=128, description="Originating system")
    message: str = Field(..., min_length=1, max_length=10000, description="Raw log message")

    @field_validator("id", "source", "message")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value


class ClassificationSignal(BaseModel):
    """A deterministic output from one classification stage."""

    model_config = ConfigDict(frozen=True)

    stage: Literal["regex", "bert", "llm"]
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched: bool
    reason: str


class FinalDecision(BaseModel):
    """The deterministic classification selected from the stage signals."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    conflict_detected: bool
    selected_stage: Literal["regex", "bert", "llm"] | None
    resolution_reason: str


class TemporalReconciliation(BaseModel):
    """State-transition result after placing an event in its time-ordered history."""

    was_late: bool
    state_updated: bool
    effective_event_timestamp: datetime
    replayed_event_count: int = Field(ge=1)
    reason: str


class AuditRecord(BaseModel):
    event_id: str
    event_timestamp: datetime
    signals: list[ClassificationSignal]
    final_decision: FinalDecision
    decision_timestamp: datetime
    reconciliation: TemporalReconciliation
    configuration_version: str = "pipeline-v1"


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("event_ids")
    @classmethod
    def reject_duplicate_event_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("event_ids must not contain duplicates")
        return value


class ReplayedState(BaseModel):
    """The state rebuilt from a replayed event history for one log ID."""

    effective_event_timestamp: datetime
    final_decision: FinalDecision


class ReplayResponse(BaseModel):
    status: str = "replayed"
    audit_records: list[AuditRecord]
    reconstructed_state: dict[str, ReplayedState]


class EventClassificationResponse(BaseModel):
    """Phase-2 response before the final conflict resolver is introduced."""

    status: str = "classified"
    event: LogEvent
    signals: list[ClassificationSignal]
    final_decision: FinalDecision
    reconciliation: TemporalReconciliation
    audit: AuditRecord
    detail: str = "Signals generated and resolved deterministically."
