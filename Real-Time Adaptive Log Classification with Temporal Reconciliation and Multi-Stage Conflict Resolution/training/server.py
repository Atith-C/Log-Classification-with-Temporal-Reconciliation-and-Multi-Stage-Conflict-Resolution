import os
from collections import defaultdict
from datetime import timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

try:
    from .schemas import AuditRecord, EventClassificationResponse, LogEvent, ReplayRequest, ReplayResponse, ReplayedState, TemporalReconciliation
except ImportError:  # Supports `uvicorn server:app` from the training directory.
    from schemas import AuditRecord, EventClassificationResponse, LogEvent, ReplayRequest, ReplayResponse, ReplayedState, TemporalReconciliation

app = FastAPI(
    title="Real-Time Adaptive Log Classification API",
    version="0.1.0",
    description="Deterministic event ingestion API. Phase 1 provides contract validation.",
)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_, exc: RequestValidationError) -> JSONResponse:
    """Translate FastAPI's default validation response to the required HTTP 400."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})


def generate_signals(source: str, message: str):
    """Lazy import keeps schema-only API checks independent from ML model loading."""
    try:
        from .classify import classify_event
    except ImportError:
        from classify import classify_event
    return classify_event(source, message)


def resolve_event_signals(signals, timestamp):
    """Lazy import keeps API schema checks independent from resolver imports."""
    try:
        from .resolver import resolve_signals
    except ImportError:
        from resolver import resolve_signals
    return resolve_signals(signals, timestamp)


_state_store = None


def get_state_store():
    """Create the local SQLite store on first use for this process."""
    global _state_store
    if _state_store is None:
        try:
            from .state_store import SQLiteStateStore
        except ImportError:
            from state_store import SQLiteStateStore
        _state_store = SQLiteStateStore()
    return _state_store


@app.post(
    "/events",
    response_model=EventClassificationResponse,
    status_code=status.HTTP_200_OK,
    tags=["events"],
)
async def receive_event(event: LogEvent) -> EventClassificationResponse:
    """Validate one event and produce its deterministic classification signals."""
    store = get_state_store()
    if store.has_event(event):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate event: this id and timestamp have already been processed.",
        )
    signals = generate_signals(event.source, event.message)
    final_decision = resolve_event_signals(signals, event.timestamp)
    try:
        from .state_store import DuplicateEventError
    except ImportError:
        from state_store import DuplicateEventError
    try:
        reconciliation, audit = store.record_event(event, signals, final_decision)
    except DuplicateEventError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate event: this id and timestamp have already been processed.",
        ) from error
    return EventClassificationResponse(
        event=event,
        signals=signals,
        final_decision=final_decision,
        reconciliation=reconciliation,
        audit=audit,
    )


@app.post("/replay", response_model=ReplayResponse, tags=["events"])
async def replay_events(request: ReplayRequest) -> ReplayResponse:
    """Rebuild and verify every decision, audit record, and final state requested."""
    store = get_state_store()
    stored = store.get_replay_events(request.event_ids)
    found_ids = {event.id for event, _ in stored}
    missing_ids = [event_id for event_id in request.event_ids if event_id not in found_ids]
    if missing_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown event IDs: {missing_ids}")
    histories: dict[str, list[tuple[LogEvent, object]]] = defaultdict(list)
    rebuilt_audits: list[AuditRecord] = []
    rebuilt_state: dict[str, ReplayedState] = {}

    for event, audit in stored:
        signals = generate_signals(event.source, event.message)
        decision = resolve_event_signals(signals, event.timestamp)
        history = histories[event.id]
        previous_effective_timestamp = max(
            (historic_event.timestamp.astimezone(timezone.utc) for historic_event, _ in history),
            default=None,
        )
        event_timestamp = event.timestamp.astimezone(timezone.utc)
        history.append((event, decision))
        effective_event, effective_decision = max(
            history, key=lambda item: item[0].timestamp.astimezone(timezone.utc)
        )
        was_late = previous_effective_timestamp is not None and event_timestamp < previous_effective_timestamp
        reconciliation = TemporalReconciliation(
            was_late=was_late,
            state_updated=previous_effective_timestamp is None or event_timestamp > previous_effective_timestamp,
            effective_event_timestamp=effective_event.timestamp,
            replayed_event_count=len(history),
            reason=(
                "Event arrived after the current effective timestamp; state was reconciled from chronological history."
                if was_late else "State was reconciled from chronological event history."
            ),
        )
        rebuilt_audit = AuditRecord(
            event_id=event.id,
            event_timestamp=event.timestamp,
            signals=signals,
            final_decision=decision,
            decision_timestamp=event.timestamp,
            reconciliation=reconciliation,
        )
        if rebuilt_audit != audit:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Replay mismatch for event {event.id}.")
        rebuilt_audits.append(rebuilt_audit)
        rebuilt_state[event.id] = ReplayedState(
            effective_event_timestamp=effective_event.timestamp,
            final_decision=effective_decision,
        )

    for event_id, reconstructed in rebuilt_state.items():
        persisted = store.get_current_state(event_id)
        if persisted is None or (
            persisted["effective_event_timestamp"] != reconstructed.effective_event_timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            or persisted["label"] != reconstructed.final_decision.label
            or persisted["confidence"] != reconstructed.final_decision.confidence
            or bool(persisted["conflict_detected"]) != reconstructed.final_decision.conflict_detected
            or persisted["selected_stage"] != reconstructed.final_decision.selected_stage
            or persisted["resolution_reason"] != reconstructed.final_decision.resolution_reason
        ):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Replayed state mismatch for event {event_id}.")
    return ReplayResponse(audit_records=rebuilt_audits, reconstructed_state=rebuilt_state)

@app.post("/classify/")
async def classify_logs(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    try:
        # Read uploaded CSV into DataFrame
        df = pd.read_csv(file.file)
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        # Perform classification
        try:
            from .classify import classify
        except ImportError:
            from classify import classify
        df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

        print("Processed DataFrame:", df.head().to_dict())

        # Ensure resources folder exists
        output_dir = "resources"
        os.makedirs(output_dir, exist_ok=True)

        # Save classified results
        output_file = os.path.join(output_dir, "output.csv")
        df.to_csv(output_file, index=False)
        print(f"File saved to {output_file}")

        return FileResponse(output_file, media_type="text/csv", filename="output.csv")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        file.file.close()
