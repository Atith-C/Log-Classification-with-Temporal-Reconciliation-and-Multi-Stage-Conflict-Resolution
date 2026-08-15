"""SQLite-backed deterministic state for processed classification events."""

import sqlite3
import json
from datetime import timezone
from pathlib import Path

try:
    from .schemas import AuditRecord, ClassificationSignal, FinalDecision, LogEvent, TemporalReconciliation
except ImportError:
    from schemas import AuditRecord, ClassificationSignal, FinalDecision, LogEvent, TemporalReconciliation


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "classification_state.sqlite3"


class DuplicateEventError(Exception):
    """Raised when an event ID and timestamp have already been recorded."""


class SQLiteStateStore:
    """Persist original events and their current deterministic decision locally."""

    def __init__(self, database_path: Path | str = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    PRIMARY KEY (event_id, event_timestamp)
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    event_id TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    conflict_detected INTEGER NOT NULL,
                    selected_stage TEXT,
                    resolution_reason TEXT NOT NULL,
                    PRIMARY KEY (event_id, event_timestamp),
                    FOREIGN KEY (event_id, event_timestamp)
                        REFERENCES events (event_id, event_timestamp)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS signals (
                    event_id TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    matched INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY (event_id, event_timestamp, stage),
                    FOREIGN KEY (event_id, event_timestamp)
                        REFERENCES events (event_id, event_timestamp)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS log_state (
                    event_id TEXT PRIMARY KEY,
                    effective_event_timestamp TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    conflict_detected INTEGER NOT NULL,
                    selected_stage TEXT,
                    resolution_reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_records (
                    event_id TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    audit_json TEXT NOT NULL,
                    PRIMARY KEY (event_id, event_timestamp),
                    FOREIGN KEY (event_id, event_timestamp) REFERENCES events (event_id, event_timestamp) ON DELETE CASCADE
                );
                """
            )

    @staticmethod
    def _timestamp(event: LogEvent) -> str:
        """Canonical UTC timestamp which sorts correctly in SQLite text order."""
        return event.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def has_event(self, event: LogEvent) -> bool:
        """Return whether this exact id-and-timestamp event was processed."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM events WHERE event_id = ? AND event_timestamp = ?",
                (event.id, self._timestamp(event)),
            ).fetchone()
        return row is not None

    def record_event(
        self, event: LogEvent, signals: list[ClassificationSignal], decision: FinalDecision
    ) -> tuple[TemporalReconciliation, AuditRecord]:
        """Atomically persist an event and rebuild its time-ordered log state."""
        timestamp = self._timestamp(event)
        try:
            with self._connect() as connection:
                previous_state = connection.execute(
                    "SELECT effective_event_timestamp FROM log_state WHERE event_id = ?", (event.id,)
                ).fetchone()
                connection.execute(
                    "INSERT INTO events (event_id, event_timestamp, source, message) VALUES (?, ?, ?, ?)",
                    (event.id, timestamp, event.source, event.message),
                )
                connection.execute(
                    """INSERT INTO decisions
                    (event_id, event_timestamp, label, confidence, conflict_detected, selected_stage, resolution_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.id, timestamp, decision.label, decision.confidence,
                        int(decision.conflict_detected), decision.selected_stage, decision.resolution_reason,
                    ),
                )
                connection.executemany(
                    """INSERT INTO signals
                    (event_id, event_timestamp, stage, label, confidence, matched, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (event.id, timestamp, signal.stage, signal.label, signal.confidence, int(signal.matched), signal.reason)
                        for signal in signals
                    ],
                )
                ordered_history = connection.execute(
                    """SELECT e.event_timestamp, d.label, d.confidence, d.conflict_detected,
                              d.selected_stage, d.resolution_reason
                       FROM events AS e
                       JOIN decisions AS d
                         ON d.event_id = e.event_id AND d.event_timestamp = e.event_timestamp
                       WHERE e.event_id = ?
                       ORDER BY e.event_timestamp ASC""",
                    (event.id,),
                ).fetchall()
                effective = ordered_history[-1]
                connection.execute(
                    """INSERT INTO log_state
                    (event_id, effective_event_timestamp, label, confidence, conflict_detected, selected_stage, resolution_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        effective_event_timestamp = excluded.effective_event_timestamp,
                        label = excluded.label,
                        confidence = excluded.confidence,
                        conflict_detected = excluded.conflict_detected,
                        selected_stage = excluded.selected_stage,
                        resolution_reason = excluded.resolution_reason""",
                    (
                        event.id, effective["event_timestamp"], effective["label"], effective["confidence"],
                        effective["conflict_detected"], effective["selected_stage"], effective["resolution_reason"],
                    ),
                )
                was_late = previous_state is not None and timestamp < previous_state["effective_event_timestamp"]
                state_updated = previous_state is None or effective["event_timestamp"] != previous_state["effective_event_timestamp"]
                reason = (
                    "Event arrived after the current effective timestamp; state was reconciled from chronological history."
                    if was_late else "State was reconciled from chronological event history."
                )
                reconciliation = TemporalReconciliation(
                    was_late=was_late,
                    state_updated=state_updated,
                    effective_event_timestamp=effective["event_timestamp"],
                    replayed_event_count=len(ordered_history),
                    reason=reason,
                )
                audit = AuditRecord(
                    event_id=event.id, event_timestamp=event.timestamp, signals=signals,
                    final_decision=decision, decision_timestamp=event.timestamp,
                    reconciliation=reconciliation,
                )
                connection.execute(
                    "INSERT INTO audit_records (event_id, event_timestamp, audit_json) VALUES (?, ?, ?)",
                    (event.id, timestamp, json.dumps(audit.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))),
                )
            return reconciliation, audit
        except sqlite3.IntegrityError as error:
            if "UNIQUE constraint failed: events.event_id, events.event_timestamp" in str(error):
                raise DuplicateEventError from error
            raise

    def get_current_state(self, event_id: str) -> dict | None:
        """Return the deterministic current state for a log ID."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM log_state WHERE event_id = ?", (event_id,)).fetchone()
        return dict(row) if row else None

    def get_replay_events(self, event_ids: list[str]) -> list[tuple[LogEvent, AuditRecord]]:
        """Return stored events in their original ingestion order for each requested ID."""
        result = []
        with self._connect() as connection:
            for event_id in event_ids:
                rows = connection.execute(
                    "SELECT rowid AS ingestion_order, event_timestamp, source, message "
                    "FROM events WHERE event_id = ? ORDER BY ingestion_order",
                    (event_id,),
                ).fetchall()
                for row in rows:
                    audit_row = connection.execute(
                        "SELECT audit_json FROM audit_records WHERE event_id = ? AND event_timestamp = ?", (event_id, row["event_timestamp"])
                    ).fetchone()
                    audit = AuditRecord.model_validate_json(audit_row["audit_json"])
                    result.append((
                        # Preserve the original timestamp representation from the audit so
                        # replay can verify the complete canonical audit byte-for-byte.
                        LogEvent(id=event_id, timestamp=audit.event_timestamp, source=row["source"], message=row["message"]),
                        audit,
                    ))
        return result

    def event_count(self) -> int:
        """Small read helper used by automated tests and diagnostics."""
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
