# API Reference

Run from the repository root with `cd training; uvicorn server:app --reload`.

## POST /events

Accepts an event JSON object with non-empty `id`, ISO-8601 `timestamp`,
non-empty `source`, and non-empty `message`. Extra fields are rejected.

Successful requests return `200` with stage signals, a reconciled decision,
temporal reconciliation metadata, and the persisted canonical audit record.

- `400`: malformed JSON or invalid/missing fields.
- `409`: an event with the same ID and canonical UTC timestamp already exists.

## POST /replay

Accepts `{ "event_ids": ["id-1", "id-2"] }`. Each stored event is
classified and resolved again in its original ingestion order. The rebuilt
audit, reconciliation result, and final per-ID state must equal their persisted
records. The response returns rebuilt audit records and `reconstructed_state`;
replay does not write new state.

- `400`: missing/invalid event-ID list, including duplicate IDs.
- `404`: an ID has no stored event.
- `409`: deterministic replay result differs from its saved record.

## Determinism policy

The local LLM mock is a version-controlled JSON rule file. Resolver weights and
tie-breakers are fixed in code. Audit decision time is the canonical event time
rather than wall-clock time, making repeated processing byte-for-byte stable.
