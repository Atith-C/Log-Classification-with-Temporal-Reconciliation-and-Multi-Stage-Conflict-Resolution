# Real-Time Adaptive Log Classification

A deterministic FastAPI service for classifying individual log events in real
time. It combines regex rules, Sentence-BERT embeddings with a supplied
Logistic Regression classifier, and a local mocked LLM response file. The
service persists every decision in SQLite, detects duplicate submissions,
handles out-of-order events, and can replay saved histories to verify their
audit trail and final state.

## How it works

Each valid `POST /events` request passes through this synchronous, single-
process pipeline:

1. Validate `id`, `timestamp`, `source`, and `message`.
2. Produce a regex classification signal from predefined patterns.
3. Produce a Sentence-BERT embedding and Logistic Regression prediction.
4. Use the local mocked LLM fallback when the signals are ambiguous, disagree,
   or have low ML confidence. It never calls an external LLM API.
5. Resolve conflicting matched signals deterministically using fixed weighted
   voting. Ties use confidence, fixed stage priority (`regex`, `bert`, `llm`),
   then label order.
6. Save the event, signals, final decision, temporal reconciliation result,
   and canonical audit record in local SQLite.

An event that arrives later than a newer event with the same ID is marked as
late. Current per-ID state is always derived from the event with the latest
timestamp.

## Project layout

- `training/server.py` — FastAPI endpoints.
- `training/classify.py` — hybrid classification pipeline.
- `training/processor_regex.py` — rule-based stage.
- `training/processor_bert.py` — Sentence-BERT and Logistic Regression stage.
- `training/processor_llm.py` — deterministic local LLM mock.
- `training/resolver.py` — deterministic conflict resolver.
- `training/state_store.py` — SQLite events, state, and audits.
- `fixtures/edge_case_events.json` — 25 edge-case fixture events.
- `tests/` — automated API, classifier, state, reconciliation, and replay tests.

## Prerequisites

- Windows PowerShell (commands below are written for Windows).
- Python **3.12**. Python 3.14 is not supported by the pinned dependency stack.
- Internet access on the first BERT request if `all-MiniLM-L6-v2` has not
  already been cached locally. The LLM itself remains fully local and mocked.

## Setup

From the repository root, enter these commands **one line at a time**:

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal and then
repeat the activation command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Start the server

From the repository root, activate the environment and enter:

```powershell
cd training
```

```powershell
uvicorn server:app --reload
```

Keep that terminal open. When it shows `Uvicorn running on
http://127.0.0.1:8000`, open the interactive API interface:

```text
http://127.0.0.1:8000/docs
```

`http://127.0.0.1:8000/` intentionally returns `{"detail":"Not Found"}`
because this project has no homepage endpoint. Stop the server with `Ctrl+C`.

## API usage

### Submit an event

Use `POST /events` in `/docs`, or run this in a second activated PowerShell
terminal:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/events -ContentType application/json -Body '{"id":"demo-001","timestamp":"2026-08-15T10:30:00Z","source":"ModernCRM","message":"Backup completed successfully."}' | ConvertTo-Json -Depth 10
```

A successful response is HTTP `200` and contains `signals`, `final_decision`,
`reconciliation`, and `audit`. For the sample backup event, the final label
should be `System Notification`.

- `400 Bad Request` — missing, invalid, blank, or extra event fields.
- `409 Conflict` — the same ID and canonical UTC timestamp were already
  processed. This is the expected idempotency behavior.

The supported event shape is:

```json
{
  "id": "evt-001",
  "timestamp": "2026-08-15T10:30:00Z",
  "source": "ModernCRM",
  "message": "Backup completed successfully."
}
```

### Replay saved decisions

After submitting an event, replay it with:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/replay -ContentType application/json -Body '{"event_ids":["demo-001"]}' | ConvertTo-Json -Depth 10
```

Replay re-runs classification in the original ingestion order, rebuilds the
full audit and reconciliation metadata, verifies it against the stored audit,
verifies reconstructed state against SQLite, and returns `reconstructed_state`.
It does not modify saved events or decisions.

- `400 Bad Request` — invalid request or duplicate IDs in the replay list.
- `404 Not Found` — an event ID has not been stored.
- `409 Conflict` — replayed classification, audit, or state differs from its
  saved deterministic result.

The legacy `POST /classify/` CSV endpoint remains available for batch CSV
compatibility. It expects `source` and `log_message` columns.

## State and audit data

Runtime state is stored locally in `training/data/classification_state.sqlite3`.
It is ignored by Git. Each successful event stores its original payload, stage
signals, final decision, current per-ID state, and a JSON audit record.

## Tests

From the repository root, with `.venv` activated, run:

```powershell
python -m pytest -q
```

The suite covers validation, duplicate handling, classification signals,
conflict resolution, SQLite persistence, late/out-of-order events, replay,
and fixtures.

## Benchmark

Run the end-to-end application benchmark from the repository root:

```powershell
python scripts\benchmark.py --count 100
```

It warms the BERT model, sends 100 unique events through FastAPI `/events`,
uses temporary SQLite persistence, measures steady-state throughput and process
RAM, and writes the result to `examples/benchmark_result.json`. The result is
hardware- and model-cache-dependent. See the JSON output to determine whether
the 100 events/sec and 512 MB targets were met on your machine.
