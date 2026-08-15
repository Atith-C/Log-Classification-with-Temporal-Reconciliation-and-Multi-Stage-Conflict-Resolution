"""Measure end-to-end event-ingestion throughput and process memory locally."""

import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fastapi.testclient import TestClient

from training import server
from training.schemas import LogEvent
from training.state_store import SQLiteStateStore


def process_rss_bytes() -> int:
    """Return current resident memory without a third-party dependency."""
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process_memory_info = ctypes.WinDLL("Psapi.dll", use_last_error=True).GetProcessMemoryInfo
        process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        process_memory_info.restype = wintypes.BOOL
        process_handle = ctypes.WinDLL("kernel32", use_last_error=True).GetCurrentProcess()
        success = process_memory_info(process_handle, ctypes.byref(counters), counters.cb)
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)

    import resource
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak_kib * 1024 if sys.platform != "darwin" else peak_kib)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the full FastAPI + SQLite event pipeline.")
    parser.add_argument("--events", type=Path, default=Path("fixtures/edge_case_events.json"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("examples/benchmark_result.json"))
    arguments = parser.parse_args()
    if arguments.count < 1:
        parser.error("--count must be at least 1")

    entries = json.loads(arguments.events.read_text(encoding="utf-8"))
    valid = []
    for entry in entries:
        if not {"id", "timestamp", "source", "message"} <= entry.keys():
            continue
        payload = {field: entry[field] for field in ("id", "timestamp", "source", "message")}
        try:
            LogEvent.model_validate(payload)
        except ValueError:
            continue
        valid.append(payload)
    if not valid:
        parser.error("fixture file contains no valid events")

    temporary_directory = Path(tempfile.mkdtemp(prefix="log-classification-benchmark-"))
    original_store = server._state_store
    try:
        server._state_store = SQLiteStateStore(temporary_directory / "benchmark.sqlite3")
        client = TestClient(server.app)
        warmup = valid[0]
        warmup_response = client.post(
            "/events", json={**warmup, "id": "benchmark-warmup", "timestamp": "2026-12-31T23:59:59Z"}
        )
        if warmup_response.status_code != 200:
            raise RuntimeError(f"Warm-up request failed: {warmup_response.status_code} {warmup_response.text}")

        baseline_rss = process_rss_bytes()
        peak_rss = baseline_rss
        start = time.perf_counter()
        for index in range(arguments.count):
            entry = valid[index % len(valid)]
            response = client.post(
                "/events",
                json={
                    **entry,
                    "id": f"benchmark-{index}",
                    "timestamp": f"2026-12-30T00:{index // 60:02d}:{index % 60:02d}Z",
                },
            )
            if response.status_code != 200:
                raise RuntimeError(f"Benchmark request {index} failed: {response.status_code} {response.text}")
            peak_rss = max(peak_rss, process_rss_bytes())
        elapsed = time.perf_counter() - start

        result = {
            "benchmark": "FastAPI TestClient POST /events with full classification and temporary SQLite persistence",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "events": arguments.count,
            "seconds": round(elapsed, 4),
            "events_per_second": round(arguments.count / elapsed, 2),
            "baseline_rss_mb": round(baseline_rss / 1024**2, 2),
            "peak_rss_mb": round(peak_rss / 1024**2, 2),
            "memory_limit_mb": 512,
            "memory_limit_met": peak_rss <= 512 * 1024**2,
            "throughput_target_events_per_second": 100,
            "throughput_target_met": arguments.count / elapsed >= 100,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        server._state_store = original_store
        shutil.rmtree(temporary_directory, ignore_errors=True)


if __name__ == "__main__":
    main()
