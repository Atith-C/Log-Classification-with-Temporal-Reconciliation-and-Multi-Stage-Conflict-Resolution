import json
from pathlib import Path


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "edge_case_events.json"


def test_fixture_set_has_25_documented_interacting_edge_cases() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    names = {case["case"] for case in cases}

    assert len(cases) >= 25
    assert len(names) == len(cases)
    assert {"duplicate_repeat", "same_id_late", "conflicting_signals", "replay_target_one"} <= names


def test_fixture_set_covers_all_required_invalid_payload_types() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    names = {case["case"] for case in cases}

    assert {"missing_id", "missing_timestamp", "missing_source", "missing_message", "blank_source", "bad_timestamp", "unexpected_field"} <= names
