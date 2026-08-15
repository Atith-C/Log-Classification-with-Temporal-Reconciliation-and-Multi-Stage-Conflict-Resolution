import pytest

from training import server
from training.state_store import SQLiteStateStore


@pytest.fixture(autouse=True)
def isolated_sqlite_state(monkeypatch, tmp_path):
    """Keep API tests independent from the persistent developer database."""
    store = SQLiteStateStore(tmp_path / "test_state.sqlite3")
    monkeypatch.setattr(server, "get_state_store", lambda: store)
    return store
