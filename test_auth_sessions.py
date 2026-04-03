from contextlib import contextmanager

import app_services


class _FakeCursor:
    def __init__(self, fetchone_values=None):
        self.fetchone_values = list(fetchone_values or [])
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        if self.fetchone_values:
            return self.fetchone_values.pop(0)
        return None


@contextmanager
def _cursor_cm(cursor):
    yield cursor


def test_create_auth_session_stores_hashed_token(monkeypatch):
    cursor = _FakeCursor()
    monkeypatch.setattr(app_services, "db_cursor", lambda: _cursor_cm(cursor))
    monkeypatch.setattr(app_services.secrets, "token_urlsafe", lambda n: "raw-session-token")

    token = app_services.create_auth_session("brandon")

    assert token == "raw-session-token"
    assert len(cursor.calls) == 1
    _, params = cursor.calls[0]
    assert params[0] != token
    assert len(params[0]) == 64
    assert params[1] == "brandon"


def test_get_auth_session_restores_user_and_refreshes_expiry(monkeypatch):
    cursor = _FakeCursor(fetchone_values=[("brandon", "admin", 19)])
    monkeypatch.setattr(app_services, "db_cursor", lambda: _cursor_cm(cursor))

    session = app_services.get_auth_session("raw-session-token")

    assert session == {
        "username": "brandon",
        "role": "admin",
        "credits_per_week": 19,
    }
    assert len(cursor.calls) == 3
    assert "DELETE FROM memo_chef_auth_sessions" in cursor.calls[0][0]
    assert "SELECT u.username, u.role, u.credits_per_week" in cursor.calls[1][0]
    assert "UPDATE memo_chef_auth_sessions" in cursor.calls[2][0]


def test_get_auth_session_returns_none_when_missing(monkeypatch):
    cursor = _FakeCursor(fetchone_values=[None])
    monkeypatch.setattr(app_services, "db_cursor", lambda: _cursor_cm(cursor))

    session = app_services.get_auth_session("missing-token")

    assert session is None
    assert len(cursor.calls) == 2


def test_revoke_auth_session_skips_empty_token(monkeypatch):
    called = False

    @contextmanager
    def _unexpected_cursor():
        nonlocal called
        called = True
        yield _FakeCursor()

    monkeypatch.setattr(app_services, "db_cursor", _unexpected_cursor)

    app_services.revoke_auth_session(None)

    assert not called
