import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import psycopg
import pytest
from fastapi.testclient import TestClient

from stationwatch.app import create_app
from stationwatch.postgres import PostgresEventStore


def test_health_is_public_and_never_calls_database():
    store = Mock()
    store.counts.side_effect = AssertionError("liveness must not query the database")
    with TestClient(create_app(None, store, "private-dashboard-token")) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz").json() == {"ok": True}
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/events").status_code == 401
        assert client.post("/api/events/test/reviewed").status_code == 401
        assert store.mock_calls == []
    store.close.assert_called_once()


@pytest.mark.parametrize("error", [psycopg.OperationalError, psycopg.errors.UndefinedTable])
def test_database_outage_does_not_break_liveness_or_expose_credentials(monkeypatch, error):
    connect = Mock(side_effect=error("postgresql://secret-password@private-host/db"))
    monkeypatch.setattr(psycopg, "connect", connect)
    with TestClient(create_app(None, PostgresEventStore(), "test-token")) as client:
        assert client.get("/healthz").status_code == 200
        connect.assert_not_called()
        response = client.get("/api/events", headers={"Authorization": "Bearer test-token"})
        assert response.status_code == 503
        assert "secret-password" not in response.text
        assert "private-host" not in response.text
        assert client.get("/healthz").status_code == 200


def test_postgres_env_fields_override_uri_without_connecting(monkeypatch):
    env = {
        "PGUSER": "hub",
        "PGDATABASE": "alerts",
        "PGHOST": "database.example",
        "PGPORT": "25060",
        "PGSSLMODE": "require",
        "PGPASSWORD": "password:@/?#'",
        "TIMESCALE_SERVICE_URL": "postgresql://fallback:secret@fallback.example/fallback",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    connect = Mock()
    monkeypatch.setattr(psycopg, "connect", connect)
    store = PostgresEventStore.from_env()
    assert store.conninfo == env["TIMESCALE_SERVICE_URL"]
    assert store.connect_kwargs == {
        "user": "hub",
        "dbname": "alerts",
        "host": "database.example",
        "port": "25060",
        "sslmode": "require",
        "password": "password:@/?#'",
        "connect_timeout": 5,
        "options": "-c statement_timeout=10000",
    }
    connect.assert_not_called()


def test_uri_only_configuration(monkeypatch):
    for name in ("PGUSER", "PGDATABASE", "PGHOST", "PGPORT", "PGSSLMODE", "PGPASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIMESCALE_SERVICE_URL", "postgresql://user:secret@host/db?sslmode=require")
    store = PostgresEventStore.from_env()
    assert store.conninfo.endswith("/db?sslmode=require")
    assert set(store.connect_kwargs) == {"connect_timeout", "options"}


def test_hub_startup_has_no_camera_ml_or_database_dependency():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys

class BlockCameraImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('cv2', 'numpy', 'mediapipe', 'sklearn', 'joblib', 'ai_edge_litert',
                   'pymongo', 'stationwatch.camera', 'stationwatch.pipeline',
                   'stationwatch.features', 'stationwatch.model', 'stationwatch.image_model')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Unexpected hub import: ' + fullname)

sys.meta_path.insert(0, BlockCameraImports())
import psycopg
def no_database(*args, **kwargs):
    raise AssertionError('Hub startup connected to the database')
psycopg.connect = no_database
import stationwatch.hub
from fastapi.testclient import TestClient
with TestClient(stationwatch.hub.app) as client:
    assert client.get('/healthz').status_code == 200
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


@pytest.mark.parametrize("port", [None, "9123"])
def test_entrypoint_binds_public_interface_and_uses_port(monkeypatch, port):
    import uvicorn

    from stationwatch import hub

    if port is None:
        monkeypatch.delenv("PORT", raising=False)
    else:
        monkeypatch.setenv("PORT", port)
    run = Mock()
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    monkeypatch.setattr(uvicorn, "run", run)
    hub.main()
    assert run.call_args.kwargs["host"] == "0.0.0.0"
    assert run.call_args.kwargs["port"] == (8080 if port is None else int(port))
    assert run.call_args.kwargs["workers"] == 1
    assert run.call_args.kwargs["proxy_headers"] is True
    assert run.call_args.kwargs["forwarded_allow_ips"] == "*"


def test_entrypoint_reads_dashboard_token(monkeypatch):
    from stationwatch import hub

    monkeypatch.setenv("DASHBOARD_TOKEN", "deployment-secret")
    importlib.reload(hub)
    try:
        with TestClient(hub.app) as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/api/events").status_code == 401
            assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
            assert client.post("/api/login", json={"token": "deployment-secret"}).status_code == 200
    finally:
        monkeypatch.delenv("DASHBOARD_TOKEN")
        importlib.reload(hub)


def test_login_behind_https_proxy_preserves_origin_and_secure_cookie():
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    app = ProxyHeadersMiddleware(create_app(None, Mock(), "test-token"), trusted_hosts="*")
    with TestClient(app, base_url="http://hub.example") as client:
        response = client.post(
            "/api/login",
            json={"token": "test-token"},
            headers={"Origin": "https://hub.example", "X-Forwarded-Proto": "https"},
        )
        assert response.status_code == 200
        assert "Secure" in response.headers["set-cookie"]


@pytest.fixture
def postgres_connection(monkeypatch):
    connection = MagicMock()
    connect = Mock(return_value=connection)
    connection.__enter__.return_value = connection
    monkeypatch.setattr(psycopg, "connect", connect)
    return connection


def test_postgres_read_and_review_contract(postgres_connection):
    store = PostgresEventStore()
    connection = postgres_connection
    event_id = "an-id-with-'quotes"
    row = {"id": event_id, "payload": {"id": "ignored", "station_id": "press-01"}, "reviewed": True}
    connection.execute.return_value.fetchall.return_value = [row]
    connection.execute.return_value.fetchone.return_value = row
    expected = {"id": event_id, "station_id": "press-01", "reviewed": True}
    assert store.list() == [expected]
    assert connection.execute.call_args.args[1] == (100,)
    assert store.get(event_id) == expected
    assert connection.execute.call_args.args[1] == (event_id,)
    assert event_id not in connection.execute.call_args.args[0]
    connection.execute.return_value.fetchone.return_value = None
    assert store.get("missing") is None
    connection.execute.return_value.rowcount = 0
    assert store.mark_reviewed("missing") is False
    connection.execute.return_value.rowcount = 1
    assert store.mark_reviewed(event_id) is True
    assert connection.execute.call_args.args[1] == (event_id,)


def test_postgres_count_contract(postgres_connection):
    postgres_connection.execute.return_value.fetchone.return_value = {"total": 3, "pending": 2, "stations": 2}
    assert PostgresEventStore().counts() == {"total": 3, "pending": 2, "reviewed": 1, "stations": 2}


def test_postgres_evidence_cache_and_video_ranges(tmp_path, postgres_connection):
    store = PostgresEventStore(cache_dir=tmp_path)
    connection = postgres_connection
    event = {"id": "event-1", "payload": {"station_id": "press-01"}, "reviewed": False}

    def execute(query, params):
        cursor = Mock()
        cursor.fetchone.return_value = {"content": b"recorded-video"} if "content" in query else event
        return cursor

    connection.execute.side_effect = execute
    with TestClient(create_app(None, store, "test-token")) as client:
        client.headers["Authorization"] = "Bearer test-token"
        response = client.get("/api/events/event-1/video", headers={"Range": "bytes=0-3"})
        assert response.status_code == 206
        assert response.content == b"reco"
        connection.execute.reset_mock()
        assert store.media_path("event-1", "clip").read_bytes() == b"recorded-video"
        connection.execute.assert_not_called()
        with pytest.raises(ValueError):
            store.media_path("event-1", "../../secrets")


def test_postgres_missing_evidence(tmp_path, postgres_connection):
    store = PostgresEventStore(cache_dir=tmp_path)
    postgres_connection.execute.return_value.fetchone.return_value = None
    with pytest.raises(FileNotFoundError):
        store.media_path("missing", "snapshot")
    assert list(tmp_path.iterdir()) == []
