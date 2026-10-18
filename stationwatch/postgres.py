"""PostgreSQL/Timescale alert reader for the camera-free cloud hub.

Schema is provisioned separately with deploy/postgres.sql. No connection or DDL
is performed at import/startup, so liveness does not depend on the database.
"""

import hashlib
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .storage import DatabaseUnavailable


class PostgresEventStore:
    def __init__(self, conninfo="", connect_kwargs=None, cache_dir=None):
        self.conninfo = conninfo
        self.connect_kwargs = connect_kwargs or {}
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "stationwatch-hub-cache"
        self.cache_lock = threading.Lock()

    @classmethod
    def from_env(cls):
        # Pass fields separately rather than interpolating passwords into a URI.
        # Explicit PG* values override matching fields in the Timescale URI.
        fields = {
            "PGUSER": "user",
            "PGDATABASE": "dbname",
            "PGHOST": "host",
            "PGPORT": "port",
            "PGSSLMODE": "sslmode",
            "PGPASSWORD": "password",
        }
        kwargs = {field: os.environ[name] for name, field in fields.items() if os.environ.get(name)}
        kwargs.update(connect_timeout=5, options="-c statement_timeout=10000")
        return cls(os.environ.get("TIMESCALE_SERVICE_URL", ""), kwargs)

    @contextmanager
    def _connection(self):
        try:
            with psycopg.connect(self.conninfo, row_factory=dict_row, **self.connect_kwargs) as connection:
                yield connection
        except psycopg.Error:
            # Driver errors can contain credentials/hosts. Never expose them to HTTP clients.
            raise DatabaseUnavailable("Alert database unavailable") from None

    @staticmethod
    def _event(row):
        if row is None:
            return None
        return dict(row["payload"], id=row["id"], reviewed=row["reviewed"])

    def list(self, limit=100):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, payload, reviewed FROM stationwatch_alerts ORDER BY captured_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [self._event(row) for row in rows]

    def get(self, event_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, payload, reviewed FROM stationwatch_alerts WHERE id = %s", (event_id,)
            ).fetchone()
        return self._event(row)

    def mark_reviewed(self, event_id):
        with self._connection() as connection:
            return (
                connection.execute(
                    "UPDATE stationwatch_alerts SET reviewed = TRUE WHERE id = %s", (event_id,)
                ).rowcount
                > 0
            )

    def counts(self):
        with self._connection() as connection:
            row = connection.execute(
                """SELECT count(*) AS total, count(*) FILTER (WHERE NOT reviewed) AS pending,
                count(DISTINCT payload->>'station_id') AS stations FROM stationwatch_alerts"""
            ).fetchone()
        return dict(row, reviewed=row["total"] - row["pending"])

    def publish_alert(self, event, snapshot):
        if event.get("mode") == "demo" or not event.get("local", {}).get("flagged"):
            raise ValueError("Only live, flagged events may be uploaded")
        if event.get("review", {}).get("decision") != "inattentive":
            raise ValueError("Only Gemini-confirmed events may be uploaded")
        content = snapshot.read_bytes()
        with self._connection() as connection:
            # Retries preserve both the original evidence and the manager's review status.
            connection.execute(
                """INSERT INTO stationwatch_alerts(id, captured_at, payload)
                VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING""",
                (event["id"], event["captured_at"], Jsonb(dict(event, clip_state="pending"))),
            )
            connection.execute(
                """INSERT INTO stationwatch_evidence(event_id, kind, content)
                VALUES (%s, 'snapshot', %s) ON CONFLICT (event_id, kind) DO NOTHING""",
                (event["id"], content),
            )

    def publish_clip(self, event, clip):
        content = clip.read_bytes()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO stationwatch_evidence(event_id, kind, content)
                VALUES (%s, 'clip', %s) ON CONFLICT (event_id, kind) DO NOTHING""",
                (event["id"], content),
            )
            connection.execute(
                """UPDATE stationwatch_alerts SET payload = jsonb_set(payload, '{clip_state}', '"ready"')
                WHERE id = %s""",
                (event["id"],),
            )

    def media_path(self, event_id, kind):
        if kind not in {"snapshot", "clip"}:
            raise ValueError("Unknown evidence kind")
        # Cache keys use an event ID digest, never a client-supplied path.
        digest = hashlib.sha256(event_id.encode()).hexdigest()
        path = self.cache_dir / (digest + (".jpg" if kind == "snapshot" else ".mp4"))
        with self.cache_lock:
            if path.is_file():
                return path
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT content FROM stationwatch_evidence WHERE event_id = %s AND kind = %s",
                    (event_id, kind),
                ).fetchone()
            if row is None:
                raise FileNotFoundError("Evidence has not arrived yet")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".partial")
            try:
                temporary.write_bytes(row["content"])
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def close(self):
        # Connections are scoped to each operation and closed by their context manager.
        pass
