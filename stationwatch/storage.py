import json
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path


class EventStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (root / "snapshots").mkdir(exist_ok=True)
        (root / "recordings").mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(root / "events.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY, captured_at TEXT NOT NULL, reviewed INTEGER NOT NULL DEFAULT 0,
            segment TEXT NOT NULL, payload TEXT NOT NULL)""")
        self.db.commit()
        self.db.execute("""CREATE TABLE IF NOT EXISTS uploads (
            event_id TEXT PRIMARY KEY, alert_sent INTEGER NOT NULL DEFAULT 0,
            clip_sent INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at REAL NOT NULL DEFAULT 0, last_error TEXT)""")
        # Existing confirmed events become eligible when MongoDB uploads are first enabled.
        self.db.execute("INSERT OR IGNORE INTO uploads(event_id) SELECT id FROM events")
        self.db.commit()

    def add(self, event, jpeg):
        snapshot = self.root / "snapshots" / (event["id"] + ".jpg")
        snapshot.write_bytes(jpeg)
        try:
            with self.lock, self.db:
                self.db.execute(
                    "INSERT INTO events(id, captured_at, segment, payload) VALUES (?, ?, ?, ?)",
                    (event["id"], event["captured_at"], event["segment"], json.dumps(event)),
                )
                self.db.execute("INSERT INTO uploads(event_id) VALUES (?)", (event["id"],))
        except Exception:
            snapshot.unlink(missing_ok=True)
            raise

    def list(self, limit=100):
        with self.lock:
            rows = self.db.execute(
                "SELECT payload, reviewed FROM events ORDER BY captured_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(json.loads(payload), reviewed=bool(reviewed)) for payload, reviewed in rows]

    def get(self, event_id):
        with self.lock:
            row = self.db.execute("SELECT payload, reviewed FROM events WHERE id = ?", (event_id,)).fetchone()
        return dict(json.loads(row[0]), reviewed=bool(row[1])) if row else None

    def mark_reviewed(self, event_id):
        with self.lock, self.db:
            return self.db.execute("UPDATE events SET reviewed=1 WHERE id=?", (event_id,)).rowcount > 0

    def retained_segments(self):
        with self.lock:
            return {row[0] for row in self.db.execute("SELECT DISTINCT segment FROM events")}

    def counts(self):
        with self.lock:
            total, pending = self.db.execute(
                "SELECT count(*), coalesce(sum(CASE WHEN reviewed=0 THEN 1 ELSE 0 END), 0) FROM events"
            ).fetchone()
            rows = self.db.execute("SELECT payload FROM events").fetchall()
        stations = {json.loads(row[0]).get("station_id") for row in rows}
        return {"total": total, "pending": pending, "reviewed": total - pending, "stations": len(stations)}

    def pending_uploads(self, now=None, limit=20):
        with self.lock:
            rows = self.db.execute(
                """SELECT e.payload, u.alert_sent, u.attempts FROM uploads u
                JOIN events e ON e.id=u.event_id WHERE u.clip_sent=0 AND u.next_retry_at<=?
                ORDER BY u.next_retry_at, e.captured_at LIMIT ?""",
                (time.time() if now is None else now, limit),
            ).fetchall()
        return [{"event": json.loads(row[0]), "alert_sent": bool(row[1]), "attempts": row[2]} for row in rows]

    def upload_progress(self, event_id, *, alert_sent=False, clip_sent=False, retry_at=0, error=None):
        with self.lock, self.db:
            self.db.execute(
                """UPDATE uploads SET alert_sent=max(alert_sent,?), clip_sent=max(clip_sent,?),
                attempts=CASE WHEN ? IS NULL THEN 0 ELSE attempts+1 END,
                next_retry_at=?, last_error=? WHERE event_id=?""",
                (int(alert_sent), int(clip_sent), error, retry_at, error, event_id),
            )

    def upload_counts(self):
        with self.lock:
            pending, failed = (
                self.db.execute("""SELECT count(*), coalesce(sum(CASE WHEN last_error IS NOT NULL
                THEN 1 ELSE 0 END),0) FROM uploads WHERE clip_sent=0""").fetchone()
            )
        return {"pending": pending, "failed": failed}

    def close(self):
        self.db.close()


def make_event(event_id, packet, settings, features, prediction, review, mode, confirmed_at):
    return {
        "id": event_id,
        "station_id": settings.station_id,
        "station_name": settings.station_name,
        "session_id": packet.session_id,
        "mode": mode,
        "captured_at": packet.captured_at,
        "confirmed_at": confirmed_at,
        "elapsed_s": packet.elapsed_s,
        "segment": packet.segment,
        "video_offset_s": packet.video_offset_s,
        "features": asdict(features),
        "local": asdict(prediction),
        "review": review.model_dump(),
        "review_model": "synthetic_demo" if mode == "demo" else settings.gemini_model,
        "analysis_backend": "image" if mode == "image" else "pose",
        "pose_backend": None
        if mode == "image"
        else "synthetic_demo"
        if mode == "demo"
        else settings.pose_backend,
    }


class DatabaseUnavailable(RuntimeError):
    """Sanitized database outage used by the hub HTTP layer."""
