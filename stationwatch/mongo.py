"""Confirmed alerts in MongoDB; evidence in GridFS; no frames or normal observations are uploaded."""

import hashlib
import threading
import uuid
from functools import wraps

import gridfs
from pymongo import MongoClient, timeout
from pymongo.errors import PyMongoError

from .storage import DatabaseUnavailable
from .uploads import EvidenceUploader as MongoUploader  # Backwards-compatible CLI import.  # noqa: F401


def database_operation(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            with timeout(self.timeout_s):
                return function(self, *args, **kwargs)
        except PyMongoError:
            # Never surface connection strings/credentials through HTTP errors or logs.
            raise DatabaseUnavailable("MongoDB is unavailable; check connection and permissions") from None

    return wrapped


class MongoEventStore:
    def __init__(self, database, cache_dir, timeout_s=15, client=None):
        self.database = database
        self.alerts = database["alerts"]
        self.files = gridfs.GridFS(database, collection="evidence")
        self.cache_dir = cache_dir
        self.timeout_s = timeout_s
        self.client = client
        self.cache_lock = threading.Lock()

    @classmethod
    def from_uri(cls, uri, settings):
        client = MongoClient(
            uri,
            connect=False,
            timeoutMS=int(settings.mongo_timeout_s * 1000),
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=5000,
        )
        return cls(
            client[settings.mongo_database], settings.data_dir / "hub-cache", settings.mongo_timeout_s, client
        )

    @database_operation
    def initialize_hub(self):
        self.database.command("ping")
        self.alerts.create_index([("captured_at", -1)])
        self.alerts.create_index([("reviewed", 1), ("captured_at", -1)])

    def _upload_file(self, file_id, path, content_type):
        if self.files.exists(file_id):
            return
        # Recover an interrupted upload. IDs belong to one event from one Pi, with one uploader process.
        self.files.delete(file_id)
        with path.open("rb") as handle:
            self.files.put(handle, _id=file_id, filename=path.name, contentType=content_type)

    @database_operation
    def publish_alert(self, event, snapshot):
        if event.get("mode") == "demo" or not event.get("local", {}).get("flagged"):
            raise ValueError("Only live, locally flagged events may be uploaded")
        if event.get("review", {}).get("decision") != "inattentive":
            raise ValueError("Only Gemini-confirmed events may be uploaded")
        snapshot_id = "snapshot:" + event["id"]
        self._upload_file(snapshot_id, snapshot, "image/jpeg")
        # Retries must not overwrite the manager's review decision or a previously uploaded clip.
        document = dict(event, reviewed=False, snapshot_file_id=snapshot_id, clip_state="pending")
        document.pop("_id", None)
        self.alerts.update_one({"_id": event["id"]}, {"$setOnInsert": document}, upsert=True)

    @database_operation
    def publish_clip(self, event, clip):
        if not self.alerts.find_one({"_id": event["id"]}, {"_id": 1}):
            raise ValueError("Upload the confirmed alert before its video")
        clip_id = "clip:" + event["id"]
        self._upload_file(clip_id, clip, "video/mp4")
        self.alerts.update_one(
            {"_id": event["id"]}, {"$set": {"clip_file_id": clip_id, "clip_state": "ready"}}
        )

    @staticmethod
    def _event(document):
        if document is None:
            return None
        return {key: value for key, value in document.items() if key != "_id"}

    @database_operation
    def list(self, limit=100):
        return [self._event(doc) for doc in self.alerts.find().sort("captured_at", -1).limit(limit)]

    @database_operation
    def get(self, event_id):
        return self._event(self.alerts.find_one({"_id": event_id}))

    @database_operation
    def mark_reviewed(self, event_id):
        return self.alerts.update_one({"_id": event_id}, {"$set": {"reviewed": True}}).matched_count > 0

    @database_operation
    def counts(self):
        total = self.alerts.count_documents({})
        pending = self.alerts.count_documents({"reviewed": False})
        return {
            "total": total,
            "pending": pending,
            "reviewed": total - pending,
            "stations": len(self.alerts.distinct("station_id")),
        }

    @database_operation
    def media_path(self, event_id, kind):
        if kind not in {"snapshot", "clip"}:
            raise ValueError("Unknown evidence kind")
        event = self.alerts.find_one({"_id": event_id})
        file_id = event.get(kind + "_file_id") if event else None
        if not file_id:
            raise FileNotFoundError("Evidence has not arrived yet")
        digest = hashlib.sha256(file_id.encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (digest + (".jpg" if kind == "snapshot" else ".mp4"))
        with self.cache_lock:
            if path.is_file():
                return path
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".partial")
            try:
                with self.files.get(file_id) as source, temporary.open("wb") as destination:
                    while chunk := source.read(256 * 1024):
                        destination.write(chunk)
                temporary.replace(path)
            except gridfs.NoFile:
                raise FileNotFoundError("Evidence is missing from GridFS") from None
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def close(self):
        if self.client is not None:
            self.client.close()
