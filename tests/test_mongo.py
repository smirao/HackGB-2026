from dataclasses import asdict
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("pymongo")
mongomock = pytest.importorskip("mongomock")
from mongomock.gridfs import enable_gridfs_integration

from stationwatch.app import create_app
from stationwatch.features import Features
from stationwatch.mongo import MongoEventStore, MongoUploader
from stationwatch.storage import EventStore


@pytest.fixture
def remote(tmp_path):
    enable_gridfs_integration()
    database = mongomock.MongoClient().stationwatch
    return MongoEventStore(database, tmp_path / "hub-cache")


def confirmed(event_id="event-1"):
    return {
        "id": event_id,
        "captured_at": "2026-10-17T18:22:00+00:00",
        "station_id": "press-01",
        "station_name": "Press 01",
        "segment": "segment.mp4",
        "video_offset_s": 3,
        "elapsed_s": 12,
        "mode": "trained",
        "features": asdict(Features(90, 130, 0.4, 0.5, 1, 0.8, 0.5, 3)),
        "local": {"flagged": True, "score": 0.95, "source": "trained_classifier"},
        "review": {"decision": "inattentive", "confidence": 0.95, "reason": "Visible orientation away"},
        "review_model": "gemini-test",
    }


def test_no_database_calls_without_confirmed_events(tmp_path):
    local = EventStore(tmp_path)
    remote = Mock()
    try:
        MongoUploader(local, remote).run_once(now=0)
        assert remote.mock_calls == []
    finally:
        local.close()


@pytest.mark.parametrize("backend", ["pose", "image"])
def test_restart_uploads_clip_later_without_duplicate_alert_or_resetting_review(tmp_path, remote, backend):
    local = EventStore(tmp_path / "pi")
    event = confirmed()
    if backend == "image":
        from stationwatch.image_model import ImageObservation

        event.update(mode="image", features=asdict(ImageObservation(5.5, 5, (0, 0, 1, 1))))
    local.add(event, b"snapshot-evidence")
    uploader = MongoUploader(local, remote, poll_s=1)
    uploader.run_once(now=0)
    assert remote.counts()["total"] == 1
    assert remote.get(event["id"])["clip_state"] == "pending"
    remote.mark_reviewed(event["id"])
    assert remote.media_path(event["id"], "snapshot").read_bytes() == b"snapshot-evidence"
    local.close()

    restarted = EventStore(tmp_path / "pi")
    try:
        (restarted.root / "recordings" / event["segment"]).write_bytes(b"finalized-clip")
        uploader = MongoUploader(restarted, remote, poll_s=1)
        uploader.run_once(now=2)
        uploader.run_once(now=20)
        assert remote.counts() == {"total": 1, "pending": 0, "reviewed": 1, "stations": 1}
        assert remote.get(event["id"])["clip_state"] == "ready"
        assert remote.media_path(event["id"], "clip").read_bytes() == b"finalized-clip"
        assert remote.database.evidence.files.count_documents({}) == 2
        assert restarted.upload_counts()["pending"] == 0
    finally:
        restarted.close()


def test_network_failure_remains_durable_for_retry(tmp_path, remote):
    local = EventStore(tmp_path)
    local.add(confirmed(), b"snapshot")
    offline = Mock()
    offline.publish_alert.side_effect = TimeoutError("Database offline")
    try:
        MongoUploader(local, offline, poll_s=1).run_once(now=0)
        assert local.upload_counts() == {"pending": 1, "failed": 1}
        assert local.pending_uploads(now=0.5) == []
        MongoUploader(local, remote, poll_s=1).run_once(now=2)
        assert remote.counts()["total"] == 1
        assert local.upload_counts() == {"pending": 1, "failed": 0}
    finally:
        local.close()


def test_retry_after_remote_commit_is_idempotent(tmp_path, remote):
    local = EventStore(tmp_path)
    event = confirmed()
    local.add(event, b"snapshot")
    (local.root / "recordings" / event["segment"]).write_bytes(b"clip")
    uploader = MongoUploader(local, remote, poll_s=1)
    original = local.upload_progress
    calls = 0

    def interrupted_ack(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Interrupted before saving local acknowledgement")
        return original(*args, **kwargs)

    local.upload_progress = interrupted_ack
    try:
        uploader.run_once(now=0)
        remote.mark_reviewed(event["id"])
        uploader.run_once(now=2)
        assert remote.get(event["id"])["reviewed"] is True
        assert remote.database.alerts.count_documents({}) == 1
        assert remote.database.evidence.files.count_documents({}) == 2
        assert local.upload_counts()["pending"] == 0
    finally:
        local.close()


def test_interrupted_gridfs_chunks_are_recovered(tmp_path, remote):
    event = confirmed()
    path = tmp_path / "snapshot.jpg"
    path.write_bytes(b"complete snapshot")
    remote.database.evidence.chunks.insert_one(
        {"files_id": "snapshot:event-1", "n": 0, "data": b"incomplete"}
    )
    remote.publish_alert(event, path)
    assert remote.files.get("snapshot:event-1").read() == b"complete snapshot"


@pytest.mark.parametrize(
    "change", [{"mode": "demo"}, {"local": {"flagged": False}}, {"review": {"decision": "uncertain"}}]
)
def test_only_confirmed_live_alerts_can_be_published(tmp_path, remote, change):
    event = dict(confirmed(), **change)
    snapshot = tmp_path / "snapshot.jpg"
    snapshot.write_bytes(b"snapshot")
    with pytest.raises(ValueError):
        remote.publish_alert(event, snapshot)
    assert remote.database.alerts.count_documents({}) == 0


def test_hub_reads_gridfs_evidence_and_supports_video_ranges_without_a_camera(tmp_path, remote):
    event = confirmed()
    snapshot, clip = tmp_path / "snapshot.jpg", tmp_path / "clip.mp4"
    snapshot.write_bytes(b"snapshot")
    clip.write_bytes(b"recorded-video")
    remote.publish_alert(event, snapshot)
    token = "private-hub-token"
    with TestClient(create_app(None, remote, token)) as client:
        assert client.get("/api/events").status_code == 401
        client.headers["Authorization"] = "Bearer " + token
        assert client.get("/api/status").json()["role"] == "hub"
        assert client.get("/api/stream").status_code == 404
        assert client.get("/api/events/event-1/video").status_code == 409
        assert client.get("/api/events/event-1/snapshot").content == b"snapshot"
        remote.publish_clip(event, clip)
        response = client.get("/api/events/event-1/video", headers={"Range": "bytes=0-3"})
        assert response.status_code == 206
        assert response.content == b"reco"
        assert client.post("/api/events/event-1/reviewed").status_code == 200
        assert client.get("/api/events").json()[0]["reviewed"] is True
