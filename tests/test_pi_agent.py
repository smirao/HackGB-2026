import asyncio
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import httpx
import numpy as np
import pytest

from stationwatch.config import Settings
from stationwatch.gemini import Review
from stationwatch.pi_agent import CameraFeed, PiAgent, Sample
from stationwatch.postgres import PostgresEventStore
from stationwatch.storage import EventStore
from stationwatch.temporal import SustainedFlag
from stationwatch.uploads import EvidenceUploader


def result(decision="off_task", confidence=0.95):
    return {"decision": decision, "confidence": confidence, "model": "jev-test"}


@pytest.fixture
def agent_setup(tmp_path):
    local = EventStore(tmp_path)
    reviewer = Mock()
    reviewer.review.return_value = Review(decision="inattentive", confidence=0.95, reason="Unrelated activity")
    agent = PiAgent(Settings(task_description="Assemble parts", local_threshold=0.6), local, reviewer, "key")
    sample = Sample(b"snapshot", "2026-10-18T18:00:00+00:00", time.monotonic() - 10,
                    0, "segment.mp4", 0, "session")
    yield agent, local, reviewer, sample
    local.close()


def test_pi_duration_gate_includes_exactly_five_seconds():
    gate = SustainedFlag(5, 1.5, inclusive=True)
    for captured in range(5):
        assert not gate.update(True, captured)
    assert gate.update(True, 5)
    assert not gate.update(False, 6)
    assert not gate.update(True, 7)
    assert not gate.update(True, 10)


def test_five_second_streak_produces_dashboard_event_and_original_evidence(agent_setup):
    agent, local, reviewer, sample = agent_setup

    async def exercise():
        for i in range(5):
            agent.accept(replace(sample, sequence=i, monotonic_s=sample.monotonic_s + i), result())
        reviewer.review.assert_not_called()
        agent.accept(replace(sample, sequence=5, monotonic_s=sample.monotonic_s + 5,
                             elapsed_s=5, video_offset_s=5), result())
        await agent.review_task
        agent.accept(replace(sample, sequence=6, monotonic_s=sample.monotonic_s + 6), result())
        await agent.review_task

    asyncio.run(exercise())
    reviewer.review.assert_called_once()
    event = local.list()[0]
    assert event["features"]["sustained_flag_s"] == 5
    assert event["features"]["source"] == "jev"
    assert event["mode"] == "image" and event["local"]["source"] == "jev"
    assert event["jev"]["model"] == "jev-test"
    assert event["task_description"] == "Assemble parts"
    assert event["captured_at"] == sample.captured_at and event["video_offset_s"] == 5
    assert event["session_id"] == sample.session_id
    assert (local.root / "snapshots" / (event["id"] + ".jpg")).read_bytes() == sample.jpeg
    assert local.upload_counts()["pending"] == 1


@pytest.mark.parametrize("interruption", ["on_task", "error", "skipped"])
def test_interrupted_observations_restart_streak(agent_setup, interruption):
    agent, local, reviewer, sample = agent_setup

    async def exercise():
        for i in range(8):
            if i == 3 and interruption == "skipped":
                continue
            answer = result()
            if i == 3:
                if interruption == "on_task":
                    answer = result(interruption)
                elif interruption == "error":
                    answer = None
            agent.accept(replace(sample, sequence=i, monotonic_s=sample.monotonic_s + i), answer)
        assert agent.review_task is None

    asyncio.run(exercise())
    reviewer.review.assert_not_called()
    assert local.list() == []


def test_stale_responses_cannot_trigger_gemini(agent_setup):
    agent, _, reviewer, sample = agent_setup
    for i in range(6):
        agent.accept(replace(sample, sequence=i, monotonic_s=time.monotonic() - 40 + i), result())
    reviewer.review.assert_not_called()
    assert agent.gate.elapsed_s == 0


def test_binary_off_task_streak_counts_low_confidence_choices(agent_setup):
    agent, local, reviewer, sample = agent_setup

    async def exercise():
        for i in range(6):
            agent.accept(replace(sample, sequence=i, monotonic_s=sample.monotonic_s + i),
                         result(confidence=0.4))
        await agent.review_task

    asyncio.run(exercise())
    reviewer.review.assert_called_once()
    assert local.list()[0]['local']['score'] == 0.4


@pytest.mark.parametrize("decision,confidence", [("attentive", 0.99), ("uncertain", 0.99),
                                                   ("inattentive", 0.79), ("error", 0.99)])
def test_only_gemini_confirmation_creates_alert(agent_setup, decision, confidence):
    agent, local, reviewer, sample = agent_setup
    if decision == "error":
        reviewer.review.side_effect = RuntimeError("secret-provider-body")
    else:
        reviewer.review.return_value = Review(decision=decision, confidence=confidence, reason="Test")

    async def exercise():
        for i in range(6):
            agent.accept(replace(sample, sequence=i, monotonic_s=sample.monotonic_s + i), result())
        await agent.review_task

    asyncio.run(exercise())
    assert local.list() == []


def test_busy_gemini_does_not_block_later_observations_or_start_duplicate_reviews(agent_setup):
    agent, local, reviewer, sample = agent_setup
    entered, release = threading.Event(), threading.Event()

    def slow_review(*args):
        entered.set()
        assert release.wait(3)
        return Review(decision="inattentive", confidence=0.99, reason="Unrelated activity")

    reviewer.review.side_effect = slow_review

    async def exercise():
        for i in range(6):
            agent.accept(replace(sample, sequence=i, monotonic_s=sample.monotonic_s + i), result())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            for i in range(6, 9):
                agent.accept(replace(sample, sequence=i, monotonic_s=sample.monotonic_s + i), result())
            assert agent.gate.elapsed_s == 8
            assert reviewer.review.call_count == 1
        finally:
            release.set()
        await agent.review_task

    asyncio.run(exercise())
    assert len(local.list()) == 1


def test_real_jev_client_samples_concurrently_and_consumes_in_capture_order(agent_setup):
    agent, _, _, _ = agent_setup
    agent.max_in_flight = 2
    ok, encoded = cv2.imencode(".jpg", np.zeros((300, 300, 3), np.uint8))
    assert ok
    consumed, completed = [], []
    calls = active = peak = 0
    agent.accept = lambda sample, answer: consumed.append(sample.sequence)
    feed = SimpleNamespace(error=False, snapshot=lambda: Sample(
        encoded.tobytes(), "2026-10-18T18:00:00Z", time.monotonic(), 0, "segment.mp4", 0, "s"))

    async def respond(request):
        nonlocal calls, active, peak
        index = calls
        calls += 1
        active += 1
        peak = max(peak, active)
        try:
            assert request.headers["authorization"] == "Bearer key"
            await asyncio.sleep(1.3 if index == 0 else 0.01)
            completed.append(index)
            return httpx.Response(200, json={"model": "jev-test", "answers": {"task_status": {
                "type": "choice", "choice": "off_task", "confidence": 0.95,
                "probabilities": {"on_task": 0.05, "off_task": 0.95},
            }}})
        finally:
            active -= 1

    async def exercise():
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(2.3, stop.set)
        await agent.run(feed, stop, transport=httpx.MockTransport(respond))

    asyncio.run(exercise())
    assert calls == 3 and peak == 2
    assert completed[:2] == [1, 0]
    assert consumed == [0, 1, 2]


def test_request_limit_skips_ticks_and_resets_streak(agent_setup):
    agent, _, reviewer, _ = agent_setup
    agent.max_in_flight = 1
    consumed = []
    agent.accept = lambda sample, answer: consumed.append(sample.sequence)
    feed = SimpleNamespace(error=False, snapshot=lambda: Sample(
        b"jpeg", "2026-10-18T18:00:00Z", time.monotonic(), 0, "segment.mp4", 0, "s"))
    calls = 0

    async def classify(client, sample):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1.2)
        return result()

    agent.classify = classify

    async def exercise():
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(3.4, stop.set)
        await agent.run(feed, stop)

    asyncio.run(exercise())
    assert calls == 2 and consumed == [0, 2]
    reviewer.review.assert_not_called()


def test_provided_mjpeg_feed_uses_timeouts_and_releases_camera(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = True
    factory = Mock(return_value=capture)
    monkeypatch.setattr("stationwatch.pi_agent.cv2.VideoCapture", factory)
    recorder = Mock()
    feed = CameraFeed(Settings(camera_source="http://127.0.0.1:5000/video_feed"), recorder)
    assert factory.call_args.args == (
        "http://127.0.0.1:5000/video_feed", cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
    )
    feed.close()
    capture.release.assert_called_once()
    recorder.close.assert_called_once()


def test_capture_failure_finalizes_recording(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.side_effect = [(True, np.zeros((120, 160, 3), np.uint8)), (False, None)]
    monkeypatch.setattr("stationwatch.pi_agent.cv2.VideoCapture", lambda source: capture)
    recorder = Mock()
    recorder.write.return_value = ("segment.mp4", 0)
    feed = CameraFeed(Settings(width=160, height=120, fps=60), recorder)
    feed.start()
    feed.thread.join(timeout=2)
    assert feed.error and feed.snapshot().jpeg.startswith(b"\xff\xd8")
    feed.close()
    capture.release.assert_called_once()
    recorder.close.assert_called_once()


def test_postgres_alert_and_snapshot_are_atomic_and_clip_retry_does_not_reset_review(tmp_path, monkeypatch):
    connection = Mock()
    context = Mock()
    context.__enter__ = Mock(return_value=connection)
    context.__exit__ = Mock(return_value=False)
    store = PostgresEventStore()
    monkeypatch.setattr(store, "_connection", lambda: context)
    event = {"id": "test-'quoted", "captured_at": "2026-10-18T18:00:00Z", "mode": "image",
             "local": {"flagged": True}, "review": {"decision": "inattentive"}}
    snapshot = tmp_path / "snapshot.jpg"
    snapshot.write_bytes(b"jpeg")
    store.publish_alert(event, snapshot)
    queries = connection.execute.call_args_list
    assert len(queries) == 2
    assert all("DO NOTHING" in call.args[0] for call in queries)
    assert all(event["id"] not in call.args[0] for call in queries)
    assert queries[0].args[1][2].obj["clip_state"] == "pending"
    context.__enter__.assert_called_once()
    context.__exit__.assert_called_once()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mp4")
    connection.reset_mock()
    store.publish_clip(event, clip)
    assert "jsonb_set" in connection.execute.call_args.args[0]
    assert "reviewed" not in connection.execute.call_args.args[0]
    assert connection.execute.call_args.args[1] == (event["id"],)


@pytest.mark.parametrize("change", [{"mode": "demo"}, {"local": {"flagged": False}},
                                     {"review": {"decision": "uncertain"}}])
def test_postgres_rejects_unconfirmed_events_without_connecting(change, monkeypatch):
    store = PostgresEventStore()
    connect = Mock()
    monkeypatch.setattr(store, "_connection", connect)
    event = {"mode": "image", "local": {"flagged": True}, "review": {"decision": "inattentive"}}
    with pytest.raises(ValueError):
        store.publish_alert(dict(event, **change), Path("unused"))
    connect.assert_not_called()


def test_upload_outage_survives_restart_and_waits_for_finalized_video(tmp_path):
    local = EventStore(tmp_path)
    event = {"id": "event", "captured_at": "2026-10-18T18:00:00Z", "mode": "image", "segment": "clip.mp4"}
    local.add(event, b"jpeg")
    remote = Mock()
    remote.publish_alert.side_effect = RuntimeError("secret-database-details")
    EvidenceUploader(local, remote, poll_s=1).run_once(now=0)
    assert local.upload_counts() == {"pending": 1, "failed": 1}
    local.close()
    restarted = EventStore(tmp_path)
    try:
        remote.publish_alert.side_effect = None
        uploader = EvidenceUploader(restarted, remote, poll_s=1)
        uploader.run_once(now=2)
        remote.publish_clip.assert_not_called()
        remote.publish_alert.assert_called_with(event, tmp_path / "snapshots" / "event.jpg")
        (tmp_path / "recordings" / "clip.mp4").write_bytes(b"mp4")
        uploader.run_once(now=4)
        uploader.run_once(now=8)
        remote.publish_clip.assert_called_once_with(event, tmp_path / "recordings" / "clip.mp4")
        assert restarted.upload_counts() == {"pending": 0, "failed": 0}
    finally:
        restarted.close()


def test_no_database_traffic_until_confirmed_and_shutdown_flushes_queue(agent_setup):
    _, local, _, _ = agent_setup
    remote = Mock()
    uploader = EvidenceUploader(local, remote)
    uploader.run_once()
    assert remote.mock_calls == []
    uploader.close(flush=True)
    remote.close.assert_called_once()


def test_forced_alert_bypasses_models_and_uploads_honestly_labeled_evidence(agent_setup):
    from stationwatch.pi_agent import send_test_alert

    agent, local, reviewer, sample = agent_setup
    sample = replace(sample, monotonic_s=time.monotonic())
    remote = Mock()

    def finish_clip():
        (local.root / 'recordings' / sample.segment).write_bytes(b'recorded-test-video')

    feed = SimpleNamespace(error=False, snapshot=lambda: sample, close=finish_clip)
    status = asyncio.run(send_test_alert(agent.settings, local, remote, feed, record_seconds=0))
    assert status == 0
    reviewer.review.assert_not_called()
    event = local.list()[0]
    assert event['test_alert'] is True and event['alert_source'] == 'manual_test'
    assert event['local'] == {'flagged': False, 'score': 0, 'source': 'manual_test'}
    assert event['review']['decision'] == 'uncertain' and event['review']['confidence'] == 0
    assert event['review_model'] == 'manual_test' and event['jev'] is None
    assert event['features']['source'] == 'manual_test'
    assert 'MANUAL TEST ALERT' in event['review']['reason']
    payload = {key: value for key, value in event.items() if key != 'reviewed'}
    remote.publish_alert.assert_called_once_with(payload, local.root / 'snapshots' / (event['id'] + '.jpg'))
    remote.publish_clip.assert_called_once_with(payload, local.root / 'recordings' / sample.segment)
    assert local.upload_counts() == {'pending': 0, 'failed': 0}


@pytest.mark.parametrize('failure', ['alert', 'clip'])
def test_forced_alert_upload_failure_is_durable_and_retried(agent_setup, failure, caplog):
    from stationwatch.pi_agent import send_test_alert

    agent, local, _, sample = agent_setup
    sample = replace(sample, monotonic_s=time.monotonic())
    clip = local.root / 'recordings' / sample.segment
    feed = SimpleNamespace(error=False, snapshot=lambda: sample, close=lambda: clip.write_bytes(b'video'))
    remote = Mock()
    getattr(remote, 'publish_' + failure).side_effect = RuntimeError('secret-password-at-private-host')
    assert asyncio.run(send_test_alert(agent.settings, local, remote, feed, record_seconds=0)) == 1
    assert local.upload_counts() == {'pending': 1, 'failed': 1}
    assert 'secret-password' not in caplog.text
    assert 'remains queued' in caplog.text
    restarted_remote = Mock()
    EvidenceUploader(local, restarted_remote).run_once(now=time.time() + 6)
    assert local.upload_counts() == {'pending': 0, 'failed': 0}
    restarted_remote.publish_clip.assert_called_once()
    if failure == 'clip':
        restarted_remote.publish_alert.assert_not_called()
    else:
        restarted_remote.publish_alert.assert_called_once()


@pytest.mark.parametrize('missing_marker', [None, 'test_alert', 'alert_source', 'review_model', 'source'])
def test_postgres_requires_complete_manual_test_marker_to_bypass_ai(missing_marker, tmp_path, monkeypatch):
    connection = Mock()
    context = Mock(__enter__=Mock(return_value=connection), __exit__=Mock(return_value=False))
    store = PostgresEventStore()
    monkeypatch.setattr(store, '_connection', lambda: context)
    event = {
        'id': 'manual-test', 'captured_at': '2026-10-18T18:00:00Z', 'mode': 'image',
        'test_alert': True, 'alert_source': 'manual_test', 'review_model': 'manual_test',
        'local': {'flagged': False, 'source': 'manual_test'},
        'review': {'decision': 'uncertain', 'confidence': 0},
    }
    if missing_marker == 'source':
        event['local'].pop('source')
    elif missing_marker:
        event.pop(missing_marker)
    snapshot = tmp_path / 'test.jpg'
    snapshot.write_bytes(b'jpeg')
    if missing_marker is None:
        store.publish_alert(event, snapshot)
        assert connection.execute.call_count == 2
        assert connection.execute.call_args_list[0].args[1][2].obj['test_alert'] is True
    else:
        with pytest.raises(ValueError):
            store.publish_alert(event, snapshot)
        connection.execute.assert_not_called()


def test_force_alert_cli_works_without_api_keys_while_normal_directory_is_locked(tmp_path, monkeypatch):
    import fcntl

    import stationwatch.pi_agent as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, 'load_dotenv', Mock())
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('TASK_DESCRIPTION', raising=False)
    for name in ('PGUSER', 'PGPASSWORD', 'PGDATABASE', 'PGHOST', 'PGPORT', 'PGSSLMODE'):
        monkeypatch.setenv(name, 'test')
    monkeypatch.setattr('sys.argv', [
        'pi_agent.py', '--force-alert', '--source', 'http://127.0.0.1:5000/video_feed',
        '--station-id', 'workbench-01',
    ])
    remote = Mock()
    monkeypatch.setattr(module.PostgresEventStore, 'from_env', lambda: remote)
    no_ai = Mock(side_effect=AssertionError('Test mode must not initialize or call AI clients'))
    monkeypatch.setattr(module, 'GeminiReviewer', no_ai)
    monkeypatch.setattr(module, 'PiAgent', no_ai)
    monkeypatch.setattr(module, 'SegmentRecorder', Mock())

    def create_feed(settings, recorder):
        sample = Sample(b'jpeg', '2026-10-18T18:00:00Z', time.monotonic(), 0, 'clip.mp4', 0, 'test')
        def close():
            (settings.data_dir / 'recordings' / sample.segment).write_bytes(b'video')
        return SimpleNamespace(error=False, start=lambda: None, snapshot=lambda: sample, close=close)

    monkeypatch.setattr(module, 'CameraFeed', create_feed)
    send_test = module.send_test_alert

    async def fast_test(settings, store, remote, feed):
        return await send_test(settings, store, remote, feed, record_seconds=0)

    monkeypatch.setattr(module, 'send_test_alert', fast_test)
    normal_dir = tmp_path / 'data' / 'pi-workbench-01'
    normal_dir.mkdir(parents=True)
    with (normal_dir / 'agent.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module.main() == 0
    no_ai.assert_not_called()
    assert remote.publish_alert.call_count == 1
    event = remote.publish_alert.call_args.args[0]
    assert event['test_alert'] is True and event['task_description'] == 'Manual alert delivery test'
    assert remote.publish_alert.call_args.args[1].parent.parent == Path('data/pi-workbench-01-manual-test')
