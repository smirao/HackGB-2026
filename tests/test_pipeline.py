import logging
import threading
import time
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from stationwatch.config import Settings
from stationwatch.features import DwellTracker, Features
from stationwatch.gemini import Review
from stationwatch.model import BaselineModel, Prediction
from stationwatch.pipeline import Packet, Pipeline
from stationwatch.storage import EventStore

FEATURES = Features(90, 130, 0.4, 0.5, 1, 0.8, 0.5)


def packet(**updates):
    now = time.monotonic()
    return replace(
        Packet(
            np.zeros((120, 160, 3), np.uint8),
            b"original-jpeg",
            "2026-10-17T18:22:00+00:00",
            now,
            12.5,
            "original.mp4",
            2.5,
            "session-test",
        ),
        **updates,
    )


@pytest.fixture
def setup(tmp_path):
    store = EventStore(tmp_path)
    extractor = Mock()
    extractor.extract.return_value = FEATURES
    model = Mock()
    model.predict.return_value = Prediction(True, 0.92, "test_model")
    reviewer = Mock()
    reviewer.review.return_value = Review(
        decision="inattentive", confidence=0.95, reason="Visible orientation away"
    )
    settings = Settings(data_dir=tmp_path, review_interval_s=0.01, event_cooldown_s=30, flag_duration_s=0)
    pipeline = Pipeline(settings, Mock(), extractor, model, reviewer, Mock(), store, "baseline")
    yield pipeline, store, reviewer, extractor
    pipeline.close()
    store.close()


def drain(pipeline):
    pipeline.local_pool.submit(lambda: None).result(timeout=3)
    pipeline.review_pool.submit(lambda: None).result(timeout=3)


@pytest.mark.parametrize(
    "decision,confidence,count",
    [
        ("inattentive", 0.95, 1),
        ("inattentive", 0.79, 0),
        ("uncertain", 0.99, 0),
        ("attentive", 0.99, 0),
    ],
)
def test_only_corroborated_confident_flags_are_saved(setup, decision, confidence, count):
    pipeline, store, reviewer, _ = setup
    reviewer.review.return_value = Review(decision=decision, confidence=confidence, reason="Test observation")
    original = packet()
    assert pipeline.offer_sample(original)
    drain(pipeline)
    assert len(store.list()) == count
    if count:
        event = store.list()[0]
        assert event["captured_at"] == original.captured_at
        assert event["segment"] == "original.mp4"
        assert event["video_offset_s"] == 2.5
        assert event["elapsed_s"] == 12.5
        assert (store.root / "snapshots" / (event["id"] + ".jpg")).read_bytes() == b"original-jpeg"


def test_review_failure_creates_no_event(setup):
    pipeline, store, reviewer, _ = setup
    reviewer.review.side_effect = RuntimeError("network failed")
    pipeline.offer_sample(packet())
    drain(pipeline)
    assert store.list() == []
    assert pipeline.status()["review_errors"] == 1


def test_unflagged_local_prediction_never_calls_gemini(setup):
    pipeline, store, reviewer, _ = setup
    pipeline.model.predict.return_value = Prediction(False, 0.1, "test_model")
    pipeline.offer_sample(packet())
    drain(pipeline)
    reviewer.review.assert_not_called()
    assert store.list() == []


@pytest.mark.parametrize("flagged,score,label", [(True, 0.92, "DISTRACTED"), (False, 0.1, "ON TASK")])
def test_console_logs_local_prediction_before_review(setup, caplog, flagged, score, label):
    pipeline, _, reviewer, _ = setup
    # The prediction is logged even while the duration gate suppresses Gemini review.
    pipeline.settings = pipeline.settings.model_copy(update={"flag_duration_s": 5})
    pipeline.flag_gate.duration_s = 5
    pipeline.model.predict.return_value = Prediction(flagged, score, "test_model")
    with caplog.at_level(logging.INFO, logger="stationwatch.pipeline"):
        pipeline.offer_sample(packet())
        drain(pipeline)
    assert f"Local model: {label}" in caplog.text
    assert f"distraction_score={score:.3f}" in caplog.text
    assert "threshold=0.800" in caplog.text
    assert "sustained=0.0s" in caplog.text
    reviewer.review.assert_not_called()


def test_missing_pose_is_unknown_and_not_reviewed(setup):
    pipeline, _, reviewer, extractor = setup
    extractor.extract.return_value = None
    pipeline.offer_sample(packet())
    drain(pipeline)
    assert pipeline.status()["last_observation"] == "unknown"
    reviewer.review.assert_not_called()


def test_stale_capture_is_skipped(setup):
    pipeline, _, reviewer, extractor = setup
    pipeline.offer_sample(packet(monotonic_s=time.monotonic() - 10))
    drain(pipeline)
    extractor.extract.assert_not_called()
    reviewer.review.assert_not_called()
    assert pipeline.status()["skipped_samples"] == 1


def test_slow_local_job_has_no_unbounded_queue(setup):
    pipeline, _, _, extractor = setup
    entered, release = threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        assert release.wait(3)
        return FEATURES

    extractor.extract.side_effect = slow
    try:
        assert pipeline.offer_sample(packet())
        assert entered.wait(1)
        for _ in range(10):
            assert not pipeline.offer_sample(packet())
        assert pipeline.status()["skipped_samples"] == 10
    finally:
        release.set()
    drain(pipeline)


def test_capture_can_continue_analysis_while_review_is_busy(setup):
    pipeline, store, reviewer, _ = setup
    entered, release = threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        assert release.wait(3)
        return Review(decision="inattentive", confidence=0.99, reason="Away")

    reviewer.review.side_effect = slow
    try:
        pipeline.offer_sample(packet())
        assert entered.wait(1)
        pipeline.local_pool.submit(lambda: None).result(1)
        assert pipeline.offer_sample(packet())
        pipeline.local_pool.submit(lambda: None).result(1)
        assert pipeline.status()["analyzed_frames"] == 2
        assert reviewer.review.call_count == 1
    finally:
        release.set()
    drain(pipeline)
    assert len(store.list()) == 1


def test_cooldown_prevents_duplicate_events(setup):
    pipeline, store, reviewer, _ = setup
    pipeline.offer_sample(packet())
    drain(pipeline)
    pipeline.last_attempt = float("-inf")
    pipeline.offer_sample(packet())
    drain(pipeline)
    assert reviewer.review.call_count == 1
    assert len(store.list()) == 1


def test_camera_failure_stops_stream_and_finalizes_video(setup):
    pipeline, _, _, _ = setup
    pipeline.camera.read.side_effect = [np.zeros((120, 160, 3), np.uint8), RuntimeError("Disconnected")]
    pipeline.recorder.write.return_value = ("test.mp4", 0)
    pipeline.start()
    pipeline.thread.join(timeout=2)
    assert not pipeline.thread.is_alive()
    assert not pipeline.status()["running"]
    assert "stopped" in pipeline.status()["last_error"]
    pipeline.camera.close.assert_called_once()
    pipeline.recorder.close.assert_called_once()


def test_dwell_uses_capture_times_and_resets_after_unknown_pose_or_gap():
    tracker = DwellTracker()
    assert tracker.update(FEATURES, 10).dwell_s == 0
    assert tracker.update(FEATURES, 10.5).dwell_s == 0.5
    assert tracker.update(FEATURES, 11).dwell_s == 1
    assert tracker.update(None, 11.5) is None
    assert tracker.update(FEATURES, 12).dwell_s == 0
    assert tracker.update(FEATURES, 15).dwell_s == 0
    assert tracker.update(replace(FEATURES, head_angle_deg=15), 15.5).dwell_s == 0


def test_baseline_requires_head_angle_dwell_and_workstation():
    model = BaselineModel(Settings())
    assert model.predict(replace(FEATURES, dwell_s=2)).flagged
    for value in [
        FEATURES,
        replace(FEATURES, head_angle_deg=10, dwell_s=3),
        replace(FEATURES, in_workstation=0, dwell_s=3),
    ]:
        assert not model.predict(value).flagged
