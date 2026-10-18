import csv
import json
import os
import shutil
import time

import cv2
import numpy as np
import pytest

from stationwatch.config import Settings
from stationwatch.features import FEATURE_NAMES, Features
from stationwatch.model import TrainedModel
from stationwatch.recording import SegmentRecorder
from stationwatch.storage import EventStore
from stationwatch.training import train


@pytest.mark.parametrize("algorithm", ["random-forest", "logistic"])
def test_training_holds_out_groups_and_model_loads(tmp_path, algorithm):
    pytest.importorskip("sklearn")
    path = tmp_path / "features.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_NAMES + ["label", "group"])
        writer.writeheader()
        # Fixture data tests the artifact contract, not real-world accuracy.
        for group in range(8):
            for i in range(10):
                label = i % 2
                values = [90, 130 if label else 10, 0.4, 0.5, 1, 0.8, 0.5, 3]
                writer.writerow(dict(zip(FEATURE_NAMES, values), label=label, group=f"worker-{group}"))
    output = tmp_path / "classifier.joblib"
    train(path, output, algorithm=algorithm, threshold=0.8)
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert not set(report["training_groups"]) & set(report["evaluation_groups"])
    assert report["algorithm"] == algorithm
    assert report["runtime_threshold"] == 0.8
    assert "report_at_runtime_threshold" in report
    model = TrainedModel(Settings(classifier_path=output))
    assert model.predict(Features(90, 130, 0.4, 0.5, 1, 0.8, 0.5, 3)).flagged
    assert not model.predict(Features(90, 10, 0.4, 0.5, 1, 0.8, 0.5, 3)).flagged


def test_forest_learns_orientation_and_dwell_together(tmp_path):
    pytest.importorskip("sklearn")
    path = tmp_path / "interactions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_NAMES + ["label", "group"])
        writer.writeheader()
        # A contract fixture, not an estimate of accuracy: only sustained orientation away is positive.
        for group in range(8):
            for _ in range(5):
                for head, dwell, label in [(10, 0.5, 0), (10, 5, 0), (130, 0.5, 0), (130, 5, 1)]:
                    values = [90, head, 0.4, 0.5, 1, 0.8, 0.5, dwell]
                    writer.writerow(dict(zip(FEATURE_NAMES, values), label=label, group=f"worker-{group}"))
    output = tmp_path / "interactions.joblib"
    train(path, output)
    model = TrainedModel(Settings(classifier_path=output))
    assert model.predict(Features(90, 130, 0.4, 0.5, 1, 0.8, 0.5, 5)).flagged
    assert not model.predict(Features(90, 130, 0.4, 0.5, 1, 0.8, 0.5, 0.5)).flagged
    assert not model.predict(Features(90, 10, 0.4, 0.5, 1, 0.8, 0.5, 5)).flagged


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")
def test_recording_offsets_atomic_finalization_and_retention(tmp_path):
    store = EventStore(tmp_path)
    recorder = SegmentRecorder(Settings(width=160, height=120, fps=10, segment_seconds=1), store)
    image = np.zeros((120, 160, 3), np.uint8)
    try:
        first, offset = recorder.write(image, 100)
        assert offset == 0
        assert not (tmp_path / "recordings" / first).exists()
        assert recorder.write(image, 100.1)[1] == 0.1
        second, offset = recorder.write(image, 101.1)
        assert first != second and offset == 0
        path = tmp_path / "recordings" / first
        assert path.exists()
        capture = cv2.VideoCapture(str(path))
        try:
            assert capture.read()[0]
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        finally:
            capture.release()
        store.add({"id": "retained", "captured_at": "2026-10-17T00:00:00Z", "segment": first}, b"jpeg")
        obsolete = tmp_path / "recordings" / "unflagged.mp4"
        obsolete.write_bytes(b"expired footage")
        old_time = time.time() - 48 * 3600
        for old_path in [path, obsolete]:
            os.utime(old_path, (old_time, old_time))
        recorder.cleanup()
        assert path.exists()
        assert not obsolete.exists()
    finally:
        recorder.close()
        store.close()


def test_invalid_geometry_and_odd_recording_dimensions_are_rejected():
    for overrides in [
        {"width": 961},
        {"machine_center": [1.2, 0.5]},
        {"workstation_polygon": [[0, 0], [0.5, 0.5], [1, 1]]},
    ]:
        with pytest.raises(ValueError):
            Settings(**overrides)
