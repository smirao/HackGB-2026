import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stationwatch.annotation import create_annotation_app
from stationwatch.config import Settings
from stationwatch.dataset import (
    checked_extraction,
    evaluate,
    extract,
    file_hash,
    import_video,
    initialize,
    label_at,
    record,
    train_dataset,
    validate_intervals,
    write_json,
)
from stationwatch.features import Features
from stationwatch.model import TrainedModel
from stationwatch.recording import SegmentRecorder


@pytest.fixture
def video(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for a real browser-playable video fixture")
    root = tmp_path / "source"
    (root / "recordings").mkdir(parents=True)
    store = SimpleNamespace(root=root, retained_segments=lambda: set())
    recorder = SegmentRecorder(Settings(width=160, height=120, fps=4), store)
    try:
        for i in range(8):
            recorder.write(np.full((120, 160, 3), i * 20, np.uint8), i / 4)
    finally:
        recorder.close()
    return root / "recordings" / recorder.segment


@pytest.fixture
def dataset(tmp_path):
    pose = tmp_path / "pose.tflite"
    pose.write_bytes(b"fixture-pose-model")
    config = tmp_path / "source.config.json"
    settings = Settings(width=160, height=120, fps=4, pose_backend="movenet", pose_model_path=pose)
    config.write_text(settings.model_dump_json())
    root = tmp_path / "dataset"
    initialize(root, config)
    return root


def calibrate(root):
    with TestClient(create_annotation_app(root)) as client:
        geometry = client.get("/api/calibration").json()
        assert client.post("/api/calibration", json=geometry).status_code == 200


def add_clip(root, source, person="person1", name="take1"):
    return import_video(root, source, name, person, "session1")


def test_annotation_rejects_overlap_invalid_geometry_and_cross_origin(dataset, video):
    add_clip(dataset, video)
    with TestClient(create_annotation_app(dataset)) as client:
        assert client.get("/api/clips/take1/video", headers={"Range": "bytes=0-3"}).status_code == 206
        assert client.get("/api/clips/unknown/labels").status_code == 404
        labels = {
            "intervals": [{"start_s": 0, "end_s": 1, "label": 0}, {"start_s": 0.5, "end_s": 2, "label": 1}]
        }
        assert client.post("/api/clips/take1/labels", json=labels).status_code == 400
        labels["intervals"][1]["start_s"] = 1
        assert (
            client.post(
                "/api/clips/take1/labels", json=labels, headers={"Origin": "https://other.test"}
            ).status_code
            == 403
        )
        assert client.post("/api/clips/take1/labels", json=labels).status_code == 200
        assert client.get("/api/clips/take1/labels").json() == labels
        assert (
            client.post(
                "/api/calibration",
                json={"machine_center": [2, 0.5], "workstation_polygon": [[0, 0], [1, 0], [1, 1]]},
            ).status_code
            == 400
        )
    assert not json.loads((dataset / "dataset.json").read_text())["calibration_confirmed"]


def test_label_boundaries_are_half_open_and_uncertain_gaps_remain_unlabeled():
    labels = validate_intervals(
        [{"start_s": 0, "end_s": 0.5, "label": 0}, {"start_s": 1, "end_s": 2, "label": 1}], 2
    )
    assert label_at(labels, 0.49) == 0
    assert label_at(labels, 0.5) is None
    assert label_at(labels, 1) == 1
    assert label_at(labels, 2) is None
    with pytest.raises(ValueError):
        validate_intervals([{"start_s": 0, "end_s": float("nan"), "label": 0}], 2)


def test_extraction_aligns_labels_with_video_and_dwell_with_capture_time(dataset, video, monkeypatch):
    info = add_clip(dataset, video)
    with pytest.raises(ValueError, match="Save calibration"):
        extract(dataset)
    calibrate(dataset)
    write_json(
        dataset / "labels/take1.json",
        {"intervals": [{"start_s": 0, "end_s": 0.5, "label": 0}, {"start_s": 1, "end_s": 2, "label": 1}]},
    )
    info["timestamps"] = "timestamps/take1.csv"
    with (dataset / info["timestamps"]).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "captured_s", "video_s"])
        writer.writeheader()
        for i, captured in enumerate([0, 0.25, 0.5, 1, 1.5, 2.5, 2.75, 3]):
            writer.writerow({"frame": i, "captured_s": captured, "video_s": i / 4})
    write_json(dataset / "clips/take1.json", info)
    extractor = Mock()
    extractor.extract.return_value = Features(90, 130, 0.4, 0.5, 1, 0.8, 0.5)
    monkeypatch.setattr("stationwatch.dataset.create_extractor", lambda _: extractor)
    report = extract(dataset)
    with (dataset / "training.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["label"]) for row in rows] == [0, 1, 1, 1]
    assert float(rows[-1]["video_s"]) == 1.75
    assert float(rows[-1]["captured_s"]) == 3
    assert float(rows[-1]["dwell_s"]) == 3  # Not the compressed 1.75-second playback clock.
    assert report["unlabeled_samples"] == 2
    assert {row["group"] for row in rows} == {"person1"}
    extractor.close.assert_called_once()
    checked_extraction(dataset)
    write_json(dataset / "labels/take1.json", {"intervals": []})
    with pytest.raises(ValueError, match="changed"):
        checked_extraction(dataset)


def test_unknown_pose_resets_dwell_in_offline_extraction(dataset, video, monkeypatch):
    add_clip(dataset, video)
    calibrate(dataset)
    write_json(dataset / "labels/take1.json", {"intervals": [{"start_s": 0, "end_s": 2, "label": 0}]})
    extractor = Mock()
    extractor.extract.side_effect = lambda image, t: (
        None if t == 1 else Features(90, 130, 0.4, 0.5, 1, 0.8, 0.5)
    )
    monkeypatch.setattr("stationwatch.dataset.create_extractor", lambda _: extractor)
    report = extract(dataset)
    with (dataset / "training.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert report["unknown_pose_samples"] == 1
    assert float(rows[-1]["dwell_s"]) == 0


def test_video_dataset_to_model_export_and_calibration_compatibility(dataset, video, monkeypatch):
    pytest.importorskip("sklearn")
    calibrate(dataset)
    for person in ["p01", "p02", "p03", "p04"]:
        for take in ["a", "b"]:
            name = person + "-" + take
            add_clip(dataset, video, person, name)
            write_json(
                dataset / "labels" / (name + ".json"),
                {
                    "intervals": [
                        {"start_s": 0, "end_s": 1, "label": 0},
                        {"start_s": 1, "end_s": 2, "label": 1},
                    ]
                },
            )
    extractor = Mock()
    extractor.extract.side_effect = lambda image, t: Features(
        90, 130 if t >= 1 else 10, 0.4, 0.5, 1, 0.8, 0.5
    )
    monkeypatch.setattr("stationwatch.dataset.create_extractor", lambda _: extractor)
    extract(dataset)
    output = dataset / "model.joblib"
    report = train_dataset(dataset, output, 0.8, holdout=["p04"])
    assert report["evaluation_groups"] == ["p04"]
    assert set(report["training_groups"]) == {"p01", "p02", "p03"}
    config = Settings.model_validate_json(output.with_suffix(".config.json").read_text())
    model = TrainedModel(config)
    assert model.predict(Features(90, 130, 0.4, 0.5, 1, 0.8, 0.5, 1)).flagged
    assert not model.predict(Features(90, 10, 0.4, 0.5, 1, 0.8, 0.5, 0)).flagged
    with pytest.raises(ValueError, match="differs"):
        TrainedModel(config.model_copy(update={"machine_center": (0.1, 0.1)}))
    with pytest.raises(ValueError, match="training groups"):
        evaluate(dataset, output, dataset / "evaluation.json")
    digest = file_hash(output)
    evaluation = evaluate(dataset, output, dataset / "evaluation.json", groups_filter=["p04"])
    assert evaluation["groups"] == ["p04"]
    assert evaluation["samples"] == 8
    assert file_hash(output) == digest  # Evaluation never modifies the model.


def test_recording_creates_video_and_matching_capture_timestamps(dataset, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for recording")
    camera = Mock()
    camera.read.return_value = np.zeros((120, 160, 3), np.uint8)
    monkeypatch.setattr("stationwatch.dataset.create_camera", lambda _: camera)
    info = record(dataset, "recorded-take", "p01", "s01", 0.7)
    camera.close.assert_called_once()
    with (dataset / info["timestamps"]).open(newline="") as handle:
        timeline = list(csv.DictReader(handle))
    assert len(timeline) == info["frames"] > 0
    assert all(0 <= float(row["captured_s"]) < 0.7 for row in timeline)
    assert json.loads((dataset / "labels/recorded-take.json").read_text())["intervals"] == []


@pytest.mark.skipif(os.name != "posix" or not shutil.which("ffmpeg"), reason="POSIX recording interrupt test")
def test_ctrl_c_saves_a_decodable_take_with_complete_timestamps(dataset):
    script = f"""
from pathlib import Path
import numpy as np
from stationwatch import dataset
class Camera:
    def read(self): return np.zeros((120,160,3),np.uint8)
    def close(self): pass
dataset.create_camera=lambda _:Camera()
dataset.record(Path({str(dataset)!r}), 'interrupted', 'p01', 's01', 20)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timestamps = dataset / "timestamps/interrupted.csv"
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if timestamps.is_file() and len(timestamps.read_text().splitlines()) >= 3:
                break
            time.sleep(0.02)
        else:
            pytest.fail("Recording did not produce frames in time")
        os.killpg(process.pid, signal.SIGINT)  # Same process-group signal sent by terminal Ctrl-C.
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        info = json.loads((dataset / "clips/interrupted.json").read_text())
        assert info["frames"] >= 2
        assert len(list(csv.DictReader(timestamps.open(newline="")))) == info["frames"]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
