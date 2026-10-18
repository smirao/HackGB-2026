import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stationwatch.annotation import create_annotation_app
from stationwatch.config import Settings
from stationwatch.dataset import (
    evaluate,
    extract,
    file_hash,
    import_video,
    initialize,
    train_dataset,
    write_json,
)
from stationwatch.gemini import Review
from stationwatch.image_model import EMBEDDING_SIZE, ImageExtractor, ImageModel, preprocess_image
from stationwatch.model import Prediction
from stationwatch.pipeline import Packet, Pipeline
from stationwatch.recording import SegmentRecorder
from stationwatch.storage import EventStore
from stationwatch.temporal import SustainedFlag


def test_image_preprocessing_uses_metadata_normalization_rgb_and_crop():
    image = np.zeros((20, 40, 3), np.uint8)
    image[:, :20] = [255, 0, 0]
    image[:, 20:] = [0, 0, 255]
    tensor = preprocess_image(image, (0.5, 0, 1, 1))
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert np.all(tensor[..., 0] == 1) and np.all(tensor[..., 1:] == 0)
    with pytest.raises(ValueError):
        Settings(image_crop=(0.8, 0, 0.2, 1))


def test_duration_gate_strictly_exceeds_five_seconds_and_resets():
    gate = SustainedFlag(5, 1.5)
    for time_s in np.arange(10, 15.5, 0.5):
        assert not gate.update(True, time_s)
    assert gate.elapsed_s == 5
    assert gate.update(True, 15.5)
    assert not gate.update(False, 16)
    assert gate.elapsed_s == 0
    assert not gate.update(True, 16.5)
    assert not gate.update(True, 20)  # Missing observations cannot establish continuity.
    assert gate.elapsed_s == 0
    assert not gate.update(True, 19)  # Nonmonotonic capture timestamp starts over.
    assert not gate.update(True, float("nan"))


def test_image_pipeline_waits_then_reviews_and_resets_on_failure(tmp_path, monkeypatch):
    clock = SimpleNamespace(monotonic=lambda: clock.now, now=100)
    monkeypatch.setattr("stationwatch.pipeline.time", clock)
    settings = Settings(analysis_backend="image", data_dir=tmp_path, flag_duration_s=5)
    store = EventStore(tmp_path)
    extractor, model, reviewer = Mock(), Mock(), Mock()
    extractor.extract.return_value = np.ones(EMBEDDING_SIZE)
    model.predict.return_value = Prediction(True, 0.95, "mobilenet_image_classifier")
    reviewer.review.return_value = Review(
        decision="inattentive", confidence=0.95, reason="Visible disengagement"
    )
    pipeline = Pipeline(settings, Mock(), extractor, model, reviewer, Mock(), store, "image")
    original = Packet(np.zeros((120, 160, 3), np.uint8), b"jpeg", "capture-time", 100, 0, "clip.mp4", 0, "s1")

    def sample(t, **updates):
        clock.now = t
        assert pipeline.offer_sample(replace(original, monotonic_s=t, elapsed_s=t - 100, **updates))
        pipeline.local_pool.submit(lambda: None).result(3)
        pipeline.review_pool.submit(lambda: None).result(3)

    try:
        for t in np.arange(100, 105.5, 0.5):
            sample(float(t))
        reviewer.review.assert_not_called()
        sample(105.5)
        reviewer.review.assert_called_once()
        event = store.list()[0]
        assert event["features"]["sustained_flag_s"] == 5.5
        assert "head_angle_deg" not in event["features"]
        assert event["pose_backend"] is None
        assert event["local"]["source"] == "mobilenet_image_classifier"
        sample(106)  # Cooldown suppresses duplicates.
        assert len(store.list()) == 1
        extractor.extract.side_effect = RuntimeError("bad inference")
        sample(106.5)
        assert pipeline.flag_gate.elapsed_s == 0
        extractor.extract.side_effect = None
        sample(107)
        assert pipeline.flag_gate.elapsed_s == 0
        clock.now = 200
        assert pipeline.offer_sample(replace(original, monotonic_s=108))
        pipeline.local_pool.submit(lambda: None).result(3)
        assert pipeline.flag_gate.elapsed_s == 0
    finally:
        pipeline.close()
        store.close()


@pytest.mark.parametrize("people", [["p01", "p02", "p03"], ["p01", "p02", "p03", "p04"]])
def test_labeled_video_to_portable_image_classifier(tmp_path, monkeypatch, people):
    pytest.importorskip("sklearn")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    context = {"fixture": "image-encoder", "crop": [0, 0, 1, 1]}
    monkeypatch.setattr("stationwatch.image_training.image_context", lambda _: context)
    monkeypatch.setattr("stationwatch.image_model.image_context", lambda _: context)
    extractor = Mock()
    extractor.extract.side_effect = lambda image, t: np.full(EMBEDDING_SIZE, 1 if t >= 2 else -1, float)
    monkeypatch.setattr("stationwatch.image_training.ImageExtractor", lambda _: extractor)
    source_config = tmp_path / "config.json"
    source_config.write_text(Settings(width=160, height=120, fps=4).model_dump_json())
    root = tmp_path / "dataset"
    initialize(root, source_config, backend="image")
    for i, person in enumerate(people):
        source = tmp_path / person
        (source / "recordings").mkdir(parents=True)
        store = SimpleNamespace(root=source, retained_segments=lambda: set())
        recorder = SegmentRecorder(Settings(width=160, height=120, fps=4), store)
        try:
            for frame in range(16):
                recorder.write(np.full((120, 160, 3), frame * 10 + i * 2, np.uint8), frame / 4)
        finally:
            recorder.close()
        import_video(root, source / "recordings" / recorder.segment, person, person, "s01")
        write_json(
            root / "labels" / (person + ".json"),
            {
                "intervals": [
                    {"start_s": 0, "end_s": 2, "label": 0},
                    {"start_s": 2, "end_s": 4, "label": 1},
                ]
            },
        )
    with TestClient(create_annotation_app(root)) as client:
        assert client.get("/api/setup").json()["analysis_backend"] == "image"
    report = extract(root)  # Does not require pose calibration.
    assert report["samples"] == 8 * len(people)
    output = root / "classifier.json"
    heldout = people[-1]
    report = train_dataset(root, output, 0.8, [heldout])
    assert report["evaluation_groups"] == [heldout]
    assert set(report["training_groups"]) == set(people[:-1])
    deployment = Settings.model_validate_json(output.with_suffix(".config.json").read_text())
    model = ImageModel(deployment)
    assert model.predict(np.ones(EMBEDDING_SIZE)).flagged
    assert not model.predict(-np.ones(EMBEDDING_SIZE)).flagged
    digest = file_hash(output)
    with pytest.raises(ValueError, match="training groups"):
        evaluate(root, output, root / "eval.json")
    report = evaluate(root, output, root / "eval.json", groups_filter=[heldout])
    assert report["runtime_confusion_matrix_labels_0_1"] == [[4, 0], [0, 4]]
    assert file_hash(output) == digest
    write_json(root / "labels" / (heldout + ".json"), {"intervals": []})
    with pytest.raises(ValueError, match="changed"):
        train_dataset(root, output, 0.8)
    monkeypatch.setattr("stationwatch.image_model.image_context", lambda _: {"fixture": "changed"})
    with pytest.raises(ValueError, match="differ"):
        ImageModel(deployment)


def test_verified_encoder_runtime_smoke():
    """Opt-in real public encoder check; never downloads a model in the test suite."""
    model_path = os.environ.get("STATIONWATCH_TEST_ENCODER")
    if not model_path:
        pytest.skip("Set STATIONWATCH_TEST_ENCODER to a downloaded verified encoder")
    pytest.importorskip("ai_edge_litert")
    extractor = ImageExtractor(Settings(image_model_path=Path(model_path)))
    try:
        embedding = extractor.extract(np.zeros((120, 160, 3), np.uint8))
        assert embedding.shape == (1024,)
        assert np.isfinite(embedding).all()
        assert np.linalg.norm(embedding) == pytest.approx(1)
    finally:
        extractor.close()
