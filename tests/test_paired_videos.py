import json
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stationwatch.annotation import create_annotation_app
from stationwatch.config import Settings
from stationwatch.dataset import (
    clips,
    extract,
    import_numbered,
    import_pairs,
    import_video,
    initialize,
    labels_for,
    main,
    train_dataset,
    write_json,
)
from stationwatch.image_model import EMBEDDING_SIZE
from stationwatch.recording import SegmentRecorder


@pytest.fixture
def paired_dataset(tmp_path, request):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for real video fixtures")
    config = tmp_path / "config.json"
    config.write_text(Settings(width=160, height=120, fps=4).model_dump_json())
    root, videos = tmp_path / "dataset", tmp_path / "videos"
    initialize(root, config, backend="image")
    for person in range(1, getattr(request, "param", 4) + 1):
        directory = videos / f"p{person:02}"
        (directory / "recordings").mkdir(parents=True)
        for name, brightness in [("on_task", 20), ("distracted", 200)]:
            store = SimpleNamespace(root=directory, retained_segments=lambda: set())
            recorder = SegmentRecorder(Settings(width=160, height=120, fps=4), store)
            try:
                for frame in range(12):
                    # Survive H.264 quantization so each person's video has a distinct hash.
                    image = np.full((120, 160, 3), brightness + person * 10, np.uint8)
                    recorder.write(image, frame / 4)
            finally:
                recorder.close()
            (directory / "recordings" / recorder.segment).rename(directory / (name + ".mp4"))
    return root, videos


@pytest.fixture
def numbered_videos(paired_dataset, tmp_path):
    root, paired = paired_dataset
    videos = tmp_path / "numbered"
    videos.mkdir()
    for person, old in zip(["aaron", "hunter", "owen", "john_smith"], sorted(paired.iterdir())):
        for number, behavior in [(1, "on_task"), (10, "distracted")]:
            shutil.copyfile(old / f"{behavior}.mp4", videos / f"{person}{number}.MP4")
    return root, videos


def test_numbered_import_annotation_and_training_group_by_name(numbered_videos, tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from train_model import main as train_main

    root, videos = numbered_videos
    encoder = tmp_path / "encoder.tflite"
    encoder.touch()
    settings = Settings(analysis_backend="image", width=160, height=120, fps=4, image_model_path=encoder)
    write_json(root / "config.json", settings.model_dump(mode="json"))
    monkeypatch.setattr(sys, "argv", ["training_workflow.py", "--dataset", str(root),
                                   "annotate", "--videos", str(videos)])
    serve = Mock()
    monkeypatch.setattr("uvicorn.run", serve)
    main()
    assert serve.call_count == 1
    entries = clips(root)
    assert len(entries) == 8
    assert {info["person"] for info in entries} == {"aaron", "hunter", "owen", "john_smith"}
    assert all(labels_for(root, info) == [] for info in entries)
    intervals = [{"start_s": 0, "end_s": 1.5, "label": 0}, {"start_s": 1.5, "end_s": 3, "label": 1}]
    with TestClient(serve.call_args.args[0]) as client:
        listed = client.get("/api/clips").json()
        assert {info["source_name"] for info in listed} == {path.name for path in videos.iterdir()}
        for info in entries:
            assert client.post(f"/api/clips/{info['id']}/labels", json={"intervals": intervals}).status_code == 200
    monkeypatch.setattr("stationwatch.image_training.image_context", lambda _: {"fixture": "encoder"})
    extractor = Mock()
    extractor.extract.side_effect = lambda image, t: np.full(EMBEDDING_SIZE, 1 if t >= 1.5 else -1)
    monkeypatch.setattr("stationwatch.image_training.ImageExtractor", lambda _: extractor)
    output = tmp_path / "model.json"
    report = train_main(["--videos", str(videos), "--dataset", str(root), "--holdout", "owen", "--output", str(output)])
    assert report["evaluation_groups"] == ["owen"]
    assert report["evaluation_clips"] == ["owen1", "owen10"]
    assert report["training_groups"] == ["aaron", "hunter", "john_smith"]
    assert report["runtime_confusion_matrix_labels_0_1"] == [[6, 0], [0, 6]]
    assert len(clips(root)) == 8
    assert all(labels_for(root, info) == intervals for info in clips(root))


def test_renamed_numbered_video_preserves_existing_clip_and_labels(numbered_videos):
    root, videos = numbered_videos
    old_id = "aaron-s01-on-task"
    import_video(root, videos / "aaron1.MP4", old_id, "aaron", "s01")
    path = root / "labels" / (old_id + ".json")
    labels = {"intervals": [{"start_s": 0.5, "end_s": 1, "label": 1}]}
    write_json(path, labels)
    saved = path.read_bytes()
    assert import_numbered(root, videos) == {
        "imported": 7, "reused": 1, "people": ["aaron", "hunter", "john_smith", "owen"]
    }
    assert path.read_bytes() == saved
    assert not (root / "clips/aaron1.json").exists()
    assert next(info for info in clips(root) if info["id"] == old_id)["source_name"] == "aaron1.MP4"
    assert import_numbered(root, videos)["reused"] == 8
    assert len(clips(root)) == 8 and path.read_bytes() == saved


@pytest.mark.parametrize("problem", ["invalid_name", "duplicate", "changed"])
def test_numbered_batch_preflight_preserves_dataset_on_error(numbered_videos, problem):
    root, videos = numbered_videos
    if problem == "changed":
        import_numbered(root, videos)
        shutil.copyfile(videos / "aaron10.MP4", videos / "aaron1.MP4")
    else:
        name = "bad.mp4" if problem == "invalid_name" else "another1.mp4"
        shutil.copyfile(videos / "aaron1.MP4", videos / name)
    before = {path: path.read_bytes() for folder in ["clips", "labels"] for path in (root / folder).glob("*.json")}
    with pytest.raises(ValueError):
        import_numbered(root, videos)
    after = {path: path.read_bytes() for folder in ["clips", "labels"] for path in (root / folder).glob("*.json")}
    assert before == after


@pytest.mark.parametrize("paired_dataset", [3], indirect=True)
@pytest.mark.parametrize("holdout", [[], ["--holdout", "p03"]])
@pytest.mark.parametrize("layout", ["folders", "flat"])
def test_training_script_runs_six_paired_videos_without_person_leakage(
    paired_dataset, tmp_path, monkeypatch, holdout, layout
):
    pytest.importorskip("sklearn")
    from train_model import main as train_main

    _root, videos = paired_dataset
    if layout == "flat":
        flat = tmp_path / "flat-clips"
        flat.mkdir()
        for person in sorted(videos.iterdir()):
            for behavior, suffix in [("on_task", "task"), ("distracted", "distracted")]:
                shutil.copyfile(person / f"{behavior}.mp4", flat / f"{person.name}_{suffix}.MP4")
        videos = flat
    context = {"fixture": "encoder"}
    monkeypatch.setattr("stationwatch.image_training.image_context", lambda _: context)
    extractor = Mock()
    extractor.extract.side_effect = lambda image, t: np.full(EMBEDDING_SIZE, 1 if image.mean() > 100 else -1)
    monkeypatch.setattr("stationwatch.image_training.ImageExtractor", lambda _: extractor)
    encoder = tmp_path / "encoder.tflite"
    encoder.touch()
    config = tmp_path / "image-config.json"
    config.write_text(Settings(width=160, height=120, fps=4, image_model_path=encoder).model_dump_json())
    dataset, output = tmp_path / "six-clips", tmp_path / "models/attention.json"
    args = [
        "--videos", str(videos), "--dataset", str(dataset), "--config", str(config),
        "--output", str(output), "--trim-start-s", "0.5", "--trim-end-s", "0.5", *holdout,
    ]
    report = train_main(args)
    assert len(clips(dataset)) == 6
    assert len(report["training_groups"]) == 2
    assert len(report["evaluation_groups"]) == 1
    assert not set(report["training_groups"]) & set(report["evaluation_groups"])
    assert len(report["training_clips"]) == 4
    assert len(report["evaluation_clips"]) == 2
    assert report["training_samples"] == 16 and report["evaluation_samples"] == 8
    if holdout:
        assert report["evaluation_groups"] == ["p03"]
    assert report["runtime_confusion_matrix_labels_0_1"] == [[4, 0], [0, 4]]
    artifact = json.loads(output.read_text())
    assert artifact["kind"] == "image_logistic"
    deployment = Settings.model_validate_json(output.with_suffix(".config.json").read_text())
    assert deployment.classifier_path == output
    assert deployment.analysis_backend == "image"
    for person in report["evaluation_groups"]:
        assert all(not clip.startswith(person + "-") for clip in report["training_clips"])
    # A rerun reuses the frozen dataset config and retains manually corrected intervals.
    corrected = dataset / "labels/p01-s01-on-task.json"
    write_json(corrected, {"intervals": [{"start_s": 1, "end_s": 2.5, "label": 0}]})
    config.unlink()
    train_main(args)
    assert labels_for(dataset, clips(dataset)[1]) == [{"start_s": 1.0, "end_s": 2.5, "label": 0}]


def test_training_script_preserves_explicit_legacy_csv_mode(tmp_path, monkeypatch):
    from train_model import main as train_main

    train = Mock(return_value={"algorithm": "logistic"})
    monkeypatch.setattr("stationwatch.training.train", train)
    csv_path, output = tmp_path / "training.csv", tmp_path / "model.joblib"
    assert train_main([
        "--csv", str(csv_path), "--output", str(output), "--algorithm", "logistic", "--threshold", "0.7"
    ]) == {"algorithm": "logistic"}
    train.assert_called_once_with(csv_path, output, "logistic", 0.7)


@pytest.mark.parametrize("paired_dataset", [3], indirect=True)
def test_annotated_training_uses_browser_intervals_and_refreshes_old_filename_labels(
    paired_dataset, tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    from train_model import main as train_main

    root, videos = paired_dataset
    encoder = tmp_path / "encoder.tflite"
    encoder.touch()
    write_json(root / "config.json", Settings(
        analysis_backend="image", width=160, height=120, fps=4, image_model_path=encoder
    ).model_dump(mode="json"))
    monkeypatch.setattr("stationwatch.image_training.image_context", lambda _: {"fixture": "encoder"})
    extractor = Mock()
    extractor.extract.side_effect = lambda image, t: np.full(
        EMBEDDING_SIZE, 1 if 1 <= t < 1.5 or t >= 2.5 else -1
    )
    monkeypatch.setattr("stationwatch.image_training.ImageExtractor", lambda _: extractor)
    import_pairs(root, videos)
    assert extract(root)["samples"] == 36  # Cached extraction still has the old whole-clip labels.
    intervals = [
        {"start_s": 0, "end_s": 1, "label": 0},
        {"start_s": 1, "end_s": 1.5, "label": 1},
        {"start_s": 2, "end_s": 2.5, "label": 0},
        {"start_s": 2.5, "end_s": 3, "label": 1},
    ]
    with TestClient(create_annotation_app(root)) as client:
        for info in clips(root):
            # Fully unlabeled clips must not be silently assigned their filename's class.
            body = {"intervals": [] if info["id"] == "p01-s01-on-task" else intervals}
            assert client.post(f"/api/clips/{info['id']}/labels", json=body).status_code == 200
    saved = {path.name: path.read_bytes() for path in (root / "labels").glob("*.json")}
    # Imported dataset recordings are sufficient; original source files can be absent.
    shutil.rmtree(videos)
    monkeypatch.setattr("train_model.import_training_videos", Mock(side_effect=AssertionError("Must not import videos")))
    output = tmp_path / "model.json"
    report = train_main(["--annotated", "--dataset", str(root), "--holdout", "p03", "--output", str(output)])
    assert report["training_groups"] == ["p01", "p02"]
    assert report["evaluation_groups"] == ["p03"]
    assert report["training_samples"] == 15
    assert report["evaluation_samples"] == 10
    assert report["runtime_confusion_matrix_labels_0_1"] == [[6, 0], [0, 4]]
    assert saved == {path.name: path.read_bytes() for path in (root / "labels").glob("*.json")}
    extraction = json.loads((root / "images.report.json").read_text())
    assert extraction["labels"] == {"0": 15, "1": 10}
    assert extraction["unlabeled_samples"] == 11
    with np.load(root / "images.npz", allow_pickle=False) as data:
        assert "p01-s01-on-task" not in data["clip_id"]
        for info in clips(root):
            if info["id"] == "p01-s01-on-task":
                continue
            mask = data["clip_id"] == info["id"]
            assert data["video_s"][mask].tolist() == [0, 0.5, 1, 2, 2.5]
            assert data["label"][mask].tolist() == [0, 0, 1, 0, 1]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "filename"])
def test_invalid_flat_pair_is_rejected_before_import(paired_dataset, tmp_path, problem):
    root, videos = paired_dataset
    flat = tmp_path / "flat"
    flat.mkdir()
    shutil.copyfile(videos / "p01/on_task.mp4", flat / "first_last_task.MP4")
    if problem != "missing":
        shutil.copyfile(videos / "p01/distracted.mp4", flat / "first_last_distracted.MP4")
    if problem == "duplicate":
        shutil.copyfile(videos / "p01/on_task.mp4", flat / "other_task.MP4")
        shutil.copyfile(videos / "p01/distracted.mp4", flat / "other_distracted.MP4")
    elif problem == "filename":
        shutil.copyfile(videos / "p02/on_task.mp4", flat / "unlabeled.mp4")
    with pytest.raises(ValueError):
        import_pairs(root, flat)
    assert clips(root) == []


def test_pair_import_cli_labels_trims_and_holds_out_both_fourth_person_videos(paired_dataset, monkeypatch):
    pytest.importorskip("sklearn")
    root, videos = paired_dataset
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training_workflow.py",
            "--dataset",
            str(root),
            "import-pairs",
            "--videos",
            str(videos),
            "--trim-start-s",
            "0.5",
            "--trim-end-s",
            "0.5",
        ],
    )
    main()
    entries = clips(root)
    assert len(entries) == 8
    for info in entries:
        label = 1 if info["id"].endswith("-distracted") else 0
        assert labels_for(root, info) == [{"start_s": 0.5, "end_s": 2.5, "label": label}]
        assert info["person"] == info["id"].split("-")[0]
    context = {"fixture": "encoder"}
    monkeypatch.setattr("stationwatch.image_training.image_context", lambda _: context)
    extractor = Mock()
    extractor.extract.side_effect = lambda image, t: np.full(EMBEDDING_SIZE, 1 if image.mean() > 100 else -1)
    monkeypatch.setattr("stationwatch.image_training.ImageExtractor", lambda _: extractor)
    assert extract(root)["samples"] == 32
    model_path = root / "classifier.json"
    report = train_dataset(root, model_path, 0.8, ["p04"])
    assert report["training_groups"] == ["p01", "p02", "p03"]
    assert report["evaluation_groups"] == ["p04"]
    assert len(report["training_clips"]) == 6
    assert report["evaluation_clips"] == ["p04-s01-distracted", "p04-s01-on-task"]
    assert all(not name.startswith("p04-") for name in report["training_clips"])
    assert report["runtime_confusion_matrix_labels_0_1"] == [[4, 0], [0, 4]]
    artifact = json.loads(model_path.read_text())
    assert artifact["evaluation_clips"] == report["evaluation_clips"]


def test_adding_fourth_pair_preserves_previous_manual_labels(paired_dataset, tmp_path):
    root, videos = paired_dataset
    fourth = videos / "p04"
    hidden_fourth = tmp_path / "fourth-later"
    fourth.rename(hidden_fourth)
    assert import_pairs(root, videos)["imported"] == 6
    manual_path = root / "labels/p01-s01-on-task.json"
    manual = {"intervals": [{"start_s": 1, "end_s": 2, "label": 0}]}
    write_json(manual_path, manual)
    hidden_fourth.rename(fourth)
    result = import_pairs(root, videos)
    assert result["imported"] == 2 and result["skipped"] == 6
    assert json.loads(manual_path.read_text()) == manual
    assert len(clips(root)) == 8


@pytest.mark.parametrize("problem", ["missing", "duplicate", "changed", "trim"])
def test_invalid_pair_batch_is_rejected_before_new_imports(paired_dataset, problem):
    root, videos = paired_dataset
    if problem == "changed":
        import_pairs(root, videos)
        shutil.copyfile(videos / "p02/on_task.mp4", videos / "p01/on_task.mp4")
    elif problem == "missing":
        (videos / "p04/distracted.mp4").unlink()
    elif problem == "duplicate":
        shutil.copyfile(videos / "p01/on_task.mp4", videos / "p01/distracted.mp4")
    before = len(clips(root))
    with pytest.raises(ValueError):
        import_pairs(
            root,
            videos,
            trim_start_s=1.5 if problem == "trim" else 0,
            trim_end_s=1.5 if problem == "trim" else 0,
        )
    assert len(clips(root)) == before
