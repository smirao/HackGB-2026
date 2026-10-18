"""Local, human-labeled video datasets. Training never invokes Gemini or MongoDB."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import signal
import time
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import cv2

from .camera import create_camera
from .config import load_settings
from .features import FEATURE_NAMES, DwellTracker, create_extractor, extraction_context
from .recording import SegmentRecorder

MODEL_URLS = {
    "movenet": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
    "mediapipe": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}
BEHAVIOR_LABELS = {"on-task": 0, "distracted": 1}
PAIRED_FILES = {"on_task.mp4": "on-task", "distracted.mp4": "distracted"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Use 1–80 letters, digits, underscores or hyphens for clip/person/session IDs")
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_settings(root):
    if not (root / "config.json").is_file():
        raise ValueError("Initialize this dataset first with the init command")
    return load_settings(root / "config.json")


def initialize(root, config, backend=None):
    if (root / "config.json").exists():
        raise ValueError("Dataset already exists; choose another --dataset to start a different camera setup")
    settings = load_settings(config)
    if backend:
        settings = settings.model_copy(
            update={
                "analysis_backend": backend,
                "classifier_path": Path(
                    "models/attention_image.json" if backend == "image" else "models/attention.joblib"
                ),
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    for name in ("clips", "recordings", "labels", "timestamps"):
        (root / name).mkdir(exist_ok=True)
    write_json(root / "config.json", settings.model_dump(mode="json"))
    write_json(
        root / "dataset.json",
        {
            "schema_version": 1,
            "labels": {
                "0": "Acceptable task behavior, including permitted brief glances",
                "1": "Visible task disengagement requiring review",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "calibration_confirmed": False,
        },
    )
    print(
        f"Initialized {root} ({settings.analysis_backend}). Record clips, then annotate them in the browser."
    )
    return settings


def download_pose_model(settings):
    path = settings.pose_model_path
    if path.is_file():
        print(f"Using existing pose model: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    try:
        request = urllib.request.Request(
            MODEL_URLS[settings.pose_backend], headers={"User-Agent": "StationWatch/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        with temporary.open("rb") as handle:
            header = handle.read(8)
        if settings.pose_backend == "movenet" and header[4:8] != b"TFL3":
            raise ValueError("Download was not a TFLite model. Check the official model download URL.")
        if settings.pose_backend == "mediapipe" and not zipfile.is_zipfile(temporary):
            raise ValueError("Download was not a MediaPipe task bundle")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {path}; sha256={file_hash(path)}")


def download_model(settings):
    if settings.analysis_backend == "image":
        from .image_model import download_image_model

        download_image_model(settings)
    else:
        download_pose_model(settings)


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset paths must stay inside the dataset directory")
    return path


def clip_metadata(root, clip_id):
    identifier(clip_id)
    path = root / "clips" / (clip_id + ".json")
    if not path.is_file():
        raise FileNotFoundError(f"Unknown clip: {clip_id}")
    return json.loads(path.read_text())


def clips(root):
    return [json.loads(path.read_text()) for path in sorted((root / "clips").glob("*.json"))]


def probe_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        readable, _ = capture.read()
        if not readable or not math.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError(f"Cannot read a finite video with a valid frame rate: {path}")
        return fps, frames
    finally:
        capture.release()


def save_clip(root, clip_id, person, session, video, timestamps=None):
    identifier(clip_id), identifier(person), identifier(session)
    fps, frames = probe_video(video)
    info = {
        "id": clip_id,
        "person": person,
        "session": session,
        "video": str(video.relative_to(root)),
        "fps": fps,
        "frames": frames,
        "duration_s": frames / fps,
        "timestamps": timestamps,
        "clock": "capture_timestamps" if timestamps else "encoded_video_time",
    }
    write_json(root / "clips" / (clip_id + ".json"), info)
    write_json(root / "labels" / (clip_id + ".json"), {"intervals": []})
    print(f"Saved {clip_id}: {frames} frames, {info['duration_s']:.2f}s playback. No labels assigned yet.")
    return info


def record(root, clip_id, person, session, seconds):
    settings = dataset_settings(root)
    identifier(clip_id), identifier(person), identifier(session)
    if not math.isfinite(seconds) or not 0 < seconds <= 240:
        raise ValueError("Record between 0 and 240 seconds per clip")
    if (root / "clips" / (clip_id + ".json")).exists():
        raise ValueError("That clip ID already exists; choose a new one")
    # Keep one recording segment and protect all training videos from recording cleanup.
    store = SimpleNamespace(
        root=root, retained_segments=lambda: {p.name for p in (root / "recordings").glob("*.mp4")}
    )
    recorder = SegmentRecorder(settings.model_copy(update={"segment_seconds": 300}), store)
    camera = create_camera(settings)
    timestamps = root / "timestamps" / (clip_id + ".csv")
    count = 0
    started = time.monotonic()
    next_frame = started
    interrupted = False

    def stop_recording(*_):
        nonlocal interrupted
        interrupted = True

    previous_interrupt = signal.signal(signal.SIGINT, stop_recording)
    print(
        f"Recording {clip_id} for {seconds:g}s. Camera position must match deployment. Ctrl-C ends this take."
    )
    try:
        with timestamps.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "captured_s", "video_s"])
            writer.writeheader()
            while not interrupted and time.monotonic() - started < seconds:
                time.sleep(max(0, next_frame - time.monotonic()))
                if interrupted:
                    break
                image = camera.read()
                captured = time.monotonic()
                if interrupted or captured - started >= seconds:
                    break
                image = cv2.resize(image, (settings.width, settings.height))
                _, video_s = recorder.write(image, captured)
                writer.writerow({"frame": count, "captured_s": captured - started, "video_s": video_s})
                handle.flush()
                count += 1
                next_frame = max(next_frame + 1 / settings.fps, time.monotonic())
    except KeyboardInterrupt:
        interrupted = True
    finally:
        try:
            try:
                camera.close()
            finally:
                recorder.close()
        finally:
            signal.signal(signal.SIGINT, previous_interrupt)
    if count == 0:
        raise ValueError("No frames were recorded")
    video = root / "recordings" / recorder.segment
    info = save_clip(root, clip_id, person, session, video, str(timestamps.relative_to(root)))
    if interrupted:
        print("Saved the shortened take after Ctrl-C.")
    return info


def import_video(root, source, clip_id, person, session):
    dataset_settings(root)
    identifier(clip_id), identifier(person), identifier(session)
    if (root / "clips" / (clip_id + ".json")).exists():
        raise ValueError("That clip ID already exists; choose a new one")
    if source.suffix.lower() != ".mp4":
        raise ValueError("Import a browser-playable MP4, preferably H.264")
    probe_video(source)
    destination = root / "recordings" / (uuid.uuid4().hex + ".mp4")
    shutil.copyfile(source, destination)
    return save_clip(root, clip_id, person, session, destination)


def behavior_intervals(duration, behavior, trim_start_s=0, trim_end_s=0):
    if behavior not in BEHAVIOR_LABELS:
        raise ValueError("Choose behavior on-task or distracted")
    if any(not math.isfinite(value) or value < 0 for value in (trim_start_s, trim_end_s)):
        raise ValueError("Trim times must be finite, nonnegative seconds")
    if trim_start_s + trim_end_s >= duration:
        raise ValueError("Trim times must leave a nonempty labeled part of every video")
    return validate_intervals(
        [{"start_s": trim_start_s, "end_s": duration - trim_end_s, "label": BEHAVIOR_LABELS[behavior]}],
        duration,
    )


def label_behavior(root, info, behavior, trim_start_s=0, trim_end_s=0):
    intervals = behavior_intervals(info["duration_s"], behavior, trim_start_s, trim_end_s)
    write_json(
        root / "labels" / (info["id"] + ".json"),
        {
            "intervals": intervals,
            "label_source": "user_selected_clip_behavior",
            "behavior": behavior,
        },
    )
    print(f"Labeled {info['id']}: {behavior} ({BEHAVIOR_LABELS[behavior]}). Review exceptions in annotate.")


def paired_sources(videos):
    """Find complete pairs in flat name_task.MP4 or person/on_task.mp4 layouts."""
    if not videos.is_dir():
        raise ValueError("--videos must be a directory containing paired MP4 clips")
    flat = sorted(path for path in videos.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
    people = sorted(path for path in videos.iterdir() if path.is_dir() and not path.name.startswith("."))
    pairs = {}
    if flat:
        if people:
            raise ValueError("Use either flat name_task.mp4/name_distracted.mp4 files or person folders")
        for source in flat:
            match = re.fullmatch(r"(.+)_(task|distracted)", source.stem, flags=re.IGNORECASE)
            if not match:
                raise ValueError(f"{source.name}: expected name_task.mp4 or name_distracted.mp4")
            person = identifier(match[1])
            behavior = "on-task" if match[2].lower() == "task" else "distracted"
            pair = pairs.setdefault(person, {})
            if behavior in pair:
                raise ValueError(f"{person}: multiple {behavior} clips; use a separate session for new takes")
            pair[behavior] = source
    else:
        for person in people:
            identifier(person.name)
            pair = pairs.setdefault(person.name, {})
            for source in sorted(person.iterdir()):
                behavior = PAIRED_FILES.get(source.name.lower())
                if source.is_file() and behavior:
                    if behavior in pair:
                        raise ValueError(f"{person.name}: multiple {behavior} clips")
                    pair[behavior] = source
    if not pairs:
        raise ValueError("No paired videos found in --videos")
    for person, pair in pairs.items():
        if set(pair) != set(BEHAVIOR_LABELS):
            raise ValueError(f"{person}: provide both task and distracted clips")
    return pairs


def import_pairs(root, videos, session="s01", trim_start_s=0, trim_end_s=0):
    """Import paired behavior clips without overwriting existing labels."""
    dataset_settings(root)
    identifier(session)
    pairs = paired_sources(videos)
    # Check all pairs, timestamps, duplicates, and existing IDs before copying any videos.
    plans, fingerprints = [], {}
    for existing in clips(root):
        digest = file_hash(safe_path(root, existing["video"]))
        fingerprints.setdefault(digest, set()).add(existing["id"])
    for person, pair in sorted(pairs.items()):
        for behavior, source in pair.items():
            clip_id = identifier(f"{person}-{session}-{behavior}")
            fps, frames = probe_video(source)
            behavior_intervals(frames / fps, behavior, trim_start_s, trim_end_s)
            digest = file_hash(source)
            duplicates = fingerprints.get(digest, set()) - {clip_id}
            if duplicates:
                raise ValueError(f"{source}: video duplicates another clip: {sorted(duplicates)}")
            fingerprints.setdefault(digest, set()).add(clip_id)
            exists = (root / "clips" / (clip_id + ".json")).is_file()
            if exists:
                existing = clip_metadata(root, clip_id)
                if (
                    existing["person"] != person
                    or existing["session"] != session
                    or file_hash(safe_path(root, existing["video"])) != digest
                ):
                    raise ValueError(
                        f"{clip_id}: already exists with different footage; use a new session ID"
                    )
            plans.append((source, clip_id, person, behavior, exists))
    imported = skipped = 0
    for source, clip_id, person, behavior, exists in plans:
        if exists:
            print(f"Keeping {clip_id}: same footage already imported; existing labels preserved.")
            skipped += 1
            continue
        info = import_video(root, source, clip_id, person, session)
        label_behavior(root, info, behavior, trim_start_s, trim_end_s)
        imported += 1
    print(f"Imported {imported} clips; kept {skipped} existing clips across {len(pairs)} people.")
    return {"imported": imported, "skipped": skipped, "people": sorted(pairs)}


def validate_intervals(intervals, duration):
    result = []
    for interval in intervals:
        if not isinstance(interval, dict) or set(interval) != {"start_s", "end_s", "label"}:
            raise ValueError("Each label interval needs start_s, end_s, and label")
        start, end, label = interval["start_s"], interval["end_s"], interval["label"]
        if (
            type(label) is not int
            or label not in (0, 1)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < min(end, duration)
            or end > duration + 0.001
        ):
            raise ValueError("Use label 0/1 and finite interval times: 0 <= start < end <= clip duration")
        result.append({"start_s": float(start), "end_s": min(float(end), duration), "label": label})
    result.sort(key=lambda interval: interval["start_s"])
    if any(b["start_s"] < a["end_s"] for a, b in pairwise(result)):
        raise ValueError("Label intervals cannot overlap; leave uncertain sections unlabeled")
    return result


def labels_for(root, info):
    path = root / "labels" / (info["id"] + ".json")
    value = json.loads(path.read_text()) if path.is_file() else {"intervals": []}
    return validate_intervals(value["intervals"], info["duration_s"])


def label_at(intervals, video_s):
    return next(
        (interval["label"] for interval in intervals if interval["start_s"] <= video_s < interval["end_s"]),
        None,
    )


def input_signature(root):
    values = {"config": json.loads((root / "config.json").read_text()), "clips": []}
    for info in clips(root):
        entry = {
            "info": info,
            "labels": labels_for(root, info),
            "video_sha256": file_hash(safe_path(root, info["video"])),
        }
        if info["timestamps"]:
            entry["timestamps_sha256"] = file_hash(safe_path(root, info["timestamps"]))
        values["clips"].append(entry)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def frame_times(root, info):
    if not info["timestamps"]:
        return [(i / info["fps"], i / info["fps"]) for i in range(info["frames"])]
    with safe_path(root, info["timestamps"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != info["frames"]:
        raise ValueError(f"Capture timestamps do not match frame count for {info['id']}")
    result = []
    for i, row in enumerate(rows):
        captured, video = float(row["captured_s"]), float(row["video_s"])
        if (
            int(row["frame"]) != i
            or not all(math.isfinite(v) and v >= 0 for v in (captured, video))
            or (result and (captured <= result[-1][0] or video <= result[-1][1]))
        ):
            raise ValueError(f"Invalid capture timeline for {info['id']}")
        result.append((captured, video))
    return result


def extract(root, group_by="person"):
    settings = dataset_settings(root)
    if settings.analysis_backend == "image":
        from .image_training import extract_images

        return extract_images(root, group_by)
    manifest = json.loads((root / "dataset.json").read_text())
    if not manifest.get("calibration_confirmed"):
        raise ValueError("Open annotate and Save calibration before extracting features")
    context = extraction_context(settings)
    signature = input_signature(root)
    entries = clips(root)
    if not entries:
        raise ValueError("Record or import videos first")
    output = root / "training.csv"
    temporary = root / "training.csv.tmp"
    fields = FEATURE_NAMES + ["label", "group", "clip_id", "video_s", "captured_s"]
    report = {
        "group_by": group_by,
        "input_signature": signature,
        "feature_context": context,
        "clips": [],
        "labeled_samples": 0,
        "unknown_pose_samples": 0,
        "unlabeled_samples": 0,
    }
    extractor = create_extractor(settings)
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for info in entries:
                intervals = labels_for(root, info)
                timeline = frame_times(root, info)
                capture = cv2.VideoCapture(str(safe_path(root, info["video"])))
                tracker = DwellTracker(settings.dwell_gap_s)
                next_sample = 0.0
                summary = {
                    "id": info["id"],
                    "sampled": 0,
                    "saved": 0,
                    "unknown": 0,
                    "unlabeled": 0,
                    "labels": {"0": 0, "1": 0},
                }
                try:
                    for captured_s, video_s in timeline:
                        ok, image = capture.read()
                        if not ok:
                            raise ValueError(f"Could not decode all frames of {info['id']}")
                        if captured_s + 1e-8 < next_sample:
                            continue
                        next_sample = (
                            math.floor(captured_s / settings.sample_interval_s) + 1
                        ) * settings.sample_interval_s
                        summary["sampled"] += 1
                        image = cv2.resize(image, (settings.width, settings.height))
                        features = tracker.update(extractor.extract(image, captured_s), captured_s)
                        label = label_at(intervals, video_s)
                        if features is None:
                            summary["unknown"] += 1
                        if label is None:
                            summary["unlabeled"] += 1
                        if features is None or label is None:
                            continue
                        group = (
                            info["person"] if group_by == "person" else info["person"] + ":" + info["session"]
                        )
                        writer.writerow(
                            dict(
                                zip(FEATURE_NAMES, features.values()),
                                label=label,
                                group=group,
                                clip_id=info["id"],
                                video_s=video_s,
                                captured_s=captured_s,
                            )
                        )
                        summary["saved"] += 1
                        summary["labels"][str(label)] += 1
                finally:
                    capture.release()
                report["clips"].append(summary)
                report["labeled_samples"] += summary["saved"]
                report["unknown_pose_samples"] += summary["unknown"]
                report["unlabeled_samples"] += summary["unlabeled"]
                print(
                    f"{info['id']}: saved {summary['saved']}/{summary['sampled']}; "
                    f"unknown pose={summary['unknown']}, unlabeled={summary['unlabeled']}"
                )
        temporary.replace(output)
        report["feature_csv_sha256"] = file_hash(output)
        write_json(root / "extraction.report.json", report)
    finally:
        extractor.close()
        temporary.unlink(missing_ok=True)
    print(f"Saved {report['labeled_samples']} labeled samples to {output}")
    return report


def checked_extraction(root):
    settings = dataset_settings(root)
    report_path = root / "extraction.report.json"
    if not report_path.is_file():
        raise ValueError("Run extract before training/evaluation")
    report = json.loads(report_path.read_text())
    if (
        report["input_signature"] != input_signature(root)
        or report["feature_context"] != extraction_context(settings)
        or report["feature_csv_sha256"] != file_hash(root / "training.csv")
    ):
        raise ValueError("Videos, labels, pose model or calibration changed. Run extract again.")
    return report


def print_metrics(report):
    metrics = report["report_at_runtime_threshold"]
    matrix = report["runtime_confusion_matrix_labels_0_1"]
    away = metrics.get("1", {})
    print(
        f"At threshold {report['runtime_threshold']:.2f}: review-class precision={away.get('precision', 0):.1%}, "
        f"recall={away.get('recall', 0):.1%}; false flags={matrix[0][1]}, missed positives={matrix[1][0]}."
    )
    print("These are labeled-frame metrics, not event-level accuracy or proof of attention.")


def train_dataset(root, output, threshold, holdout=None):
    if dataset_settings(root).analysis_backend == "image":
        from .image_training import train_images

        return train_images(root, output, threshold, holdout)
    from .training import train

    extraction = checked_extraction(root)
    report = train(
        root / "training.csv",
        output,
        threshold=threshold,
        evaluation_groups=holdout,
        feature_context=extraction["feature_context"],
        verbose=False,
    )
    config = dataset_settings(root).model_copy(
        update={"classifier_path": output, "local_threshold": threshold}
    )
    deployment = output.with_suffix(".config.json")
    write_json(deployment, config.model_dump(mode="json"))
    print(f"Training groups: {', '.join(report['training_groups'])}")
    print(f"Held-out groups: {', '.join(report['evaluation_groups'])}")
    print_metrics(report)
    print(f"Model: {output}\nReport: {output.with_suffix('.report.json')}\nPi config: {deployment}")
    return report


def evaluate(root, model_path, output, threshold=None, groups_filter=None):
    if dataset_settings(root).analysis_backend == "image":
        from .image_training import evaluate_images

        return evaluate_images(root, model_path, output, threshold, groups_filter)
    import joblib
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    extraction = checked_extraction(root)
    artifact = joblib.load(model_path)
    if artifact.get("features") != FEATURE_NAMES or artifact.get("schema_version") != 1:
        raise ValueError("Incompatible classifier schema")
    if artifact.get("feature_context") != extraction["feature_context"]:
        raise ValueError("Evaluation extraction/calibration does not match the model")
    with (root / "training.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if groups_filter:
        unknown = set(groups_filter) - {row["group"] for row in rows}
        if unknown:
            raise ValueError(f"Evaluation groups are not in the dataset: {sorted(unknown)}")
        rows = [row for row in rows if row["group"] in groups_filter]
    if not rows:
        raise ValueError("No labeled samples to evaluate")
    groups = {row["group"] for row in rows}
    if not artifact.get("training_groups"):
        raise ValueError("This model lacks training-group metadata; retrain through this workflow")
    if groups & set(artifact["training_groups"]):
        raise ValueError(
            "Evaluation includes training groups. Reserve different people/sessions for this test."
        )
    cutoff = artifact["evaluated_threshold"] if threshold is None else threshold
    if not 0 <= cutoff <= 1:
        raise ValueError("Threshold must be in [0,1]")
    x = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows])
    y = np.array([int(row["label"]) for row in rows])
    classifier = artifact["classifier"]
    scores = classifier.predict_proba(x)[:, list(classifier.classes_).index(1)]
    predictions = ((scores >= cutoff) & (x[:, FEATURE_NAMES.index("in_workstation")] > 0)).astype(int)
    report = {
        "runtime_threshold": cutoff,
        "groups": sorted(groups),
        "samples": len(rows),
        "report_at_runtime_threshold": classification_report(
            y, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
        "runtime_confusion_matrix_labels_0_1": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }
    write_json(output, report)
    predictions_path = output.with_suffix(".predictions.csv")
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip_id", "video_s", "captured_s", "group", "label", "score", "predicted"]
        )
        writer.writeheader()
        for row, score, prediction in zip(rows, scores, predictions):
            writer.writerow(
                {
                    **{key: row[key] for key in ["clip_id", "video_s", "captured_s", "group", "label"]},
                    "score": float(score),
                    "predicted": int(prediction),
                }
            )
    print_metrics(report)
    print(f"Saved {output} and {predictions_path}. No fitting occurred.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/training_dataset"))
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("init", help="Create a dataset and freeze a copy of the Pi settings")
    setup.add_argument("--config", type=Path, default=Path("config.pi.example.json"))
    setup.add_argument("--backend", choices=["image", "pose"], default="image")
    setup.add_argument("--download-model", action="store_true")
    commands.add_parser("download-model", help="Download the selected pretrained image or pose encoder")
    paired = commands.add_parser(
        "import-pairs", help="Import name_task.mp4/name_distracted.mp4 or person-folder pairs with labels"
    )
    paired.add_argument("--videos", type=Path, required=True)
    paired.add_argument("--session", default="s01")
    paired.add_argument(
        "--trim-start-s", type=float, default=0, help="Leave this many opening seconds unlabeled"
    )
    paired.add_argument(
        "--trim-end-s", type=float, default=0, help="Leave this many closing seconds unlabeled"
    )
    for name in ("record", "import-video"):
        command = commands.add_parser(name)
        command.add_argument("--clip", required=True)
        command.add_argument("--person", required=True)
        command.add_argument("--session", required=True)
        command.add_argument(
            "--behavior", choices=list(BEHAVIOR_LABELS), help="Assign a label to the whole clip"
        )
        if name == "record":
            command.add_argument("--seconds", type=float, default=60)
        else:
            command.add_argument("--video", type=Path, required=True)
    commands.add_parser("list", help="List clips and labeled intervals")
    annotate = commands.add_parser("annotate", help="Open a local video annotation/calibration server")
    annotate.add_argument("--port", type=int, default=8002)
    extraction = commands.add_parser("extract", help="Extract visual features from human-labeled video")
    extraction.add_argument("--group-by", choices=["person", "session"], default="person")
    fit = commands.add_parser("train", help="Train, evaluate, and export model plus matching Pi config")
    fit.add_argument("--output", type=Path)
    fit.add_argument("--threshold", type=float, default=0.8)
    fit.add_argument("--holdout", nargs="+", help="Explicit groups to exclude from fitting")
    check = commands.add_parser("evaluate", help="Evaluate additional held-out recordings without fitting")
    check.add_argument("--model", type=Path)
    check.add_argument("--output", type=Path)
    check.add_argument("--threshold", type=float)
    check.add_argument("--groups", nargs="+", help="Evaluate only these held-out groups, e.g. a new person")
    args = parser.parse_args()
    root = args.dataset
    try:
        if args.command == "init":
            settings = initialize(root, args.config, args.backend)
            if args.download_model:
                download_model(settings)
        elif args.command == "download-model":
            download_model(dataset_settings(root))
        elif args.command == "record":
            info = record(root, args.clip, args.person, args.session, args.seconds)
            if args.behavior:
                label_behavior(root, info, args.behavior)
        elif args.command == "import-video":
            info = import_video(root, args.video, args.clip, args.person, args.session)
            if args.behavior:
                label_behavior(root, info, args.behavior)
        elif args.command == "import-pairs":
            import_pairs(root, args.videos, args.session, args.trim_start_s, args.trim_end_s)
        elif args.command == "list":
            dataset_settings(root)
            for info in clips(root):
                print(
                    f"{info['id']}: person={info['person']}, session={info['session']}, "
                    f"{info['duration_s']:.1f}s, intervals={len(labels_for(root, info))}"
                )
        elif args.command == "annotate":
            import uvicorn

            from .annotation import create_annotation_app

            dataset_settings(root)
            print(f"Annotation tool: http://127.0.0.1:{args.port}. Stop with Ctrl-C before extracting.")
            uvicorn.run(create_annotation_app(root), host="127.0.0.1", port=args.port, access_log=False)
        elif args.command == "extract":
            extract(root, args.group_by)
        elif args.command == "train":
            train_dataset(
                root, args.output or dataset_settings(root).classifier_path, args.threshold, args.holdout
            )
        elif args.command == "evaluate":
            evaluate(
                root,
                args.model or dataset_settings(root).classifier_path,
                args.output or root / "evaluation.report.json",
                args.threshold,
                args.groups,
            )
    except (ValueError, FileNotFoundError, OSError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
