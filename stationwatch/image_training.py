"""Training and held-out evaluation from human-labeled videos, without body tracking."""

import csv
import json
import math

import cv2
import numpy as np

from .dataset import (
    clips,
    dataset_settings,
    file_hash,
    frame_times,
    input_signature,
    label_at,
    labels_for,
    print_metrics,
    safe_path,
    write_json,
)
from .image_model import EMBEDDING_SIZE, ImageExtractor, image_context
from .temporal import SustainedFlag


def extract_images(root, group_by="person"):
    settings = dataset_settings(root)
    context = image_context(settings)
    signature = input_signature(root)
    extractor = ImageExtractor(settings)
    rows, vectors, sources, summaries = [], [], {}, []
    try:
        for info in clips(root):
            group = info["person"] if group_by == "person" else info["person"] + ":" + info["session"]
            source_hash = file_hash(safe_path(root, info["video"]))
            if source_hash in sources and sources[source_hash] != group:
                raise ValueError("Identical videos assigned to different groups would leak into evaluation")
            sources[source_hash] = group
            intervals, timeline = labels_for(root, info), frame_times(root, info)
            capture = cv2.VideoCapture(str(safe_path(root, info["video"])))
            next_sample, count = 0.0, 0
            summary = {
                "id": info["id"], "source_name": info.get("source_name", info["id"]),
                "labels": {"0": 0, "1": 0}, "unlabeled_samples": 0,
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
                    label = label_at(intervals, video_s)
                    if label is None:
                        summary["unlabeled_samples"] += 1
                        continue
                    image = cv2.resize(image, (settings.width, settings.height))
                    vectors.append(extractor.extract(image, captured_s))
                    rows.append(
                        {
                            "clip_id": info["id"],
                            "group": group,
                            "label": label,
                            "video_s": video_s,
                            "captured_s": captured_s,
                        }
                    )
                    count += 1
                    summary["labels"][str(label)] += 1
            finally:
                capture.release()
            summaries.append({**summary, "samples": count})
            print(
                f"{summary['source_name']}: {count} labeled images "
                f"(on task={summary['labels']['0']}, distracted={summary['labels']['1']}, "
                f"unlabeled/skipped={summary['unlabeled_samples']})"
            )
        if not rows:
            raise ValueError("No labeled images: record/import videos and save annotation intervals first")
        # Numeric arrays only: no executable pickle serialization in the image path.
        path = root / "images.npz"
        temporary = root / "images.npz.tmp"
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    embeddings=np.asarray(vectors),
                    **{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        report = {
            "kind": "image_embeddings",
            "group_by": group_by,
            "samples": len(rows),
            "clips": summaries,
            "labels": {key: sum(clip["labels"][key] for clip in summaries) for key in ("0", "1")},
            "unlabeled_samples": sum(clip["unlabeled_samples"] for clip in summaries),
            "input_signature": signature,
            "image_context": context,
            "arrays_sha256": file_hash(path),
        }
        write_json(root / "images.report.json", report)
        print(f"Saved {len(rows)} labeled image embeddings to {path}. No pose model was used.")
        return report
    finally:
        extractor.close()


def load_images(root):
    settings = dataset_settings(root)
    path, report_path = root / "images.npz", root / "images.report.json"
    if not path.is_file() or not report_path.is_file():
        raise ValueError("Run extract before training/evaluation")
    report = json.loads(report_path.read_text())
    if (
        report["input_signature"] != input_signature(root)
        or report["image_context"] != image_context(settings)
        or report["arrays_sha256"] != file_hash(path)
    ):
        raise ValueError("Videos, labels, encoder or crop changed. Run extract again.")
    with np.load(path, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    x, y = data["embeddings"], data["label"]
    if x.shape != (len(y), EMBEDDING_SIZE) or not np.isfinite(x).all() or not set(y) <= {0, 1}:
        raise ValueError("Invalid image training arrays")
    if any(len(value) != len(y) for value in data.values()):
        raise ValueError("Image rows and embeddings are misaligned")
    return settings, report, data


def split_groups(y, groups, holdout):
    from sklearn.model_selection import GroupShuffleSplit

    if len(y) < 20 or len(set(groups)) < 3:
        raise ValueError("Record at least 20 labeled frames across three independent people/session groups")
    if holdout:
        unknown = set(holdout) - set(groups)
        if unknown:
            raise ValueError(f"Unknown holdout groups: {sorted(unknown)}")
        mask = np.isin(groups, holdout)
        candidates = [(np.flatnonzero(~mask), np.flatnonzero(mask))]
    else:
        candidates = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=42).split(y, y, groups)
    for train, test in candidates:
        if set(y[train]) == set(y[test]) == {0, 1}:
            return train, test
    raise ValueError("Both training and held-out groups need examples of each label")


def split_clips(y, clip_ids, people, holdout=None, seed=42):
    """Hold out one whole clip per known person, without choosing by model performance."""
    if len(y) < 20:
        raise ValueError("Record at least 20 labeled frames before training")
    by_person = {person: sorted(set(clip_ids[people == person])) for person in sorted(set(people))}
    insufficient = [str(person) for person, ids in by_person.items() if len(ids) < 2]
    if insufficient:
        raise ValueError(f"Clip splitting needs at least two labeled clips per person: {insufficient}")
    if holdout and set(holdout) - set(clip_ids):
        raise ValueError("With --split-by clip, --holdout must contain existing clip IDs, not person names")
    rng = np.random.default_rng(seed)
    for _ in range(1 if holdout else 30):
        selected = holdout or [rng.choice(ids) for ids in by_person.values()]
        mask = np.isin(clip_ids, selected)
        train, test = np.flatnonzero(~mask), np.flatnonzero(mask)
        if set(people[train]) != set(by_person) or set(people[test]) != set(by_person):
            raise ValueError("Every person must have whole clips in both training and evaluation")
        if set(y[train]) == set(y[test]) == {0, 1}:
            return train, test
    raise ValueError("Cannot hold out whole clips from each person with both labels in training and evaluation")


def score_images(artifact, x):
    logits = np.asarray(x) @ np.asarray(artifact["weights"]) + artifact["bias"]
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def metrics(y, scores, threshold):
    from sklearn.metrics import classification_report, confusion_matrix

    predicted = (scores >= threshold).astype(int)
    return {
        "runtime_threshold": threshold,
        "report_at_runtime_threshold": classification_report(
            y, predicted, labels=[0, 1], output_dict=True, zero_division=0
        ),
        "runtime_confusion_matrix_labels_0_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def duration_candidates(data, indices, scores, settings, threshold):
    """Show local gate candidates on held-out clips; Gemini does not run during training."""
    eligible = np.zeros(len(indices), dtype=bool)
    spans = []
    for clip in sorted(set(data["clip_id"][indices])):
        locations = [i for i, index in enumerate(indices) if data["clip_id"][index] == clip]
        locations.sort(key=lambda i: data["captured_s"][indices[i]])
        gate = SustainedFlag(settings.flag_duration_s, settings.dwell_gap_s)
        previous_eligible = False
        for i in locations:
            index = indices[i]
            eligible[i] = gate.update(scores[i] >= threshold, float(data["captured_s"][index]))
            if eligible[i] and not previous_eligible:
                spans.append(
                    {
                        "clip_id": str(clip),
                        "video_s": float(data["video_s"][index]),
                        "captured_s": float(data["captured_s"][index]),
                        "label_at_candidate": int(data["label"][index]),
                    }
                )
            previous_eligible = eligible[i]
    return eligible, spans


def save_predictions(path, data, indices, scores, eligible, threshold):
    keys = ["clip_id", "group", "label", "video_s", "captured_s"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys + ["score", "predicted", "duration_gate_passed"])
        writer.writeheader()
        for index, score, passed in zip(indices, scores, eligible):
            writer.writerow(
                {
                    **{key: data[key][index].item() for key in keys},
                    "score": float(score),
                    "predicted": int(score >= threshold),
                    "duration_gate_passed": int(passed),
                }
            )


def train_images(root, output, threshold=0.8, holdout=None, split_by="person", seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if output.suffix != ".json" or not 0 <= threshold <= 1:
        raise ValueError("Use a .json output and a threshold in [0,1]")
    if split_by not in {"person", "clip"}:
        raise ValueError("Choose --split-by person or clip")
    settings, extraction, data = load_images(root)
    metadata = {info["id"]: info for info in clips(root)}
    people = np.asarray([metadata[clip_id]["person"] for clip_id in data["clip_id"]])
    if split_by == "clip":
        data = {**data, "group": data["clip_id"]}
    x, y, groups = data["embeddings"], data["label"], data["group"]
    if split_by == "clip":
        train, test = split_clips(y, groups, people, holdout, seed)
        hashes = [
            {file_hash(safe_path(root, metadata[clip_id]["video"])) for clip_id in set(groups[indices])}
            for indices in [train, test]
        ]
        if hashes[0] & hashes[1]:
            raise ValueError("Identical video content in training and held-out clips would leak into evaluation")
    else:
        train, test = split_groups(y, groups, holdout)
    scaler = StandardScaler().fit(x[train])
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    classifier.fit(scaler.transform(x[train]), y[train])
    # Fold normalization into the head so Pi inference needs only NumPy + LiteRT.
    weights = classifier.coef_[0] / scaler.scale_
    artifact = {
        "schema_version": 1,
        "kind": "image_logistic",
        "image_context": extraction["image_context"],
        "weights": weights.tolist(),
        "bias": float(classifier.intercept_[0] - weights @ scaler.mean_),
        "training_groups": sorted(set(groups[train])),
        "evaluation_groups": sorted(set(groups[test])),
        "training_clips": sorted(set(data["clip_id"][train])),
        "evaluation_clips": sorted(set(data["clip_id"][test])),
        "training_people": sorted(set(people[train])),
        "evaluation_people": sorted(set(people[test])),
        "split_by": "clip" if split_by == "clip" else extraction["group_by"],
        "split_seed": seed if split_by == "clip" else 42,
        "evaluated_threshold": threshold,
        "input_signature": extraction["input_signature"],
    }
    scores = score_images(artifact, x[test])
    eligible, spans = duration_candidates(data, test, scores, settings, threshold)
    report = {
        **metrics(y[test], scores, threshold),
        "algorithm": "frozen_mobilenet_logistic",
        "training_groups": artifact["training_groups"],
        "evaluation_groups": artifact["evaluation_groups"],
        "training_clips": artifact["training_clips"],
        "evaluation_clips": artifact["evaluation_clips"],
        "training_people": artifact["training_people"],
        "evaluation_people": artifact["evaluation_people"],
        "split_by": artifact["split_by"],
        "split_seed": artifact["split_seed"],
        "evaluation_scope": "New clips of known people" if split_by == "clip" else "Held-out people/session groups",
        "training_samples": len(train),
        "evaluation_samples": len(test),
        "local_candidate_spans": spans,
        "note": "Frame metrics and local duration candidates only; no Gemini/event-level validation.",
    }
    write_json(output, artifact)
    write_json(output.with_suffix(".report.json"), report)
    deployment = settings.model_copy(
        update={
            "analysis_backend": "image",
            "classifier_path": output,
            "local_threshold": threshold,
        }
    )
    write_json(output.with_suffix(".config.json"), deployment.model_dump(mode="json"))
    save_predictions(output.with_suffix(".predictions.csv"), data, test, scores, eligible, threshold)
    print_metrics(report)
    print(f"Training: {artifact['training_groups']}; held out: {artifact['evaluation_groups']}")
    print(f"Evaluation scope: {report['evaluation_scope']}")
    print(f"Classifier: {output}\nPi config: {output.with_suffix('.config.json')}")
    return report


def evaluate_images(root, model_path, output, threshold=None, groups_filter=None):
    settings, extraction, data = load_images(root)
    artifact = json.loads(model_path.read_text())
    if artifact.get("kind") != "image_logistic" or artifact.get("schema_version") != 1:
        raise ValueError("Expected an image classifier from this workflow")
    if artifact.get("image_context") != extraction["image_context"]:
        raise ValueError("Evaluation image encoder/crop does not match training")
    if not artifact.get("training_groups"):
        raise ValueError("Missing training-group metadata")
    if artifact.get("split_by") == "clip":
        data = {**data, "group": data["clip_id"]}
    if groups_filter and set(groups_filter) - set(data["group"]):
        raise ValueError("Unknown evaluation groups")
    selected = np.isin(data["group"], groups_filter) if groups_filter else np.ones(len(data["label"]), bool)
    indices = np.flatnonzero(selected)
    groups = set(data["group"][indices])
    if not len(indices) or groups & set(artifact["training_groups"]):
        raise ValueError("Select held-out evaluation groups; evaluation cannot include training groups")
    threshold = artifact["evaluated_threshold"] if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    scores = score_images(artifact, data["embeddings"][indices])
    eligible, spans = duration_candidates(data, indices, scores, settings, threshold)
    report = {
        **metrics(data["label"][indices], scores, threshold),
        "groups": sorted(groups),
        "samples": len(indices),
        "local_candidate_spans": spans,
    }
    write_json(output, report)
    save_predictions(output.with_suffix(".predictions.csv"), data, indices, scores, eligible, threshold)
    print_metrics(report)
    print(f"Saved {output}. No fitting occurred.")
    return report
