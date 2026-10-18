"""Collect real pose features and train with a group-isolated evaluation split."""

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .camera import create_camera
from .features import FEATURE_NAMES, DwellTracker, create_extractor


def collect(settings, output: Path, label: int, group: str, duration_s: float):
    extractor = create_extractor(settings)
    try:
        camera = create_camera(settings)
    except Exception:
        extractor.close()
        raise
    dwell = DwellTracker(settings.dwell_gap_s)
    fields = FEATURE_NAMES + ["label", "group"]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = output.is_file() and output.stat().st_size > 0
        if existing:
            with output.open(newline="") as handle:
                if next(csv.reader(handle)) != fields:
                    raise ValueError("Existing CSV header does not match feature schema")
        with output.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not existing:
                writer.writeheader()
            started = time.monotonic()
            next_sample = started
            count = 0
            while time.monotonic() - started < duration_s:
                # Drain the stream continuously instead of reading a stale queued frame.
                image = camera.read()
                now = time.monotonic()
                if now < next_sample:
                    time.sleep(min(0.01, next_sample - now))
                    continue
                next_sample = now + settings.sample_interval_s
                features = dwell.update(extractor.extract(image, now - started), now)
                if features is not None:
                    writer.writerow(dict(asdict(features), label=label, group=group))
                    handle.flush()
                    count += 1
            print(f"Collected {count} visible single-worker samples in {output}")
    finally:
        camera.close()
        extractor.close()


def train(csv_path: Path, output: Path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FEATURE_NAMES + ["label", "group"]).issubset(reader.fieldnames or []):
            raise ValueError("CSV must include all feature columns, label, and group")
        rows = list(reader)
    if len(rows) < 20:
        raise ValueError(
            "Collect at least 20 real labeled samples; substantially more are needed for validation"
        )
    x = np.array([[float(row[key]) for key in FEATURE_NAMES] for row in rows])
    y = np.array([int(row["label"]) for row in rows])
    groups = np.array([row["group"].strip() for row in rows])
    if not np.isfinite(x).all() or set(y) != {0, 1} or "" in groups:
        raise ValueError("Use finite feature values, both labels 0 and 1, and nonempty group identifiers")
    if len(set(groups)) < 4:
        raise ValueError(
            "Collect at least four independent groups; group by worker/session to prevent frame leakage"
        )
    split = None
    for train_idx, test_idx in GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=42).split(
        x, y, groups
    ):
        if len(set(y[train_idx])) == len(set(y[test_idx])) == 2:
            split = train_idx, test_idx
            break
    if split is None:
        raise ValueError(
            "Cannot create a group-isolated split containing both labels in train and evaluation"
        )
    train_idx, test_idx = split
    classifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    classifier.fit(x[train_idx], y[train_idx])
    predictions = classifier.predict(x[test_idx])
    report = {
        "samples": len(rows),
        "training_samples": len(train_idx),
        "evaluation_samples": len(test_idx),
        "training_groups": sorted(set(groups[train_idx])),
        "evaluation_groups": sorted(set(groups[test_idx])),
        "confusion_matrix_labels_0_1": confusion_matrix(y[test_idx], predictions, labels=[0, 1]).tolist(),
        "report_at_threshold_0_5": classification_report(
            y[test_idx], predictions, output_dict=True, zero_division=0
        ),
        "note": "Evaluation split held out. Runtime threshold is configured separately; validate it before use.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 1, "features": FEATURE_NAMES, "classifier": classifier}, output)
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved classifier to {output}; held-out groups were not used for fitting")
