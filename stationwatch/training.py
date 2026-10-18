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


def train(
    csv_path: Path,
    output: Path,
    algorithm="random-forest",
    threshold=0.8,
    evaluation_groups=None,
    feature_context=None,
    verbose=True,
):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if algorithm not in {"random-forest", "logistic"} or not 0 <= threshold <= 1:
        raise ValueError("Choose random-forest or logistic and a threshold in [0, 1]")
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
    if evaluation_groups:
        unknown = set(evaluation_groups) - set(groups)
        if unknown:
            raise ValueError(f"Evaluation groups are not in the dataset: {sorted(unknown)}")
        test_mask = np.isin(groups, evaluation_groups)
        train_idx, test_idx = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        if len(set(y[train_idx])) == len(set(y[test_idx])) == 2:
            split = train_idx, test_idx
    else:
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
    if algorithm == "random-forest":
        classifier = RandomForestClassifier(
            n_estimators=128,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    else:
        classifier = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
        )
    classifier.fit(x[train_idx], y[train_idx])
    predictions = classifier.predict(x[test_idx])
    scores = classifier.predict_proba(x[test_idx])[:, list(classifier.classes_).index(1)]
    # Match deployment: no local flag outside the workstation, even with a high model score.
    in_station = x[test_idx, FEATURE_NAMES.index("in_workstation")] > 0
    runtime_predictions = ((scores >= threshold) & in_station).astype(int)
    report = {
        "algorithm": algorithm,
        "runtime_threshold": threshold,
        "samples": len(rows),
        "training_samples": len(train_idx),
        "evaluation_samples": len(test_idx),
        "training_groups": sorted(set(groups[train_idx])),
        "evaluation_groups": sorted(set(groups[test_idx])),
        "confusion_matrix_labels_0_1": confusion_matrix(y[test_idx], predictions, labels=[0, 1]).tolist(),
        "report_at_threshold_0_5": classification_report(
            y[test_idx], predictions, output_dict=True, zero_division=0
        ),
        "report_at_runtime_threshold": classification_report(
            y[test_idx], runtime_predictions, output_dict=True, zero_division=0
        ),
        "runtime_confusion_matrix_labels_0_1": confusion_matrix(
            y[test_idx], runtime_predictions, labels=[0, 1]
        ).tolist(),
        "note": "Held-out groups were not fitted. Validate with independent workstation footage before deployment.",
    }
    if algorithm == "random-forest":
        report["feature_importances"] = dict(zip(FEATURE_NAMES, classifier.feature_importances_.tolist()))
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "features": FEATURE_NAMES,
            "classifier": classifier,
            "algorithm": algorithm,
            "evaluated_threshold": threshold,
            "training_groups": report["training_groups"],
            "evaluation_groups": report["evaluation_groups"],
            "feature_context": feature_context,
        },
        output,
    )
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    if verbose:
        print(json.dumps(report, indent=2))
        print(f"Saved classifier to {output}; held-out groups were not used for fitting")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the local workstation classifier from labeled pose features"
    )
    parser.add_argument("--csv", type=Path, default=Path("data/training.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/attention.joblib"))
    parser.add_argument("--algorithm", choices=["random-forest", "logistic"], default="random-forest")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    train(args.csv, args.output, args.algorithm, args.threshold)


if __name__ == "__main__":
    main()
