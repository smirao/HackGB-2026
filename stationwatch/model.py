from dataclasses import dataclass

import numpy as np

from .config import Settings
from .features import FEATURE_NAMES, Features, extraction_context


@dataclass(frozen=True)
class Prediction:
    flagged: bool
    score: float
    source: str


class BaselineModel:
    """Explicit heuristic starting point; score is a rule indicator, not a probability."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def predict(self, features: Features) -> Prediction:
        flagged = bool(
            features.in_workstation
            and features.head_angle_deg >= self.settings.baseline_angle_deg
            and features.dwell_s >= self.settings.baseline_dwell_s
        )
        return Prediction(flagged, float(flagged), "orientation_baseline")


class TrainedModel:
    def __init__(self, settings: Settings):
        import joblib

        # Load only artifacts produced by your own training command; joblib is executable serialization.
        artifact = joblib.load(settings.classifier_path)
        if artifact.get("schema_version") != 1 or artifact.get("features") != FEATURE_NAMES:
            raise ValueError("Classifier feature schema does not match this service")
        if artifact.get("feature_context") is not None and artifact["feature_context"] != extraction_context(
            settings
        ):
            raise ValueError(
                "Pose model/calibration differs from training. Use the exported deployment config or retrain."
            )
        self.classifier = artifact["classifier"]
        self.threshold = settings.local_threshold
        classes = list(self.classifier.classes_)
        if set(classes) != {0, 1}:
            raise ValueError("Classifier must contain classes 0 and 1")
        self.positive_index = classes.index(1)

    def predict(self, features: Features) -> Prediction:
        score = float(self.classifier.predict_proba(np.array([features.values()]))[0][self.positive_index])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Classifier returned an invalid score")
        return Prediction(
            bool(features.in_workstation and score >= self.threshold), score, "trained_classifier"
        )
