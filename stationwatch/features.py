import math
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from .config import Settings


@dataclass(frozen=True)
class Features:
    body_angle_deg: float
    head_angle_deg: float
    person_x: float
    person_y: float
    in_workstation: float
    machine_x: float
    machine_y: float
    dwell_s: float = 0.0

    def values(self):
        return list(asdict(self).values())


FEATURE_NAMES = list(Features.__dataclass_fields__)


def angle_deg(vector, target):
    norm = np.linalg.norm(vector) * np.linalg.norm(target)
    if norm < 1e-5:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(vector, target) / norm, -1, 1))))


def pose_features(points, settings, image_shape, min_confidence):
    """Seven [x, y, confidence] points: nose, ears, shoulders, hips."""
    points = np.asarray(points, dtype=float)
    if points.shape != (7, 3) or not np.isfinite(points).all() or np.any(points[:, 2] < min_confidence):
        return None
    nose, left_ear, right_ear, left_shoulder, right_shoulder, left_hip, right_hip = points[:, :2]
    hip = (left_hip + right_hip) / 2
    shoulder = (left_shoulder + right_shoulder) / 2
    ears = (left_ear + right_ear) / 2
    machine = np.array(settings.machine_center)
    scale = np.array([image_shape[1], image_shape[0]])
    body = angle_deg((shoulder - hip) * scale, (machine - hip) * scale)
    head = angle_deg((nose - ears) * scale, (machine - ears) * scale)
    if body is None or head is None or np.linalg.norm((nose - ears) * scale) < 3:
        return None
    if not all(0 <= coordinate <= 1 for coordinate in hip):
        return None
    inside = (
        cv2.pointPolygonTest(
            np.array(settings.workstation_polygon, np.float32), tuple(float(v) for v in hip), False
        )
        >= 0
    )
    return Features(body, head, float(hip[0]), float(hip[1]), float(inside), *machine.tolist())


class PoseExtractor:
    def __init__(self, settings: Settings):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.settings = settings
        self.detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(settings.pose_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=2,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
            )
        )

    def extract(self, image, elapsed_s) -> Features | None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        # One station/one visible worker. Ambiguous frames must never imply an attention result.
        if len(result.pose_landmarks) != 1:
            return None
        pose = result.pose_landmarks[0]
        points = [
            [pose[i].x, pose[i].y, min(pose[i].visibility, pose[i].presence)]
            for i in [0, 7, 8, 11, 12, 23, 24]
        ]
        return pose_features(points, self.settings, image.shape, 0.6)

    def close(self):
        self.detector.close()


class MoveNetExtractor:
    """SinglePose Lightning via ARM64-capable LiteRT; requires one worker in view."""

    def __init__(self, settings):
        from ai_edge_litert.interpreter import Interpreter

        self.settings = settings
        self.interpreter = Interpreter(model_path=str(settings.pose_model_path), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        shape = self.input["shape"]
        if (
            tuple(shape) != (1, 192, 192, 3)
            or self.input["dtype"] not in (np.uint8, np.int32)
            or tuple(self.output["shape"]) != (1, 1, 17, 3)
        ):
            raise ValueError("Expected MoveNet SinglePose Lightning 192x192 integer-input TFLite model")

    def extract(self, image, elapsed_s):
        height, width = image.shape[:2]
        scale = min(192 / width, 192 / height)
        resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (192 - resized_w) // 2, (192 - resized_h) // 2
        resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (resized_w, resized_h))
        padded = np.zeros((192, 192, 3), np.uint8)
        padded[top : top + resized_h, left : left + resized_w] = resized
        self.interpreter.set_tensor(self.input["index"], padded[None].astype(self.input["dtype"]))
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(self.output["index"])[0, 0]
        points = [
            [
                (keypoints[i, 1] * 192 - left) / resized_w,
                (keypoints[i, 0] * 192 - top) / resized_h,
                keypoints[i, 2],
            ]
            for i in [0, 3, 4, 5, 6, 11, 12]
        ]
        return pose_features(points, self.settings, image.shape, 0.3)

    def close(self):
        self.interpreter = None


def create_extractor(settings):
    return MoveNetExtractor(settings) if settings.pose_backend == "movenet" else PoseExtractor(settings)


class DwellTracker:
    """Duration of a similar visible pose, measured using capture time, never inference time."""

    def __init__(self, max_gap_s=1.5):
        self.max_gap_s = max_gap_s
        self.anchor = None
        self.started = 0.0
        self.last = 0.0

    def update(self, features: Features | None, captured_s: float) -> Features | None:
        if features is None:
            self.anchor = None
            return None
        a = self.anchor
        changed = a is None or captured_s - self.last > self.max_gap_s or captured_s <= self.last
        if a is not None:
            changed |= (
                abs(a.body_angle_deg - features.body_angle_deg) > 12
                or abs(a.head_angle_deg - features.head_angle_deg) > 12
                or math.hypot(a.person_x - features.person_x, a.person_y - features.person_y) > 0.08
                or a.in_workstation != features.in_workstation
            )
        if changed:
            self.anchor, self.started = features, captured_s
        self.last = captured_s
        return replace(features, dwell_s=max(0, captured_s - self.started))


class DemoExtractor:
    def extract(self, image, elapsed_s):
        away = int(elapsed_s) % 16 >= 8
        return Features(90, 130 if away else 15, 0.46, 0.5, 1, 0.78, 0.5)

    def close(self):
        pass
