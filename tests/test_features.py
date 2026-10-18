from unittest.mock import Mock

import numpy as np
import pytest

from stationwatch.config import Settings
from stationwatch.features import MoveNetExtractor, pose_features


def test_angles_use_pixel_aspect_ratio_and_machine_direction():
    # Nose points directly right toward the machine; torso points vertically up.
    points = [
        [0.55, 0.4, 1],
        [0.48, 0.4, 1],
        [0.52, 0.4, 1],
        [0.4, 0.5, 1],
        [0.6, 0.5, 1],
        [0.45, 0.8, 1],
        [0.55, 0.8, 1],
    ]
    features = pose_features(points, Settings(machine_center=(0.8, 0.4)), (500, 1000, 3), 0.6)
    assert features.head_angle_deg == pytest.approx(0)
    assert features.body_angle_deg == pytest.approx(56.3099, abs=0.001)
    assert features.person_x == 0.5 and features.person_y == 0.8 and features.in_workstation == 1
    points[0][2] = 0.2
    assert pose_features(points, Settings(), (500, 1000, 3), 0.6) is None


def test_degenerate_head_vector_cannot_be_an_attention_result():
    points = [
        [0.5, 0.4, 1],
        [0.48, 0.4, 1],
        [0.52, 0.4, 1],
        [0.4, 0.5, 1],
        [0.6, 0.5, 1],
        [0.45, 0.8, 1],
        [0.55, 0.8, 1],
    ]
    assert pose_features(points, Settings(), (500, 1000, 3), 0.6) is None


def test_movenet_letterbox_is_removed_before_using_machine_coordinates():
    extractor = MoveNetExtractor.__new__(MoveNetExtractor)
    extractor.settings = Settings(machine_center=(0.8, 0.4))
    extractor.input = {"index": 0, "dtype": np.uint8}
    extractor.output = {"index": 1}
    extractor.interpreter = Mock()
    output = np.zeros((1, 1, 17, 3), np.float32)
    points = [
        [0.55, 0.4, 1],
        [0.48, 0.4, 1],
        [0.52, 0.4, 1],
        [0.4, 0.5, 1],
        [0.6, 0.5, 1],
        [0.45, 0.8, 1],
        [0.55, 0.8, 1],
    ]
    # A 2:1 frame is resized to 192x96, padded 48 pixels at top/bottom.
    for index, (x, y, confidence) in zip([0, 3, 4, 5, 6, 11, 12], points):
        output[0, 0, index] = [(y * 96 + 48) / 192, x, confidence]
    extractor.interpreter.get_tensor.return_value = output
    features = extractor.extract(np.zeros((500, 1000, 3), np.uint8), 0)
    assert features.person_x == pytest.approx(0.5)
    assert features.person_y == pytest.approx(0.8)
    assert features.head_angle_deg == pytest.approx(0, abs=0.001)
