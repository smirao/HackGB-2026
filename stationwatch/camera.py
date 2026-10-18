"""Camera adapters return uint8 BGR arrays. Only the capture thread calls read()."""

import importlib
import time
from typing import Protocol

import cv2
import numpy as np

from .config import Settings


class Camera(Protocol):
    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


class OpenCVCamera:
    def __init__(self, settings: Settings):
        self.capture = cv2.VideoCapture(settings.camera_source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        self.capture.set(cv2.CAP_PROP_FPS, settings.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open camera source {settings.camera_source!r}")

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Camera read failed or video ended")
        return frame

    def close(self):
        self.capture.release()


class PiCamera:
    def __init__(self, settings: Settings):
        from picamera2 import Picamera2

        self.camera = Picamera2()
        # libcamera's RGB888 format is BGR byte order, as expected by OpenCV.
        self.camera.configure(
            self.camera.create_video_configuration(
                main={"size": (settings.width, settings.height), "format": "RGB888"},
                controls={"FrameRate": settings.fps},
            )
        )
        self.camera.start()

    def read(self):
        return self.camera.capture_array("main")

    def close(self):
        self.camera.stop()
        self.camera.close()


class DemoCamera:
    """Synthetic frames; never represents observed worker behavior."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.start = time.monotonic()

    def read(self):
        s = self.settings
        img = np.full((s.height, s.width, 3), (31, 35, 41), dtype=np.uint8)
        for x in range(0, s.width, 48):
            cv2.line(img, (x, 0), (x, s.height), (45, 49, 55), 1)
        for y in range(0, s.height, 48):
            cv2.line(img, (0, y), (s.width, y), (45, 49, 55), 1)
        poly = np.array(s.workstation_polygon) * [s.width, s.height]
        cv2.polylines(img, [poly.astype(np.int32)], True, (100, 156, 124), 2)
        mx, my = (int(v * size) for v, size in zip(s.machine_center, [s.width, s.height]))
        cv2.rectangle(img, (mx - 55, my - 100), (mx + 55, my + 110), (70, 90, 104), -1)
        x, y = int(s.width * 0.46), int(s.height * 0.50)
        away = int(time.monotonic() - self.start) % 16 >= 8
        color = (80, 176, 242) if away else (168, 211, 133)
        cv2.circle(img, (x, y - 70), 25, color, 3)
        for a, b in [
            ((x, y - 45), (x, y + 55)),
            ((x - 45, y - 10), (x + 45, y - 10)),
            ((x, y + 55), (x - 35, y + 130)),
            ((x, y + 55), (x + 35, y + 130)),
        ]:
            cv2.line(img, a, b, color, 4)
        dx = -75 if away else 75
        cv2.arrowedLine(img, (x, y - 70), (x + dx, y - 70), color, 3)
        cv2.putText(
            img, "SYNTHETIC DEMO / NO CAMERA", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (215, 225, 234), 1
        )
        cv2.putText(
            img,
            "Scripted pose: away" if away else "Scripted pose: toward machine",
            (24, s.height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
        )
        return img

    def close(self):
        pass


def create_camera(settings: Settings, demo=False) -> Camera:
    if demo:
        return DemoCamera(settings)
    if settings.camera_backend == "custom":
        module, factory = settings.camera_factory.rsplit(":", 1)
        return getattr(importlib.import_module(module), factory)(settings)
    return PiCamera(settings) if settings.camera_backend == "picamera2" else OpenCVCamera(settings)
