from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    station_id: str = "press-01"
    station_name: str = "Hydraulic press · Station 01"
    camera_backend: Literal["opencv", "picamera2", "custom"] = "opencv"
    camera_source: int | str = 0
    camera_factory: str | None = None
    width: int = Field(960, ge=160, le=3840)
    height: int = Field(540, ge=120, le=2160)
    fps: int = Field(15, ge=1, le=60)
    sample_interval_s: float = Field(0.5, ge=0.1, le=60)
    machine_center: tuple[float, float] = (0.78, 0.5)
    workstation_polygon: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.15, 0.15), (0.9, 0.15), (0.9, 0.95), (0.15, 0.95)]
    )
    pose_backend: Literal["mediapipe", "movenet"] = "mediapipe"
    pose_model_path: Path = Path("models/pose_landmarker_lite.task")
    analysis_backend: Literal["pose", "image"] = "pose"
    image_model_path: Path = Path("models/mobilenet_v3_small.tflite")
    image_crop: tuple[float, float, float, float] = (0, 0, 1, 1)
    task_description: str = Field(
        "One worker performing the assigned task at the work surface", min_length=1, max_length=2000
    )
    classifier_path: Path = Path("models/attention.joblib")
    local_threshold: float = Field(0.8, ge=0, le=1)
    gemini_model: str = Field("gemini-2.5-flash", pattern=r"^[A-Za-z0-9._-]+$")
    gemini_threshold: float = Field(0.8, ge=0, le=1)
    gemini_timeout_s: float = Field(20, gt=0, le=120)
    review_interval_s: float = Field(5, gt=0)
    event_cooldown_s: float = Field(30, ge=0)
    max_analysis_age_s: float = Field(1.5, gt=0)
    dwell_gap_s: float = Field(1.5, gt=0)
    flag_duration_s: float = Field(5, ge=0, le=300)
    baseline_angle_deg: float = Field(60, ge=0, le=180)
    baseline_dwell_s: float = Field(2, gt=0)
    segment_seconds: float = Field(30, ge=1, le=300)
    unflagged_retention_hours: float = Field(24, gt=0)
    mongo_database: str = Field("stationwatch", pattern=r"^[A-Za-z0-9_-]+$")
    mongo_poll_s: float = Field(5, ge=0.5, le=60)
    mongo_timeout_s: float = Field(15, ge=1, le=60)
    data_dir: Path = Path("data")

    @model_validator(mode="after")
    def validate_geometry(self):
        left, top, right, bottom = self.image_crop
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("image_crop must be normalized [left, top, right, bottom] with positive area")
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 recording requires even width and height")
        points = [self.machine_center, *self.workstation_polygon]
        if len(self.workstation_polygon) < 3 or any(not 0 <= v <= 1 for p in points for v in p):
            raise ValueError("Use at least three polygon points and normalized coordinates in [0, 1]")
        area = sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(
                self.workstation_polygon, self.workstation_polygon[1:] + self.workstation_polygon[:1]
            )
        )
        if abs(area) < 0.001:
            raise ValueError("Workstation polygon must enclose an area")
        if self.camera_backend == "custom" and not self.camera_factory:
            raise ValueError("custom camera requires camera_factory='your_module:create_camera'")
        return self


def load_settings(path: Path) -> Settings:
    return Settings.model_validate_json(path.read_text())
