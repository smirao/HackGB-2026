"""Loopback-only annotation UI. All labels and calibration choices come from people."""

import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from .config import Settings
from .dataset import (
    clip_metadata,
    clips,
    dataset_settings,
    frame_times,
    labels_for,
    safe_path,
    validate_intervals,
    write_json,
)


def create_annotation_app(root):
    app = FastAPI(title="StationWatch training labels", docs_url=None, redoc_url=None, openapi_url=None)

    def same_origin(request: Request):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Cross-origin writes rejected")

    def find(clip_id):
        try:
            return clip_metadata(root, clip_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Clip not found") from None

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "annotate.html")

    @app.get("/api/clips")
    def list_clips():
        return [{**info, "intervals": len(labels_for(root, info))} for info in clips(root)]

    @app.get("/api/clips/{clip_id}/video")
    def video(clip_id: str):
        return FileResponse(safe_path(root, find(clip_id)["video"]), media_type="video/mp4")

    @app.get("/api/clips/{clip_id}/timeline")
    def timeline(clip_id: str):
        return frame_times(root, find(clip_id))

    @app.get("/api/clips/{clip_id}/labels")
    def get_labels(clip_id: str):
        return {"intervals": labels_for(root, find(clip_id))}

    class Annotations(BaseModel):
        model_config = ConfigDict(extra="forbid")
        intervals: list[dict]

    @app.post("/api/clips/{clip_id}/labels", dependencies=[Depends(same_origin)])
    def save_labels(clip_id: str, body: Annotations):
        info = find(clip_id)
        try:
            intervals = validate_intervals(body.intervals, info["duration_s"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        write_json(root / "labels" / (clip_id + ".json"), {"intervals": intervals})
        return {"intervals": intervals}

    @app.get("/api/calibration")
    def calibration():
        settings = dataset_settings(root)
        return {
            "machine_center": settings.machine_center,
            "workstation_polygon": settings.workstation_polygon,
        }

    @app.get("/api/setup")
    def setup():
        settings = dataset_settings(root)
        return {
            "analysis_backend": settings.analysis_backend,
            "image_crop": settings.image_crop,
            "task_description": settings.task_description,
        }

    class Geometry(BaseModel):
        model_config = ConfigDict(extra="forbid")
        machine_center: tuple[float, float]
        workstation_polygon: list[tuple[float, float]]

    @app.post("/api/calibration", dependencies=[Depends(same_origin)])
    def save_calibration(body: Geometry):
        values = dataset_settings(root).model_dump(mode="json")
        values.update(body.model_dump(mode="json"))
        try:
            settings = Settings.model_validate(values)
        except ValidationError:
            raise HTTPException(
                400, "Use normalized points and a workstation polygon enclosing an area"
            ) from None
        write_json(root / "config.json", settings.model_dump(mode="json"))
        manifest = json.loads((root / "dataset.json").read_text())
        manifest["calibration_confirmed"] = True
        write_json(root / "dataset.json", manifest)
        return {"ok": True}

    return app
