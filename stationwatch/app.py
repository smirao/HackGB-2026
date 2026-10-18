import asyncio
import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .storage import DatabaseUnavailable


def create_app(pipeline, store, dashboard_token=None, uploader=None):
    @asynccontextmanager
    async def lifespan(app):
        if pipeline is not None:
            pipeline.start()
        if uploader is not None:
            uploader.start()
        try:
            yield
        finally:
            if pipeline is not None:
                pipeline.close()
            if uploader is not None:
                uploader.close()
            store.close()

    app = FastAPI(title="StationWatch", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(DatabaseUnavailable)
    async def database_unavailable(request, exc):
        return JSONResponse(status_code=503, content={"detail": "Alert database unavailable. Please retry."})

    @app.middleware("http")
    async def privacy_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    def authorize(request: Request):
        if not dashboard_token:
            return
        bearer = request.headers.get("authorization", "")
        supplied = (
            bearer[7:] if bearer.startswith("Bearer ") else request.cookies.get("stationwatch_session", "")
        )
        if not hmac.compare_digest(supplied.encode(), dashboard_token.encode()):
            raise HTTPException(401, "Dashboard access token required")

    def same_origin(request: Request):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Cross-origin write rejected")

    class Login(BaseModel):
        token: str

    @app.post("/api/login", dependencies=[Depends(same_origin)])
    def login(body: Login, request: Request, response: Response):
        if dashboard_token and not hmac.compare_digest(body.token.encode(), dashboard_token.encode()):
            raise HTTPException(401, "Incorrect access token")
        if dashboard_token:
            response.set_cookie(
                "stationwatch_session",
                dashboard_token,
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
                max_age=8 * 3600,
            )
        return {"ok": True}

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/status", dependencies=[Depends(authorize)])
    def status():
        state = pipeline.status() if pipeline is not None else {"role": "hub", "mode": "hub"}
        return dict(state, events=store.counts())

    @app.get("/api/events", dependencies=[Depends(authorize)])
    def events():
        return store.list()

    @app.post("/api/events/{event_id}/reviewed", dependencies=[Depends(authorize), Depends(same_origin)])
    def reviewed(event_id: str):
        if not store.mark_reviewed(event_id):
            raise HTTPException(404, "Event not found")
        return {"ok": True}

    def find_event(event_id):
        event = store.get(event_id)
        if not event:
            raise HTTPException(404, "Event not found")
        return event

    @app.get("/api/events/{event_id}/snapshot", dependencies=[Depends(authorize)])
    def snapshot(event_id: str):
        find_event(event_id)
        if pipeline is None:
            try:
                path = store.media_path(event_id, "snapshot")
            except FileNotFoundError:
                raise HTTPException(404, "Snapshot unavailable") from None
        else:
            path = store.root / "snapshots" / (event_id + ".jpg")
        if not path.is_file():
            raise HTTPException(404, "Snapshot unavailable")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/events/{event_id}/video", dependencies=[Depends(authorize)])
    def video(event_id: str):
        event = find_event(event_id)
        if pipeline is None:
            try:
                path = store.media_path(event_id, "clip")
            except FileNotFoundError:
                raise HTTPException(409, "Video is uploading or unavailable; retry shortly") from None
        else:
            path = store.root / "recordings" / event["segment"]
        if not path.is_file():
            raise HTTPException(409, "Recording is still finalizing or unavailable; retry after segment end")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/stream", dependencies=[Depends(authorize)])
    async def stream(request: Request):
        if pipeline is None:
            raise HTTPException(404, "The alert hub does not receive a continuous camera stream")

        async def frames():
            previous = None
            while not await request.is_disconnected():
                with pipeline.lock:
                    jpeg = pipeline.latest_jpeg
                if jpeg is not None and jpeg is not previous:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    previous = jpeg
                if not pipeline.status()["running"]:
                    break
                await asyncio.sleep(0.1)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
