"""Run the local Jev photo classifier: python worker_webpage.py."""

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from stationwatch.images import prepare_image
from stationwatch.jev import classify_jpeg

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

def create_app(api_key=None, model="jev-latest", transport=None):
    key = (api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")).strip()
    busy = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=60, transport=transport) as client:
            app.state.client = client
            yield

    app = FastAPI(title="StationWatch · Jev photo review", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def privacy_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "stationwatch/static/worker_upload.html")

    @app.get("/api/setup")
    def setup():
        return {"configured": bool(key), "model": model, "max_upload_bytes": MAX_UPLOAD_BYTES}

    @app.post("/api/classify")
    async def classify(
        request: Request,
        task: str = Query(min_length=1, max_length=2000),
    ):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Cross-origin uploads are not allowed.")
        if not key:
            raise HTTPException(503, "Set TYPESAFE_API_KEY in your terminal, then restart this webpage.")
        if not task.strip():
            raise HTTPException(422, "Describe the worker's assigned task first.")
        if request.headers.get("content-type", "").split(";")[0] not in {
            "image/jpeg", "image/png", "image/webp",
        }:
            raise HTTPException(415, "Choose a JPEG, PNG, or WebP image.")
        if busy.locked():
            raise HTTPException(429, "Another photo is being reviewed. Try again shortly.")
        async with busy:
            data = bytearray()
            async for chunk in request.stream():
                if len(data) + len(chunk) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Choose an image smaller than 10 MB.")
                data.extend(chunk)
            jpeg = await run_in_threadpool(prepare_image, bytes(data))
            return await classify_jpeg(app.state.client, jpeg, task, key, model)

    return app


def main():
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default=os.environ.get("TYPESAFE_MODEL", "jev-latest"))
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port} in your browser.")
    uvicorn.run(create_app(model=args.model), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
