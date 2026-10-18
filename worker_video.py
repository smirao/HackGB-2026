"""Classify a webcam, video file, or stream with Jev once per second."""

import argparse
import asyncio
import getpass
import math
import os
import sys
from pathlib import Path

import cv2
import httpx
from dotenv import dotenv_values, set_key
from fastapi import HTTPException

from stationwatch.images import prepare_image
from stationwatch.jev import classify_jpeg

ENV_FILE = Path(__file__).resolve().with_name(".env")


def get_api_key(env_path=None):
    """Load the saved key, or save an exported/prompted key for later runs."""
    env_path = Path(env_path) if env_path is not None else ENV_FILE
    saved_key = (dotenv_values(env_path, interpolate=False).get("TYPESAFE_API_KEY") or "").strip()
    key = os.environ.get("TYPESAFE_API_KEY", "").strip() or saved_key
    if not key:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "No TypeSafe API key saved. Run this script in a terminal to enter and save it, "
                "or add TYPESAFE_API_KEY to the project's .env file."
            )
        key = getpass.getpass("TypeSafe API key (will be saved to .env): ").strip()
    if not key or any(char in key for char in "\r\n\0"):
        raise RuntimeError("The TypeSafe API key must be a nonempty, single-line value.")
    if key != saved_key:
        # New files are private; set_key preserves other settings and quotes the value.
        env_path.touch(mode=0o600, exist_ok=True)
        set_key(env_path, "TYPESAFE_API_KEY", key, quote_mode="always")
        print(f"Saved TYPESAFE_API_KEY to {env_path}", file=sys.stderr)
    return key


async def report_frame(client, frame, elapsed, task, key, model):
    """Print a result tagged with capture time, even if requests finish out of order."""
    try:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Could not encode the captured frame.")
        jpeg = await asyncio.to_thread(prepare_image, encoded.tobytes())
        result = await classify_jpeg(client, jpeg, task, key, model)
    except (HTTPException, cv2.error, ValueError) as exc:
        # Do not print raw provider bodies, connection URLs, or credentials.
        detail = exc.detail if isinstance(exc, HTTPException) else "Could not prepare the captured frame."
        print(f"[{elapsed:8.2f}s] ERROR: {detail}", file=sys.stderr, flush=True)
        return
    print(
        f"[{elapsed:8.2f}s] {result['decision']} "
        f"confidence={result['confidence']:.3f} "
        f"model_choice={result['model_choice']}",
        flush=True,
    )


async def watch(source, task, key, model="jev-latest", interval=1.0, max_in_flight=8, transport=None):
    """Drain the feed continuously and submit current frames on a monotonic clock."""
    capture = cv2.VideoCapture(source)
    pending = set()
    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open the video source. Check the source and camera permissions.")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        is_file = isinstance(source, str) and Path(source).is_file()
        fps = capture.get(cv2.CAP_PROP_FPS) if is_file else None
        if is_file and (not math.isfinite(fps) or fps <= 0):
            raise RuntimeError("Video file has no usable frame rate; use a live camera or stream instead.")
        loop = asyncio.get_running_loop()
        start = None
        frame_index = 0
        next_sample = 0.0
        async with httpx.AsyncClient(timeout=60, transport=transport) as client:
            try:
                while True:
                    if is_file and start is not None:
                        # Play files at their recorded rate instead of decoding the whole file at once.
                        await asyncio.sleep(max(0, start + frame_index / fps - loop.time()))
                    ok, frame = await asyncio.to_thread(capture.read)
                    if not ok:
                        if is_file:
                            break
                        raise RuntimeError("Video feed stopped or a frame could not be read.")
                    now = loop.time()
                    if start is None:
                        start = now
                    elapsed = frame_index / fps if is_file else now - start
                    frame_index += 1
                    if elapsed + 1e-9 < next_sample:
                        continue
                    # Missed sampling times never trigger a burst of requests for stale frames.
                    next_sample = (math.floor((elapsed + 1e-9) / interval) + 1) * interval
                    if len(pending) >= max_in_flight:
                        print(
                            f"[{elapsed:8.2f}s] SKIPPED: {max_in_flight} Jev requests still in flight.",
                            file=sys.stderr, flush=True,
                        )
                        continue
                    job = asyncio.create_task(report_frame(client, frame, elapsed, task, key, model))
                    pending.add(job)
                    job.add_done_callback(pending.discard)
                # Let the final video-file samples finish before closing the HTTP client.
                if pending:
                    await asyncio.gather(*pending)
            finally:
                for job in pending:
                    job.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index (default: 0), video path, or stream URL")
    parser.add_argument("--task", required=True, help="Describe the worker's assigned task")
    parser.add_argument("--model", default=os.environ.get("TYPESAFE_MODEL", "jev-latest"))
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples (default: 1)")
    parser.add_argument("--max-in-flight", type=int, default=8, help="Concurrent request limit (default: 8)")
    args = parser.parse_args()
    if not 1 <= len(args.task.strip()) <= 2000:
        parser.error("--task must contain between 1 and 2000 characters.")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be a finite positive number.")
    if args.max_in_flight < 1:
        parser.error("--max-in-flight must be at least 1.")
    try:
        key = get_api_key()
    except (RuntimeError, OSError) as exc:
        parser.error(str(exc))
    except (EOFError, KeyboardInterrupt):
        print("API key setup cancelled.", file=sys.stderr)
        return 130
    source = int(args.source) if args.source.isdecimal() else args.source
    print(f"Sampling every {args.interval:g}s. Sending frames to Jev. Press Ctrl+C to stop.", flush=True)
    try:
        asyncio.run(watch(source, args.task.strip(), key, args.model, args.interval, args.max_in_flight))
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except cv2.error:
        # OpenCV errors can include the source URL and embedded credentials.
        print("ERROR: Could not open or read the video feed. Check the source and camera permissions.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
