"""Live camera → Jev every second → five-second gate → Gemini → PostgreSQL."""

import argparse
import asyncio
import fcntl
import logging
import math
import os
import re
import signal
import socket
import threading
import time
import uuid
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import ValidationError

from .config import Settings
from .gemini import GeminiReviewer
from .images import prepare_image
from .jev import classify_jpeg
from .postgres import PostgresEventStore
from .recording import SegmentRecorder
from .storage import EventStore, make_event
from .temporal import SustainedFlag
from .uploads import EvidenceUploader

log = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Sample:
    jpeg: bytes
    captured_at: str
    monotonic_s: float
    elapsed_s: float
    segment: str
    video_offset_s: float
    session_id: str
    sequence: int = 0


@dataclass(frozen=True)
class JevObservation:
    sustained_flag_s: float
    required_flag_s: float = 5.0
    kind: str = "image_classification"
    source: str = "jev"


@dataclass(frozen=True)
class JevPrediction:
    flagged: bool
    score: float
    source: str = "jev"


class CameraFeed:
    """Own the camera and recorder on one thread; API calls never block capture."""

    def __init__(self, settings, recorder):
        self.settings, self.recorder = settings, recorder
        self.session_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = False
        self.thread = None
        source = settings.camera_source
        if isinstance(source, str) and source.startswith(("http://", "https://", "rtsp://")):
            # The supplied Flask /video_feed endpoint is an FFmpeg-readable MJPEG stream.
            self.capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ])
        else:
            self.capture = cv2.VideoCapture(source)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
            self.capture.set(cv2.CAP_PROP_FPS, settings.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Could not open camera. Check its index or the /video_feed URL.")

    def start(self):
        self.thread = threading.Thread(target=self._capture, name="pi-camera", daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.latest

    def _capture(self):
        deadline = time.monotonic()
        try:
            while not self.stop.is_set():
                if self.stop.wait(max(0, deadline - time.monotonic())):
                    break
                ok, image = self.capture.read()
                now, captured_at = time.monotonic(), utc_now()
                if not ok or image is None:
                    raise RuntimeError("Camera frame unavailable")
                image = cv2.resize(image, (self.settings.width, self.settings.height))
                segment, offset = self.recorder.write(image, now)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError("Camera frame encoding failed")
                sample = Sample(encoded.tobytes(), captured_at, now, now - self.started,
                                segment, offset, self.session_id)
                with self.lock:
                    self.latest = sample
                deadline = max(deadline + 1 / self.settings.fps, time.monotonic())
        except Exception as exc:  # noqa: BLE001 — sanitize failures from camera/encoder adapters.
            self.error = True
            log.error("Camera/recording stopped (%s)", type(exc).__name__)
        finally:
            self.capture.release()
            try:
                self.recorder.close()
            except Exception as exc:  # noqa: BLE001 — report finalization failure without exposing paths.
                self.error = True
                log.error("Recording finalization failed (%s)", type(exc).__name__)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=25)
            if self.thread.is_alive():
                raise RuntimeError("Camera did not stop within its timeout")
        else:
            self.capture.release()
            self.recorder.close()


class PiAgent:
    def __init__(self, settings, local_store, reviewer, jev_key, jev_model="jev-latest", max_in_flight=8):
        self.settings, self.store, self.reviewer = settings, local_store, reviewer
        self.jev_key, self.jev_model, self.max_in_flight = jev_key, jev_model, max_in_flight
        self.gate = SustainedFlag(5, 1.5, inclusive=True)
        self.previous_sequence = None
        self.last_attempt = self.last_confirmed = float("-inf")
        self.review_task = None

    async def classify(self, client, sample):
        try:
            jpeg = await asyncio.to_thread(prepare_image, sample.jpeg)
            return await asyncio.wait_for(
                classify_jpeg(client, jpeg, self.settings.task_description, self.jev_key, self.jev_model),
                timeout=15,
            )
        except (HTTPException, asyncio.TimeoutError, cv2.error, ValueError) as exc:
            log.warning("Jev sample unavailable (%s); off-task streak will reset", type(exc).__name__)
            return None

    def accept(self, sample, result):
        """Only the ordered consumer calls this; timestamps belong to capture, not API completion."""
        if self.previous_sequence is not None and sample.sequence != self.previous_sequence + 1:
            self.gate.reset()
        self.previous_sequence = sample.sequence
        fresh = time.monotonic() - sample.monotonic_s <= 20
        flagged = bool(result and result["decision"] == "off_task"
                       and result["confidence"] >= self.settings.local_threshold and fresh)
        sustained = self.gate.update(flagged, sample.monotonic_s)
        log.info("Jev: %s | off-task streak=%.1fs | capture=%s",
                 result["decision"] if result else "unavailable", self.gate.elapsed_s, sample.captured_at)
        now = time.monotonic()
        if (not sustained or now - self.last_attempt < self.settings.review_interval_s
                or now - self.last_confirmed < self.settings.event_cooldown_s
                or (self.review_task is not None and not self.review_task.done())):
            return
        self.last_attempt = now
        features = JevObservation(self.gate.elapsed_s)
        self.review_task = asyncio.create_task(self.confirm(sample, features, result))

    async def confirm(self, sample, features, result):
        try:
            review = await asyncio.to_thread(self.reviewer.review, sample.jpeg, features)
            if review.decision != "inattentive" or review.confidence < self.settings.gemini_threshold:
                log.info("Gemini did not confirm off-task behavior; no alert created")
                return
            event_id = uuid.uuid4().hex
            event = make_event(event_id, sample, self.settings, features,
                               JevPrediction(True, result["confidence"]), review, "image", utc_now())
            event.update(jev=result, task_description=self.settings.task_description)
            # SQLite + snapshot are durable before the independent database uploader sees them.
            await asyncio.to_thread(self.store.add, event, sample.jpeg)
            self.last_confirmed = time.monotonic()
            log.info("Confirmed alert %s queued for PostgreSQL", event_id)
        except Exception as exc:  # noqa: BLE001 — isolate provider/storage failures from capture.
            log.error("Gemini review or local alert save failed (%s)", type(exc).__name__)

    async def run(self, feed, stop, transport=None):
        pending = deque()
        stop_wait = asyncio.create_task(stop.wait())
        loop = asyncio.get_running_loop()
        origin = next_sample = loop.time()
        last_capture = None
        async with httpx.AsyncClient(timeout=15, transport=transport) as client:
            try:
                while not stop.is_set():
                    if feed.error:
                        raise RuntimeError("Camera/recording stopped; restart after checking the feed and disk")
                    while pending and pending[0][1].done():
                        sample, task = pending.popleft()
                        self.accept(sample, task.result())
                    now = loop.time()
                    if now >= next_sample:
                        sequence = math.floor(now - origin)
                        next_sample = origin + sequence + 1
                        sample = feed.snapshot()
                        if sample is not None and sample.monotonic_s != last_capture:
                            last_capture = sample.monotonic_s
                            if now - sample.monotonic_s <= 1.5 and len(pending) < self.max_in_flight:
                                sample = replace(sample, sequence=sequence)
                                pending.append((sample, asyncio.create_task(self.classify(client, sample))))
                            else:
                                log.warning("Jev sample skipped: stale frame or request limit reached")
                        # A skipped tick leaves a sequence gap, resetting the next accepted streak.
                    waiting = [stop_wait] + ([pending[0][1]] if pending else [])
                    await asyncio.wait(waiting, timeout=max(0, next_sample - loop.time()),
                                       return_when=asyncio.FIRST_COMPLETED)
            finally:
                for _, task in pending:
                    task.cancel()
                await asyncio.gather(*(task for _, task in pending), return_exceptions=True)
                stop_wait.cancel()
                await asyncio.gather(stop_wait, return_exceptions=True)
                if self.review_task is not None:
                    await self.review_task


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="USB camera index or http://127.0.0.1:5000/video_feed")
    parser.add_argument("--task", default=os.environ.get("TASK_DESCRIPTION"), help="Worker's assigned task")
    parser.add_argument("--station-id", default=os.environ.get("STATION_ID", socket.gethostname()))
    parser.add_argument("--station-name", default=os.environ.get("STATION_NAME"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--model", default=os.environ.get("TYPESAFE_MODEL", "jev-latest"))
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--jev-threshold", type=float, default=0.6)
    parser.add_argument("--gemini-threshold", type=float, default=0.8)
    parser.add_argument("--cooldown", type=float, default=30, help="Minimum seconds between confirmed alerts")
    parser.add_argument("--max-in-flight", type=int, default=8)
    args = parser.parse_args()
    if not args.task or not args.task.strip():
        parser.error("Set --task or TASK_DESCRIPTION to describe the worker's assigned task")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.station_id):
        parser.error("--station-id must use letters, numbers, underscores, or hyphens")
    if not 1 <= args.max_in_flight <= 32:
        parser.error("--max-in-flight must be between 1 and 32")
    if not 0.6 <= args.jev_threshold <= 1:
        parser.error("--jev-threshold must be between 0.6 and 1")
    if not math.isfinite(args.cooldown):
        parser.error("--cooldown must be finite")
    for key in ("TYPESAFE_API_KEY", "GEMINI_API_KEY"):
        if not os.environ.get(key, "").strip():
            parser.error(f"Set {key} in .env or the environment")
    if not os.environ.get("TIMESCALE_SERVICE_URL"):
        missing = [key for key in ("PGUSER", "PGPASSWORD", "PGDATABASE", "PGHOST", "PGPORT", "PGSSLMODE")
                   if not os.environ.get(key)]
        if missing:
            parser.error("Missing database settings: " + ", ".join(missing))
    source = int(args.source) if args.source.isdecimal() else args.source
    if isinstance(source, str) and not source.startswith(("http://", "https://", "rtsp://", "/dev/video")):
        parser.error("--source must be a live USB camera index, /dev/video device, or stream URL")
    try:
        settings = Settings(
            station_id=args.station_id, station_name=args.station_name or args.station_id,
            task_description=args.task.strip(), camera_source=source,
            width=args.width, height=args.height, fps=args.fps,
            data_dir=args.data_dir or Path("data") / ("pi-" + args.station_id),
            sample_interval_s=1, flag_duration_s=5, analysis_backend="image",
            gemini_model=args.gemini_model, local_threshold=args.jev_threshold,
            gemini_threshold=args.gemini_threshold, event_cooldown_s=args.cooldown,
        )
    except ValidationError:
        parser.error("Invalid configuration: check dimensions, model name, thresholds, and cooldown")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Suppress provider HTTP request URLs in normal console output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        with ExitStack() as stack:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((settings.data_dir / "agent.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another Pi agent is already using this data directory") from None
            local = EventStore(settings.data_dir)
            stack.callback(local.close)
            reviewer = GeminiReviewer(settings, os.environ["GEMINI_API_KEY"])
            stack.callback(reviewer.close)
            remote = PostgresEventStore.from_env()
            uploader = EvidenceUploader(local, remote, poll_s=5)
            stack.callback(uploader.close)
            recorder = SegmentRecorder(settings, local)
            stack.callback(recorder.close)
            feed = CameraFeed(settings, recorder)
            stack.callback(feed.close)
            agent = PiAgent(settings, local, reviewer, os.environ["TYPESAFE_API_KEY"],
                            args.model, args.max_in_flight)

            async def monitor():
                stop = asyncio.Event()
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, stop.set)
                uploader.start()
                feed.start()
                try:
                    await agent.run(feed, stop)
                finally:
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        loop.remove_signal_handler(sig)

            log.info("Monitoring station %s: Jev every second; Gemini after at least 5s off-task",
                     settings.station_id)
            asyncio.run(monitor())
            feed.close()  # Finalize the active video before the last upload attempt.
            uploader.close(flush=True)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 — keep credentials and stream URLs out of error output.
        log.error("Pi agent stopped (%s). Check camera, ffmpeg, keys, and local disk.", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
