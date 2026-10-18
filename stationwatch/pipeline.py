import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np

from .features import DwellTracker
from .storage import make_event
from .temporal import SustainedFlag

log = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Packet:
    image: np.ndarray
    jpeg: bytes
    captured_at: str
    monotonic_s: float
    elapsed_s: float
    segment: str
    video_offset_s: float
    session_id: str


class Pipeline:
    def __init__(self, settings, camera, extractor, model, reviewer, recorder, store, mode):
        self.settings, self.camera, self.extractor, self.model = settings, camera, extractor, model
        self.reviewer, self.recorder, self.store, self.mode = reviewer, recorder, store, mode
        self.session_id = uuid.uuid4().hex
        self.dwell = DwellTracker(settings.dwell_gap_s)
        self.flag_gate = SustainedFlag(settings.flag_duration_s, settings.dwell_gap_s)
        # Single local worker preserves pose chronology. Each pool accepts at most one in-flight task.
        self.local_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-inference")
        self.review_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemini-review")
        self.local_slot, self.review_slot = threading.BoundedSemaphore(1), threading.BoundedSemaphore(1)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.last_attempt = float("-inf")
        self.last_confirmed = float("-inf")
        self.started = time.monotonic()
        self.thread = None
        self.closed = False
        self.state = {
            "running": False,
            "mode": mode,
            "station_id": settings.station_id,
            "station_name": settings.station_name,
            "session_id": self.session_id,
            "sample_interval_s": settings.sample_interval_s,
            "captured_frames": 0,
            "analyzed_frames": 0,
            "skipped_samples": 0,
            "review_requests": 0,
            "review_errors": 0,
            "confirmed_events": 0,
            "last_frame_at": None,
            "last_observation": "waiting",
            "last_error": None,
            "last_features": None,
            "sustained_flag_s": 0.0,
        }

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def increment(self, key):
        with self.lock:
            self.state[key] += 1

    def status(self):
        with self.lock:
            result = self.state.copy()
        result["uptime_s"] = round(time.monotonic() - self.started, 1)
        return result

    def start(self):
        self.thread = threading.Thread(target=self._capture, name="camera-capture", daemon=True)
        self.thread.start()

    def _capture(self):
        next_frame = next_sample = time.monotonic()
        self.update(running=True)
        try:
            while not self.stop_event.is_set():
                self.stop_event.wait(max(0, next_frame - time.monotonic()))
                if self.stop_event.is_set():
                    break
                image = self.camera.read()
                now, captured_at = time.monotonic(), utc_now()
                if (
                    not isinstance(image, np.ndarray)
                    or image.dtype != np.uint8
                    or image.ndim != 3
                    or image.shape[2] != 3
                ):
                    raise ValueError("Camera adapter must return a uint8 BGR HxWx3 array")
                image = cv2.resize(image, (self.settings.width, self.settings.height))
                segment, offset = self.recorder.write(image, now)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    raise RuntimeError("Could not encode camera frame")
                jpeg = encoded.tobytes()
                with self.lock:
                    self.latest_jpeg = jpeg
                self.increment("captured_frames")
                self.update(last_frame_at=captured_at)
                if now >= next_sample:
                    packet = Packet(
                        image.copy(),
                        jpeg,
                        captured_at,
                        now,
                        now - self.started,
                        segment,
                        offset,
                        self.session_id,
                    )
                    self.offer_sample(packet)
                    # Skip missed ticks instead of submitting a burst of stale screenshots.
                    next_sample += (
                        int((now - next_sample) / self.settings.sample_interval_s) + 1
                    ) * self.settings.sample_interval_s
                next_frame = max(next_frame + 1 / self.settings.fps, time.monotonic())
        except Exception as exc:  # noqa: BLE001 — camera adapters form an external failure boundary.
            log.error("Capture stopped (%s)", type(exc).__name__)
            self.update(
                last_error="Camera/recording stopped. Check camera, encoder and disk space.",
                last_observation="unavailable",
            )
        finally:
            self.update(running=False)
            try:
                try:
                    self.camera.close()
                finally:
                    self.recorder.close()
            except Exception:  # noqa: BLE001 — shutdown must report encoder failures without hiding capture status.
                log.error("Recording finalization failed")
                self.update(last_error="Recording finalization failed; video may be unavailable.")

    def offer_sample(self, packet):
        if not self.local_slot.acquire(blocking=False):
            self.increment("skipped_samples")
            return False
        self.local_pool.submit(self._analyze, packet)
        return True

    def _analyze(self, packet):
        from dataclasses import asdict

        try:
            if time.monotonic() - packet.monotonic_s > self.settings.max_analysis_age_s:
                self.dwell.update(None, packet.monotonic_s)
                self.flag_gate.reset()
                self.increment("skipped_samples")
                self.update(last_observation="unknown", last_features=None, sustained_flag_s=0)
                return
            extracted = self.extractor.extract(packet.image, packet.elapsed_s)
            features = extracted if self.mode == "image" else self.dwell.update(extracted, packet.monotonic_s)
            self.increment("analyzed_frames")
            if features is None:
                self.flag_gate.reset()
                self.update(last_observation="unknown", last_features=None, sustained_flag_s=0)
                return
            prediction = self.model.predict(features)
            now = time.monotonic()
            fresh = now - packet.monotonic_s <= self.settings.max_analysis_age_s
            sustained = self.flag_gate.update(prediction.flagged and fresh, packet.monotonic_s)
            if self.mode == "image":
                from .image_model import ImageObservation

                features = ImageObservation(
                    self.flag_gate.elapsed_s, self.settings.flag_duration_s, self.settings.image_crop
                )
            self.update(
                last_features=asdict(features),
                last_observation="local_flag" if prediction.flagged else "no_flag",
                sustained_flag_s=self.flag_gate.elapsed_s,
            )
            log.info(
                "Local model: %s | distraction_score=%.3f threshold=%.3f | "
                "sustained=%.1fs | capture=%s | fresh=%s",
                "DISTRACTED" if prediction.flagged else "ON TASK",
                prediction.score,
                self.settings.local_threshold,
                self.flag_gate.elapsed_s,
                packet.captured_at,
                fresh,
            )
            # Slow extraction must not send stale captures for a new review.
            if (
                not sustained
                or not fresh
                or now - self.last_attempt < self.settings.review_interval_s
                or now - self.last_confirmed < self.settings.event_cooldown_s
            ):
                return
            if not self.review_slot.acquire(blocking=False):
                return
            self.last_attempt = now
            self.increment("review_requests")
            self.review_pool.submit(self._review, packet, features, prediction)
        except Exception as exc:  # noqa: BLE001 — failed model adapters must not produce confirmed events.
            self.dwell.update(None, packet.monotonic_s)
            self.flag_gate.reset()
            log.error("Local inference failed (%s)", type(exc).__name__)
            self.update(
                last_observation="unknown",
                last_features=None,
                sustained_flag_s=0,
                last_error="Local inference failed; no event created.",
            )
        finally:
            self.local_slot.release()

    def _review(self, packet, features, prediction):
        try:
            review = self.reviewer.review(packet.jpeg, features)
            if review.decision == "inattentive" and review.confidence >= self.settings.gemini_threshold:
                event_id = uuid.uuid4().hex
                event = make_event(
                    event_id, packet, self.settings, features, prediction, review, self.mode, utc_now()
                )
                self.store.add(event, packet.jpeg)
                self.last_confirmed = time.monotonic()
                self.increment("confirmed_events")
        except Exception as exc:  # noqa: BLE001 — all review/storage failures fail closed at the worker boundary.
            log.error("Review failed (%s)", type(exc).__name__)
            self.increment("review_errors")
            self.update(
                last_error="Review unavailable or invalid; no event created. Check API key/quota/network."
            )
        finally:
            self.review_slot.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)
            if self.thread.is_alive():
                log.error("Camera adapter did not stop; its read() must have a finite timeout")
        else:
            self.camera.close()
            self.recorder.close()
        self.local_pool.shutdown(wait=True)
        self.review_pool.shutdown(wait=True)
        self.extractor.close()
        self.reviewer.close()
