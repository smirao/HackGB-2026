"""Durable background uploads of confirmed alerts and evidence."""

import logging
import threading
import time

log = logging.getLogger(__name__)


class EvidenceUploader:
    """Restart-safe SQLite outbox. Database outages never block capture/inference."""

    def __init__(self, local_store, remote_store, poll_s=5):
        self.local = local_store
        self.remote = remote_store
        self.poll_s = poll_s
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="evidence-uploads", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 — keep the local queue alive after storage/driver failures.
                log.error("Upload queue failed (%s)", type(exc).__name__)
            self.stop_event.wait(self.poll_s)

    def run_once(self, now=None):
        now = time.time() if now is None else now
        for item in self.local.pending_uploads(now):
            if self.stop_event.is_set():
                break
            event = item["event"]
            event_id = event["id"]
            if event.get("mode") == "demo":
                self.local.upload_progress(event_id, retry_at=now + 3600)
                continue
            try:
                if not item["alert_sent"]:
                    self.remote.publish_alert(event, self.local.root / "snapshots" / (event_id + ".jpg"))
                    self.local.upload_progress(event_id, alert_sent=True)
                if self.stop_event.is_set():
                    break
                clip = self.local.root / "recordings" / event["segment"]
                if clip.is_file():
                    self.remote.publish_clip(event, clip)
                    self.local.upload_progress(event_id, clip_sent=True)
                else:
                    self.local.upload_progress(event_id, retry_at=now + self.poll_s)
            except Exception as exc:  # noqa: BLE001 — confirmed evidence remains queued until a successful retry.
                backoff = min(300, self.poll_s * 2 ** min(item["attempts"], 8))
                self.local.upload_progress(event_id, retry_at=now + backoff, error=type(exc).__name__)
                log.warning("Alert evidence upload deferred (%s)", type(exc).__name__)

    def close(self, flush=False):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        if flush:
            self.stop_event.clear()
            try:
                self.run_once()
            finally:
                self.stop_event.set()
        self.remote.close()
