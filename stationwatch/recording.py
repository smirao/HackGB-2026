"""H.264 segments, finalized atomically so browsers never read an incomplete MP4."""

import shutil
import subprocess
import time
import uuid


class SegmentRecorder:
    def __init__(self, settings, store, duration_by_frames=False):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for browser-playable H.264 video recording")
        self.settings, self.store = settings, store
        self.duration_by_frames = duration_by_frames
        self.process = None
        self.segment = None
        self.frame_count = 0
        self.started_s = 0
        self.last_cleanup = 0

    def _start(self, now):
        self.segment = uuid.uuid4().hex + ".mp4"
        self.temporary = self.store.root / "recordings" / (self.segment + ".partial.mp4")
        s = self.settings
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{s.width}x{s.height}",
                "-framerate",
                str(s.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "25",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.temporary),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Ctrl-C belongs to the capture process; let it close the encoder pipe and finalize the MP4.
            start_new_session=True,
        )
        self.frame_count, self.started_s = 0, now

    def write(self, frame, now):
        duration_reached = (
            self.frame_count >= round(self.settings.segment_seconds * self.settings.fps)
            if self.duration_by_frames
            else now - self.started_s >= self.settings.segment_seconds
        )
        if self.process and duration_reached:
            self._finish()
        if not self.process:
            self._start(now)
        offset = self.frame_count / self.settings.fps
        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                "Video encoder failed; check ffmpeg/libx264 availability and free disk space"
            ) from exc
        self.frame_count += 1
        if now - self.last_cleanup >= 60:
            self.cleanup()
            self.last_cleanup = now
        return self.segment, offset

    def _finish(self):
        process, self.process = self.process, None
        pipe_failed = False
        try:
            process.stdin.close()
        except BrokenPipeError:
            pipe_failed = True
        try:
            result = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("Video encoder did not finalize in time")
        if result != 0 or pipe_failed:
            raise RuntimeError("Video encoder could not finalize the recording")
        self.temporary.replace(self.store.root / "recordings" / self.segment)

    def cleanup(self):
        retained = self.store.retained_segments()
        cutoff = time.time() - self.settings.unflagged_retention_hours * 3600
        # Protect recent in-flight reviews even if retention is configured very small.
        cutoff = min(cutoff, time.time() - self.settings.gemini_timeout_s - 60)
        for path in (self.store.root / "recordings").glob("*.mp4"):
            if ".partial." in path.name or path.name == self.segment or path.name in retained:
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink()

    def close(self):
        if self.process:
            self._finish()
