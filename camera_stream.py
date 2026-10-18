#!/usr/bin/env python3
"""Share one USB camera with browser viewers and the StationWatch Pi agent.

Install: python -m pip install -e '.[camera]'
Run:     python camera_stream.py --camera 0 --host 0.0.0.0 --port 5000
Browser: http://<pi-tailscale-ip>:5000/
Agent:   python pi_agent.py --source http://127.0.0.1:5000/video_feed
"""

import argparse
import threading
import time

import cv2
from flask import Flask, Response, render_template_string

PAGE = """
<!doctype html>
<html lang="en">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StationWatch camera</title>
<style>
body { margin: 0; background: #111; display: flex; justify-content: center;
       align-items: center; min-height: 100vh; }
img { max-width: 100%; max-height: 100vh; }
</style>
<img src="{{ url_for('video_feed') }}" alt="Live Raspberry Pi camera feed">
</html>
"""


class Camera:
    """One capture thread provides the latest JPEG to every client."""

    def __init__(self, index=0, width=640, height=480, fps=15, quality=80):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open camera {index}; stop other processes using it")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.quality = quality
        self.frame = None
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._update, name="camera-stream", daemon=True)
        self.thread.start()

    def _update(self):
        last_success = time.monotonic()
        try:
            while not self.stop.is_set():
                ok, image = self.cap.read()
                if ok:
                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    if ok:
                        with self.condition:
                            self.frame = encoded.tobytes()
                            self.condition.notify_all()
                        last_success = time.monotonic()
                        continue
                if time.monotonic() - last_success >= 5:
                    break
                self.stop.wait(0.05)
        except cv2.error:
            pass
        finally:
            self.stop.set()
            with self.condition:
                self.condition.notify_all()

    def get_frame(self, previous=None):
        with self.condition:
            self.condition.wait_for(lambda: self.stop.is_set() or self.frame is not previous, timeout=1)
            return None if self.stop.is_set() else self.frame

    def release(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=1)
        self.cap.release()


def create_app(camera):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/video_feed")
    def video_feed():
        def frames():
            previous = None
            while not camera.stop.is_set():
                frame = camera.get_frame(previous)
                if frame is None or frame is previous:
                    continue
                previous = frame
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n")

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if min(args.width, args.height, args.fps) < 1:
        parser.error("--width, --height, and --fps must be positive")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    try:
        camera = Camera(args.camera, args.width, args.height, args.fps, args.quality)
    except (RuntimeError, cv2.error):
        print("Could not open the camera. Check its index, connection, permissions, and other camera processes.")
        return 1
    try:
        create_app(camera).run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    finally:
        camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
