# StationWatch

Raspberry Pi workstation observation prototype: live camera → a frame every **0.5 seconds** → local orientation model → Gemini vision review → timestamped manager evidence. It records an event **only when the local stage flags the frame and Gemini returns `inattentive` at the configured confidence threshold**. Uncertain responses and API failures do not become confirmed events.

The project includes camera adapters, an isolated synthetic demo, a usable orientation-rule baseline, a pretrained pose-model integration, data collection and classifier training, video recording, SQLite persistence, and a browser dashboard. No attention model or labeled training data is bundled.

## Try it without hardware or API credentials

Use Python **3.11 or 3.12** for the widest MediaPipe/Raspberry Pi compatibility. The synthetic demo does not require MediaPipe. Install `ffmpeg` with H.264 encoding support (`sudo apt install ffmpeg` on Raspberry Pi OS; `brew install ffmpeg` on macOS).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
stationwatch serve --demo
```

Open **http://127.0.0.1:8000**. A synthetic worker alternates between poses every eight seconds; the first synthetic event appears after roughly 10–12 seconds. Demo frames and reviews are generated locally; **no Gemini requests run**. Demo data lives separately in `data/demo/`. Recordings become available after their 30-second segment closes, or after graceful shutdown.

## Recommended starting model

Use **MoveNet SinglePose Lightning with LiteRT on a 64-bit Raspberry Pi**, or **MediaPipe Pose Landmarker Lite on a supported desktop**. Both are pretrained **pose detectors**, not validated industrial attention classifiers. Both adapters are implemented. The baseline turns landmarks into a visible-orientation flag when the head is angled away by at least 60° for two seconds inside the configured workstation. This lets you inspect results before training your own classifier.

For Raspberry Pi OS 64-bit and a CSI camera:

```sh
sudo apt install python3-picamera2 ffmpeg libgl1 libglib2.0-0
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[lite,train]'
mkdir -p models
curl --fail --location \
  'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite' \
  --output models/movenet_lightning.tflite
cp config.pi.example.json config.json
export GEMINI_API_KEY='your-key'
stationwatch --config config.json serve --baseline
```

For a USB camera, change `camera_backend` to `opencv` and `camera_source` to `0`. LiteRT publishes Linux ARM64 wheels; use a Python version supported by both LiteRT and your OS camera packages. The default Pi configuration uses MoveNet's 192×192 integer-input model. Download endpoints may redirect; the extractor validates the tensor shape/type rather than accepting arbitrary models.

For a desktop with MediaPipe support:

```sh
python -m pip install -e '.[pose,train]'
mkdir -p models
curl --fail --location \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task \
  --output models/pose_landmarker_lite.task
cp config.example.json config.json
export GEMINI_API_KEY='your-key'
stationwatch --config config.json serve --baseline
```

The currently tested MediaPipe release does not publish an official Linux ARM64 wheel; use the MoveNet/LiteRT path on the Pi. Measure throughput on the target Pi; a 0.5-second sample target does not guarantee 0.5-second inference latency. The implementation reports skipped samples instead of silently growing a backlog. Model files are ignored by Git; rerun the corresponding download command on each new machine.

On a minimal/headless Linux installation, OpenCV may need `sudo apt install libgl1 libglib2.0-0`. The project uses the same OpenCV contrib distribution as MediaPipe so two competing `cv2` packages are not installed.

MediaPipe documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

MoveNet documentation: https://www.tensorflow.org/hub/tutorials/movenet

Gemini model names and availability can change; set `gemini_model` in the config to a vision-capable model available to your API project. The default is `gemini-2.5-flash`. The implementation calls the REST API directly and sends only locally flagged JPEGs plus workstation geometry and measured features. Provide the API key through the environment; it is never sent to the dashboard.

## Camera setup and future script integration

For a USB/UVC camera, keep `camera_backend: "opencv"` and set `camera_source` to the camera index (usually `0`). An OpenCV-supported stream URL also works; set finite read timeouts in a custom adapter for network sources. Video files are read at the configured capture FPS; end-of-file stops monitoring and is shown on the dashboard.

For a Pi CSI camera, install Picamera2 from the OS package manager:

```sh
sudo apt install python3-picamera2 ffmpeg
# Create the environment with --system-site-packages so it can import the OS's Picamera2.
python3 -m venv --system-site-packages .venv
```

Set `camera_backend` to `picamera2`. Use a Python version compatible with your system Picamera2/libcamera packages.

A future camera script can plug in through `examples/custom_camera.py`: implement `read()` to return a fresh **BGR uint8 H×W×3 NumPy frame**, and `close()` to release resources. `read()` must have a finite timeout. Set:

```json
{
  "camera_backend": "custom",
  "camera_factory": "your_module:create_camera"
}
```

The factory receives the settings object. The pipeline owns capture timing, sampling, threading, streaming, and recording, so the camera script only supplies frames. See the example adapter for the exact contract. The example intentionally raises until a real script is connected.

## Calibrate a workstation

Edit `config.json` for a fixed camera looking at **one workstation and one visible worker**:

- `machine_center`: the task/machine point, normalized `[x, y]` in the camera image; top-left is `[0, 0]`.
- `workstation_polygon`: normalized polygon enclosing the operating area. The worker's hip midpoint determines whether they are inside.
- `width`, `height`, `fps`: camera recording dimensions and target rate. Use even width/height for H.264.
- `baseline_angle_deg`, `baseline_dwell_s`: starting heuristic thresholds. Verify using representative task footage, including legitimate turning, reaching, breaks, and occlusion.
- `local_threshold`, `gemini_threshold`: classifier score and Gemini reported confidence cutoffs. Gemini's reported confidence is not a calibrated probability.
- `event_cooldown_s`: minimum spacing after a confirmed event (default 30 seconds). `review_interval_s` limits API attempts even after a failed/negative review (default five seconds).

Features are **2D camera-plane proxies**, not true 3D body or gaze angles. Body angle compares the hip-to-shoulder torso vector with the hip-to-machine vector; head angle compares the ear-midpoint-to-nose vector with the ear-to-machine vector. Camera perspective matters. The baseline uses **head angle only**, plus workstation membership and dwell; the trainable classifier receives all features. A change larger than 12° or 0.08 image units resets similar-pose dwell; missing/ambiguous poses and gaps over 1.5 seconds also reset it. With no identity tracking, a worker swap without a missing frame can share dwell. MediaPipe rejects multiple detected people. **MoveNet SinglePose cannot detect/reject multiple people**, so its camera must contain only one worker. Both adapters reject low-confidence or degenerate required landmarks.

Machine position is manually calibrated and fixed; this prototype does not detect or track moving machines. Actual attention cannot be established from a still image. Events are observations for human review. This prototype is not a machine interlock or a validated protective control.

## Collect data and train the local classifier

Collect **real, independently labeled demonstrations** under controlled conditions. Label `0` for task-oriented behavior and `1` for visible orientation away from the task. Every sample in one collection run receives the supplied label, so keep the behavior consistent throughout that run. Never stage unsafe actions around operating equipment.

```sh
stationwatch --config config.json collect --label 0 --group workerA-session1 --seconds 30
stationwatch --config config.json collect --label 1 --group workerA-session1 --seconds 30
# Repeat both labels across several workers/sessions and camera conditions.
# Keep ALL footage from the same worker/session in the same group.
stationwatch --config config.json train --csv data/training.csv --output models/attention.joblib
stationwatch --config config.json serve
```

The CSV contains `body_angle_deg, head_angle_deg, person_x, person_y, in_workstation, machine_x, machine_y, dwell_s, label, group`. Collection skips frames without a clear single pose. Training requires both labels and at least four independent groups, fits a standardized logistic regression, and writes a held-out classification report next to the artifact. Twenty samples is only a format/startup minimum, not evidence of sufficient data. If you want to test generalization to new workers, use the **worker ID** as `group` across all their sessions. Do not split neighboring frames randomly between training and evaluation.

The saved model excludes held-out groups. Use the same pose backend and camera calibration for collection and deployment. The report uses a 0.5 score threshold; the runtime default is 0.8, which must be assessed separately on held-out footage. Evaluate missed events, false alerts, camera changes, task-specific exceptions, and per-worker coverage before relying on the results. Only load trusted, locally created `.joblib` files.

## Concurrency, evidence, and failure behavior

- Capture records continuously and exposes an MJPEG stream; it offers a screenshot every 0.5 seconds.
- One local inference worker processes screenshots chronologically to preserve dwell time. There is at most one local job in flight; busy samples are skipped and counted.
- A separate worker handles Gemini requests, with one request in flight and a finite HTTP timeout. Capture and local inference continue during review. Frames older than `max_analysis_age_s` are excluded from new API reviews.
- Both stages must flag before an event is saved. Snapshot time, session elapsed time, video segment name, exact encoded-frame offset, local features, scores, and Gemini's reason are persisted. Gemini latency never substitutes for capture time.
- H.264 MP4 files use atomic finalization. Active segments return HTTP 409 until closed; the dashboard provides a retry action. Playback can start two seconds before the event within the same segment. Cross-segment pre-roll is not implemented.
- Unflagged closed recordings expire after 24 hours by default. Segments referenced by events and their snapshots are retained until manually removed. Plan storage capacity; there is no automatic deletion of event evidence. Interrupted `.partial.mp4` files are not served or automatically recovered.
- Camera/recording failures stop capture and show an error. Missing poses are `unknown`. API errors and malformed/blocked responses are counted and never treated as agreement. Errors remain visible for investigation. No automatic restart is included.

Recording uses a constant encoded FPS; under capture lag, playback duration can differ from wall time. Event video offsets use the encoded frame index, while UTC timestamps and session elapsed seconds preserve capture time. For network cameras, timestamps represent local frame receipt, not the camera's hardware clock.

## Manager access

The service defaults to loopback. To access it from another computer:

```sh
export DASHBOARD_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# Store the token securely and enter it in the dashboard's login dialog.
stationwatch --config config.json serve --baseline --host 0.0.0.0
```

Binding to a non-loopback address requires an access token of at least 24 characters. Status, events, media, live stream, and review actions share that token; the dashboard stores it in an HttpOnly SameSite cookie, not in URLs or browser storage. API clients may instead send `Authorization: Bearer <token>`. Put remote access behind HTTPS and a trusted network/reverse proxy. This prototype has one shared manager credential; it has no user roles, per-person audit log, or internet-facing login rate limiting.

The service runs **one process/one camera**; do not enable Uvicorn multi-worker mode or reload. Use Ctrl-C for graceful shutdown and video finalization. Multiple stations would need separate instances and a central event-ingestion service, which are not implemented here.

## Checks

```sh
python -m pip install -e '.[dev,train]'
pytest -q
ruff check .
```

Tests use fake cameras/reviewers and generated features; they verify pipeline gates, bounded concurrency, dwell resets, evidence timestamps, API authorization, and group-separated training. They do not validate attention accuracy. Hardware camera/Pi performance and real Gemini calls require their respective hardware and API credentials. Both pretrained runtimes have also been smoke-tested locally using Google's public pose test image; low-confidence MoveNet observations correctly return unknown.

`uv.lock` records dependency versions. If you use uv, `uv sync --locked --extra lite --extra train --extra dev` installs the Pi inference/training dependencies, or replace `lite` with `pose` for MediaPipe. Picamera2 remains an OS package.
