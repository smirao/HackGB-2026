# StationWatch

### Raspberry Pi: camera → Jev → PostgreSQL dashboard

The agent calls Jev every second and sends an alert, snapshot, and 5-second video
to PostgreSQL after **more than five seconds** of consecutive Jev off-task
observations. The Pi agent does not use Gemini.

#### 1. Put the project on the Pi

Copy or clone the repository onto the Raspberry Pi. Keep the project together:
`pi_agent.py` imports the `stationwatch/` package, and `pyproject.toml` defines its
dependencies. Copy your `.env` separately; Git does not include it. Create the
virtual environment on the Pi using the steps below.

Open a terminal in the project folder. For example, if you put it in your home
directory:

```sh
cd ~/HackGB-2026
```

#### 2. Install dependencies and create the virtual environment

Run these commands from the project folder on Raspberry Pi OS:

```sh
sudo apt update
sudo apt install -y python3-venv ffmpeg libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e '.[camera]'
```

#### 3. Configure `.env`

Put `.env` alongside `pi_agent.py`. If you already copied it to the Pi, edit that
file to add any missing settings. For a new file, copy the template first:

```sh
cp .env.pi.example .env
nano .env
```

Fill in your real credentials and station details:

```dotenv
TYPESAFE_API_KEY=your-typesafe-api-key
PGPASSWORD=your-postgres-password
PGUSER=your-postgres-user
PGDATABASE=your-postgres-database
PGHOST=your-postgres-host
PGPORT=5432
PGSSLMODE=require
STATION_ID=workbench-01
STATION_NAME="Workbench 01"
TASK_DESCRIPTION="Operate the joystick/controller while facing and observing the laptop display"
```

Use the same PostgreSQL connection settings as the dashboard, including your
provider's actual port and TLS mode. The database tables have already been
provisioned for this project. Each Pi should have its own `STATION_ID`.
The agent loads `.env` automatically; exported environment variables take
precedence. Keep `.env` out of Git.

#### 4. Start the agent with your camera

**Direct USB camera:** stop any other script using that camera, then run:

```sh
python pi_agent.py --source 0
```

Use `--source 1` for another camera. The task and station come from `.env`; you
can also supply them explicitly:

```sh
python pi_agent.py --source 0 \
  --task "Operate the joystick/controller while facing and observing the laptop display" \
  --station-id workbench-01 --station-name "Workbench 01"
```

**Camera streamer for browser viewing:** the repository includes
[`camera_stream.py`](camera_stream.py). Activate the same virtual environment
in a separate terminal and start it:

```sh
cd ~/HackGB-2026
source .venv/bin/activate
python -m pip install -e '.[camera]'
python camera_stream.py --camera 0 --host 0.0.0.0 --port 5000
```

Leave that terminal running. In the agent's terminal, run:

```sh
python pi_agent.py --source http://127.0.0.1:5000/video_feed
```

Use `/video_feed`, not the streamer's HTML root page. Replace `127.0.0.1` if the
streamer runs on another device. In this mode, the streamer owns the camera and
the agent reads its HTTP feed; browser viewers can continue using the Flask page.
From a laptop connected to the same Tailscale network, open
`http://<pi-tailscale-ip>:5000/` to watch the feed. Run `tailscale ip -4` on the
Pi to find that address. Keep both the streamer and the agent running in their
respective terminals. If the agent reports `Connection refused`, check that the
streamer is running on the same Pi and port.

#### 5. View alerts, stop, and restart

Open the project's dashboard to view confirmed alerts. To run the dashboard
locally on the computer hosting it, use its project environment and PostgreSQL
settings:

```sh
python -m stationwatch.hub
```

Open **http://127.0.0.1:8080** on that computer. Alerts appear when Jev's off-task
observations span more than five seconds. Snapshots upload first; video becomes available once
its 5-second recording segment closes. Playback starts at the beginning so the
full clip is visible. Database failures keep confirmed alerts
in a local queue for retry.

Press **Ctrl+C** in the agent terminal to stop, finalize the active recording,
and attempt a final upload. When you open a new terminal on the Pi, reactivate
the environment before starting again:

```sh
cd ~/HackGB-2026
source .venv/bin/activate
python pi_agent.py --source 0
```

Use the `/video_feed` source again if you are using the Flask streamer. Keep the
same station ID and `data/pi-<station-id>/` directory so queued uploads survive
restarts. Run `deactivate` when you want to leave the virtual environment.

See [the Pi setup guide](deploy/RASPBERRY_PI.md) for detailed retry and recording
behavior, camera options, and an optional systemd service that starts at boot.

#### Controller/laptop task rules

When `--task` or `TASK_DESCRIPTION` names a **joystick** or **controller**, Jev
uses a detailed workstation rubric:

- **On task:** a hand is in an operating position on the controls **and** the
  person's head/face is directed toward the machine/laptop display.
- **Off task:** the person clearly faces away from the machine, even while
  holding/operating the controller; leaves the controls visibly untouched; or
  looks at/uses a separate phone instead of observing the machine.

Jev chooses only **on task** or **off task**, using the best-supported label
when the image is ambiguous and reporting its confidence separately. A clear
violation of either requirement supports off task even if the other is hidden.

A still frame does not need to prove joystick movement, button presses, precise
eye fixation, or readable screen content. The five-second gate handles duration;
the image classifier assesses the current frame. Other task descriptions keep
the general task-behavior prompt. An alert requires **more than five seconds** of
off-task observations; exactly five seconds is insufficient. Restart the agent after updating its code or
task description.

The console prints Jev's `model_choice` and `confidence` alongside the decision.
There is no Jev confidence cutoff or `--jev-threshold` option: every valid
off-task decision counts toward the streak. API failures, missing frames, and
on-task decisions break it. The Pi agent requires only `TYPESAFE_API_KEY` and the
database credentials; no Gemini key or confirmation is required.

#### Send a manual test alert

With `camera_stream.py` running, open another terminal on the Pi and run:

```sh
cd ~/HackGB-2026
source .venv/bin/activate
python pi_agent.py --force-alert \
  --source http://127.0.0.1:5000/video_feed \
  --station-id workbench-01
```

This records a 5-second video, sends **one manual test
alert** with its snapshot and video to PostgreSQL, then exits. It bypasses Jev
and the duration gate, so no API key is required for this command.
The dashboard shows **Manual test alert** and **AI review: Bypassed for test**.
The event does not claim that either AI found off-task behavior.

You can run this alongside the normal agent when both consume the HTTP stream.
Use `--source 0` instead if the USB camera is free. Test uploads have their own
local queue at `data/pi-<station-id>-manual-test/`. An upload failure returns a
nonzero exit status and retains the evidence there. Running the command again
retries queued tests and creates one new test alert. Normal monitoring continues
to require more than five seconds of consecutive Jev off-task observations.

### Classify a video feed with Jev

```bash
uv sync --locked --extra camera
.venv/bin/python worker_video.py --source 0 --task "Assemble the parts on the workbench"
```

On the first run, enter your TypeSafe API key at the hidden prompt. The script
saves it as `TYPESAFE_API_KEY` in this project's `.env` and loads it automatically
on future runs, including from a new terminal. If the key is already exported in
your terminal, it is saved automatically instead. An exported key overrides and
updates the saved key; other `.env` entries are preserved. You can also edit the
`TYPESAFE_API_KEY` entry in `.env` directly. The file is ignored by Git.

`worker_video.py` reads the webcam continuously and sends a current frame to Jev
every second, starting with the first frame. Each printed result includes capture
time, `on_task` or `off_task`, confidence, and Jev's original choice.
It uses the same binary API request, image preparation, and response validation
as the photo webpage. Frames are sent to TypeSafe and are
not saved locally.

Use `--source 1` for another camera, `--source 'rtsp://host/stream'` for a stream,
or `--source clips/example.mp4` for a video file played at its recorded frame rate.
Pass `--model` (or set `TYPESAFE_MODEL`) to select a model, and `--interval 2` to
sample every two seconds. Requests run concurrently so API latency does not pause
capture; results may print out of order and are tagged with the sampled frame's
time. At most eight requests are in flight (`--max-in-flight`); if they all remain
busy, the script prints `SKIPPED` for that sample instead of building a backlog.
API failures print `ERROR` and sampling continues. Ctrl+C stops capture; reaching
the end of a video file waits for outstanding results and exits.

### Upload a photo and classify it with Jev

Run the standalone local webpage (no camera, trained local model, or MongoDB required):

```bash
# From this project directory, install the photo webpage dependencies:
uv sync --locked --extra camera
export TYPESAFE_API_KEY='your TypeSafe API key'
.venv/bin/python worker_webpage.py
```

The webpage also loads this project's `.env`, so you can omit `export` once the
video script has saved your key.

Open **http://127.0.0.1:8001**. Upload a JPEG, PNG, or WebP image (up to 10 MB and 200 MP),
describe the worker's assigned task, and select **Classify photo**. The page shows
**On task** or **Off task**, plus Jev's confidence and the two outcome probabilities.
Low-confidence results keep their chosen label. Confidence is reported by the
model and is not a measured accuracy guarantee. A photo only supports an
observation of visible behavior at one moment.

Get a key from the [TypeSafe console](https://console.typesafe.ai/). The Python
server keeps it out of the browser and sends the photo to the
[Jev image API](https://docs.typesafe.ai/images); uploaded photos are not saved by
this app. Jev image support is experimental. Photos are converted to upright
baseline JPEGs, resized and padded to the API's image limits. Use
`--port 8002` to change the local port or `--model jev-latest` to choose a model
(also configurable through `TYPESAFE_MODEL`). Stop the server with Ctrl+C.

The older local-model workflow described below uses a frame every **0.5 seconds**,
a trained local classifier, **more than five seconds of consecutive confident
flags**, Gemini review, and MongoDB uploads. The Jev/PostgreSQL Pi agent above
does not require that training workflow. Uncertain reviews and API failures do
not become confirmed alerts in either workflow.

The project includes USB/Picamera2/custom camera adapters, a synthetic demo, labeled-video training, pretrained MobileNet image features, an optional MoveNet/MediaPipe pose path, bounded inference workers, video recording, a durable upload queue, MongoDB/GridFS storage, and a black/gray alert dashboard. **No real labeled worker data or trained task-behavior classifier is bundled.**

## Which file creates and runs the ML model?

Use **[training_workflow.py](training_workflow.py)** and follow **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for the exact recording → labeling → extraction → training → evaluation → Pi deployment steps. Image mode is the recommended starting experiment for the fixed side-view demo camera; a body tracker is optional.

- **[stationwatch/image_training.py](stationwatch/image_training.py)** trains a small logistic classifier from human-labeled video frames, keeping whole people/sessions separate during evaluation. It exports JSON weights, held-out reports/predictions, and a matching Pi config.
- **[stationwatch/image_model.py](stationwatch/image_model.py)** runs a verified pretrained MobileNetV3 Small image encoder and the learned classifier on the Pi. The encoder is generic visual feature extraction, not an off-the-shelf attention detector.
- **[stationwatch/temporal.py](stationwatch/temporal.py)** requires sustained local flags using capture timestamps. The five-second timer is independent of training labels.
- **[stationwatch/pipeline.py](stationwatch/pipeline.py)** owns capture, bounded inference/review workers, and saving confirmed evidence.
- **[train_model.py](train_model.py)** trains directly from saved mixed-behavior annotations with `--annotated --holdout owen`, refreshing extraction without importing or changing labels. With `--videos clips`, it imports numbered `aaron1.mp4`, `aaron2.mp4`, etc., grouping by person and preserving saved labels. Legacy behavior pairs and `--csv` pose training also work. Uppercase `.MP4` extensions work too. **[features.py](stationwatch/features.py)** runs MoveNet/MediaPipe and **[model.py](stationwatch/model.py)** scores those pose features.

For numbered clips already in `clips/`, use the initialized dataset:

```sh
python training_workflow.py annotate
# Save interval labels in http://127.0.0.1:8002, then stop the annotator.
python train_model.py --annotated --holdout owen
```

The annotator automatically discovers `name#.mp4` files in `clips/`, including files added while it is open. The selector refreshes every five seconds without changing the current video or unsaved intervals; **Refresh recordings** checks immediately. Use `annotate --videos PATH` to watch another folder. The number identifies a take, not a behavior. New videos start unlabeled. Renamed copies preserve existing annotations by matching video content.

To test new clips from the same people, hold out one whole clip per person:

```sh
python train_model.py --annotated --split-by clip --seed 42 --output models/attention_image_known_people.json
```

This measures performance on known people in new recordings. The default person split measures workers excluded from training. Both options keep whole clips together, and clip selection is reproducible.

Image training starts here after installation and camera configuration:

```sh
python -m pip install -e '.[lite,train,mongo]'
cp config.pi.example.json config.json
# Edit task_description, camera settings, and optional image_crop first.
python training_workflow.py init --config config.json --download-model
python training_workflow.py record --clip p01-s01-a --person p01 --session s01 --seconds 60
# Record both behaviors across at least three independent people/session groups.
python training_workflow.py annotate
# Save human labels in http://127.0.0.1:8002, then stop the annotation server.
python training_workflow.py extract
python training_workflow.py train --holdout p03
```

Label `0` for acceptable task behavior and `1` for visible disengagement **from its onset**. Do not wait five seconds to start label 1. Leave ambiguity unlabeled. The trained image path processes frames rather than learning motion from complete video sequences; if the distinction requires motion or machine-state information, evaluate a temporal model or add task signals. Scores and visible behavior do not establish mental attention.

## Try it without hardware or API credentials

Use Python **3.11 or 3.12** for the widest MediaPipe/Raspberry Pi compatibility. The synthetic demo does not require MediaPipe. Install `ffmpeg` with H.264 encoding support (`sudo apt install ffmpeg` on Raspberry Pi OS; `brew install ffmpeg` on macOS).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[camera,dev]'
stationwatch serve --demo
```

Open **http://127.0.0.1:8000**. A synthetic worker alternates between poses every eight seconds; the first synthetic event appears after roughly 10–12 seconds. Demo frames and reviews are generated locally; **no Gemini requests run**. Demo data lives separately in `data/demo/`. Recordings become available after their 30-second segment closes, or after graceful shutdown.

## Model/runtime choices

The Pi example uses `analysis_backend: "image"`, a verified 224×224 MobileNetV3 Small LiteRT encoder, and learned `models/attention_image.json` weights. The full guide explains cropping, independent labels, holdout groups, deployment, and failure behavior. Training needs scikit-learn; Pi inference needs only LiteRT/NumPy plus the standard camera/upload dependencies. Use a supported 64-bit OS/Python combination and measure throughput on the actual Pi.

MoveNet SinglePose Lightning and MediaPipe Pose Landmarker Lite remain **optional pose detectors**. To use that route, initialize a separate dataset with `--backend pose`, as described in the guide, and calibrate the machine point/workstation polygon. It trains `models/attention.joblib` from eight body/head/position/pose-dwell features. The rule baseline requires an away-facing head angle and pose dwell, followed by the separate runtime flag-duration gate. MediaPipe currently lacks an official Linux ARM64 wheel; use LiteRT/MoveNet on a supported Pi.

On minimal/headless Linux, OpenCV may need `sudo apt install libgl1 libglib2.0-0`. Model files are ignored by Git and must be copied/downloaded to each deployment. Image model downloads verify the expected checksum; preprocessing matches the model metadata (RGB divided by 255).

Gemini defaults to `gemini-2.5-flash`; choose a vision-capable model available to your project. Image mode sends the flagged JPEG, configured task description, crop, and local duration context. Pose mode includes calibrated geometry and measured pose features. Provide `GEMINI_API_KEY` through the environment; the dashboard never receives it. A still-image review cannot independently confirm the whole duration.

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

## Optional pose-mode workstation calibration

Edit `config.json` for a fixed camera looking at **one workstation and one visible worker**:

- `machine_center`: the task/machine point, normalized `[x, y]` in the camera image; top-left is `[0, 0]`.
- `workstation_polygon`: normalized polygon enclosing the operating area. The worker's hip midpoint determines whether they are inside.
- `width`, `height`, `fps`: camera recording dimensions and target rate. Use even width/height for H.264.
- `baseline_angle_deg`, `baseline_dwell_s`: starting heuristic thresholds. Verify using representative task footage, including legitimate turning, reaching, breaks, and occlusion.
- `local_threshold`, `gemini_threshold`: classifier score and Gemini reported confidence cutoffs. Gemini's reported confidence is not a calibrated probability.
- `event_cooldown_s`: minimum spacing after a confirmed event (default 30 seconds). `review_interval_s` limits API attempts even after a failed/negative review (default five seconds).

Features are **2D camera-plane proxies**, not true 3D body or gaze angles. Body angle compares the hip-to-shoulder torso vector with the hip-to-machine vector; head angle compares the ear-midpoint-to-nose vector with the ear-to-machine vector. Camera perspective matters. The baseline uses **head angle only**, plus workstation membership and dwell; the trainable classifier receives all features. A change larger than 12° or 0.08 image units resets similar-pose dwell; missing/ambiguous poses and gaps over 1.5 seconds also reset it. With no identity tracking, a worker swap without a missing frame can share dwell. MediaPipe rejects multiple detected people. **MoveNet SinglePose cannot detect/reject multiple people**, so its camera must contain only one worker. Both adapters reject low-confidence or degenerate required landmarks.

Machine position is manually calibrated and fixed; this prototype does not detect or track moving machines. Actual attention cannot be established from a still image. Events are observations for human review. This prototype is not a machine interlock or a validated protective control.

## Legacy pose-feature CSV collection/training

For image mode, use the labeled-video workflow above. This older path requires a pose-mode config (`analysis_backend: "pose"`, a downloaded pose model, and `classifier_path: "models/attention.joblib"`).

Collect **real, independently labeled demonstrations** under controlled conditions. Label `0` for task-oriented behavior and `1` for visible orientation away from the task. Every sample in one collection run receives the supplied label, so keep the behavior consistent throughout that run. Never stage unsafe actions around operating equipment.

```sh
stationwatch --config config.json collect --label 0 --group workerA-session1 --seconds 30
stationwatch --config config.json collect --label 1 --group workerA-session1 --seconds 30
# Repeat both labels across several workers/sessions and camera conditions.
# Keep ALL footage from the same worker/session in the same group.
python train_model.py --csv data/training.csv --output models/attention.joblib --threshold 0.8
# For local inspection; use agent for MongoDB deployment.
stationwatch --config config.json serve
```

The CSV contains `body_angle_deg, head_angle_deg, person_x, person_y, in_workstation, machine_x, machine_y, dwell_s, label, group`. Collection skips frames without a clear single pose. Training requires both labels and at least four independent groups, fits a 128-tree random forest with depth capped at eight (or standardized logistic regression), and writes a held-out report next to the artifact. The forest uses one CPU worker for predictable Pi inference. Twenty samples is only a format/startup minimum, not evidence of sufficient data. If you want to test generalization to new workers, use the **worker ID** as `group` across all their sessions. Do not split neighboring frames randomly between training and evaluation.

The saved model excludes held-out groups. Use the same pose backend and camera calibration for collection and deployment. The report includes results at both 0.5 and the supplied runtime threshold, including the workstation-membership gate. The default threshold is 0.8; keep `local_threshold` in your Pi config aligned with the threshold you evaluated. Forest reports also include feature importance. Evaluate missed events, false alerts, camera changes, task-specific exceptions, and per-worker coverage before relying on the results. Only load trusted, locally created `.joblib` files.

## Pi → MongoDB → central alert hub

For the camera-free PostgreSQL/Timescale hub on DigitalOcean App Platform, see
[the deployment guide](deploy/DIGITALOCEAN.md). Its root Procfile runs
`python -m stationwatch.hub`; the MongoDB CLI workflow below is separate.

The Pi runs the webcam, local ML, and Gemini. **Only confirmed alerts, their snapshots, and their recorded segments are uploaded.** Normal frames, local-only flags, uncertain reviews, and per-frame statistics stay on the Pi. The central hub reads MongoDB and runs no camera, pose detector, or Gemini calls.

On the Pi, copy the image encoder, trained classifier, and exported config as described in the training guide. Install inference/upload dependencies and set both credentials locally:

```sh
python -m pip install -e '.[lite,mongo]'
export GEMINI_API_KEY='your Gemini key'
export MONGODB_URI='your MongoDB connection string'
# Your USB webcam is already selected by this example config.
stationwatch --config models/attention_image.config.json agent
```

`agent` is the Pi runtime entry point in [stationwatch/cli.py](stationwatch/cli.py). It runs without a dashboard/web server. A camera/recording failure stops the agent with an error; Ctrl-C or SIGTERM finalizes the active recording before exit. Give each Pi a distinct `station_id` and use exactly one agent process per Pi/local data directory.

On the central computer, use the **same MongoDB database** and start the hub:

```sh
python -m pip install -e '.[mongo]'
export MONGODB_URI='your MongoDB connection string'
stationwatch --config config.hub.example.json hub
```

Open http://127.0.0.1:8000 on the central computer. For access from another computer, set `DASHBOARD_TOKEN` and pass `--host 0.0.0.0` as described below. The hub only needs database access; it does not need `GEMINI_API_KEY`, a trained model, or a webcam. The browser polls the hub for stored alerts every three seconds. The inbox defaults to unreviewed alerts, supports station/status filters, and shows capture time, the corroborating reason, snapshot, video, and a review action. It has no continuous camera feed or per-frame analysis statistics.

[stationwatch/mongo.py](stationwatch/mongo.py) implements the storage and background uploader:

- The `alerts` collection uses the event UUID as `_id` and stores the station, timestamps, feature measurements, local score, Gemini review, and manager review status.
- GridFS stores JPEG/MP4 evidence in `evidence.files` and `evidence.chunks`. Large videos are not embedded inside an alert document, avoiding MongoDB's 16 MiB document limit.
- The snapshot and alert are sent on the next upload pass (default five seconds). The recording segment is attached after it closes (default 30 seconds); the alert exists with `clip_state: "pending"` until then. These are the recording segments containing the flag, not individually trimmed clips.
- A SQLite outbox is committed with each local confirmed event. It survives restarts and retries failed uploads with backoff up to five minutes. Retries use stable IDs and do not duplicate alerts/media or reset a manager's reviewed status. Pending segments remain protected from local recording cleanup.
- The Pi's upload worker is separate from capture, local inference, and Gemini review. When there are no confirmed events queued, it performs no application-level MongoDB reads/writes. Synthetic demo evidence is never exported to MongoDB.
- `MONGODB_URI` lives only in the environment. Configure `mongo_database` in JSON (default `stationwatch`), plus `mongo_poll_s` and `mongo_timeout_s` as needed. Both computers must be allowed to connect to your chosen MongoDB deployment.
- The hub caches downloaded media under `data/hub-cache` so its authenticated video endpoint supports browser seeking. GridFS media, alert records, and this cache have no automatic expiry yet; set a retention policy before long-running deployment. Interrupted Pi recording segments remain pending until recovered or removed manually.

The same upload path can be enabled alongside local inspection with `stationwatch --config config.json serve --mongo`. `serve --demo` remains entirely local and cannot be combined with `--mongo`.

## Concurrency, evidence, and failure behavior

- Capture records continuously and exposes an MJPEG stream; it offers a screenshot every 0.5 seconds.
- One local inference worker processes screenshots chronologically to preserve capture-time flag streaks and optional pose dwell. There is at most one local job in flight; busy samples are skipped and counted.
- A separate worker handles Gemini requests, with one request in flight and a finite HTTP timeout. Capture and local inference continue during review. Frames older than `max_analysis_age_s` are excluded from new API reviews.
- The local duration gate must pass and Gemini must agree before an event is saved. Snapshot time, session elapsed time, video segment name, exact encoded-frame offset, local features, scores, and Gemini's reason are persisted. Gemini latency never substitutes for capture time.
- H.264 MP4 files use atomic finalization. Active segments return HTTP 409 until closed; the dashboard provides a retry action. Playback can start two seconds before the event within the same segment. Cross-segment pre-roll is not implemented.
- Unflagged closed recordings expire after 24 hours by default. Segments referenced by events and their snapshots are retained until manually removed. Plan storage capacity; there is no automatic deletion of event evidence. Interrupted `.partial.mp4` files are not served or automatically recovered.
- Camera/recording failures stop capture and show an error. Missing poses in pose mode are `unknown`; image mode has no person detector and assumes one worker in its fixed view. API errors and malformed/blocked responses are counted and never treated as agreement. Errors remain visible for investigation. No automatic restart is included.

Recording uses a constant encoded FPS; under capture lag, playback duration can differ from wall time. Event video offsets use the encoded frame index, while UTC timestamps and session elapsed seconds preserve capture time. For network cameras, timestamps represent local frame receipt, not the camera's hardware clock.

## Manager access

The `serve` and MongoDB `hub` CLI commands default to loopback. To access them from another computer:

```sh
export DASHBOARD_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# Store the token securely and enter it in the dashboard's login dialog.
stationwatch --config config.json serve --baseline --host 0.0.0.0
```

Binding those CLI commands to a non-loopback address requires an access token of at least 24 characters. The App Platform entrypoint always binds to `0.0.0.0`; set `DASHBOARD_TOKEN` explicitly there because authentication is skipped when it is unset. Status, events, media, live stream, and review actions share that token; the dashboard stores it in an HttpOnly SameSite cookie, not in URLs or browser storage. API clients may instead send `Authorization: Bearer <token>`. Put remote access behind HTTPS and a trusted network/reverse proxy. This prototype has one shared manager credential; it has no user roles, per-person audit log, or internet-facing login rate limiting.

Run **one agent per Pi/camera** and one central `hub` instance connected to the shared MongoDB database. `serve` is the combined local inspection mode and must use one process; do not enable Uvicorn multi-worker mode or reload on a camera process. Use Ctrl-C for graceful shutdown and video finalization.

## Checks

```sh
python -m pip install -e '.[dev,train,mongo]'
pytest -q
ruff check .
```

Tests use fake cameras/reviewers, generated features, and mongomock's GridFS integration. They cover pipeline gates, bounded concurrency, dwell resets, evidence timestamps, authorization, group-separated training, upload retry/restart behavior, duplicate handling, preservation of review state, incomplete GridFS recovery, and hub video range requests. They do not validate attention accuracy or a real MongoDB deployment. Hardware camera/Pi performance and real Gemini/MongoDB connections require their respective hardware, deployments, and credentials. The pretrained image encoder has also passed a local LiteRT smoke test. No real worker-classification accuracy or Pi timing has been established.

`uv.lock` records dependency versions. A plain `uv sync --locked` installs only the lightweight PostgreSQL hub dependencies. `uv sync --locked --extra camera` adds OpenCV, NumPy, and the HTTP client for the camera/demo and photo webpage. `uv sync --locked --extra lite --extra train --extra mongo --extra dev` installs the Pi inference/training/upload dependencies, or replace `lite` with `pose` for MediaPipe; `lite`, `pose`, and `train` each include the `camera` extra. Picamera2 remains an OS package when using a CSI camera.
