# Raspberry Pi: Jev → Gemini → PostgreSQL

The entrypoint is `pi_agent.py` (also `python -m stationwatch.pi_agent`). It runs
without a web server or a trained local model. It captures a live USB camera or
the MJPEG `/video_feed` endpoint from the included `camera_stream.py`.

## Install and configure

From the project directory on the Pi:

```sh
sudo apt update
sudo apt install python3-venv ffmpeg libgl1 libglib2.0-0
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[camera]'
```

Put the settings from [`.env.pi.example`](../.env.pi.example) in this checkout's
`.env`. Supply `TYPESAFE_API_KEY`, `GEMINI_API_KEY`, and the same six `PG*` settings
used by the dashboard. Use your database provider's actual port and TLS mode.
The database must already have the schema from `deploy/postgres.sql`.
The producer database role needs SELECT/INSERT on `stationwatch_alerts` and
`stationwatch_evidence`, and UPDATE on `stationwatch_alerts`. The role used when
provisioning the tables already owns them. The Pi needs network access to the
database and both model APIs. Exported settings override `.env` settings.

Use a distinct `STATION_ID` for each camera. Set `TASK_DESCRIPTION` to the actual
assigned task so both models evaluate the same behavior. Credentials stay on the
Pi and are never included in an alert's payload or printed by the agent.

## Run with the USB camera directly

```sh
.venv/bin/python pi_agent.py --source 0 \
  --task "Assemble the parts on the workbench" \
  --station-id workbench-01 --station-name "Workbench 01"
```

If the task and station are set in `.env`, only `--source 0` is needed. Use
`--source 1` for another camera. The default recording size is 640×480 at 15 FPS;
`--width`, `--height`, and `--fps` override it. Width and height must be even.
This process owns the USB camera, so stop your Flask streamer before selecting
the same camera directly.

## Run with the included Flask camera streamer

In one terminal, start the included streamer (Flask is installed by the `camera` extra):

```sh
.venv/bin/python -m pip install -e '.[camera]'
.venv/bin/python camera_stream.py --camera 0 --host 0.0.0.0 --port 5000
```

From a laptop on the same Tailscale network, open `http://<pi-tailscale-ip>:5000/`.
Find the Pi's address by running `tailscale ip -4` on it.

Leave the streamer running and start the agent in another terminal:

```sh
.venv/bin/python pi_agent.py \
  --source http://127.0.0.1:5000/video_feed \
  --task "Assemble the parts on the workbench" \
  --station-id workbench-01 --station-name "Workbench 01"
```

Use the raw `/video_feed` URL, not the streamer's HTML root page. If the streamer
is on another device, replace `127.0.0.1` with its address. The streamer owns the
camera, and the agent consumes one HTTP stream. Other browser viewers can keep
using the Flask page. Network streams have five-second open/read timeouts.

## What becomes an alert

- The agent continuously reads and records the camera while sending a current
  frame to Jev every second. Requests can overlap, up to eight in flight.
- Results are processed in capture order even when API responses arrive out of
  order. A confident `off_task` result starts or extends a streak; `on_task`,
  `uncertain`, API failures, skipped sample ticks, and capture gaps over 1.5 seconds
  break it. Responses older than 20 seconds do not extend a streak. Jev requests
  time out after 15 seconds; missed sampling times are skipped without catch-up
  bursts. Sampling continues while Gemini or database uploads are busy.
- Gemini is called once the first and latest consecutive off-task samples span
  **at least five seconds**. With one sample per second, this normally means six
  observations at approximately t=0, 1, 2, 3, 4, and 5. Capture times determine the
  duration. Both models evaluate visible task behavior; Gemini receives the last
  flagged snapshot and the streak context, rather than a video of the whole streak.
- Only Gemini's `inattentive` result at confidence ≥0.8 creates an alert. Its
  `attentive`, `uncertain`, invalid, or failed responses create none. Jev's default
  confidence threshold is 0.6. These are configurable prototype thresholds, not
  measured accuracy guarantees. `--jev-threshold` can raise Jev's threshold above
  its shared client's minimum of 0.6; `--gemini-threshold` changes Gemini's cutoff.
- Gemini attempts are at least five seconds apart, with one request in flight.
  A confirmed event starts a 30-second cooldown (`--cooldown` overrides it), so a
  sustained episode does not create an alert every second.

Alerts include station and session IDs, UTC capture/confirmation timestamps, the
assigned task, Jev decision/confidence/probabilities/model, Gemini reason and
confidence/model, streak duration, a JPEG snapshot, recording name, and the
encoded video offset. The existing dashboard can display and review them.

## Uploads, recording, and restarts

Confirmed alerts and snapshots are committed to a local SQLite outbox before
upload. The uploader sends the alert plus snapshot in one PostgreSQL transaction,
then sends its video once the 30-second H.264 MP4 segment is finalized. Until then,
the website can display the alert and snapshot; its video action asks you to retry.
Database outages retry with backoff up to five minutes. Stable IDs make retries
idempotent and preserve the manager's reviewed status. No database calls run when
there is nothing queued. Normal observations and continuous video are not sent
to PostgreSQL; only confirmed alerts and their containing segments are uploaded.
Jev receives the sampled frames, and Gemini receives snapshots selected for review.

Local state defaults to `data/pi-<station-id>/`; override it with `--data-dir`.
Use the same directory after restarting so queued uploads survive. One agent may
use a directory at a time; a process lock enforces that. Unflagged finalized
segments expire after 24 hours. Confirmed events and their recordings are retained
locally and in PostgreSQL until manually removed. Allow sufficient disk space.
A containing segment can include normal behavior around the alert. Encoded video
offsets follow the actual frame count; under capture lag playback speed can differ
from wall-clock duration.

Ctrl+C and SIGTERM stop sampling, finish an active review, finalize the recording,
and attempt a final upload. Camera or encoder failure exits with an error;
a service can restart it. A hard power loss may leave a `.partial.mp4` recording
that needs manual recovery; completed queued alerts and snapshots still survive.
No real camera, Jev/Gemini accuracy, or Raspberry Pi performance is certified by
the automated tests.

## Start at boot (optional)

Edit the user and checkout paths in `deploy/stationwatch-pi.service`, then:

```sh
sudo cp deploy/stationwatch-pi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stationwatch-pi
journalctl -u stationwatch-pi -f
```

The service reads the checkout's `.env`; set `TASK_DESCRIPTION` there. Its default
source is camera 0. Change `ExecStart` to use the Flask `/video_feed` URL if desired;
start the Flask streamer separately before that service. Restart after editing
settings: `sudo systemctl restart stationwatch-pi`.
