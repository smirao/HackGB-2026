# Train from labeled video, then run on the Pi

Use **`training_workflow.py annotate --videos clips`** to import numbered clips and annotate their behavior intervals. Use **`train_model.py --annotated`** to refresh extraction and train from those saved annotations. `training_workflow.py` also provides recording, individual extraction/training steps, and held-out evaluation. Run all commands from the project directory. No labeled worker data or trained task-behavior classifier is supplied.

## What we are training

For your fixed side-view camera, start with **image-based transfer learning**:

1. Record video from the webcam in its final position.
2. Mark intervals of acceptable task behavior (`0`) and visible task disengagement (`1`).
3. Sample labeled frames every 0.5 seconds. A pretrained **MobileNetV3 Small** image encoder turns each frame into 1,024 visual features. It does not track a skeleton.
4. Train a small logistic classifier on those features. The encoder stays pretrained; the classifier learns your task and camera view.
5. On the Pi, classify a frame every 0.5 seconds. Require consecutive confident flags spanning **more than five capture-time seconds**, then request Gemini review.
6. Save and upload an alert/snapshot/video segment to MongoDB only if Gemini also flags it at the configured confidence threshold. The central hub displays those alerts.

This is a frame classifier with a separate duration rule. It does not learn motion from a whole video sequence. That is a practical first experiment when head/body orientation and interaction with a work item are visibly different. If acceptable work and disengagement look identical in individual frames, more frame labels may not solve it; evaluate a temporal model using several seconds of frames, or add task/machine signals. Pose tracking remains an optional alternative, not a requirement.

The labels describe **visible behavior**, not a measurement of someone's thoughts. Define the task and legitimate exceptions before labeling.

## 1. Fix the camera view and configuration

Your photo shows a useful side view of the worker, hands, and work surface. Give the head a little more space at the top/left. Keep the work item and hands visible. If the laptop is part of the task, include it; if it is unrelated, define that consistently. Keep the mount, camera orientation, resolution, and task location the same for recording and the demo.

A large part of the right side is empty background. You can crop it out if the crop still includes every legitimate task area. The classifier resizes its crop to 224×224, so unnecessary background reduces useful worker detail. Crop choice is optional; start with the full frame if unsure.

Copy `config.pi.example.json` to a local configuration:

```sh
cp config.pi.example.json config.json
```

Edit `config.json` before initializing the dataset:

- `camera_source`: USB webcam index, usually `0`.
- `analysis_backend`: `"image"`.
- `image_model_path`: `"models/mobilenet_v3_small.tflite"`.
- `classifier_path`: `"models/attention_image.json"`.
- `image_crop`: `[left, top, right, bottom]`, normalized to 0–1; `[0, 0, 1, 1]` uses the whole image.
- `task_description`: explain the real demo task and legitimate reaching/checking actions. Gemini receives this description.
- `station_id` / `station_name`: identify this Pi's station.
- `flag_duration_s`: `5`.

Image mode needs no machine-point or workstation-polygon calibration. Those fields belong to the optional pose path.

## 2. Install and initialize

Use Python **3.11 or 3.12** and a 64-bit Raspberry Pi OS installation. On the Pi:

```sh
sudo apt update
sudo apt install ffmpeg python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[lite,train,mongo]'
python training_workflow.py init --config config.json --download-model
```

On macOS, install FFmpeg with `brew install ffmpeg` instead of the two `apt` commands, and use Python 3.11/3.12. No TensorFlow training installation, GPU, or MediaPipe pose package is required for image mode. LiteRT must have a wheel for your OS/Python/CPU; choose a supported 64-bit environment if installation fails.

Initialization creates `data/training_dataset/` and freezes a copy of your settings there. The encoder download comes from Google's public MediaPipe image-embedder model storage and is checked against a pinned SHA-256. It is a generic pretrained image encoder, **not** a pretrained attention detector. Downloads need internet access; training/labeling do not use Gemini or MongoDB.

You can record on the Pi and train on a laptop. Copy the **whole dataset directory**, including recordings, labels, timestamps, and config, to the same relative location in the laptop checkout. Also copy/download the encoder under `models/`. Run subsequent commands from that checkout's project root. Paths in these example configs are relative to the working directory.

To use another dataset directory, place `--dataset PATH` **before** the subcommand in every command.

## 3. Record independent examples

Record a take:

```sh
python training_workflow.py record --clip p01-s01-a --person p01 --session s01 --seconds 60
```

The script uses the configured USB webcam, saves H.264 video, and records capture timestamps. Ctrl-C ends and saves a shortened take. IDs must be unique for clips. The same person keeps the same `--person` ID across all their clips and sessions.

Record additional takes, with both behaviors in each person's footage:

```sh
python training_workflow.py record --clip p01-s01-b --person p01 --session s01 --seconds 60
python training_workflow.py record --clip p02-s01-a --person p02 --session s01 --seconds 60
python training_workflow.py record --clip p03-s01-a --person p03 --session s01 --seconds 60
python training_workflow.py record --clip p04-s01-a --person p04 --session s01 --seconds 60
```

Repeat takes rather than treating these four minutes as a sufficient dataset. A reasonable pilot starts with several minutes per person, multiple takes/sessions, and both labels represented for everyone. Add data based on held-out mistakes; there is no guaranteed number of videos that establishes accuracy.

Include:

- Normal handling of the demo work item, looking down, reaching, task-related laptop checks, and other legitimate actions.
- Clearly staged disengagement, including sustained turns toward unrelated areas or activities.
- Short and long episodes, especially examples near the five-second boundary.
- Different clothing, people, lighting, and normal task positions. Each class needs comparable variety.

Use a harmless mock task for staged demonstrations. Keep the camera fixed. Avoid making every positive take use one hoodie or one lighting condition: the classifier can learn those shortcuts.

Image training requires at least **three independent groups** and 20 labeled frames as a startup check. That minimum does not imply useful accuracy. By default groups are people, so an entire person's footage is held out together. If you only have one participant, collect at least three genuinely separate sessions and use `--group-by session` later. That evaluates new sessions of a known person, not generalization to new workers. The legacy pose trainer retains its four-group minimum.

### Numbered mixed-behavior videos

Your current layout is:

```text
clips/
  aaron1.MP4
  aaron2.MP4
  aaron3.mp4
  hunter1.MP4
  hunter2.MP4
  hunter3.mp4
  owen1.MP4
  owen2.MP4
```

The prefix identifies the person; trailing digits identify the clip. `aaron1`, `aaron2`, and `aaron3` all belong to person `aaron`. The number does not assign a behavior. Any number of clips per person is supported, including multi-digit numbers and `.MP4`/`.mp4` extensions. Use the same name prefix consistently, and use a new number for new footage.

With the existing initialized dataset, start annotation:

```sh
python training_workflow.py annotate --videos clips
```

This imports new numbered videos with no labels, reuses videos already in the dataset, and displays the current source filenames in numerical order. If you renamed an existing video, its content hash links it to the existing clip and preserves every saved annotation. Internal clip IDs may retain earlier names so saved labels and references continue to work. Existing whole-clip labels are retained too; remove them in the annotator when replacing them with mixed-behavior intervals.

To import without starting the annotation server:

```sh
python training_workflow.py import-clips --videos clips
```

After saving each clip's intervals and stopping the annotator:

```sh
python train_model.py --annotated --holdout owen
```

All Owen clips stay together in evaluation. `python train_model.py --videos clips --holdout owen` can also import/reuse numbered clips and then train from saved intervals; newly imported clips remain excluded until annotated. Filenames never assign labels to numbered clips. The original source files stay unchanged.

### Earlier paired behavior videos (optional)

The `clips/` folder can contain the six files directly, using the same name for each person's pair. Both `.mp4` and `.MP4` extensions work:

```text
clips/
  aaron_task.MP4
  aaron_distracted.MP4
  hunter_task.MP4
  hunter_distracted.MP4
  owen_task.MP4
  owen_distracted.MP4
```

```sh
python train_model.py --videos clips --holdout owen --download-model
```

This labels `_task` clips **0** and `_distracted` clips **1**, trains on Aaron and Hunter, and evaluates both Owen clips. The source videos stay in place. Raw `clips/` footage and generated `data/`/`models/` files are ignored by Git.

The earlier folder-per-person layout also works:

```text
data/worker_videos/
  p01/
    on_task.mp4
    distracted.mp4
  p02/
    on_task.mp4
    distracted.mp4
  p03/
    on_task.mp4
    distracted.mp4
```

After installing `.[lite,train]`, run:

```sh
python train_model.py --videos data/worker_videos --holdout p03 --download-model
```

This initializes an image dataset if needed, imports all six videos, assigns `on_task.mp4` label **0** and `distracted.mp4` label **1**, samples frames every 0.5 seconds, and fits a classifier on frozen MobileNetV3 Small features. Both of `p03`'s clips are held out: training uses four clips from `p01`/`p02`, evaluation uses two from `p03`. Without `--holdout`, the split is reproducible and still keeps entire people together. Three people are sufficient to run this workflow; a fourth is optional.

Outputs default to `models/attention_image.json`, its `.config.json`, `.report.json`, and `.predictions.csv` companions. Run the Pi with the exported config. The MobileNet encoder stays frozen; only the small classification head is trained on your footage. Training does not call Gemini.

Optional `--trim-start-s 1 --trim-end-s 1` leaves setup/ending footage unlabeled. These trims must leave part of every clip. `--config config.json` supplies settings when creating a dataset; an existing dataset reuses its saved settings. Rerunning preserves imported videos and any manual label corrections, then extracts and trains again. Use `--session s02` for new recordings of the same people. Use `--dataset PATH` for a separate dataset and `--output PATH.json` for a different model output.

The paired importer assumes each clip consistently shows its named behavior. Review setup, transitions, or exceptions in the annotation tool before relying on those labels:

```sh
python training_workflow.py annotate
# Save corrections and stop the server, then rerun train_model.py with the same arguments.
```

If your filenames differ, either rename the source files to the layout above or import each video with its person's ID and an explicit behavior:

```sh
python training_workflow.py import-video --video /path/to/person1-on-task.mp4 --clip p01-on --person p01 --session s01 --behavior on-task
python training_workflow.py import-video --video /path/to/person1-distracted.mp4 --clip p01-away --person p01 --session s01 --behavior distracted
# Repeat both imports for p02 and p03, using their own files and unique clip IDs.
python training_workflow.py annotate
```

With `import-video`, `--behavior` assigns the label; omitting it leaves the clip unlabeled for manual annotation. `import-pairs` and `train_model.py --videos` assign labels from the paired filenames.

After saving labels and stopping the annotation server, sample frames and run three leave-one-person-out checks:

```sh
python training_workflow.py extract
python training_workflow.py train --holdout p01 --output models/check_p01.json
python training_workflow.py train --holdout p02 --output models/check_p02.json
python training_workflow.py train --holdout p03 --output models/check_p03.json
```

Each command trains on two people's four videos and tests on the remaining person's two videos. Inspect all three reports, including differences between people. Do not randomly split frames from the same videos across training and testing: adjacent images are very similar and would inflate apparent accuracy. Each exported model/config belongs to its particular split and excludes the held-out person.

Six videos can establish whether the idea works in this demo view, but many sampled frames are still only six recording sessions. Use the results to find mistakes, then record fresh mixed-behavior videos containing normal task-related glances and short distractions as well as long ones. Reserve additional untouched footage for a final test after choosing thresholds. To fit using all three people with the existing training command, add footage from a fourth held-out person and use `--holdout p04`.

To import an existing H.264 MP4:

```sh
python training_workflow.py import-video --video /path/to/take.mp4 --clip p01-s02-a --person p01 --session s02
python training_workflow.py list
```

Import keeps the source video unchanged. Without a capture timestamp sidecar, playback time is used as elapsed time. Avoid time-lapse, edited, variable-speed, or heavily dropped-frame videos for duration evaluation. Do not copy the same video into different groups; extraction rejects exact duplicates across groups.

## 4. Label the videos in the browser

Mixed-behavior clips are supported: each saved interval supplies the frame label, regardless of the clip's filename. A single clip can alternate between `0` and `1` any number of times.

```sh
python training_workflow.py annotate
```

Open **http://127.0.0.1:8002** on that computer. The annotation server is loopback-only. If recording on the Pi, copy the dataset to your laptop to label locally, or use an SSH port forward.

For each recording:

1. Play/pause and find where a behavior starts. Click **Set start here**.
2. Find where it ends. Click **Set end here**.
3. Choose `0 · Acceptable task behavior` or `1 · Visible task disengagement`.
4. Click **Add interval**. Repeat as the behavior changes.
5. Click **Save all intervals** before switching recordings.

**Start label 1 when the visible disengagement begins, not five seconds later.** The runtime timer provides the grace period. A brief clearly unrelated turn can be label 1 for its visible duration; the timer suppresses short episodes. Normal task-related glances remain label 0. Resolve inconsistent definitions before training; identical-looking frames with conflicting labels cannot be reliably separated by this model.

Leave ambiguous/occluded sections unlabeled. The extractor excludes them; it does not automatically assign labels. Intervals cannot overlap and use `[start, end)` boundaries. Labels use video playback time; the tool also shows capture elapsed time for recordings made by this script.

In image mode, the preview outlines the configured crop and hides pose calibration. Stop the annotation server with Ctrl-C before extracting.

For the six clips already imported from `clips/`, remove the old whole-clip interval in the annotator before adding the mixed-behavior intervals; otherwise they overlap. Save each clip before switching. Leave uncertain sections without an interval. Training reads the dataset's saved labels, not unsaved edits in the browser or the original filename.

To add a new mixed-behavior recording, import it without `--behavior` (any MP4 filename works):

```sh
python training_workflow.py import-video --video clips/owen_mixed.MP4 --clip owen-s02-mixed --person owen --session s02
python training_workflow.py annotate
```

Use the same person ID across that person's clips to keep evaluation separate from training.

## 5. Extract features and train

After saving your interval annotations and stopping the annotation server, retrain your existing image dataset in one command:

```sh
python train_model.py --annotated --holdout owen
```

This refreshes extracted features from the current saved intervals, skips unlabeled frames (including completely unlabeled clips), and trains/evaluates using those interval labels. It does not import videos, infer labels from filenames, or modify your saved annotations. It works with any clip names and any number of clips per person. Entire people remain held out together; both training and evaluation need both labels and at least three labeled people in total. Add `--dataset PATH` if you used another dataset directory.

Extraction prints on-task, distracted, and unlabeled/skipped sample counts per clip and saves them in `images.report.json`. Unchanged old whole-clip labels still count as annotations; review every clip you want to relabel. Model/config/report outputs have the same paths as before.

The separate extraction and training commands still work:

```sh
python training_workflow.py extract
python training_workflow.py train --holdout owen
```

This excludes all of `owen` from fitting and evaluates on that person's frames. Both training and held-out footage must include both labels. If you omit `--holdout`, the script chooses a reproducible group-separated split. Neighboring frames are never randomly scattered between training and evaluation. With separate commands, edits to labels invalidate the previous extraction and require another `extract` before training.

For the one-person, multiple-session option:

```sh
python training_workflow.py extract --group-by session
python training_workflow.py train --holdout p01:s04
```

Expected outputs for image mode:

- `data/training_dataset/images.npz`: labeled numeric visual embeddings and sample metadata.
- `models/attention_image.json`: your learned classifier weights and group/preprocessing metadata.
- `models/attention_image.config.json`: matching Pi deployment config.
- `models/attention_image.report.json`: held-out precision, recall, confusion matrix, and local duration-gate candidate spans.
- `models/attention_image.predictions.csv`: held-out scores, labels, timestamps, and duration-gate results for inspection.

The classifier standardizes features using training data only, fits logistic regression, and folds that standardization into portable JSON weights. Pi inference uses NumPy plus the same LiteRT encoder; scikit-learn is needed only for training/evaluation. No executable `.joblib` artifact is used by this image path.

Model/crop/resolution and raw-input hashes guard against stale extraction. Changing labels, videos, crop, or encoder requires another `extract` and training run. Physically moving the camera cannot be detected by those hashes; collect/relabel/evaluate the new view.

The report measures frame classification and local duration candidates, not final Gemini-confirmed event accuracy. A classifier score of 0.8 is a cutoff, not established 80% real-world reliability. Do not repeatedly tune against your final test set.

## 6. Test new footage without retraining

Record/import and label another independent person/session, then extract again:

```sh
python training_workflow.py record --clip p05-s01-a --person p05 --session s01 --seconds 60
python training_workflow.py annotate
# Save p05 labels, stop the annotation server, then:
python training_workflow.py extract
python training_workflow.py evaluate --model models/attention_image.json --groups p05
```

Evaluation refuses groups used in fitting and does not alter the model. Review the prediction CSV alongside complete clips. Check missed disengagement, false flags during legitimate work, and when the duration gate first fires. Unlabeled sections are excluded, so offline duration candidates are incomplete where observations are missing.

If you change thresholds after examining `p04`, reserve `p05` or later footage for a final independent test. For example, train with `--threshold 0.9`, then evaluate untouched footage at that same cutoff. More data is useful when it covers actual mistakes rather than many adjacent copies of the same pose.

## 7. Run the real Pi agent

If you trained on a laptop, copy these files to the Pi under the same project-relative paths:

- `models/mobilenet_v3_small.tflite`
- `models/attention_image.json`
- `models/attention_image.config.json`

On the Pi, inference/upload needs only:

```sh
python -m pip install -e '.[lite,mongo]'
# Set GEMINI_API_KEY and MONGODB_URI in the Pi environment privately.
stationwatch --config models/attention_image.config.json agent
```

Use the exported config so training and inference use the same crop, resolution, and encoder. `task_description` should describe the actual task. Neither an API key nor a database URI belongs in source control.

Runtime defaults:

- Sample every **0.5 seconds** and flag frames with score ≥ **0.8**.
- Print an INFO console line for every completed local prediction: `Local model: DISTRACTED` or `Local model: ON TASK`, including distraction score, threshold, sustained duration, capture timestamp, and freshness. These are local predictions; a logged distraction does not by itself mean Gemini confirmed an alert.
- Start the duration timer at the first confident flag. Exactly five seconds is insufficient; with perfect 0.5-second sampling, the first eligible sample is **5.5 seconds after that first flag**.
- A below-threshold prediction, unknown result, error, or stale capture resets the timer. A gap over `dwell_gap_s` (default **1.5 seconds**) or a non-increasing capture timestamp starts a new streak. Missing samples within that tolerance are tolerated; continuity means consecutive analyzed samples, not proof of every intervening frame.
- Gemini receives the flagged snapshot, task description, crop, and local duration context. It independently judges visible behavior in that still; it cannot verify the full five-second episode.
- Only Gemini decision `inattentive` with reported confidence ≥ **0.8** creates an alert. Uncertain/negative reviews and API errors do not. Reviews are rate limited; confirmed events have a **30-second** cooldown.
- SQLite preserves confirmed alerts during network outages. The background worker retries MongoDB uploads without blocking capture. Normal observations are not sent to the hub.

Benchmark on your actual Pi: no Pi timing has been validated here. If analysis takes longer than the sample interval, samples are skipped rather than spawning unbounded threads. Long gaps prevent the duration gate from firing. Reduce capture resolution before collecting a new dataset or choose a faster device if needed.

Evidence currently uses **the H.264 recording segment containing the flag**, not a precisely trimmed five-second clip. The default segment length is 30 seconds; uploads attach video once that segment closes. An event near a segment boundary can have its onset in the preceding segment; cross-segment pre-roll is not implemented. The alert snapshot and exact timestamp/offset are available independently.

## 8. Start the central alert dashboard

On the central computer:

```sh
python -m pip install -e '.[mongo]'
# Set MONGODB_URI to the same database deployment used by the Pi.
stationwatch --config config.hub.example.json hub
```

Open **http://127.0.0.1:8000**. The hub needs no webcam, Gemini key, or ML weights. MongoDB stores alert documents in `alerts` and evidence in GridFS. The dashboard shows confirmed alerts and their evidence; it does not receive normal footage. See README for authenticated network access.

## Optional pose alternative

The earlier MoveNet path is retained. Use a separate dataset:

```sh
python training_workflow.py --dataset data/pose_dataset init --config config.pi.example.json --backend pose --download-model
```

Record/import with that dataset, use the annotation tool to set the machine point and workstation polygon, then **Save calibration**, label intervals, extract, and train. It exports `models/attention.joblib` and a matching config. This trains a classifier from body/head/position/pose-dwell features rather than pixels. It has different failure modes; the new image workflow does not require it.

## Implementation files and verification

- `training_workflow.py` / `stationwatch/dataset.py`: recording, datasets, annotation entry point, command dispatch.
- `stationwatch/image_training.py`: visual extraction, group-separated fitting/evaluation, JSON/config export.
- `stationwatch/image_model.py`: verified pretrained encoder, preprocessing, Pi classifier inference.
- `stationwatch/temporal.py` / `stationwatch/pipeline.py`: duration gate, bounded inference/review workers.
- `stationwatch/mongo.py`: confirmed-alert and evidence uploads; central store.

Tests exercise generated video, human interval formats, group isolation, portable inference, timer resets, mocked Gemini, and mocked MongoDB. A real pretrained encoder can be smoke-tested with `STATIONWATCH_TEST_ENCODER=models/mobilenet_v3_small.tflite pytest -q`. Synthetic fixtures test software mechanics, not attention accuracy. Real worker footage, the actual Pi, Gemini credentials, and a MongoDB deployment are still needed for end-to-end validation. This prototype is not a validated machinery safety control.
