import argparse
import logging
import os
import signal
import threading
from pathlib import Path

from .config import load_settings


def run_agent(pipeline, uploader, store):
    """No web server on the Pi; confirmed events leave through the MongoDB uploader."""
    stop = threading.Event()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: stop.set())
    failed = False
    try:
        pipeline.start()
        uploader.start()
        while not stop.wait(0.5):
            if not pipeline.status()["running"]:
                failed = True
                break
    finally:
        pipeline.close()
        uploader.close()
        store.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return failed


def main():
    parser = argparse.ArgumentParser(description="StationWatch Pi analysis and central alert hub")
    parser.add_argument("--config", type=Path, default=Path("config.example.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run analysis with a local alert dashboard")
    mode = serve.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Synthetic camera and reviews; no external APIs")
    mode.add_argument("--baseline", action="store_true", help="Use an explicit orientation/dwell heuristic")
    serve.add_argument("--mongo", action="store_true", help="Also upload confirmed evidence to MongoDB")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    agent = sub.add_parser("agent", help="Run Pi capture/ML/Gemini and upload confirmed alerts to MongoDB")
    agent.add_argument(
        "--baseline", action="store_true", help="Use orientation rules before training a classifier"
    )
    hub = sub.add_parser("hub", help="Show MongoDB alerts on the central computer; no camera or ML needed")
    hub.add_argument("--host", default="127.0.0.1")
    hub.add_argument("--port", type=int, default=8000)
    capture = sub.add_parser("collect", help="Append labeled feature samples to a training CSV")
    capture.add_argument("--output", type=Path, default=Path("data/training.csv"))
    capture.add_argument("--label", type=int, choices=[0, 1], required=True)
    capture.add_argument(
        "--group", required=True, help="Worker/session group kept together during evaluation"
    )
    capture.add_argument("--seconds", type=float, default=30)
    fit = sub.add_parser(
        "train", help="Create a model artifact from labeled body/head/position/dwell features"
    )
    fit.add_argument("--csv", type=Path, default=Path("data/training.csv"))
    fit.add_argument("--output", type=Path, default=Path("models/attention.joblib"))
    fit.add_argument("--algorithm", choices=["random-forest", "logistic"], default="random-forest")
    fit.add_argument("--threshold", type=float, help="Evaluate at this runtime cutoff; defaults to config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    if args.command == "train":
        from .training import train

        train(
            args.csv,
            args.output,
            args.algorithm,
            settings.local_threshold if args.threshold is None else args.threshold,
        )
        return
    if args.command == "collect":
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        from .training import collect

        collect(settings, args.output, args.label, args.group, args.seconds)
        return

    token = os.environ.get("DASHBOARD_TOKEN")
    if (
        args.command != "agent"
        and args.host not in {"127.0.0.1", "localhost", "::1"}
        and (not token or len(token) < 24)
    ):
        parser.error("Network dashboard access requires DASHBOARD_TOKEN of at least 24 characters")
    mongo_enabled = args.command in {"agent", "hub"} or getattr(args, "mongo", False)
    remote = None
    if mongo_enabled:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            parser.error("Set MONGODB_URI on this computer to connect to your MongoDB deployment")
        if getattr(args, "demo", False):
            parser.error("Synthetic demo evidence cannot be uploaded; omit --mongo when using --demo")
        try:
            from .mongo import MongoEventStore
        except ImportError:
            parser.error("Install MongoDB support with: python -m pip install -e '.[mongo]'")
        try:
            remote = MongoEventStore.from_uri(uri, settings)
        except Exception:  # noqa: BLE001 — database configuration errors must not expose credentials.
            parser.error("Invalid MongoDB configuration. Check MONGODB_URI and mongo_database.")
    if args.command == "hub":
        import uvicorn

        from .app import create_app

        try:
            remote.initialize_hub()
        except Exception:  # noqa: BLE001 — never print connection strings from driver errors.
            remote.close()
            parser.error("Cannot reach MongoDB. Check the connection, network access and credentials.")
        uvicorn.run(
            create_app(None, remote, token),
            host=args.host,
            port=args.port,
            workers=1,
            access_log=False,
            timeout_graceful_shutdown=3,
        )
        return

    demo = getattr(args, "demo", False)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not demo and not api_key:
        parser.error("Set GEMINI_API_KEY on the Pi for second-stage review, or use serve --demo")
    image_mode = not demo and not args.baseline and settings.analysis_backend == "image"
    if image_mode and not settings.image_model_path.is_file():
        parser.error("Download the MobileNet encoder with training_workflow.py init --download-model")
    if not demo and not image_mode and not settings.pose_model_path.is_file():
        parser.error(
            f"Download the {settings.pose_backend} pose model to {settings.pose_model_path}; see README"
        )
    if not demo and not args.baseline and not settings.classifier_path.is_file():
        parser.error(
            "No trained classifier found. Follow TRAINING_GUIDE.md and run training_workflow.py train"
        )

    from .camera import create_camera
    from .features import DemoExtractor, create_extractor
    from .gemini import DemoReviewer, GeminiReviewer
    from .model import BaselineModel, TrainedModel
    from .pipeline import Pipeline
    from .recording import SegmentRecorder
    from .storage import EventStore

    if demo:
        settings = settings.model_copy(update={"data_dir": settings.data_dir / "demo", "flag_duration_s": 0})
    store = EventStore(settings.data_dir)
    resources = [remote] if remote is not None else []
    try:
        recorder = SegmentRecorder(settings, store)
        resources.append(recorder)
        if image_mode:
            from .image_model import ImageExtractor, ImageModel

            extractor = ImageExtractor(settings)
        else:
            extractor = DemoExtractor() if demo else create_extractor(settings)
        resources.append(extractor)
        model = (
            ImageModel(settings)
            if image_mode
            else BaselineModel(settings)
            if demo or args.baseline
            else TrainedModel(settings)
        )
        reviewer = DemoReviewer() if demo else GeminiReviewer(settings, api_key)
        resources.append(reviewer)
        camera = create_camera(settings, demo=demo)
        resources.append(camera)
    except Exception:
        for resource in reversed(resources):
            resource.close()
        store.close()
        raise
    pipeline = Pipeline(
        settings,
        camera,
        extractor,
        model,
        reviewer,
        recorder,
        store,
        "demo" if demo else "image" if image_mode else "baseline" if args.baseline else "trained",
    )
    uploader = None
    if remote is not None:
        from .mongo import MongoUploader

        uploader = MongoUploader(store, remote, settings.mongo_poll_s)
    if args.command == "agent":
        if run_agent(pipeline, uploader, store):
            parser.exit(1, "Camera/recording stopped; inspect the Pi logs before restarting.\n")
        return

    import uvicorn

    from .app import create_app

    uvicorn.run(
        create_app(pipeline, store, token, uploader),
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
