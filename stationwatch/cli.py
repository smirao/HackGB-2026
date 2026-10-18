import argparse
import logging
import os
from pathlib import Path

from .config import load_settings


def main():
    parser = argparse.ArgumentParser(description="StationWatch workstation observation prototype")
    parser.add_argument("--config", type=Path, default=Path("config.example.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run camera pipeline and manager dashboard")
    mode = serve.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Synthetic camera and reviews; no external APIs")
    mode.add_argument("--baseline", action="store_true", help="Use an explicit orientation/dwell heuristic")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    capture = sub.add_parser("collect", help="Append labeled feature samples to a training CSV")
    capture.add_argument("--output", type=Path, default=Path("data/training.csv"))
    capture.add_argument("--label", type=int, choices=[0, 1], required=True)
    capture.add_argument(
        "--group", required=True, help="Worker/session group kept together during evaluation"
    )
    capture.add_argument("--seconds", type=float, default=30)
    fit = sub.add_parser("train", help="Fit a classifier using a group-isolated evaluation split")
    fit.add_argument("--csv", type=Path, default=Path("data/training.csv"))
    fit.add_argument("--output", type=Path, default=Path("models/attention.joblib"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    if args.command == "train":
        from .training import train

        train(args.csv, args.output)
        return
    if args.command == "collect":
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        from .training import collect

        collect(settings, args.output, args.label, args.group, args.seconds)
        return
    token = os.environ.get("DASHBOARD_TOKEN")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and (not token or len(token) < 24):
        parser.error("Network access requires a DASHBOARD_TOKEN of at least 24 characters")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not args.demo and not api_key:
        parser.error("Set GEMINI_API_KEY for live review, or use --demo")
    if not args.demo and not settings.pose_model_path.is_file():
        parser.error(
            f"Download the {settings.pose_backend} pose model to {settings.pose_model_path}; see README"
        )
    if not args.demo and not args.baseline and not settings.classifier_path.is_file():
        parser.error("No trained classifier found. Start with --baseline or train a classifier; see README")

    import uvicorn

    from .app import create_app
    from .camera import create_camera
    from .features import DemoExtractor, create_extractor
    from .gemini import DemoReviewer, GeminiReviewer
    from .model import BaselineModel, TrainedModel
    from .pipeline import Pipeline
    from .recording import SegmentRecorder
    from .storage import EventStore

    # Synthetic records always live in a separate database and media directory.
    if args.demo:
        settings = settings.model_copy(update={"data_dir": settings.data_dir / "demo"})
    store = EventStore(settings.data_dir)
    resources = []
    try:
        recorder = SegmentRecorder(settings, store)
        resources.append(recorder)
        extractor = DemoExtractor() if args.demo else create_extractor(settings)
        resources.append(extractor)
        model = BaselineModel(settings) if args.demo or args.baseline else TrainedModel(settings)
        reviewer = DemoReviewer() if args.demo else GeminiReviewer(settings, api_key)
        resources.append(reviewer)
        camera = create_camera(settings, demo=args.demo)
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
        "demo" if args.demo else "baseline" if args.baseline else "trained",
    )
    uvicorn.run(
        create_app(pipeline, store, token),
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
