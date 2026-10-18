"""Train MobileNet from annotated video intervals or paired clips, or train from legacy pose CSVs."""

import argparse
from pathlib import Path

import cv2

from stationwatch.dataset import (
    clips,
    dataset_settings,
    download_model,
    extract,
    import_training_videos,
    initialize,
    train_dataset,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--annotated", action="store_true", help="Train from saved annotator intervals in --dataset; no video import"
    )
    source.add_argument("--videos", type=Path, help="Numbered name1.mp4/name2.mp4 clips or legacy behavior pairs")
    source.add_argument("--csv", type=Path, help="Use the legacy pose-feature trainer instead")
    parser.add_argument("--output", type=Path, help="Default: models/attention_image.json (or attention.joblib)")
    parser.add_argument("--threshold", type=float, default=0.8)
    image = parser.add_argument_group("image training")
    image.add_argument("--dataset", type=Path, default=Path("data/training_dataset"))
    image.add_argument("--config", type=Path, default=Path("config.pi.example.json"))
    image.add_argument("--download-model", action="store_true", help="Download/verify the pretrained MobileNet")
    image.add_argument("--session", default="s01")
    image.add_argument("--trim-start-s", type=float, default=0, help="Opening seconds to leave unlabeled")
    image.add_argument("--trim-end-s", type=float, default=0, help="Closing seconds to leave unlabeled")
    image.add_argument("--holdout", nargs="+", help="People excluded from fitting; default: reproducible split")
    parser.add_argument("--algorithm", choices=["random-forest", "logistic"], help="Legacy CSV training only")
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.threshold <= 1:
            raise ValueError("Threshold must be in [0,1]")
        if args.csv is not None:
            from stationwatch.training import train

            return train(
                args.csv,
                args.output or Path("models/attention.joblib"),
                args.algorithm or "random-forest",
                args.threshold,
            )
        if args.algorithm:
            raise ValueError("--algorithm is for --csv; video training uses a frozen MobileNet and logistic head")
        output = args.output or Path("models/attention_image.json")
        if output.suffix != ".json":
            raise ValueError("Image training requires a .json output")
        if args.annotated and not (args.dataset / "config.json").is_file():
            raise ValueError("Initialize/import videos with training_workflow.py, then save annotator intervals first")
        if args.videos is not None and not args.videos.is_dir():
            raise ValueError("--videos must contain numbered name1.mp4/name2.mp4 clips or legacy behavior pairs")
        if not (args.dataset / "config.json").exists():
            initialize(args.dataset, args.config, backend="image")
        settings = dataset_settings(args.dataset)
        if settings.analysis_backend != "image":
            raise ValueError("Choose a separate --dataset for MobileNet image training; this dataset uses pose")
        if args.videos is not None:
            import_training_videos(args.dataset, args.videos, args.session, args.trim_start_s, args.trim_end_s)
        people = {info["person"] for info in clips(args.dataset)}
        if len(people) < 3:
            raise ValueError("Provide annotated footage from at least three people, with both labels in each split")
        if args.holdout and (set(args.holdout) - people or not people - set(args.holdout)):
            raise ValueError("--holdout must name existing people and leave people available for training")
        if args.download_model:
            download_model(settings)
        elif not settings.image_model_path.is_file():
            raise ValueError("MobileNet encoder is missing; rerun with --download-model")
        # Re-extract saved intervals, including both labels within a clip and unlabeled gaps.
        # Keep every clip/session from a person in the same split.
        extract(args.dataset, group_by="person")
        return train_dataset(args.dataset, output, args.threshold, args.holdout)
    except (ValueError, FileNotFoundError, OSError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
