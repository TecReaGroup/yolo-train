"""Command-line entry point."""

import argparse
from collections.abc import Sequence

from yolo_train import __version__
from yolo_train.config import Settings
from yolo_train.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(prog="yolo-train")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("command", choices=("train", "app", "camera", "yolov8n"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the application."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    settings = Settings.from_env()
    logger = configure_logging(settings.log_level)
    logger.info("application_started environment=%s", settings.environment)
    try:
        if arguments.command == "app":
            from yolo_train.preview.window import run_preview

            return run_preview()

        if arguments.command == "camera":
            from yolo_train.camera.preview import run_camera

            run_camera()
            return 0

        from yolo_train.training.runner import train_yolov8n

        train_yolov8n()
    except Exception:
        logger.exception("命令执行失败 command=%s", arguments.command)
        return 1
    return 0
