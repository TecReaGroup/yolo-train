"""Train one YOLOv8n detector per input dataset."""

import json
import logging
import os
import random
import shutil
import tempfile
import tomllib
from contextlib import chdir
from pathlib import Path
from typing import Any

from yolo_train.config import CONFIG_PATH, PROJECT_DIRECTORY

INPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "input"
TEMP_DIRECTORY = PROJECT_DIRECTORY / "temp"
RUNTIME_DIRECTORY = TEMP_DIRECTORY / "runtime"
CONFIG_DIRECTORY = TEMP_DIRECTORY / "config"
MODEL_PATH = TEMP_DIRECTORY / "model" / "yolov8n.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger("training")


def prepare_dataset(source: Path, workspace: Path, fraction: float, seed: int) -> Path:
    """Stage annotated images and create disjoint, reproducible training splits."""
    names = (source / "labels" / "labels.txt").read_text(encoding="utf-8-sig").splitlines()
    names = [name.strip() for name in names]
    if not names or any(not name for name in names):
        raise ValueError(f"{source}: labels/labels.txt must contain one class per line")
    images = sorted(
        image for image in (source / "images").rglob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
    )
    annotated = [
        image for image in images
        if (source / "labels" / image.relative_to(source / "images")).with_suffix(".txt").is_file()
    ]
    if len(annotated) < 2:
        raise ValueError(f"{source}: at least two images with matching labels are required")
    LOGGER.info("数据集=%s 已标注=%d 跳过缺少标注=%d", source, len(annotated), len(images) - len(annotated))
    staged_images = []
    for image in annotated:
        relative = image.relative_to(source / "images")
        staged_image = workspace / "images" / relative
        staged_label = (workspace / "labels" / relative).with_suffix(".txt")
        staged_image.parent.mkdir(parents=True, exist_ok=True)
        staged_label.parent.mkdir(parents=True, exist_ok=True)
        try:
            staged_image.hardlink_to(image)
        except OSError:
            shutil.copy2(image, staged_image)
        shutil.copy2((source / "labels" / relative).with_suffix(".txt"), staged_label)
        staged_images.append(staged_image.as_posix())
    random.Random(seed).shuffle(staged_images)
    validation_count = max(1, min(len(staged_images) - 1, round(len(staged_images) * fraction)))
    for split, paths in (
        ("val", staged_images[:validation_count]),
        ("train", staged_images[validation_count:]),
    ):
        (workspace / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    specification = workspace / "dataset.yaml"
    # JSON is valid YAML and safely preserves Unicode class names and Windows paths.
    specification.write_text(json.dumps({
        "path": workspace.as_posix(),
        "train": (workspace / "train.txt").as_posix(),
        "val": (workspace / "val.txt").as_posix(),
        "names": names,
    }, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("训练集=%d 验证集=%d", len(staged_images) - validation_count, validation_count)
    return specification


def train_yolov8n() -> None:
    """Train each immediate input directory with fresh pretrained weights."""
    with CONFIG_PATH.open("rb") as config_file:
        training = tomllib.load(config_file)["training"]
    for key in ("epochs", "imgsz", "batch", "workers", "seed"):
        value = training[key]
        minimum = 0 if key in ("workers", "seed") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"training.{key} must be an integer >= {minimum}")
    fraction = training["validation_fraction"]
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("training.validation_fraction must be between 0 and 1")
    datasets = sorted(directory for directory in INPUT_DIRECTORY.iterdir() if directory.is_dir())
    if not datasets:
        raise ValueError("data/input contains no dataset directories")
    for directory in (RUNTIME_DIRECTORY, CONFIG_DIRECTORY, MODEL_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
    # Ultralytics appends its own directory and falls back to CWD if the parent is absent.
    os.environ["YOLO_CONFIG_DIR"] = str(CONFIG_DIRECTORY)
    os.environ["TORCH_HOME"] = str(TEMP_DIRECTORY / "torch")
    os.environ["MPLCONFIGDIR"] = str(TEMP_DIRECTORY / "matplotlib")

    # Third-party checks may download auxiliary models relative to the working directory.
    with chdir(RUNTIME_DIRECTORY):
        train_dataset_collection(datasets, training)


def train_dataset_collection(datasets: list[Path], training: dict[str, Any]) -> None:
    """Own library initialization and training inside the isolated runtime directory."""

    import torch
    from ultralytics import YOLO, settings
    from ultralytics.utils import LOGGER as ultralytics_logger

    ultralytics_logger.handlers.clear()
    ultralytics_logger.propagate = True
    settings.update({
        "datasets_dir": str(TEMP_DIRECTORY / "dataset"),
        "weights_dir": str(MODEL_PATH.parent),
        "runs_dir": str(TEMP_DIRECTORY / "run"),
    })
    LOGGER.info("运行目录=%s 配置目录=%s 权重目录=%s", RUNTIME_DIRECTORY, CONFIG_DIRECTORY, MODEL_PATH.parent)
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        LOGGER.warning("CUDA 不可用，使用 CPU 训练；torch=%s", torch.__version__)
    else:
        LOGGER.info("使用 GPU: %s", torch.cuda.get_device_name(device))
    for source in datasets:
        source = source.resolve()
        LOGGER.info("开始训练 dataset=%s model=yolov8n output=%s", source, source / "train")
        with tempfile.TemporaryDirectory(prefix="yolov8n-", dir=TEMP_DIRECTORY.resolve()) as temporary:
            specification = prepare_dataset(
                source, Path(temporary), training["validation_fraction"], training["seed"]
            )
            model = YOLO(str(MODEL_PATH.resolve()))
            model.train(
                data=str(specification),
                device=device,
                project=str(source),
                name="train",
                exist_ok=True,
                epochs=training["epochs"],
                imgsz=training["imgsz"],
                batch=training["batch"],
                workers=training["workers"],
                seed=training["seed"],
            )
        LOGGER.info("训练完成 dataset=%s best=%s", source, source / "train/weights/best.pt")
    LOGGER.info("全部训练完成 count=%d", len(datasets))
