"""Read local YOLO detection annotations without modifying source files."""

import math
from dataclasses import dataclass
from pathlib import Path

from yolo_train.config import PROJECT_DIRECTORY

INPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "input"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class Detection:
    """A validated normalized YOLO bounding box."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class Dataset:
    """One input directory containing images and class names."""

    directory: Path
    names: tuple[str, ...]
    images: tuple[Path, ...]

    @classmethod
    def open(cls, directory: Path) -> "Dataset":
        """Index images and validate the class-name file."""
        names = tuple(
            line.strip()
            for line in (directory / "labels" / "labels.txt").read_text(
                encoding="utf-8-sig"
            ).splitlines()
        )
        if not names or any(not name for name in names):
            raise ValueError("labels/labels.txt 必须每行包含一个类别名称")
        images = tuple(sorted(
            image for image in (directory / "images").rglob("*")
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
        ))
        return cls(directory, names, images)

    def read_annotations(self, image: Path) -> tuple[list[Detection], str]:
        """Read boxes, reporting missing files and rejecting malformed labels."""
        relative = image.relative_to(self.directory / "images")
        label = (self.directory / "labels" / relative).with_suffix(".txt")
        if not label.exists():
            return [], "缺少标注文件"
        boxes = []
        for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                class_text, x_text, y_text, width_text, height_text = line.split()
                class_id = int(class_text)
                x, y, width, height = map(float, (x_text, y_text, width_text, height_text))
                if not (
                    0 <= class_id < len(self.names)
                    and all(math.isfinite(value) for value in (x, y, width, height))
                    and 0 <= x <= 1 and 0 <= y <= 1
                    and 0 < width <= 1 and 0 < height <= 1
                ):
                    raise ValueError("invalid class or coordinates")
            except ValueError as error:
                raise ValueError(f"{label.name} 第 {number} 行: 无效的 YOLO 检测框") from error
            boxes.append(Detection(class_id, x, y, width, height))
        return boxes, f"{len(boxes)} 个标注" if boxes else "空标注 (背景图片)"
