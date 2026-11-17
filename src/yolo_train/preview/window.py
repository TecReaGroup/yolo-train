"""Minimal desktop browser for local annotated images."""

import logging
from pathlib import Path
from typing import override

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from yolo_train.preview.dataset import INPUT_DIRECTORY, Dataset, Detection

LOGGER = logging.getLogger("preview")
CLASS_COLORS = ("#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2")
STYLE = """
QMainWindow, QWidget#sidebar { background: #f8fafc; }
QWidget { font-family: 'Segoe UI', 'Microsoft YaHei UI'; font-size: 13px; color: #334155; }
QLabel#title { font-size: 22px; font-weight: 600; color: #0f172a; }
QLabel#hint { color: #64748b; }
QLineEdit, QComboBox, QListWidget {
    background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 7px;
}
QListWidget::item { padding: 8px 4px; }
QListWidget::item:selected { background: #dbeafe; color: #1d4ed8; border-radius: 4px; }
QPushButton { background: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #eff6ff; border-color: #93c5fd; }
QPushButton:disabled { color: #94a3b8; background: #f1f5f9; }
QSplitter::handle { background: #e2e8f0; }
"""


class ImageCanvas(QGraphicsView):
    """Display an image with independently toggleable annotation overlays."""

    def __init__(self) -> None:
        super().__init__()
        self.canvas_scene = QGraphicsScene(self)
        self.setScene(self.canvas_scene)
        self.overlays: list[QGraphicsItem] = []
        self.fitted = True
        self.setBackgroundBrush(QColor("#e9eef4"))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def clear_image(self) -> None:
        """Release the current image and all overlays."""
        self.overlays.clear()
        self.canvas_scene.clear()
        self.canvas_scene.setSceneRect(0, 0, 1, 1)
        self.resetTransform()

    def show_image(self, pixmap: QPixmap, boxes: list[Detection], names: tuple[str, ...]) -> None:
        """Render normalized annotations in the original image coordinate system."""
        self.clear_image()
        self.canvas_scene.addPixmap(pixmap)
        width, height = pixmap.width(), pixmap.height()
        self.canvas_scene.setSceneRect(0, 0, width, height)
        for box in boxes:
            color = QColor(CLASS_COLORS[box.class_id % len(CLASS_COLORS)])
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            left = (box.center_x - box.width / 2) * width
            top = (box.center_y - box.height / 2) * height
            rectangle = self.canvas_scene.addRect(
                left, top, box.width * width, box.height * height, pen
            )
            caption = self.canvas_scene.addSimpleText(names[box.class_id])
            caption.setBrush(color)
            caption.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            caption.setPos(max(0, left), max(0, top))
            self.overlays.extend((rectangle, caption))
        self.fit_image()

    def set_annotations_visible(self, visible: bool) -> None:
        """Toggle annotation overlays without reloading the image."""
        for item in self.overlays:
            item.setVisible(visible)

    def fit_image(self) -> None:
        """Fit the image while preserving its aspect ratio."""
        self.fitted = True
        self.fitInView(self.canvas_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Preserve fit-to-window mode when the viewport changes."""
        super().resizeEvent(event)
        if self.fitted:
            self.fit_image()

    @override
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom around the pointer within usable scale limits."""
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.02 <= self.transform().m11() * factor <= 30:
            self.fitted = False
            self.scale(factor, factor)
        event.accept()


class PreviewWindow(QMainWindow):
    """Browse datasets, images, and their detection annotations."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset: Dataset | None = None
        self.visible_images: list[Path] = []
        self.setWindowTitle("标注预览")
        self.resize(1280, 820)
        self.setMinimumSize(880, 560)
        self.setStyleSheet(STYLE)

        self.dataset_choice = QComboBox()
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索图片名称…")
        self.image_list = QListWidget()
        self.canvas = ImageCanvas()
        self.details = QLabel("请选择数据集")
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.TextFormat.PlainText)
        self.show_boxes = QCheckBox("显示标注")
        self.show_boxes.setChecked(True)
        self.previous = QPushButton("上一张")
        self.next_image = QPushButton("下一张")
        fit_button = QPushButton("适应窗口")
        refresh_button = QPushButton("刷新")

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(12)
        title = QLabel("标注预览")
        title.setObjectName("title")
        hint = QLabel("data/input · 只读浏览")
        hint.setObjectName("hint")
        for widget in (title, hint, self.dataset_choice, self.search, self.image_list, self.legend):
            sidebar_layout.addWidget(widget)
        sidebar_layout.addWidget(refresh_button)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(16, 20, 20, 16)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.previous)
        toolbar.addWidget(self.next_image)
        toolbar.addStretch()
        toolbar.addWidget(self.show_boxes)
        toolbar.addWidget(fit_button)
        preview_layout.addLayout(toolbar)
        preview_layout.addWidget(self.canvas, 1)
        preview_layout.addWidget(self.details)
        splitter = QSplitter()
        splitter.addWidget(sidebar)
        splitter.addWidget(preview)
        splitter.setSizes([300, 980])
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("滚轮缩放 · 拖动平移 · ← / → 切换图片 · F 适应窗口")

        self.dataset_choice.currentIndexChanged.connect(self.select_dataset)
        self.search.textChanged.connect(self.filter_images)
        self.image_list.currentRowChanged.connect(self.select_image)
        self.show_boxes.toggled.connect(self.canvas.set_annotations_visible)
        self.previous.clicked.connect(self.previous_image)
        self.next_image.clicked.connect(self.advance_image)
        fit_button.clicked.connect(self.canvas.fit_image)
        refresh_button.clicked.connect(self.refresh_datasets)
        for key, callback in (
            ("Left", self.previous_image), ("Right", self.advance_image),
            ("F", self.canvas.fit_image),
        ):
            shortcut = QShortcut(QKeySequence(key), self.canvas)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(callback)
        self.refresh_datasets()

    def refresh_datasets(self) -> None:
        """Rescan the input directory while retaining the selected dataset."""
        selected = self.dataset_choice.currentText()
        self.dataset_choice.blockSignals(True)
        self.dataset_choice.clear()
        try:
            directories = sorted(
                path for path in INPUT_DIRECTORY.iterdir()
                if path.is_dir() and (path / "images").is_dir()
            )
            for directory in directories:
                self.dataset_choice.addItem(directory.name, directory)
        except OSError:
            LOGGER.exception("无法读取数据目录 %s", INPUT_DIRECTORY)
        index = self.dataset_choice.findText(selected)
        self.dataset_choice.setCurrentIndex(max(0, index))
        self.dataset_choice.blockSignals(False)
        self.select_dataset()

    def select_dataset(self) -> None:
        """Load class names and image paths for the selected dataset."""
        self.dataset = None
        self.legend.clear()
        directory = self.dataset_choice.currentData()
        try:
            if directory is not None:
                self.dataset = Dataset.open(directory)
                self.legend.setText("类别\n" + "\n".join(
                    f"{index} · {name}" for index, name in enumerate(self.dataset.names)
                ))
                LOGGER.info("打开数据集=%s 图片数=%d", directory, len(self.dataset.images))
        except (OSError, ValueError) as error:
            LOGGER.exception("无法打开数据集=%s", directory)
            self.filter_images()
            self.details.setText(f"无法打开数据集: {error}")
            return
        self.filter_images()

    def filter_images(self) -> None:
        """Filter image filenames without reading image contents."""
        query = self.search.text().strip().casefold()
        self.visible_images = [
            image for image in self.dataset.images if query in image.name.casefold()
        ] if self.dataset else []
        self.image_list.blockSignals(True)
        self.image_list.clear()
        if self.dataset:
            self.image_list.addItems([
                str(image.relative_to(self.dataset.directory / "images"))
                for image in self.visible_images
            ])
        self.image_list.blockSignals(False)
        if self.visible_images:
            self.image_list.setCurrentRow(0)
        else:
            self.select_image(-1)

    def select_image(self, index: int) -> None:
        """Display an image and surface annotation errors without stopping browsing."""
        self.previous.setEnabled(index > 0)
        self.next_image.setEnabled(0 <= index < len(self.visible_images) - 1)
        self.canvas.clear_image()
        if index < 0 or self.dataset is None:
            self.details.setText(
                "没有匹配的图片" if self.dataset else
                "未找到数据集, 请在 data/input/<数据集>/ 下放置 images/ 和 labels/labels.txt"
            )
            return
        image = self.visible_images[index]
        pixmap = QPixmap(str(image))
        if pixmap.isNull():
            self.details.setText(f"无法读取图片: {image.name}")
            LOGGER.warning("无法读取图片=%s", image)
            return
        try:
            boxes, annotation_status = self.dataset.read_annotations(image)
        except (OSError, ValueError) as error:
            boxes, annotation_status = [], f"标注错误: {error}"
            LOGGER.exception("无法读取标注 image=%s", image)
        self.canvas.show_image(pixmap, boxes, self.dataset.names)
        self.canvas.set_annotations_visible(self.show_boxes.isChecked())
        self.details.setText(
            f"{index + 1} / {len(self.visible_images)}  ·  {image.name}\n"
            f"{pixmap.width()} x {pixmap.height()}  ·  {annotation_status}"
        )

    def previous_image(self) -> None:
        """Select the previous image when available."""
        self.image_list.setCurrentRow(max(0, self.image_list.currentRow() - 1))

    def advance_image(self) -> None:
        """Select the next image when available."""
        self.image_list.setCurrentRow(min(
            len(self.visible_images) - 1, self.image_list.currentRow() + 1
        ))


def run_preview() -> int:
    """Start the desktop preview application."""
    application = QApplication([])
    application.setStyle("Fusion")
    window = PreviewWindow()
    window.show()
    LOGGER.info("标注预览已启动 input=%s", INPUT_DIRECTORY)
    return application.exec()
