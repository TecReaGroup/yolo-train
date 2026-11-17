"""Preview YOLO detections from a local camera."""

import logging
import os
import tomllib
from contextlib import chdir
from pathlib import Path

from yolo_train.config import CONFIG_PATH, PROJECT_DIRECTORY

LOGGER = logging.getLogger("camera")
TEMP_DIRECTORY = PROJECT_DIRECTORY / "temp"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
WINDOW_TITLE = "YOLO Camera - Q / Esc to quit"


def run_camera() -> None:
    """Load camera settings and preview detections until the window is closed."""
    with CONFIG_PATH.open("rb") as config_file:
        camera = tomllib.load(config_file)["camera"]
    for key, minimum in (("index", 0), ("imgsz", 1)):
        if type(camera[key]) is not int or camera[key] < minimum:
            raise ValueError(f"camera.{key} must be an integer >= {minimum}")
    confidence = camera["confidence"]
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise ValueError("camera.confidence must be between 0 and 1")
    if not isinstance(camera["model"], str) or not camera["model"].strip():
        raise ValueError("camera.model must be a non-empty .pt path")
    model_path = (PROJECT_DIRECTORY / Path(camera["model"])).resolve()
    if model_path.suffix.lower() != ".pt":
        raise ValueError("camera.model must be a .pt path")
    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    runtime_directory = TEMP_DIRECTORY / "runtime"
    config_directory = TEMP_DIRECTORY / "config"
    for directory in (runtime_directory, config_directory):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
    os.environ["TORCH_HOME"] = str(TEMP_DIRECTORY / "torch")
    os.environ["MPLCONFIGDIR"] = str(TEMP_DIRECTORY / "matplotlib")

    # Keep third-party downloads and runtime artifacts inside the project temp directory.
    with chdir(runtime_directory):
        preview_camera(model_path, camera["index"], camera["imgsz"], confidence)


def preview_camera(model_path: Path, index: int, imgsz: int, confidence: float) -> None:
    """Own model initialization, capture, rendering, and resource cleanup."""
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER as ultralytics_logger

    ultralytics_logger.handlers.clear()
    ultralytics_logger.propagate = True
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        LOGGER.warning("CUDA 不可用，使用 CPU 推理；torch=%s", torch.__version__)
    else:
        LOGGER.info("使用 GPU: %s", torch.cuda.get_device_name(device))
    LOGGER.info("加载模型=%s 摄像头=%d", model_path, index)
    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(index)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"无法打开摄像头 {index}，请检查设备编号、权限或占用情况")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        LOGGER.info("摄像头采集分辨率=%dx%d", actual_width, actual_height)
        if (actual_width, actual_height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
            LOGGER.warning(
                "摄像头未采用请求的 %dx%d，实际采集=%dx%d",
                CAMERA_WIDTH, CAMERA_HEIGHT, actual_width, actual_height,
            )
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, CAMERA_WIDTH, CAMERA_HEIGHT)
        LOGGER.info("实时预览已启动，按 Q、Esc 或关闭窗口退出")
        while True:
            captured, frame = capture.read()
            if not captured:
                raise RuntimeError(f"摄像头 {index} 读取失败，请检查设备连接")
            prediction = model.predict(
                frame, device=device, imgsz=imgsz, conf=confidence, verbose=False, save=False,
            )[0]
            cv2.imshow(WINDOW_TITLE, prediction.plot())
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        LOGGER.info("收到中断，停止实时预览")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        LOGGER.info("摄像头已释放，实时预览结束")
