UV ?= uv
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu128

.PHONY: install run yolov8n

install:
	$(UV) sync
	$(UV) pip install --upgrade --reinstall-package torch --reinstall-package torchvision torch torchvision --index-url $(TORCH_INDEX_URL)

run: yolov8n

yolov8n:
	$(UV) run --no-sync python -m yolo_train yolov8n
