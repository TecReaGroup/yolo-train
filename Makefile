UV ?= uv

.PHONY: install train app camera

install:
	$(UV) sync --locked

train:
	$(UV) run --no-sync python -m yolo_train train

app:
	$(UV) run --no-sync python -m yolo_train app

camera:
	$(UV) run --no-sync python -m yolo_train camera
