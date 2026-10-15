"""Allow the package to run with ``python -m yolo_train``."""

from yolo_train.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
