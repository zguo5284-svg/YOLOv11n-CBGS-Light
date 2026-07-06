from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from yolov11_cshc import register_cshc_modules


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with YOLOv11n-CBGS-Light.")
    parser.add_argument("--weights", default="weights/yolov11n_cbgs_light_epoch150_77p8.pt")
    parser.add_argument("--source", required=True, help="Image, folder, video, or webcam index.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="yolov11n_cbgs_light")
    args = parser.parse_args()

    register_cshc_modules()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()
