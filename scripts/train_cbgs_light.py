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
    parser = argparse.ArgumentParser(description="Train YOLOv11n-CBGS-Light.")
    parser.add_argument("--model", default="models/yolo11n_cbgs.yaml")
    parser.add_argument("--data", default="configs/marjan_balance.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="yolov11n_cbgs_light_seed2026")
    args = parser.parse_args()

    register_cshc_modules()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        project=args.project,
        name=args.name,
        mosaic=0.8,
        close_mosaic=100,
        hsv_h=0.003,
        hsv_s=0.25,
        hsv_v=0.15,
        scale=0.30,
        translate=0.05,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()
