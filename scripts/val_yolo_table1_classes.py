from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from yolov11_cshc import register_cshc_modules
except Exception:
    register_cshc_modules = None


def model_complexity(model: YOLO, imgsz: int) -> tuple[float, float | None]:
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        flops = get_flops(model.model, imgsz=imgsz)
        if flops is not None:
            flops = float(flops)
            gflops = flops / 1e9 if flops > 1e6 else flops
    except Exception:
        gflops = None
    return params_m, gflops


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO and export paper Table-1 style class AP results")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="datasets/marjan_balance.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", required=True)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    if register_cshc_modules is not None:
        register_cshc_modules()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="test",
        plots=False,
        verbose=True,
    )

    names = metrics.names
    ap50 = metrics.box.ap50
    class_ap = {names[i]: float(ap50[i]) * 100 for i in range(len(ap50))}
    params_m, gflops = model_complexity(model, args.imgsz)
    speed = metrics.speed
    total_ms = float(speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0))
    fps = 1000.0 / total_ms if total_ms > 0 else None

    row = {
        "模型": args.name,
        "Healthy coral AP@0.5/%": round(class_ap.get("Healthy Coral", 0.0), 1),
        "Bleached coral AP@0.5/%": round(class_ap.get("Bleached Coral", 0.0), 1),
        "Dead coral AP@0.5/%": round(class_ap.get("Dead Coral", 0.0), 1),
        "GFLOPS": round(gflops, 1) if gflops is not None else None,
        "FPS": round(fps, 1) if fps is not None else None,
        "mAP@0.5/%": round(float(metrics.box.map50) * 100, 1),
        "mAP@0.5-0.95/%": round(float(metrics.box.map) * 100, 1),
        "Precision/%": round(float(metrics.box.mp) * 100, 1),
        "Recall/%": round(float(metrics.box.mr) * 100, 1),
        "Params/M": round(params_m, 2),
    }

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "paper_table1_three_classes.csv"
    json_path = save_dir / "paper_table1_three_classes.json"
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

    table_cols = [
        "模型",
        "Healthy coral AP@0.5/%",
        "Bleached coral AP@0.5/%",
        "Dead coral AP@0.5/%",
        "GFLOPS",
        "FPS",
        "mAP@0.5/%",
    ]
    print(pd.DataFrame([row])[table_cols].to_markdown(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

