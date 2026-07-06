from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yolov11_cshc


yolov11_cshc.register_cshc_modules()

WEIGHTS = Path("runs/jetson_deploy/yolov11n_cbgs_light_epoch150_77p8.pt")
OUT_ROOT = Path("runs/visual_validation/extra_different_scene")

CLASS_NAMES = {
    0: "Bleached Coral",
    1: "Dead Coral",
    2: "Healthy Coral",
}

COLORS_BGR = {
    0: (0, 165, 255),
    1: (120, 120, 120),
    2: (0, 200, 80),
}


@dataclass
class DatasetConfig:
    key: str
    name: str
    image_dir: Path
    label_dir: Path
    used_csv: Path


@dataclass
class ImageMetric:
    image_path: Path
    label_path: Path
    brightness: float
    texture: float
    object_count: int
    small_count: int
    avg_box_area: float
    class_count: int
    classes: str
    box_background_contrast: float
    score: float


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        boxes.append((cid, x, y, w, h))
    return boxes


def used_stems(csv_path: Path) -> set[str]:
    stems: set[str] = set()
    if not csv_path.exists():
        return stems
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_path = row.get("image_path", "")
            if image_path:
                stems.add(Path(image_path).stem)
    return stems


def compute_metric(image_path: Path, label_dir: Path) -> ImageMetric | None:
    label_path = label_dir / f"{image_path.stem}.txt"
    boxes = read_labels(label_path)
    if not boxes:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean() / 255.0)
    texture = float(cv2.Laplacian(gray, cv2.CV_32F).std() / 255.0)

    areas = [w * h for _, _, _, w, h in boxes]
    small_count = sum(1 for area in areas if area < 0.025)
    class_ids = sorted({cid for cid, *_ in boxes})
    classes = "; ".join(CLASS_NAMES.get(cid, str(cid)) for cid in class_ids)

    global_mean = float(gray.mean())
    contrasts: list[float] = []
    for _, x, y, bw, bh in boxes:
        x1 = max(0, int((x - bw / 2) * w_img))
        y1 = max(0, int((y - bh / 2) * h_img))
        x2 = min(w_img, int((x + bw / 2) * w_img))
        y2 = min(h_img, int((y + bh / 2) * h_img))
        crop = gray[y1:y2, x1:x2]
        if crop.size:
            contrasts.append(abs(float(crop.mean()) - global_mean) / 255.0)

    avg_box_area = float(np.mean(areas))
    object_count = len(boxes)
    # Different from previous five scenes: prefer clear close-up views with few,
    # larger targets and relatively simple background.
    score = (
        2.8 * avg_box_area
        - 0.18 * object_count
        - 0.65 * texture
        - 0.25 * small_count
        - 0.35 * abs(brightness - 0.42)
        - 0.20 * max(0, len(class_ids) - 1)
    )

    return ImageMetric(
        image_path=image_path,
        label_path=label_path,
        brightness=brightness,
        texture=texture,
        object_count=object_count,
        small_count=small_count,
        avg_box_area=avg_box_area,
        class_count=len(class_ids),
        classes=classes,
        box_background_contrast=float(np.mean(contrasts)) if contrasts else 0.0,
        score=score,
    )


def select_extra_scene(config: DatasetConfig) -> ImageMetric:
    used = used_stems(config.used_csv)
    candidates: list[ImageMetric] = []
    for image_path in sorted(config.image_dir.glob("*.*")):
        if image_path.stem in used:
            continue
        metric = compute_metric(image_path, config.label_dir)
        if metric is not None:
            candidates.append(metric)

    if not candidates:
        raise RuntimeError(f"No candidate images found for {config.name}")

    # Prefer 1-3 objects and medium-large boxes. Exclude near-full-image boxes
    # because they are less informative for paper visualization.
    closeup = [m for m in candidates if 1 <= m.object_count <= 3 and 0.08 <= m.avg_box_area <= 0.65]
    pool = closeup or candidates
    return sorted(pool, key=lambda m: m.score, reverse=True)[0]


def draw_predictions(model: YOLO, metric: ImageMetric, out_path: Path) -> int:
    image = cv2.imread(str(metric.image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {metric.image_path}")

    result = model.predict(str(metric.image_path), imgsz=640, conf=0.20, device="cpu", verbose=False)[0]
    pred_count = 0
    for box in result.boxes:
        pred_count += 1
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        color = COLORS_BGR.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
        x1, y1, x2, y2 = xyxy.tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        y_text = max(0, y1 - th - base - 4)
        cv2.rectangle(image, (x1, y_text), (x1 + tw + 5, y_text + th + base + 5), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 2, y_text + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
    return pred_count


def main() -> None:
    configs = [
        DatasetConfig(
            key="marjan_balance",
            name="Marjan Balance Dataset",
            image_dir=Path("datasets/marjan_balance_paper_v5/test/images"),
            label_dir=Path("datasets/marjan_balance_paper_v5/test/labels"),
            used_csv=Path("runs/visual_validation/cshc_aligned_five_scenes/five_scene_visual_validation.csv"),
        ),
        DatasetConfig(
            key="coral_bleaching_general_v1",
            name="Coral Bleaching General v1",
            image_dir=Path("datasets/coral_bleaching_general_v1_marjan3/test/images"),
            label_dir=Path("datasets/coral_bleaching_general_v1_marjan3/test/labels"),
            used_csv=Path(
                "runs/visual_validation/cbgs_light_general_v1_five_scenes/general_v1_five_scene_visual_validation.csv"
            ),
        ),
    ]

    model = YOLO(str(WEIGHTS))
    rows: list[dict[str, str]] = []

    for config in configs:
        metric = select_extra_scene(config)
        out_path = OUT_ROOT / config.key / "06_clear_closeup_simple_scene.jpg"
        pred_count = draw_predictions(model, metric, out_path)
        rows.append(
            {
                "dataset": config.name,
                "scene_id": "06_clear_closeup_simple",
                "scene_cn": "清晰近距离单体/少量大目标场景",
                "image_path": str(metric.image_path),
                "annotated_path": str(out_path),
                "brightness": f"{metric.brightness:.4f}",
                "texture": f"{metric.texture:.4f}",
                "object_count_gt": str(metric.object_count),
                "object_count_pred": str(pred_count),
                "small_count": str(metric.small_count),
                "avg_box_area": f"{metric.avg_box_area:.5f}",
                "class_count": str(metric.class_count),
                "classes": metric.classes,
                "box_background_contrast": f"{metric.box_background_contrast:.4f}",
                "selection_score": f"{metric.score:.4f}",
            }
        )
        print(config.name)
        print(f"  selected: {metric.image_path}")
        print(f"  saved:    {out_path}")
        print(f"  gt objects={metric.object_count}, pred objects={pred_count}, classes={metric.classes}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "extra_different_scene_visual_validation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
