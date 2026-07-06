from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yolov11_cshc


yolov11_cshc.register_cshc_modules()


DATASET = Path("datasets/coral_bleaching_general_v1_marjan3")
IMAGE_DIR = DATASET / "test" / "images"
LABEL_DIR = DATASET / "test" / "labels"
WEIGHTS = Path("runs/jetson_deploy/yolov11n_cbgs_light_epoch150_77p8.pt")
OUT_DIR = Path("runs/visual_validation/cbgs_light_general_v1_five_scenes")
FINAL_DIR = OUT_DIR / "final_images"

CLASS_NAMES = {
    0: "Bleached Coral",
    1: "Dead Coral",
    2: "Healthy Coral",
}

COLORS_BGR = {
    0: (0, 165, 255),   # orange
    1: (120, 120, 120), # gray
    2: (0, 200, 80),    # green
}

SCENES = [
    ("01_dark", "昏暗环境"),
    ("02_bright_biological_interference", "光线良好且有其他生物干扰的环境"),
    ("03_similar_seabed_multiclass", "与海床颜色接近且含多类别珊瑚的海床环境"),
    ("04_small_rugged_complex", "小目标错综复杂的崎岖海床环境"),
    ("05_many_small_more_complex", "小目标众多且环境更加复杂的试验场景"),
]


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
    boxes: list[tuple[int, float, float, float, float]]


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


def image_metrics(image_path: Path) -> ImageMetric | None:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"
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

    return ImageMetric(
        image_path=image_path,
        label_path=label_path,
        brightness=brightness,
        texture=texture,
        object_count=len(boxes),
        small_count=small_count,
        avg_box_area=float(np.mean(areas)) if areas else 0.0,
        class_count=len(class_ids),
        classes=classes,
        box_background_contrast=float(np.mean(contrasts)) if contrasts else 0.0,
        boxes=boxes,
    )


def select_distinct(metrics: list[ImageMetric]) -> list[ImageMetric]:
    selected: list[ImageMetric] = []
    used: set[Path] = set()

    def pick(candidates: list[ImageMetric]) -> ImageMetric:
        for item in candidates:
            if item.image_path not in used:
                selected.append(item)
                used.add(item.image_path)
                return item
        raise RuntimeError("No unused candidate found.")

    pick(sorted(metrics, key=lambda m: (m.brightness, -m.object_count)))

    pick(
        sorted(
            metrics,
            key=lambda m: (-(m.brightness + 0.45 * m.texture + 0.015 * m.object_count), m.avg_box_area),
        )
    )

    multiclass = [m for m in metrics if m.class_count >= 2]
    pick(
        sorted(
            multiclass or metrics,
            key=lambda m: (m.box_background_contrast, -m.class_count, -m.object_count),
        )
    )

    pick(
        sorted(
            metrics,
            key=lambda m: (-m.small_count, -m.texture, -m.object_count, m.avg_box_area),
        )
    )

    pick(
        sorted(
            metrics,
            key=lambda m: (-m.object_count, -m.small_count, -m.texture, m.avg_box_area),
        )
    )

    return selected


def draw_predictions(model: YOLO, metric: ImageMetric, out_path: Path) -> None:
    image = cv2.imread(str(metric.image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {metric.image_path}")

    results = model.predict(str(metric.image_path), imgsz=640, conf=0.20, device="cpu", verbose=False)
    result = results[0]
    boxes = result.boxes

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        color = COLORS_BGR.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
        x1, y1, x2, y2 = xyxy.tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_text = max(0, y1 - th - base - 4)
        cv2.rectangle(image, (x1, y_text), (x1 + tw + 4, y_text + th + base + 4), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 2, y_text + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def make_contact_sheet(rows: list[dict[str, str]]) -> None:
    thumbs: list[Image.Image] = []
    for row in rows:
        im = Image.open(row["annotated_path"]).convert("RGB")
        im.thumbnail((360, 260))
        canvas = Image.new("RGB", (380, 330), "white")
        canvas.paste(im, ((380 - im.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
            small = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
            small = ImageFont.load_default()
        draw.text((12, 275), row["scene_cn"], fill=(0, 0, 0), font=font)
        draw.text((12, 300), f"Objects: {row['object_count']}  Classes: {row['classes']}", fill=(50, 50, 50), font=small)
        thumbs.append(canvas)

    sheet = Image.new("RGB", (380 * len(thumbs), 330), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (380 * i, 0))
    sheet.save(OUT_DIR / "general_v1_five_scene_contact_sheet.jpg", quality=95)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [m for m in (image_metrics(p) for p in sorted(IMAGE_DIR.glob("*.*"))) if m is not None]
    if not metrics:
        raise RuntimeError("No labeled test images found.")

    selected = select_distinct(metrics)
    model = YOLO(str(WEIGHTS))

    rows: list[dict[str, str]] = []
    for idx, ((scene_id, scene_cn), metric) in enumerate(zip(SCENES, selected)):
        out_path = FINAL_DIR / f"{scene_id}_image{idx}.jpg"
        draw_predictions(model, metric, out_path)
        rows.append(
            {
                "scene_id": scene_id,
                "scene_cn": scene_cn,
                "image_path": str(metric.image_path),
                "annotated_path": str(out_path),
                "brightness": f"{metric.brightness:.4f}",
                "texture": f"{metric.texture:.4f}",
                "object_count": str(metric.object_count),
                "small_count": str(metric.small_count),
                "avg_box_area": f"{metric.avg_box_area:.5f}",
                "class_count": str(metric.class_count),
                "classes": metric.classes,
                "box_background_contrast": f"{metric.box_background_contrast:.4f}",
            }
        )

    csv_path = OUT_DIR / "general_v1_five_scene_visual_validation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_contact_sheet(rows)
    print(f"Saved {len(rows)} visual images to {FINAL_DIR}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved contact sheet to {OUT_DIR / 'general_v1_five_scene_contact_sheet.jpg'}")
    for row in rows:
        print(row["scene_id"], row["scene_cn"], row["annotated_path"], row["classes"], row["object_count"])


if __name__ == "__main__":
    main()
