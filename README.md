# YOLOv11n-CBGS-Light for Coral Bleaching Detection

This repository contains the implementation of **YOLOv11n-CBGS-Light**, a lightweight coral bleaching object detection model built on YOLOv11n. The method is designed for underwater coral images and integrates four key modules:

- **RLI-FA / C2fCIB**: reduced-redundancy lightweight feature aggregation.
- **EMCA / MEFA**: edge-guided multi-scale context aggregation.
- **HGA / CGA**: hierarchical group-wise attention.
- **HBS-Head / CBGS-P2**: high-resolution boundary-guided small-object detection head.

The released weight `weights/yolov11n_cbgs_light_epoch150_77p8.pt` is the best checkpoint used in our experiments on the Marjan Balance Dataset.

## Main Result

| Model | Params/M | GFLOPs | Precision/% | Recall/% | mAP50/% | mAP50-95/% |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv11n-CBGS-Light | 3.00 | 12.1 | 82.8 | 71.8 | 77.8 | 46.6 |

## Repository Structure

```text
YOLOv11n-CBGS-Light/
├── assets/
│   ├── paper_table1_three_classes_cbgs_light_epoch150.csv
│   └── sample_cbgs_light_77p8_predict.jpg
├── configs/
│   └── marjan_balance.yaml
├── models/
│   └── yolo11n_cbgs.yaml
├── scripts/
│   ├── predict.py
│   ├── train_cbgs_light.py
│   └── val_yolo_table1_classes.py
├── weights/
│   └── yolov11n_cbgs_light_epoch150_77p8.pt
└── yolov11_cshc/
    ├── __init__.py
    └── modules.py
```

## Installation

```bash
conda create -n cbgs python=3.11 -y
conda activate cbgs
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch version that matches your CUDA runtime before installing Ultralytics.

## Dataset

The dataset is not included in this repository. Prepare the Marjan Balance Dataset in YOLO format and update `configs/marjan_balance.yaml` if your dataset path is different.

Expected layout:

```text
datasets/marjan_balance_paper_v5/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
├── test/images
└── test/labels
```

Class names:

```text
0: Bleached Coral
1: Dead Coral
2: Healthy Coral
```

## Inference

```bash
python scripts/predict.py \
  --weights weights/yolov11n_cbgs_light_epoch150_77p8.pt \
  --source path/to/images \
  --imgsz 640 \
  --conf 0.25 \
  --device 0
```

For CPU inference, set `--device cpu`.

## Training

```bash
python scripts/train_cbgs_light.py \
  --model models/yolo11n_cbgs.yaml \
  --data configs/marjan_balance.yaml \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --seed 2026
```

The training script uses the lightweight augmentation strategy used in the reported experiment:

```text
mosaic=0.8
close_mosaic=100
hsv_h=0.003
hsv_s=0.25
hsv_v=0.15
scale=0.30
translate=0.05
erasing=0.0
seed=2026
```

## Evaluation

```bash
python scripts/val_yolo_table1_classes.py \
  --weights weights/yolov11n_cbgs_light_epoch150_77p8.pt \
  --data configs/marjan_balance.yaml \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --name YOLOv11n-CBGS-Light \
  --save-dir runs/table1_test_b1
```

## Citation

If this code is helpful for your research, please cite the related paper or repository.

## License

This project is released under the MIT License.
