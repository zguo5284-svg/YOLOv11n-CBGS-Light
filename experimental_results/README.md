# Experimental Results

This directory contains the final experimental result tables reported for YOLOv11n-CBGS-Light.

| File | Experiment | Dataset / Platform |
| --- | --- | --- |
| `comparison_experiments_marjan_balance.csv` | Comparative experiments | The Marjan Balance Dataset |
| `generalization_experiments_coral_bleaching_general_v1.csv` and `.xlsx` | Generalization experiments | Coral Bleaching General v1 |
| `ablation_experiments_marjan_balance.csv` and `.xlsx` | Ablation experiments | The Marjan Balance Dataset |
| `embedded_platform_experiments_jetson_orin_nano.xlsx` | Embedded deployment experiments | NVIDIA Jetson Orin Nano 8 GB, The Marjan Balance Dataset test set |

All embedded-platform results were measured on the complete 276-image test set with an input size of 640 x 640 and a batch size of 1. FPS is calculated from the end-to-end average processing time, including preprocessing, inference, and post-processing. Except for YOLOv11n-CBGS-Light, baseline models were trained without data augmentation or multi-seed retraining.

The external Coral Bleaching General v1 dataset contains Healthy Coral and Bleached Coral categories only; therefore, its mAP values are averaged across these two classes.
