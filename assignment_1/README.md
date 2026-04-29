# YOLOv1 Cat-Dog Object Detection

A YOLOv1-inspired object detection pipeline for cat and dog head detection, built for the INFOMCV 2026 Assignment 4. The project supports training from scratch, fine-tuning a pretrained ResNet-18 backbone, and backbone pretraining via a classification task.

---

## Table of Contents

- [YOLOv1 Cat-Dog Object Detection](#yolov1-cat-dog-object-detection)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Data](#data)
  - [Configuration](#configuration)
    - [`general` section](#general-section)
    - [`jobs` section](#jobs-section)
  - [Usage](#usage)
  - [Architecture](#architecture)
    - [YOLOv1Base (custom CNN)](#yolov1base-custom-cnn)
    - [YOLOv1ResNet (pretrained ResNet-18 backbone)](#yolov1resnet-pretrained-resnet-18-backbone)
  - [Loss Function](#loss-function)
  - [Evaluation](#evaluation)
    - [Mean Average Precision (mAP)](#mean-average-precision-map)
    - [Early Stopping](#early-stopping)
  - [Choice Tasks Implemented](#choice-tasks-implemented)
  - [Disclaimers](#disclaimers)

---

## Overview

This pipeline detects cat and dog heads in images using a YOLOv1-style output format. A 7×7 spatial grid is used, with one bounding box and two class scores per cell, resulting in a 343-dimensional output vector `(7 × 7 × (1 + 4 + 2))`. Two backbone options are available:

- **YOLOv1Base** — a lightweight custom CNN (~1M parameters) trained from scratch on 112×112 images.
- **YOLOv1ResNet** — a ResNet-18 backbone (pretrained on ImageNet) with a custom detection head, using 224×224 images.

Multiple training jobs can be defined and run sequentially via a YAML config file.

---

## Project Structure

```
.
├── main.py                      # Entry point: loads config and dispatches jobs
├── train.py                     # Training and validation loop
├── visualise.py                 # Visualisation utilities (bounding boxes, losses, mAP curves)
│
├── cat_dog_dataset.py           # PyTorch Dataset for the cat-dog-detection dataset
├── data.py                      # Utility to convert Datasets into DataLoaders
├── decode.py                    # Decodes raw model output into (x, y, w, h, conf, classes)
├── handle_output.py             # Defines the timestamped output directory
│
├── yolov1_base.py               # YOLOv1Base model (custom CNN backbone)
├── yolov1_resnet.py             # YOLOv1ResNet model (ResNet-18 backbone)
├── yolov1_loss.py               # YOLOv1 loss function (5-component)
├── mean_average_precision.py    # Vectorised mAP computation
├── early_stopper.py             # Early stopping based on validation 
|
├── process_video.py             # Convert video into annotated version
├── confidence_thresholds.py     # Find the best confidence thresholds for a model
│
├── create_logger.py             # Logger factory (colour-coded, UTC+1 timestamps)
├── custom_logger_formatter.py   # Custom logging formatter
│
├── pretrain_classification.ipynb  # Notebook: backbone pretraining on classification task
│
├── config.yaml                  # Experiment configuration (jobs, hyperparameters)
├── config_validation_template.py # JSON Schema for config validation
└── requirements.txt             # Python dependencies
```

---

## Installation

Python 3.10+ is required (the codebase uses `X | Y` union type hints).

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins CUDA 13.0-compatible builds of PyTorch and torchvision. If your system uses a different CUDA version, install the appropriate wheels from [pytorch.org](https://pytorch.org/get-started/locally/) first.

---

## Data

Download the [dog-cat-detection dataset from Kaggle](https://www.kaggle.com/datasets/). Place the files so that your directory structure matches the paths set in `config.yaml`:

```
assignment_4/
└── data/
    ├── images/        # .png image files
    └── annotations/   # .xml annotation files (Pascal VOC format)
```

The `CatDogDataset` class handles:
- Parsing Pascal VOC XML annotations (bounding boxes + class labels).
- Converting `[xmin, ymin, xmax, ymax]` pixel coordinates to normalised `[cx, cy, w, h]` format.
- Building the 7×7×7 YOLO target tensor for each image.
- Applying any `torchvision` transforms (e.g. resize, normalise) passed in at construction.

Images are resized to the `input_image_size` specified per job (112 for `YOLOv1Base`, 224 for `YOLOv1ResNet`). Bounding boxes are normalised relative to the original image dimensions before resizing, so they remain valid after the transform.

The dataset is split into train / val / test using the `train_val_test_split` ratios in the config (default 72% / 18% / 10%), with stratification on the class label.

---

## Configuration

All experiments are defined in `config.yaml`. The file is validated against the JSON Schema in `config_validation_template.py` at startup.

### `general` section

| Key | Description |
|---|---|
| `data_images_path` | Path to the directory of `.png` image files |
| `data_annotations_path` | Path to the directory of `.xml` annotation files |
| `grid_size` | YOLO grid size (default: 7) |
| `num_data_workers` | Number of DataLoader worker processes |

### `jobs` section

Each key `job0`, `job1`, … defines an independent experiment run sequentially.

| Key | Description |
|---|---|
| `model` | `"yolo"` (YOLOv1Base) or `"resnet"` (YOLOv1ResNet) |
| `start_from_checkpoint_path` | Path to a `.pth` checkpoint to resume from, or `None` |
| `input_image_size` | Image size fed to the model (112 or 224) |
| `train_val_test_split` | Three floats summing to 1.0, e.g. `[0.72, 0.18, 0.10]` |
| `batch_size` | Mini-batch size |
| `n_epochs` | Maximum number of training epochs |
| `k_folds` | Number of cross-validation folds; set to `1` for a fixed split |
| `learning_rate` | Optimiser learning rate |
| `lambda_coord` | Loss weight for bounding box coordinate terms (default: 5) |
| `lambda_noobj` | Loss weight for no-object confidence term (default: 0.5) |
| `iou_thresholds` | List of IoU thresholds for mAP evaluation, e.g. `[0.5, 0.9]` |
| `conf_threshold` | Minimum confidence score to keep a prediction during evaluation |
| `plotting_conf_threshold` | Confidence threshold used when drawing bounding boxes |

---

## Usage

```bash
python main.py
```

This will:
1. Load and validate `config.yaml`.
2. Create a timestamped output directory under `assignment_4/output/`.
3. Run each job defined under `jobs` in sequence.
4. For each job: train the model with early stopping, evaluate on train/val/test, save the best checkpoint, and write loss curves and mAP plots.

To run only a specific job, adjust the `jobs` section in `config.yaml` to contain only that job.

---

## Architecture

### YOLOv1Base (custom CNN)

Designed for 112×112×3 input.

| Layer | Details |
|---|---|
| Conv 1 | 3×3, 16 kernels, padding 1 → BN → ReLU → MaxPool 2×2 |
| Conv 2 | 3×3, 32 kernels, padding 1 → BN → ReLU → MaxPool 2×2 |
| Conv 3 | 3×3, 64 kernels, padding 1 → BN → ReLU → MaxPool 2×2 |
| Conv 4 | 3×3, 64 kernels, padding 1 → BN → ReLU → MaxPool 2×2 |
| Conv 5 | 3×3, 32 kernels, padding 1 → BN → ReLU |
| Flatten | → 1568 features (7×7×32) |
| Dropout | p=0.5 |
| FC 1 | 1568 → 512, ReLU |
| FC 2 (output) | 512 → 343, Sigmoid |

All weights are initialised with Kaiming uniform initialisation.

### YOLOv1ResNet (pretrained ResNet-18 backbone)

Designed for 224×224×3 input.

- **Backbone**: ResNet-18 pretrained on ImageNet (all layers except `avgpool` and the final FC), outputting 512×7×7 feature maps.
- **Head**: Flatten → Dropout(0.5) → FC(25088 → 512, ReLU) → FC(512 → 343, Sigmoid).
- The backbone can optionally be frozen during fine-tuning (`freeze_backbone=True`).
- Head weights are initialised with Kaiming uniform; backbone weights come from the ImageNet checkpoint.

Both models expose `.save(path)` and `.load(path, logger)` class methods for checkpointing.

---

## Loss Function

`YOLOv1Loss` follows the original YOLOv1 paper, composed of five terms:

| Term | Formula | Weight |
|---|---|---|
| **XY loss** | SSE of (cx, cy) in cells with an object | `lambda_coord` (×5) |
| **WH loss** | SSE of (√w, √h) in cells with an object | `lambda_coord` (×5) |
| **Obj confidence** | SSE of objectness where object exists | ×1 |
| **Noobj confidence** | SSE of objectness where no object exists | `lambda_noobj` (×0.5) |
| **Class loss** | SSE of class scores in cells with an object | ×1 |

The square root is taken for width and height to reduce the impact of large bounding box errors relative to small ones.

---

## Evaluation

### Mean Average Precision (mAP)

`compute_map` in `mean_average_precision.py` is a fully vectorised (no Python loops) mAP implementation:

1. Decodes predictions and ground truths from the output cube.
2. Filters predictions below `conf_threshold`.
3. Computes pairwise IoU between all predictions and ground truths.
4. Sorts predictions by confidence per class and performs greedy matching.
5. Computes per-class precision-recall curves and integrates via the trapezoid rule.
6. Returns the mean AP across classes that have at least one ground truth.

The function supports evaluation at multiple IoU thresholds (e.g. mAP@50, mAP@90).

### Early Stopping

`EarlyStopper` monitors validation loss and stops training if there is no improvement greater than `min_delta` for `patience` consecutive epochs.

---

## Choice Tasks Implemented

The following optional tasks from the assignment specification are included:

- **Choice 2 — Pretrained ResNet backbone**: `YOLOv1ResNet` replaces the custom CNN backbone with ResNet-18 initialised from ImageNet weights. Select with `model: "resnet"` in the config.
- **Choice 3 — Backbone pretraining**: `pretrain_classification.ipynb` pretrains the `YOLOv1Base` backbone on a cat-dog classification dataset. The resulting checkpoint can be loaded via `start_from_checkpoint_path` in the config.

---

## Disclaimers

The following files contain code originally authored by **Joris Heemskerk** as part of a Bachelor's thesis at **Technolution BV, Gouda NL**. They are reused here with permission, with all rights reserved to the original author:

- `create_logger.py`
- `custom_logger_formatter.py`
- `config_validation_template.py`
- `config.yaml` (structure)
