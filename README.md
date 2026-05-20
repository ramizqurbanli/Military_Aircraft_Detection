# Military_Aircraft_Detection
# 🛰️ Aircraft Detection in Aerial Imagery: YOLOv11 vs. YOLOv26 Comparative Analysis

A production-ready deep learning pipeline designed to train, evaluate, and benchmark three high-performance object detection architectures on top-down aerial surveillance imagery.

| Model Variant | Bounding Box Type | Architectural Paradigm | Target Use Case |
| :--- | :--- | :--- | :--- |
| **YOLOv11s-Horizontal** | Axis-Aligned Rectangle | Decoupled Anchor-Free | Standard production pipeline integration |
| **YOLOv11s-OBB** | Oriented / Rotated Polygon | Decoupled with Angle Regression ($\theta$) | Maximum geometric precision |
| **YOLOv26n** | Axis-Aligned Rectangle | Next-Gen **NMS-Free** End-to-End | Real-time edge and mobile deployment |

---

## 📊 Dataset: MAR20 (Military Aircraft Recognition)

> [!IMPORTANT]
> **Dataset Naming Note:** In the source code and local directory structures, the folder is labeled `mil-plane`. In reality, this pipeline utilizes the standard **MAR20 (Military Aircraft Recognition) Aerial Aircraft Dataset**. 

The MAR20 dataset contains challenging, high-resolution top-down captures of aircraft sitting on runways, taxiways, open aprons, and hangars. It contains **20 specific military aircraft classes** (coded `A1` through `A20`):


```

mil-plane/ (MAR20 Dataset Root Directory)
├── JPEGImages/                  # Raw aerial images (.jpg format)
├── Annotations/
│   ├── Horizontal Bounding Boxes/   # Pascal VOC XML format (Standard bounding boxes)
│   └── Oriented Bounding Boxes/     # Pascal VOC XML format (Rotated bounding boxes)
└── ImageSets/Main/
├── train.txt                # 1,331 training indices
└── test.txt                 # 2,511 validation/test indices

```

---

## 🏗️ Pipeline & Architecture Flow

The project is engineered as an end-to-end Python pipeline running across structured Jupyter Notebook steps or separate executable modules:


```

aircraft-detection/
├── config.py              # Centralized hyperparameters & VRAM configurations
├── prepare_dataset.py     # XML Parsing & multi-head dataset normalization (YOLO / OBB format)
├── train_yolo_horiz.py    # Training execution script for YOLOv11-Horizontal
├── train_yolo_obb.py      # Training execution script for YOLOv11-OBB
├── train_yolo26.py        # Training execution script for YOLOv26 Nano (NMS-Free)
├── evaluate.py            # Custom side-by-side performance matrix generator
├── app.py                 # Gradio GUI intelligence dashboard app
└── airplane_detection.ipynb # Master pipeline execution notebook

```


```

[VOC XML Annotations] ➔ [prepare_dataset.py] ➔ 分 [horiz/ labels] ➔ YOLOv11s-Horizontal & YOLOv26n
↳ 分 [obb/ labels]   ➔ YOLOv11s-OBB

```

---

## ⚡ Unified Training Parameters

All three variations are trained under strict identical experimental conditions on an **NVIDIA Tesla T4 GPU** to isolate architectural performance gaps:

* **Epochs:** 50 Total
* **Resolution:** Normalized to $640 \times 640$ pixels
* **Batch Size:** 48 (Optimized for ~12GB/16GB VRAM limits)
* **Optimizer Tuning:** Automatically managed AdamW execution via Ultralytics framework
* **Augmentation Strategy:** Active **Mosaic Augmentation** (4-image composite) for epochs 1–40. Mosaic is explicitly closed down (`close_mosaic=10`) during the final 10 epochs to guarantee tight, unwarped boundary localization precision.

---

## 🏁 Empirical Results Benchmark

### 📊 Performance Leaderboard

| Model Architecture | Bounding Box Mode | Precision ($P$) | Recall ($R$) | mAP50 | mAP50-95 (Primary Metric) | Latency (Inference) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv11s-OBB** | Oriented / Rotated | **0.873** | 0.858 | **0.913** | **0.791** | 5.7 ms / image |
| **YOLOv11s-Horizontal**| Axis-Aligned | 0.869 | **0.866** | 0.905 | 0.693 | 5.8 ms / image |
| **YOLOv26n** | Axis-Aligned | 0.749 | 0.732 | 0.794 | 0.600 | **3.4 ms / image** |

### 🔍 Key Engineering Takeaways
1. **The OBB Localization Advantage:** YOLOv11s-OBB provides an absolute gain of **+0.098** in strict localization accuracy (mAP50-95) over its horizontal twin. This represents a **14.1% relative improvement**. By eliminating irrelevant background ground clutter within a box, the oriented box ensures higher IoU calculation scores.
2. **YOLOv26 Speed Superiority:** Despite its smaller parameter architecture footprint (~1.9M vs. ~9.4M parameters), YOLOv26n drops inference down to **3.4 ms**. Its **NMS-Free** design eliminates the processing overhead typically caused by post-prediction sorting bottlenecks.

---

## 🚀 Quick Start Guide

### 1. Environment Initialization
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Upgrade core package to ensure latest 2026 YOLO support
pip install -U pip ultralytics gradio PyTorch

```

### 2. Run Preprocessing Pipeline

Execute `prepare_dataset.py` to target your `mil-plane` folder, extract the XML metadata annotations, and map polygon variants into localized text files:

```bash
python prepare_dataset.py

```

### 3. Training Execution

Train your network heads sequentially or in parallel depending on hardware capacity:

```bash
python train_yolo_horiz.py
python train_yolo_obb.py
python train_yolo26.py

```

### 4. Deploy Gradio Intelligence UI Dashboard

Launch the visualization web-service application containing an integrated MAR20 code parser and side-by-side performance comparator:

```bash
python app.py

```

---

## 🛰️ UI Intelligence Reporting Feature

The graphical interface (`app.py`) parses raw detection output into actionable insights, calculating computational speeds and decoding target signatures directly from class mappings:

* **Input Channel options:** Interactive click-and-drop manual image uploads or local array loops through the MAR20 dataset via Google Drive paths.
* **Intelligence Reporting Summary Output:** Outputs detailed target lists using descriptive labels (e.g., mapping class `A13` directly to **F-15** and class `A1` to **SU-35**).

```
