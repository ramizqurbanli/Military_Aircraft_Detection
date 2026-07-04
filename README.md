# 🛰️ Aircraft Detection in Aerial Imagery: YOLOv11 vs. YOLOv26 Comparative Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111F68)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
[![Google Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/drive/1mHSmVubNYTysTgPzh9Fa4KP8sbpMdnyy?usp=sharing)
[![Dataset: MAR20](https://img.shields.io/badge/Dataset-Google%20Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1syZVVdd-k6zbWglQPyn_ZAwqbP2r-i1R?usp=sharing)

A deep learning pipeline designed to train, evaluate, and benchmark three high-performance object detection architectures on top-down aerial surveillance imagery. This entire project is structured, executed, and analyzed inside a single Master Jupyter Notebook.

| Model Variant | Bounding Box Type | Architectural Paradigm | Target Use Case |
| :--- | :--- | :--- | :--- |
| **YOLOv11s-Horizontal** | Axis-Aligned Rectangle | Decoupled Anchor-Free | Standard production pipeline integration |
| **YOLOv11s-OBB** | Oriented / Rotated Polygon | Decoupled with Angle Regression ($\theta$) | Maximum geometric precision |
| **YOLOv26n** | Axis-Aligned Rectangle | Next-Gen **NMS-Free** End-to-End | Real-time edge and mobile deployment |

---

## 📊 Dataset: MAR20 (Military Aircraft Recognition)

> [!IMPORTANT]
> **Dataset Naming Note:** In the source code, variables, and local directory structures, the dataset folder is named `mil-plane`. In reality, this project utilizes the standard **MAR20 (Military Aircraft Recognition) Aerial Aircraft Dataset**.

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

## 🏗️ Notebook Structure & Workflow

The entire engineering pipeline is self-contained within `airplane_detection.ipynb`. The notebook is organized into sequential, execution-ready cells:

1. **Environment Setup & Configuration:** Installation of dependencies (`ultralytics`, `gradio`, etc.) and setting path variables pointing to the `mil-plane` directory.
2. **Dataset Preprocessing & Conversion:** Parsing the raw Pascal VOC XML annotations from the `mil-plane` directory and formatting them into standard horizontal YOLO text labels and Oriented Bounding Box (OBB) labels.
3. **Model Training Pipelines:** Sequential training configurations for all three variations (`YOLOv11s`, `YOLOv11s-OBB`, and `YOLOv26n`).
4. **Evaluation & Comparative Analysis:** Generating validation metrics, computing inference speed latencies, and outputting performance comparison tables.
5. **Interactive UI Deployment:** An inline Gradio interface cell to load the trained weights and perform real-time test inferences with explicit class name mapping.

---

## ⚡ Unified Training Parameters

All three variations are trained under strict identical experimental conditions within the notebook environment to isolate architectural performance gaps:

* **Epochs:** 50 Total
* **Resolution:** Normalized to $640 \times 640$ pixels
* **Batch Size:** 48 (Optimized for VRAM efficiency)
* **Optimizer Tuning:** Automatically managed AdamW execution via the Ultralytics framework
* **Augmentation Strategy:** Active **Mosaic Augmentation** for epochs 1–40. Mosaic is explicitly closed down (`close_mosaic=10`) during the final 10 epochs to guarantee tight, unwarped boundary localization precision.

---

## 🏁 Empirical Results Benchmark

### 📊 Performance Leaderboard

| Model Architecture | Bounding Box Mode | Precision ($P$) | Recall ($R$) | mAP50 | mAP50-95 (Primary Metric) | Latency (Inference in CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv11s-OBB** | Oriented / Rotated | **0.873** | 0.858 | **0.913** | **0.791** | 5.7 ms / image |
| **YOLOv11s-Horizontal**| Axis-Aligned | 0.869 | **0.866** | 0.905 | 0.693 | 5.8 ms / image |
| **YOLOv26n** | Axis-Aligned | 0.749 | 0.732 | 0.794 | 0.600 | **3.4 ms / image** |
<img width="1484" height="583" alt="image" src="https://github.com/user-attachments/assets/31f49b41-631d-46f3-9f0a-7b7f05732106" />

### 🔍 Key Engineering Takeaways
1. **The OBB Localization Advantage:** YOLOv11s-OBB provides an absolute gain of **+0.098** in strict localization accuracy (mAP50-95) over its horizontal twin. This represents a **14.1% relative improvement**. By eliminating irrelevant background ground clutter within a box, the oriented box ensures higher IoU calculation scores.
2. **YOLOv26 Speed Superiority:** Despite its smaller parameter architecture footprint, YOLOv26n drops inference down to **3.4 ms**. Its **NMS-Free** design eliminates the processing overhead typically caused by post-prediction sorting bottlenecks.

---

## 🚀 How to Run

### 1. Environment Initialization
Before opening the notebook, ensure your environment has access to a GPU and install the required dependencies:
```bash
pip install -U pip ultralytics gradio torch torchvision

```

### 2. Dataset Placement

Ensure your dataset is unpacked and matching the naming convention required by the notebook:

```
path/to/your/workspace/mil-plane/

```

### 3. Step-by-Step Execution

1. Open `airplane_detection.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run the **Configuration** cell to map your directory paths correctly.
3. Execute the **Preprocessing** cell to parse the XML files into standard YOLO training labels.
4. Run the respective **Training** cells for the models you wish to evaluate.
5. Launch the final **Gradio UI Dashboard** cell to manually test images against the trained weights.

[Google Colab Link](https://colab.research.google.com/drive/1mHSmVubNYTysTgPzh9Fa4KP8sbpMdnyy?usp=sharing)
