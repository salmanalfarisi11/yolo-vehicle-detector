# YOLO Vehicle Detector

Custom object detection pipeline for **Cars**, **Motorcycles**, and **Trucks** in pre-recorded traffic videos, built on **YOLOv12-s**.

---

## 🔗 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)

   * [Training](#training)
   * [Inference](#inference)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

A high-performance, object detection pipeline built on **Ultralytics YOLOv12-s** to accurately identify Cars, Motorcycles, and Trucks in pre-recorded traffic video footage. Key project highlights:

* **State-of-the-art accuracy**: achieves mAP@0.5 ≥ 0.99 and mAP@0.5:0.95 ≥ 0.82 on a custom 80-image dataset
* **Robustness**: resilient to motion blur, challenging lighting conditions, and both portrait/landscape orientations
* **Efficiency**: leverages mixed-precision (FP16), rectangular multi-scale training, and cosine learning rate scheduling
* **Advanced augmentations**: includes mosaic, mixup, and copy-paste to enhance data diversity

This detector is tailored for essential traffic analytics tasks—vehicle counting, classification, and monitoring of Cars, Motorcycles, and Trucks in road scenarios.

[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://youtube.com/shorts/HRZQ_QOSOSo?si=8rZVTkii-6SOsM9T)

---

## Features

### Training
- **Custom pipeline** with mosaic, mixup & copy-paste augmentations  
- **Cosine LR scheduler** for smooth decay  
- **Mixed-precision (FP16)** to halve memory usage  
- **Rectangular & multi-scale** batches matching input aspect ratio  
- **Backbone freezing** of early layers for data-efficient convergence  

### Inference
- **Test-time augmentation (TTA)** for boosted recall  
- **Integrated tracking** (BOTSort/ByteTrack) for stable object IDs  
- **CLI scripts**  
  - `python scripts/train_custom.py`  
  - `python scripts/infer.py`  



---

## Project Structure

```
yolo-vehicle-detector/
├── data/
│   ├── raw_videos/         # Source .mov/.mp4 videos
│   └── images_labels/      # Extracted frames and CVAT .txt labels
│       ├── images/
│       └── labels/
├── configs/
│   └── data.yaml           # Dataset configuration
├── models/
│   ├── yolov12s.pt         # Pretrained YOLOv12-s weights
│   └── best.pt             # Trained model weights
├── scripts/
│   ├── train.py            # Training script
│   └── infer.py            # Inference script
├── results/
│   ├── runs/               # Training logs and checkpoints
│   └── outputs/            # Inference videos and label exports
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/salmanalfarisi11/yolo-vehicle-detector.git
   cd yolo-vehicle-detector
  ```
  
2. **Install system dependencies**  

    ```bash
    sudo apt update
    sudo apt install -y python3-venv python3-pip
    ```

3. **Create & activate a Python virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Upgrade pip and Install Python packages**

   ```bash
   pip install --upgrade pip
   pip install ultralytics opencv-python torch torchvision
   ```

---

## Configuration

Edit `configs/data.yaml` to point to your image and label directories:

```yaml
train: data/images_labels/images
val:   data/images_labels/images    # or separate val folder
nc: 3
names:
  - car
  - motorcycle
  - truck
```

---

## Usage

### Training

```bash
python scripts/train_custom.py
```

**Key arguments** (in `train_custom.py`):

* `epochs`: number of training epochs (e.g., 120)
* `imgsz`: input resolution (e.g., 1080)
* `batch`: batch size (usually 1 for limited VRAM)

Training outputs are saved under `results/runs/road_custom_v12s_max/`.

### Inference

```bash
python scripts/infer.py
```

Inference results (annotated video and `.txt` labels) are stored in `results/outputs/road_filtered/`.

---

## Results

After training on 80 images for 120 epochs with advanced augmentation and FP16:

| Metric        | Value |
| ------------- | ----- |
| Precision     | 0.955 |
| Recall        | 0.998 |
| mAP\@0.5      | 0.990 |
| mAP\@0.5:0.95 | 0.817 |

Visualize training curves in `results/runs/road_custom_v12s_max/plots`.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Submit a pull request

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
