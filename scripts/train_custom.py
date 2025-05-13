#!/usr/bin/env python3
import os
from ultralytics import YOLO

def main():
    # Project root
    project_root = "/home/salman/yolo-vehicle-detector"

    # Paths
    model_path = os.path.join(project_root, "models", "yolov12s.pt")
    data_yaml  = os.path.join(project_root, "configs", "data.yaml")
    project_dir = os.path.join(project_root, "results", "runs")  

    # Load YOLOv12-small model
    model = YOLO(model_path)

    # Training with improved parameters
    model.train(
        data=data_yaml,
        epochs=120,                 
        imgsz=1080,
        batch=1,
        lr0=0.001,
        lrf=0.1,
        cos_lr=True,
        freeze=(10,),
        augment=True,
        copy_paste=0.5,
        multi_scale=True,
        rect=True,
        half=True,
        save_period=5,
        project=project_dir,
        name="road_custom_v12s_max"
    )

if __name__ == "__main__":
    main()
