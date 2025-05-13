#!/usr/bin/env python3
import os
from ultralytics import YOLO

def main():
    # Paths
    model_path = "/home/salman/yolo-vehicle-detector/models/best.pt"
    source_video = "/home/salman/yolo-vehicle-detector/data/raw_videos/road.mov"
    output_dir = "/home/salman/yolo-vehicle-detector/results/outputs"

    os.makedirs(output_dir, exist_ok=True)

    # Load trained model
    model = YOLO(model_path)

    # Run inference
    model.predict(
        source=source_video,
        imgsz=1080,
        half=True,
        conf=0.70,
        iou=0.45,
        augment=True,
        vid_stride=1,
        tracker="botsort.yaml",
        save=True,
        save_txt=True,
        project=output_dir,
        name="road_filtered"
    )

    print(f"Inference complete. Results saved to: {output_dir}/road_filtered")

if __name__ == "__main__":
    main()

