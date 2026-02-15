from ultralytics import YOLO
import torch
import os

def start_training():
    # 1. Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")
    else:
        print("⚠ GPU not detected. Training will be VERY slow on CPU.")

    # 2. Load model
    # Use yolov8m.pt as a balanced choice, or yolov8n.pt for speed
    model = YOLO('yolov8m.pt') 

    # 3. Path to data.yaml
    data_path = 'data.yaml'

    # 4. Start training
    print(f"Starting training with data: {data_path}")
    results = model.train(
        data=data_path, 
        epochs=120, 
        imgsz=640, 
        batch=32,       # Optimized for RTX 4090 (24GB VRAM)
        device=0,       # Use the first GPU
        patience=25,    # Early stopping if no improvement
        project='RoadDetectionModel',
        name='SpeedBump_Training_v1'
    )
    print("Training complete!")

if __name__ == "__main__":
    start_training()
