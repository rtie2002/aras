from ultralytics import YOLO
import os

def visualize_test_results():
    # 1. Path to your best model weights
    model_path = r'RoadDetectionModel/SpeedBump_Training_v12/weights/best.pt'
    
    # 2. Path to your test images
    source_path = r'../dataset/test/images'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 3. Load the model
    model = YOLO(model_path)

    # 4. Run prediction and SHOW results
    # This will open a window for each image, press any key to see the next one
    print("Starting visualization. Press 'q' or 'Esc' on the image window to stop.")
    results = model.predict(source=source_path, show=True, conf=0.25, save=True)
    
    print(f"Predictions saved to: runs/detect/predict")

if __name__ == "__main__":
    visualize_test_results()
