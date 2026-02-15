from ultralytics import YOLO
import cv2
import os

def detect_video_and_frame_out():
    # 1. Path to your best model weights
    model_path = r'RoadDetectionModel/SpeedBump_Training_v12/weights/best.pt'
    
    # 2. Path to your video file
    # Change 'test_video.mp4' to your actual video filename
    video_input_path = r'test_video.mp4' 
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(video_input_path):
        print(f"Error: Video file not found: {video_input_path}")
        print("Please place your video in the folder and update the filename in this script.")
        return

    # 3. Load the model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # 4. Run prediction on video
    # show=True: Opens a window and displays the detection live
    # save=True: Saves the final annotated video to 'runs/detect/predict_video'
    print("Starting video detection. Press 'q' to stop.")
    results = model.predict(
        source=video_input_path, 
        show=True, 
        conf=0.30, 
        save=True,
        project='runs/detect',
        name='predict_video'
    )
    
    print(f"Detection finished. Annotated video saved in: runs/detect/predict_video")

if __name__ == "__main__":
    detect_video_and_frame_out()
