from ultralytics import YOLO
import cv2
import os

def detect_video_optim():
    # 1. Path to your best model weights
    model_path = r'RoadDetectionModel/SpeedBump_Training_v12/weights/best.pt'
    
    # 2. Path to your video file
    video_input_path = r'test_video.mp4' 
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(video_input_path):
        print(f"Error: Video file not found: {video_input_path}")
        return

    # 3. Load the model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # 4. Run prediction with optimization
    # - stream=True: Processes frames one by one (memory efficient)
    # - half=True: Uses FP16 inference (much faster on NVIDIA RTX GPUs)
    # - imgsz=640: Standard size, can be reduced to 480 or 320 for even more speed
    # - vid_stride=2: Skip every other frame to effectively double FPS while remaining smooth
    
    print("Starting optimized video detection. Press 'q' to stop.")
    
    # Using model.predict as a generator
    results = model.predict(
        source=video_input_path, 
        stream=True, 
        conf=0.30, 
        half=True,    # Fast inference on 4090
        imgsz=640,    # Input resolution
        vid_stride=1  # Set to 2 or 3 to skip frames for more speed
    )

    import time
    prev_time = 0
    
    for r in results:
        # Get the annotated frame
        annotated_frame = r.plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Draw FPS on frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("YOLOv8 Real-time Detection (Press 'q' to quit)", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Detection finished.")

if __name__ == "__main__":
    detect_video_optim()
