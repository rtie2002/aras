from ultralytics import YOLO
import cv2
import os

def detect_video_realtime():
    # 1. Path to your best model weights
    model_path = r'RoadDetectionModel/SpeedBump_Training_v12/weights/best.pt'
    
    # 2. Path to your video file
    video_input_path = r'test_video.mp4' 
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 3. Load the model and force GPU
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    model.to('cuda') # Ensure it's on the RTX 4090

    # 4. Open Video
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Fallback
    
    # Since the video is 120 FPS, we don't need to run inference on every frame.
    # We will target ~30 detections per second to keep it smooth but fast.
    target_detection_fps = 30
    stride = max(1, int(fps / target_detection_fps))
    
    frame_delay = 1 / fps # Time per frame in seconds
    
    print(f"Original Video FPS: {fps:.2f}")
    print(f"Detection Stride: {stride} (Every {stride}th frame)")
    print("Starting detection. Press 'q' to stop.")
    
    import time
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        # Read the frame
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Only run detection on every 'stride' frame
        if frame_count % stride == 0:
            # Run inference (using imgsz=480 for extreme speed on 120fps video)
            results = model.predict(frame, conf=0.30, half=True, verbose=False, device=0, imgsz=480)
            annotated_frame = results[0].plot()
        else:
            # For intermediate frames, just show the original or the last detection
            # To save performance, we just show the annotated frame from the last detection
            # or if you want it very smooth, just the original frame.
            # Let's just keep the last annotated frame to avoid flickering
            pass

        # Display (we update display every frame to keep 120fps feel)
        cv2.imshow("YOLOv8 Speed Bump Detection (120 FPS Video)", annotated_frame if 'annotated_frame' in locals() else frame)
        
        # Timing control to match 120fps playback
        current_playback_time = frame_count * frame_delay
        actual_elapsed_time = time.time() - start_time
        
        wait_time = 1 # minimum wait
        if actual_elapsed_time < current_playback_time:
            # We are faster than the video, wait a bit
            wait_time = int((current_playback_time - actual_elapsed_time) * 1000)
            wait_time = max(1, wait_time)
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

        # catching up logic: if we are too slow, grab next frames until we are on time
        actual_elapsed_time = time.time() - start_time
        while actual_elapsed_time > current_playback_time + frame_delay:
            cap.grab() # Skip decoding for speed
            frame_count += 1
            current_playback_time = frame_count * frame_delay
            actual_elapsed_time = time.time() - start_time

    cap.release()
    cv2.destroyAllWindows()
    print("Detection finished.")

if __name__ == "__main__":
    detect_video_realtime()
