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

    # 3. Load the model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    # model.to('cuda') - Removed to avoid "Torch not compiled with CUDA enabled" error

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
    annotated_frame = None
    
    while cap.isOpened():
        # Read the frame
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Only run detection on every 'stride' frame
        if frame_count % stride == 0:
            # Run inference (allowing auto-device selection)
            results = model.predict(frame, conf=0.30, half=False, verbose=False, imgsz=480)
            annotated_frame = results[0].plot()
            
            # --- Object Counting Logic ---
            counts = results[0].boxes.cls.unique().tolist() # Get unique class indices
            class_names = results[0].names
            current_counts = {}
            
            # Count occurrences of each class in the current frame
            for cls_idx in results[0].boxes.cls.cpu().numpy():
                name = class_names[int(cls_idx)]
                current_counts[name] = current_counts.get(name, 0) + 1
            
            # Store count string to display on subsequent frames
            count_text = " | ".join([f"{name}: {count}" for name, count in current_counts.items()])
            if not count_text: count_text = "No objects detected"
            # ---------------------------

        # Display (show annotated if ready, else original)
        if annotated_frame is not None:
            display_frame = annotated_frame.copy()
            
            # Draw the counts on the frame (top left corner)
            if 'count_text' in locals():
                # Background rectangle for better readability
                cv2.rectangle(display_frame, (10, 10), (int(len(count_text)*18) + 20, 45), (0, 0, 0), -1)
                # Text
                cv2.putText(display_frame, count_text, (15, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            display_frame = frame

        # Ensure display_frame is valid before imshow
        if display_frame is not None and display_frame.size > 0:
            cv2.imshow("YOLOv8 Speed Bump Detection (120 FPS Video)", display_frame)
        
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
