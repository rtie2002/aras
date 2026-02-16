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
    count_text = ""
    current_counts = {}
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Only run detection on every 'stride' frame
        if frame_count % stride == 0:
            results = model.predict(frame, conf=0.30, half=True if torch.cuda.is_available() else False, verbose=False, imgsz=480)
            
            # --- Autopilot Style Metadata Extraction ---
            current_counts = {}
            class_names = results[0].names
            for cls_idx in results[0].boxes.cls.cpu().numpy():
                name = class_names[int(cls_idx)]
                current_counts[name] = current_counts.get(name, 0) + 1
            
            # Get original detections but custom plot them later or use .plot() as base
            annotated_frame = results[0].plot(labels=True, conf=True, boxes=True)
        
        # --- HUD RENDERING ---
        if annotated_frame is not None:
            display_frame = annotated_frame.copy()
        else:
            display_frame = frame.copy()

        # 1. Create Transparent HUD Overlays (Top and Bottom)
        overlay = display_frame.copy()
        h, w, _ = display_frame.shape
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1) # Header
        cv2.rectangle(overlay, (0, h-40), (w, h), (20, 20, 20), -1) # Footer
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

        # 2. Header: System Status
        cv2.putText(display_frame, "AUTOPILOT SYSTEM v1.2", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        
        # System Time / Status
        status_color = (0, 255, 0) if len(current_counts) > 0 else (200, 200, 200)
        cv2.putText(display_frame, "STATUS: TRACKING ACTIVE", (w - 320, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2)

        # 3. Sidebar: Object Tracking List
        y_offset = 110
        cv2.putText(display_frame, "ENVIRONMENT SCAN:", (20, 90), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        for name, count in current_counts.items():
            # Dashboard style entries
            cv2.rectangle(display_frame, (20, y_offset-20), (200, y_offset+10), (50, 50, 50), -1)
            cv2.putText(display_frame, f"{name.upper()}: {count}", (30, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 40

        # 4. CRITICAL WARNING: Speed Bump Alert
        if 'speed bump' in [k.lower() for k in current_counts.keys()]:
            # Flashing Alert logic based on frame_count
            if (frame_count // 5) % 2 == 0:
                # Warning Box in Center-Top
                cv2.rectangle(display_frame, (w//2 - 200, 80), (w//2 + 200, 140), (0, 0, 255), -1)
                cv2.putText(display_frame, "WARNING: SPEED BUMP AHEAD", (w//2 - 180, 120), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                # Draw dynamic scanning lines to the bump
                cv2.line(display_frame, (w//2, h-40), (w//2, 140), (0, 0, 255), 1)

        # 5. Footer: Performance Metrics
        current_playback_time = frame_count * frame_delay
        actual_elapsed_time = time.time() - start_time
        fps_real = frame_count / actual_elapsed_time if actual_elapsed_time > 0 else 0
        
        cv2.putText(display_frame, f"SENSORS: CAMERA_01 | RESOLUTION: {w}x{h} | PROC_FPS: {fps_real:.1f}", 
                    (20, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Display Final HUD
        cv2.imshow("Autopilot Vision - Speed Bump Detection", display_frame)
        
        # Timing control
        wait_time = 1
        if actual_elapsed_time < current_playback_time:
            wait_time = max(1, int((current_playback_time - actual_elapsed_time) * 1000))
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

        # Catch-up logic
        actual_elapsed_time = time.time() - start_time
        while actual_elapsed_time > current_playback_time + frame_delay:
            cap.grab()
            frame_count += 1
            current_playback_time = frame_count * frame_delay
            actual_elapsed_time = time.time() - start_time

    cap.release()
    cv2.destroyAllWindows()
    print("Detection finished.")

if __name__ == "__main__":
    detect_video_realtime()
