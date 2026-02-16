import cv2
import os

video_path = r'c:\Users\Raymond Tie\Desktop\speedBumpDetection\road-anomaly-detection\test_video.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {frame_count}")
cap.release()
