import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# ----------------------------
# Load YOLOv8 model
# ----------------------------
MODEL_PATH = "best.pt"  # Path to your trained YOLOv8 model
model = YOLO(MODEL_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Hygiene Compliance Detection 🧴😷")
st.write("Detect mask, hairnet, etc. in real-time!")

use_webcam = st.checkbox("Use Webcam for Real-time Detection")

video_file = None
if not use_webcam:
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

FRAME_WINDOW = st.image([])

# ----------------------------
# Helper function to process frames
# ----------------------------
def process_frame(frame):
    # YOLOv8 expects RGB images
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict(img_rgb, verbose=False, stream=False)
    
    # Loop through results and draw bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            label = model.names[cls]

            # Skip unwanted class (gloves)
            if label.lower() == "gloves":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
            conf = box.conf[0].item()

            # Choose color: green if compliant, red if non-compliant
            if label in ["no_mask", "no_hairnet"]:
                color = (255, 0, 0)  # Red
            else:
                color = (0, 255, 0)  # Green

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# ----------------------------
# Real-time Webcam Stream
# ----------------------------
if use_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam")
            break

        img = process_frame(frame)
        stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

else:
    # ----------------------------
    # Video Upload
    # ----------------------------
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = process_frame(frame)
            FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))