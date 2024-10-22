# app.py
import streamlit as st
from config import load_yolo_model
from util import detect_plates, detect_characters, process_video
from PIL import Image
import numpy as np
import cv2
import os

# Streamlit UI
st.title("Plate Detection and Character Recognition")
upload_type = st.radio("Select Upload Type", ["Image", "Video"])

# Load models
plate_model_path = "Models\Plate-best.pt"  # Update with the path to Plate.pt
ocr_model_path = "Models\OCR-best.pt"  # Update with the path to ocr.pt
plate_model = load_yolo_model(plate_model_path)
ocr_model = load_yolo_model(ocr_model_path)

def draw_text_with_background(image, text, position, font_scale=0.5, font_thickness=1):
    """ Draw text with a black background on an image. """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Coordinates for black rectangle (background)
    x, y = position
    rectangle_pt1 = (x, y - text_size[1] - 5)  # Top-left corner of the rectangle
    rectangle_pt2 = (x + text_size[0] + 2, y)  # Bottom-right corner of the rectangle

    # Draw the black rectangle background
    cv2.rectangle(image, rectangle_pt1, rectangle_pt2, (0, 0, 0), thickness=cv2.FILLED)

    # Draw the text on top of the black rectangle
    cv2.putText(image, text, (x, y - 2), font, font_scale, (255, 255, 255), font_thickness)

if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        results = detect_plates(image, plate_model)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    plate_region = image[int(y1):int(y2), int(x1):int(x2)]
                    detected_characters = detect_characters(plate_region, ocr_model)

                    # Draw bounding box for detected plate
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Add text with black background
                    draw_text_with_background(image, detected_characters, (int(x1), int(y1) - 10))

        st.image(image, caption="Detected Plates", use_column_width=True)

elif upload_type == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    if uploaded_file is not None:
        video_file_path = uploaded_file.name
        with open(video_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(video_file_path)

        output_file_path = "output_video.mp4"
        process_video(video_file_path, plate_model, ocr_model, output_file_path, draw_text_with_background)

        st.video(output_file_path)
