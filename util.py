import cv2

from config import charclassnames


# Function to detect plates
def detect_plates(image, model):
    results = model(image)
    return results

# Function to detect characters in a plate region
def detect_characters(plate_region, model):
    plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)
    results = model(plate_region)
    detected_classes = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            detected_classes.append((int(cls), int(x1)))

    # Sort detected characters
    detected_classes.sort(key=lambda x: x[1])
    detected_characters = ''.join([charclassnames[cls] for cls, _ in detected_classes if cls < len(charclassnames)])
    
    return detected_characters

def process_video(video_file_path, plate_model, ocr_model, output_file_path, draw_text_with_background):
    cap = cv2.VideoCapture(video_file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_plates(frame, plate_model)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    plate_region = frame[int(y1):int(y2), int(x1):int(x2)]
                    detected_characters = detect_characters(plate_region, ocr_model)

                    # Draw bounding box and text with background for each plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    draw_text_with_background(frame, detected_characters, (int(x1), int(y1) - 10))

        out.write(frame)

    cap.release()
    out.release()
