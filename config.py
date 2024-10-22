from ultralytics import YOLO

# Function to load YOLO model
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

# Character class names
charclassnames = ['0', '9', 'b', 'd', 'ein', 'ein', 'g', 'gh', 'h', 'n', 'm', '1', 'malul', 'n', 's', 'sad', 't', 'ta', 'v', 'y', '2', '3', '4', '5', '6', '7', '8']
