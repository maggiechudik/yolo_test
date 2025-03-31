from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (using the 'nano' version for speed)
model = YOLO("yolov8n.pt")

# Run inference on the sample image
results = model("sample.jpg")

# Get the results for the first (and only) image
result = results[0]

# YOLOv8 models are trained on the COCO dataset.
# The class mapping is available via model.names, which is a dict like {0: 'person', 1: 'bicycle', ...}
class_names = model.names
print("Class names:", class_names)

# Find the class ID for 'person'
person_class_id = [cls_id for cls_id, name in class_names.items() if name == "person"][0]

# Extract detected class IDs for this image
detected_class_ids = result.boxes.cls.tolist()

# Count how many detections correspond to 'person'
num_people = detected_class_ids.count(person_class_id)

print(f"Number of people detected: {num_people}")
