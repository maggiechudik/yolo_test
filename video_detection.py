from ultralytics import YOLO

# Load the pretrained YOLOv8 nano model (yolov8n.pt)
model = YOLO("yolov8n.pt")
class_names = model.names

# Find the class ID for "person" from the model's class names dictionary.
person_class_id = [cls_id for cls_id, name in class_names.items() if name == "person"][0]

# Path to the video file (make sure you've downloaded it)
video_path = "factory_video.mp4"

# Use stream=True so that YOLO processes the video frame-by-frame as a generator
results_generator = model(video_path, stream=True)

frame_counter = 0

print("Processing video frames (processing every 5th frame):\n")
for result in results_generator:
    frame_counter += 1
    # Process every 5th frame to reduce computation load
    if frame_counter % 5 != 0:
        continue

    # Extract detections from the current frame result
    detected_ids = result.boxes.cls.tolist()
    num_people = detected_ids.count(person_class_id)

    print(f"Frame {frame_counter}: {num_people} people detected")
