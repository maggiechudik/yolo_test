import csv
import os
from ultralytics import YOLO

# --- SETTINGS ---
# Path to the directory containing COCO validation images (update as needed)
IMAGES_DIR = "val2017"
# Path to the CSV file containing ground truth person counts
CSV_FILE = "person_counts.csv"

# --- LOAD YOLO MODEL ---
model = YOLO("yolov8n.pt")
class_names = model.names
# In the COCO dataset, the "person" category is typically ID 0
person_class_id = [cls_id for cls_id, name in class_names.items() if name == "person"][0]

# --- READ GROUND TRUTH COUNTS FROM CSV ---
gt_counts = {}
with open(CSV_FILE, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = row["filename"]
        gt = int(row["person_count"])
        gt_counts[filename] = gt

# --- INITIALIZE METRICS ---
total_images = 0
exact_matches = 0
total_abs_error = 0

# --- PROCESS EACH IMAGE ---
print("Processing images...\n")
for filename, gt_count in gt_counts.items():
    img_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(img_path):
        print(f"{filename}: Image not found. Skipping.")
        continue

    # Run YOLO on the image
    results = model(img_path)
    result = results[0]
    # Extract detected class IDs (a list)
    detected_ids = result.boxes.cls.tolist()
    pred_count = detected_ids.count(person_class_id)

    # Update metrics
    total_images += 1
    abs_error = abs(pred_count - gt_count)
    total_abs_error += abs_error
    if pred_count == gt_count:
        exact_matches += 1

    # Print a clean summary line for this image
    print(f"{filename}: GT = {gt_count}, Predicted = {pred_count}, Error = {abs_error}")

# --- COMPUTE OVERALL METRICS ---
if total_images > 0:
    exact_accuracy = (exact_matches / total_images) * 100
    mae = total_abs_error / total_images
    print("\n--- Summary ---")
    print(f"Total images processed: {total_images}")
    print(f"Exact count match accuracy: {exact_accuracy:.2f}%")
    print(f"Mean Absolute Error (MAE): {mae:.2f} persons per image")
else:
    print("No images processed.")
