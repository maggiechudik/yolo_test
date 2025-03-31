import json
import os

# Path to the COCO annotations JSON file (update if needed)
ANNOTATIONS_FILE = "annotations/instances_val2017.json"
# Path to the folder containing validation images
IMAGES_DIR = "val2017"

# Load the annotations JSON file
with open(ANNOTATIONS_FILE, "r") as f:
    data = json.load(f)

# Build a dictionary: image_id -> image filename
image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# Build a dictionary to count persons per image_id
person_counts = {}

# In COCO, the "person" category has category_id 1.
PERSON_CATEGORY_ID = 1

# Iterate over all annotations and count persons for each image.
for ann in data["annotations"]:
    if ann["category_id"] == PERSON_CATEGORY_ID:
        image_id = ann["image_id"]
        person_counts[image_id] = person_counts.get(image_id, 0) + 1

# Now create a mapping from filename to person count.
filename_to_count = {}
for image_id, count in person_counts.items():
    filename = image_id_to_filename.get(image_id)
    if filename:
        filename_to_count[filename] = count

# Optionally, write the mapping to a CSV file for easier viewing.
import csv

with open("person_counts.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "person_count"])
    for filename, count in filename_to_count.items():
        writer.writerow([filename, count])

print("Done! Person counts saved to person_counts.csv")
