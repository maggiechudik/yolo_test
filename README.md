# QueueView YOLO Object Detection Test

This directory contains scripts and data used to test the Ultralytics YOLOv8 object detection model as part of our QueueView project. Our goal is to incrementally build a system that detects and counts people—from static images to video streams—and eventually deploys it in a cloud pipeline.

## File Structure

- **📁 Annotations**: Contains COCO 2017 annotation files with ground-truth object counts.

- **📁 val2017**: Contains thousands of sample images (COCO 2017 validation set).

- **📄 extract_person_counts.py**: Parses the COCO annotations to extract person counts and outputs `person_counts.csv`.

- **📄 live_stream_analysis.py**: Analyzes every 5th frame of a sample livestream (can view it via `ffplay "ffplay "http://webcam.st-malo.com/axis-cgi/mjpg/video.cgi\?resolution\=352x288"` after doing `brew install ffmpeg` to see what the stream actually looks like)

- **📄 person_counts.csv**: CSV file mapping each image filename to its ground-truth person count.

- **📄 main.py**: Tests YOLO on a single image (e.g., `sample.jpg`) and prints detected objects and counts.

- **📄 batch_yolo_test.py**: Processes all images in the **val2017** folder, printing each image’s filename and the number of people detected.

- **📄 factory_video.mp4**: A sample video file used for testing video detection.

- **📄 video_detection.py**: Processes a video file frame-by-frame (or every 5th frame) to detect and count people.

- **📄 view_video.py**: Plays a video file for visual inspection.

- **📄 yolov8n.pt**: Pre-trained YOLOv8 nano model weights.

## How to Run

1. **Activate the Virtual Environment:**  You might need to create a new one?
   `source .venv/bin/activate`
2. **Install Dependencies:** Ensure required packages are installed:
   * **You should be able to just install everything by doing**: `pip install requirements.txt` 
   * But, these are the things you need off the top of my head:
     * `pip install --upgrade pip setuptools wheel`
     * `pip install opencv-python`
     * `pip install --upgrade pip`
     * `pip install ultralytics`
   * There could be other issues with this so lmk if you have errors, it also took sooo long to install all this stuff so it might be a little difficult to setup the dockerfile/image with these big packages
3. **Run a script**:
* Single image test: `python main.py`
* Batch image test (test on 1000s in **val2017**): `python batch_yolo_test.py`
* Video Processing (tests the factory_video.mp4 file): `python video_detection.py`

