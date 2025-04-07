
# from ultralytics import YOLO
# import cv2

# # Load model
# model = YOLO("yolov8m.pt")
# cap = cv2.VideoCapture("vegas_video.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO detection
#     results = model(frame, conf=0.08, verbose=False)

#     # Count number of 'person' class detections
#     person_count = 0
#     if results[0].boxes is not None and results[0].boxes.data.numel() > 0:
#         for box in results[0].boxes.data.tolist():
#             _, _, _, _, score, class_id = box
#             if int(class_id) == 0:  # class 0 is 'person'
#                 person_count += 1

#     print(f"People detected in frame: {person_count}")

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import random

# Load model
model = YOLO("yolov8m.pt")
video_path = "factory_video.mp4"
cap = cv2.VideoCapture(video_path)

# Setup for random frame display
frames_to_display = 3
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
random_frames = random.sample(range(1, frame_count), frames_to_display)
print(f"Debug: Will display frames: {random_frames}")

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Run YOLO detection
    results = model(frame, conf=0.08, verbose=False)

    # Count number of 'person' class detections
    person_count = 0
    if results[0].boxes is not None and results[0].boxes.data.numel() > 0:
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if int(class_id) == 0:  # class 0 is 'person'
                person_count += 1

                # Draw person bounding boxes if this is a display frame
                if frame_number in random_frames:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {int(score * 100)}%', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"People detected in frame {frame_number}: {person_count}")

    # Show only selected random frames
    if frame_number in random_frames:
        # Add overlay text with total person count for easy comparison
        cv2.putText(frame, f"Total People: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print(f">>> Displaying Frame {frame_number} (Count: {person_count}) <<<")
        cv2.imshow("Debug Frame", frame)
        cv2.waitKey(0)  # Wait for key press before continuing

cap.release()
cv2.destroyAllWindows()

