import cv2
from ultralytics import YOLO

# Load a more accurate YOLO model
model = YOLO("yolov10m.pt")  # Try yolov8l.pt if your hardware can handle it

# Get person class ID
class_names = model.names
person_class_id = [cls_id for cls_id, name in class_names.items() if name == "person"][0]

# OR use direct m3u8 if you're okay with potential errors
stream_url = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1743477263/ei/rwXrZ6nrLvHzsfIPwZDH-Q0/ip/207.151.52.44/id/rnXIjl_Rzy4.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hls_chunk_host/rr2---sn-oxuuvn-a5ml.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/bui/AccgBcOd-76wmObPy4oa56MmRsjt3Z-DWub__9ItSCZLMDpw9lj9bMUkiHTYxzXaYbUZr6THH_SfiDo-/spc/_S3wKiQbg45AkmLZ7OqFwt861je77OV3DWX5pIdTFmykEvuhy0dDC0AdCs3jdpFbYBoiJJI/vprv/1/playlist_type/DVR/initcwndbps/4241250/met/1743455665,/mh/K2/mm/44/mn/sn-oxuuvn-a5ml/ms/lva/mv/m/mvi/2/pl/23/rms/lva,lva/dover/11/pacing/0/keepalive/yes/fexp/51355912,51435733/mt/1743455072/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,playlist_duration,manifest_duration,bui,spc,vprv,playlist_type/sig/AJfQdSswRQIgKIxRaA94FiXs1LwYj2MYfAIO7wO72Y6-3kXaqvQbaSYCIQCtvrn-CoV-TOgF80HzOLhH32-m0BHIxvzlHJb7ccxhJA%3D%3D/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/AFVRHeAwRQIhAPCXCnEVb5LIYlF57FteondN-PaGezCF38y83DWTsy9PAiAo4VOuYLfgayBNS_Cq-gTYuk51ZbIf4h2JNe8kWx7CUg%3D%3D/playlist/index.m3u8"

cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("Error: Could not open the video stream.")
    exit()

frame_counter = 0
print("Processing livestream frames (every 5th frame)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame. Retrying...")
        continue

    frame_counter += 1

    if frame_counter % 5 == 0:
        results = model(frame)
        result = results[0]
        detected_ids = result.boxes.cls.tolist()
        num_people = detected_ids.count(person_class_id)
        print(f"Frame {frame_counter}: {num_people} people detected")

        annotated = result.plot()
        cv2.imshow("YOLO Livestream Detection", annotated)
    else:
        cv2.imshow("YOLO Livestream Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
