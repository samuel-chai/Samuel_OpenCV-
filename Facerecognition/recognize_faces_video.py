from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from tqdm import tqdm
import os

# 建立參數解析器
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", type=str, help="path to input video file (optional)")
ap.add_argument("-o", "--output", type=str, help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="face detection model: `hog` or `cnn`")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# 初始化影片來源
if args["input"]:
    vs = cv2.VideoCapture(args["input"])
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vs.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 20  # fallback for安全
    is_file = True
else:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    total_frames = 0
    fps = 20
    is_file = False

print("[INFO] starting video stream...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 設定影片格式
writer = None
frame_number = 0
progress = tqdm(total=total_frames) if total_frames > 0 else None

while True:
    if is_file:
        ret, frame = vs.read()
        if not ret:
            break
    else:
        frame = vs.read()

    frame_number += 1
    if progress:
        progress.update(1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    if args["output"]:
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(args["output"], fourcc, fps, (width, height))
        writer.write(frame)

    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# 清理資源
if progress:
    progress.close()
cv2.destroyAllWindows()
if is_file:
    vs.release()
else:
    vs.stop()
if writer is not None:
    writer.release()