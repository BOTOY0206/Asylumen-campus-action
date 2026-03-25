# examples/inference_stub.py
# 假检测占位：读取视频并生成 data/sample_videos/demo_log.json（随机 bbox + label）
import cv2
import json
import time
import random
import os

VIDEO_PATH = "data/sample_videos/sample1.mp4"
OUT_LOG = "data/sample_videos/demo_log.json"

if not os.path.exists(VIDEO_PATH):
    print("视频文件不存在，请把 sample1.mp4 放在 data/sample_videos/ 或修改路径。")
    exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("无法打开视频，请检查路径：", VIDEO_PATH)
    exit(1)

results = []
frame_idx = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    ts = time.time() - start_time
    persons = []
    for pid in range(random.randint(0, 2)):
        h, w = frame.shape[:2]
        x1 = random.randint(0, max(0, w-100))
        y1 = random.randint(0, max(0, h-100))
        x2 = min(w, x1 + random.randint(30, 200))
        y2 = min(h, y1 + random.randint(30, 200))
        label = random.choice(["normal", "loiter", "phone", "fall"])
        conf = round(random.uniform(0.5, 0.99), 2)
        persons.append({"person_id": pid+1, "bbox": [x1,y1,x2,y2], "label": label, "conf": conf})
    results.append({"time": ts, "frame": frame_idx, "persons": persons})
    frame_idx += 1

os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)
with open(OUT_LOG, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("已生成假检测日志：", OUT_LOG)
