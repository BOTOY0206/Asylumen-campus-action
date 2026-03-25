# examples/read_video.py
# 最小读视频脚本：用 OpenCV 读取并显示视频（按 q 退出）
import cv2
import sys
import os

VIDEO_PATH = "data/sample_videos/sample1.mp4"  # 请确保仓库里有这个文件或修改为网盘路径

if not os.path.exists(VIDEO_PATH):
    print("视频文件不存在，请把 sample1.mp4 放在 data/sample_videos/ 或修改路径。")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("无法打开视频，请检查路径：", VIDEO_PATH)
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("frame", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
