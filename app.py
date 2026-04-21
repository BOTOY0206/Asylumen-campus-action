from flask import Flask, request, jsonify
import cv2
import os
from ultralytics import YOLO
import tempfile
import numpy as np

app = Flask(__name__)

# ====================== 加载你训练好的专属行为识别模型 ======================
MODEL_PATH = "best.pt"  # 训练完的模型放在根目录，直接读取
model = YOLO(MODEL_PATH)

# ====================== 行为标签（和你训练的完全一致） ======================
ABNORMAL_PRIORITY = [
    "fighting", "lie", "climbing", "hit", "slap",
    "running", "kick", "squat", "call", "smoke", "point"
]
NORMAL_ACTION = ["stand", "touch", "normal"]
ALL_LABELS = ABNORMAL_PRIORITY + NORMAL_ACTION

# ====================== 视频行为识别接口（前端完全兼容） ======================
@app.route("/infer_behavior", methods=["POST"])
def infer_behavior():
    video_path = None

    # 接收前端上传的视频文件
    if "video" in request.files:
        file = request.files["video"]
        if file.filename != "":
            # 创建临时文件保存视频
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            file.save(temp.name)
            temp.close()
            video_path = temp.name

    # 视频不存在校验
    if not video_path or not os.path.exists(video_path):
        return jsonify({
            "code": 404,
            "msg": "视频不存在",
            "data": None
        }), 404

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # 行为统计初始化
    action_stats = {k: 0 for k in ALL_LABELS}
    abnormal_frames = 0
    frames_result = []
    frame_idx = 0

    # 逐帧AI识别行为
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO模型自动预测行为（无任何人工规则！）
        results = model(frame, verbose=False)
        current_persons = []

        for res in results:
            if res.probs is None:
                continue

            # 获取置信度最高的行为标签
            label = res.names[res.probs.top1]
            conf = round(float(res.probs.top1conf), 2)

            current_persons.append({
                "bbox": [0, 0, frame.shape[1], frame.shape[0]],
                "label": label,
                "conf": conf
            })

            # 统计行为次数
            if label in action_stats:
                action_stats[label] += 1

        # 保存每一帧结果
        frames_result.append({
            "time": round(frame_idx / fps, 2),
            "persons": current_persons
        })

        # 判断是否为异常帧
        if any(p["label"] in ABNORMAL_PRIORITY for p in current_persons):
            abnormal_frames += 1

        frame_idx += 1

    # 释放资源 + 删除临时视频
    cap.release()
    os.unlink(video_path)

    # 计算异常比例
    abnormal_ratio = round((abnormal_frames / total_frames) * 100, 1) if total_frames else 0

    # 返回结果（和旧接口格式完全一样！前端不用改）
    return jsonify({
        "code": 200,
        "msg": "success",
        "data": {
            "total_frames": total_frames,
            "abnormal_frames": abnormal_frames,
            "abnormal_ratio": abnormal_ratio,
            "action_stats": action_stats,
            "time_series": [],
            "frames": frames_result,
            "result_video_url": ""
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)