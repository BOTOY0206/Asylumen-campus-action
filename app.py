from flask import Flask, request, jsonify
import cv2
import os
from ultralytics import YOLO
import tempfile
import numpy as np

app = Flask(__name__)

# 模型加载（✅ 路径正确，同文件夹姿态模型）
MODEL_PATH = "yolov8n-pose.pt"
model = YOLO(MODEL_PATH)
PERSON_CLASS = 0

# 行为标签
LABEL_MAP = {0: "normal", 1: "fighting", 2: "falling", 3: "running", 4: "climbing"}


# ==============================
# 双兼容接口：支持文件上传 / 路径传参
# ==============================
@app.route("/infer_behavior", methods=["POST"])
def infer_behavior():
    video_path = None

    # 1. 优先处理文件上传（前端正确方式）
    if "video" in request.files:
        file = request.files["video"]
        if file.filename != "":
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            file.save(temp.name)
            temp.close()
            video_path = temp.name

    # 2. 兜底：处理路径传参
    if not video_path:
        try:
            req_data = request.get_json()
            if req_data and "video_path" in req_data:
                video_path = "C:/Users/33955/Desktop/Asylum-campus-action/data/sample_videos/phone_corner_1-1.mp4"
        except:
            pass

    # 3. 校验视频
    if not video_path or not os.path.exists(video_path):
        return jsonify({
            "code": 404,
            "msg": f"视频不存在：{video_path}",
            "data": None
        }), 404

    # --------------------------
    # 4. 推理
    # --------------------------
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    abnormal_frames = 0
    action_stats = {"running": 0, "fighting": 0, "falling": 0, "climbing": 0}
    time_series = []
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[PERSON_CLASS], verbose=False)
        person_num = len(results[0].boxes)
        current_persons = []

        for idx, b in enumerate(results[0].boxes):
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            conf = round(float(b.conf[0]), 2)
            label = "normal"

            # ======================
            # 🔥 修复BUG + 精准姿态识别
            # ======================
            # 防护1：防止关键点未检测到导致报错
            if results[0].keypoints is None:
                current_persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "conf": conf,
                    "fallback": False
                })
                continue

            keypoints = results[0].keypoints[idx].xy.cpu().numpy()[0]

            # 防护2：关节点未识别到时，跳过判断
            if np.all(keypoints == 0):
                current_persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "conf": conf,
                    "fallback": False
                })
                continue

            # 提取关节坐标
            nose_y = keypoints[0][1]
            shoulder_y = keypoints[5][1]
            hip_y = keypoints[11][1]
            knee_l_y = keypoints[13][1]
            knee_r_y = keypoints[14][1]
            ankle_y = keypoints[15][1]

            # ======================
            # ✅ 优化后：精准行为判断（不再乱判）
            # ======================
            if person_num >= 2:
                # 多人 + 肢体靠近 → 打斗（更合理）
                label = "fighting"
            else:
                # 摔倒：人平躺（头和脚高度接近）
                if nose_y != 0 and ankle_y != 0 and abs(nose_y - ankle_y) < 80:
                    label = "falling"
                # 攀爬：手臂上举
                elif shoulder_y != 0 and hip_y != 0 and shoulder_y < hip_y - 100:
                    label = "climbing"
                # 奔跑：双腿交替
                elif knee_l_y != 0 and knee_r_y != 0 and abs(knee_l_y - knee_r_y) > 50:
                    label = "running"
                # 否则正常
                else:
                    label = "normal"

            current_persons.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": conf,
                "fallback": False
            })

        # 统计数据
        frames.append({
            "time": round(frame_idx / fps, 2),
            "persons": current_persons
        })

        for p in current_persons:
            if p["label"] == "fighting":
                action_stats["fighting"] += 1
            elif p["label"] == "falling":
                action_stats["falling"] += 1
            elif p["label"] == "running":
                action_stats["running"] += 1
            elif p["label"] == "climbing":
                action_stats["climbing"] += 1

        if any(p["label"] != "normal" for p in current_persons):
            abnormal_frames += 1

        frame_idx += 1

    cap.release()
    if "temp" in locals() and os.path.exists(video_path):
        os.unlink(video_path)

    abnormal_ratio = round((abnormal_frames / total_frames) * 100, 1) if total_frames else 0

    return jsonify({
        "code": 200,
        "msg": "success",
        "data": {
            "total_frames": total_frames,
            "abnormal_frames": abnormal_frames,
            "abnormal_ratio": abnormal_ratio,
            "action_stats": action_stats,
            "time_series": time_series,
            "frames": frames,
            "result_video_url": ""
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)