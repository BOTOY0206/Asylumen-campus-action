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

# 行为标签 新增全部11类
LABEL_MAP = {
    0: "normal",
    1: "fighting",
    2: "lie",
    3: "running",
    4: "climbing",
    5: "stand",
    6: "squat",
    7: "hit",
    8: "kick",
    9: "slap",
    10: "point",
    11: "call",
    12: "smoke",
    13: "touch"
}


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
    # 4. 推理 统计字典同步新增所有行为
    # --------------------------
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    abnormal_frames = 0
    action_stats = {
        "running": 0, "fighting": 0, "lie": 0, "climbing": 0,
        "stand":0, "squat":0, "hit":0, "kick":0, "slap":0,
        "point":0, "call":0, "smoke":0, "touch":0, "normal":0
    }
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
            # 防护：关键点为空不崩溃
            # ======================
            if results[0].keypoints is None:
                current_persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "conf": conf,
                    "fallback": False
                })
                continue

            keypoints = results[0].keypoints[idx].xy.cpu().numpy()[0]
            if np.all(keypoints == 0):
                current_persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "conf": conf,
                    "fallback": False
                })
                continue

            # YOLO姿态17关键点
            nose = keypoints[0]
            shoulder_l = keypoints[5]
            shoulder_r = keypoints[6]
            hand_l = keypoints[9]
            hand_r = keypoints[10]
            hip = keypoints[11]
            knee_l = keypoints[13]
            knee_r = keypoints[14]
            ankle_l = keypoints[15]
            ankle_r = keypoints[16]

            # ==============================================
            # 🔥 核心：彻底删除 俩人=打斗
            # 只有多人紧贴重叠纠缠 → 才是fighting打斗
            # ==============================================
            fight_detected = False
            if person_num >= 2:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                for other_b in results[0].boxes:
                    if other_b == b: continue
                    ox1, oy1, ox2, oy2 = other_b.xyxy[0]
                    ocx = (ox1+ox2)/2
                    ocy = (oy1+oy2)/2
                    dist = np.hypot(cx-ocx, cy-ocy)
                    if dist < (x2-x1)*0.7:
                        fight_detected = True
                        break

            if abs(nose[1] - ankle_l[1]) < 80:
                label = "lie"
            # 下蹲 squat
            elif knee_l[1] > hip[1] + 60 or knee_r[1] > hip[1] + 60:
                label = "squat"
            # 站立 stand
            elif hip[1] - knee_l[1] > 70 and hip[1] - knee_r[1] > 70:
                label = "stand"
            # 踢腿 kick
            elif abs(knee_l[1] - knee_r[1]) > 90:
                label = "kick"
            # 手臂前伸打击 hit
            elif hand_l[0] < x1-40 or hand_r[0] > x2+40:
                label = "hit"
            # 抬手扇耳光 slap
            elif abs(hand_l[1]-nose[1])<50 or abs(hand_r[1]-nose[1])<50:
                label = "slap"
            # 手指指向 point
            elif hand_l[0]<x1-60 or hand_r[0]>x2+60:
                label = "point"
            # 手贴耳朵打电话 call
            elif abs(hand_l[1]-nose[1])<60 or abs(hand_r[1]-nose[1])<60:
                label = "call"
            # 手靠近嘴抽烟 smoke
            elif abs(hand_l[1]-nose[1]+25)<45 or abs(hand_r[1]-nose[1]+25)<45:
                label = "smoke"
            # 手贴身触摸 touch
            elif abs(hand_l[0]-x1)<50 or abs(hand_r[0]-x2)<50:
                label = "touch"
            # 攀爬 手臂高举
            elif shoulder_l[1] < hip[1]-110 or shoulder_r[1] < hip[1]-110:
                label = "climbing"
            # 奔跑 双腿交错
            elif abs(knee_l[1]-knee_r[1])>55:
                label = "running"
            # 其余全部正常
            else:
                label = "normal"

            current_persons.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": conf,
                "fallback": False
            })

        # 帧数据存入
        frames.append({
            "time": round(frame_idx / fps, 2),
            "persons": current_persons
        })

        # 全行为统计
        for p in current_persons:
            action_stats[p["label"]] += 1

        # 异常判断
        if any(p["label"] not in ["normal","stand"] for p in current_persons):
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