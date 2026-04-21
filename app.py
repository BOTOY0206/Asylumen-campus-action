from flask import Flask, request, jsonify
import cv2
import os
from ultralytics import YOLO
import tempfile
import numpy as np

app = Flask(__name__)

# 模型加载
MODEL_PATH = "yolov8n-pose.pt"
model = YOLO(MODEL_PATH)
PERSON_CLASS = 0

# ====================== 异常优先级排序！前端图表直接按这个顺序展示 ======================
# 越靠前越危险，统计图优先显示，不会被stand/normal挤下去
ABNORMAL_PRIORITY = [
    "fighting", "lie", "climbing", "hit", "slap",
    "running", "kick", "call", "smoke", "point"
]
# 完全正常行为：不计入异常帧
NORMAL_ACTION = ["stand", "touch", "normal"]

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

slap_frame_history = {}

# ====================== 和摄像头脚本1:1完全同步行为判断 ======================
def get_video_action_type(keypoints, person_box, frame_id):
    x1, y1, x2, y2 = person_box
    body_h = y2 - y1
    body_w = x2 - x1
    body_ratio = body_h / body_w

    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    hip_avg_y = (left_hip[1] + right_hip[1]) / 2
    knee_diff = abs(left_knee[1] - right_knee[1])

    # 高危优先判断
    if left_wrist[1] < nose[1] - body_h*0.05 and right_wrist[1] < nose[1] - body_h*0.05:
        return "climbing"
    if abs(left_wrist[1] - nose[1]) < body_h*0.25 or abs(right_wrist[1] - nose[1]) < body_h*0.25:
        return "call"
    if abs(left_wrist[1] - (nose[1]+body_h*0.1)) < body_h*0.2 or abs(right_wrist[1] - (nose[1]+body_h*0.1)) < body_h*0.2:
        return "smoke"

    global slap_frame_history
    if frame_id not in slap_frame_history:
        slap_frame_history[frame_id] = [left_wrist, right_wrist]
    slap_flag = False
    if (frame_id - 1) in slap_frame_history:
        old_lw, old_rw = slap_frame_history[frame_id - 1]
        speed = max(abs(left_wrist[0] - old_lw[0]), abs(right_wrist[0] - old_rw[0]))
        if speed > body_h * 0.12 and abs(left_wrist[1] - nose[1]) < body_h*0.3:
            slap_flag = True
    if len(slap_frame_history) > 10:
        keys = sorted(slap_frame_history.keys())
        for k in keys[:-10]:
            del slap_frame_history[k]
    if slap_flag:
        return "slap"

    if left_wrist[0] < x1 - body_w*0.08 or right_wrist[0] > x2 + body_w*0.08:
        return "point"
    if body_ratio < 1.15 and abs(nose[1] - left_ankle[1]) < body_h * 0.35:
        return "lie"
    if knee_diff < body_h * 0.1:
        return "stand"
    if left_knee[1] > hip_avg_y or right_knee[1] > hip_avg_y:
        return "squat"
    if knee_diff > body_h * 0.18:
        return "kick"

    # 奔跑：必须双腿大幅错开，坐着永远不触发
    left_arm_bend = abs(left_wrist[1]-left_elbow[1]) < body_h*0.18
    right_arm_bend = abs(right_wrist[1]-right_elbow[1]) < body_h*0.18
    if left_arm_bend and right_arm_bend and knee_diff > body_h * 0.15:
        return "running"

    if nose[0] < x1+body_w*0.1 or nose[0] > x2-body_w*0.1:
        return "hit"
    if (x1<left_wrist[0]<x2) or (x1<right_wrist[0]<x2):
        return "touch"

    return "normal"


@app.route("/infer_behavior", methods=["POST"])
def infer_behavior():
    video_path = None

    if "video" in request.files:
        file = request.files["video"]
        if file.filename != "":
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            file.save(temp.name)
            temp.close()
            video_path = temp.name

    if not video_path:
        try:
            req_data = request.get_json()
            if req_data and "video_path" in req_data:
                video_path = req_data["video_path"]
        except:
            pass

    if not video_path or not os.path.exists(video_path):
        return jsonify({
            "code": 404,
            "msg": f"视频不存在：{video_path}",
            "data": None
        }), 404

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    abnormal_frames = 0
    action_stats = {k:0 for k in ABNORMAL_PRIORITY + NORMAL_ACTION}
    time_series = []
    frames = []
    frame_idx = 0
    global slap_frame_history
    slap_frame_history.clear()

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

            if results[0].keypoints is None or len(results[0].keypoints) == 0:
                current_persons.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": conf})
                continue

            keypoints = results[0].keypoints[idx].xy.cpu().numpy()[0]
            if np.all(keypoints == 0):
                current_persons.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": conf})
                continue

            # 多人打斗最高优先级
            fight_detected = False
            if person_num >= 2:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                for other_b in results[0].boxes:
                    if np.array_equal(other_b.xyxy[0].cpu().numpy(), b.xyxy[0].cpu().numpy()):
                        continue
                    ox1, oy1, ox2, oy2 = other_b.xyxy[0].cpu().numpy()
                    ocx = (ox1+ox2)/2
                    ocy = (oy1+oy2)/2
                    dist = np.hypot(cx-ocx, cy-ocy)
                    if dist < (x2-x1)*0.7:
                        fight_detected = True
                        break

            if fight_detected:
                label = "fighting"
            else:
                label = get_video_action_type(keypoints, [x1,y1,x2,y2], frame_idx)

            current_persons.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": conf})

        frames.append({"time": round(frame_idx/fps,2), "persons": current_persons})

        # 统计：异常行为优先累加
        for p in current_persons:
            action_stats[p["label"]] += 1

        # ✅ 关键：只有高危异常才算异常帧！stand/normal/touch 不算异常！
        if any(p["label"] in ABNORMAL_PRIORITY for p in current_persons):
            abnormal_frames += 1

        frame_idx += 1

    cap.release()
    if video_path and os.path.exists(video_path):
        os.unlink(video_path)

    abnormal_ratio = round((abnormal_frames / total_frames) * 100, 1) if total_frames else 0

    # 接口返回：按异常危险度排序！前端柱状图自动优先显示异常
    sorted_action_stats = {k:action_stats[k] for k in ABNORMAL_PRIORITY + NORMAL_ACTION}

    return jsonify({
        "code": 200,
        "msg": "success",
        "data": {
            "total_frames": total_frames,
            "abnormal_frames": abnormal_frames,
            "abnormal_ratio": abnormal_ratio,
            "action_stats": sorted_action_stats,
            "time_series": time_series,
            "frames": frames,
            "result_video_url": ""
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)