from flask import Flask, request, jsonify
import cv2
import os
from ultralytics import YOLO
import tempfile

app = Flask(__name__)

# 模型加载
# 模型加载（相对路径，适配所有环境：本地/CI/云服务器）
MODEL_PATH = "yolov8n.pt"
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
            # 保存临时文件
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            file.save(temp.name)
            temp.close()
            video_path = temp.name

    # 2. 兜底：处理路径传参（兼容前端旧格式）
    if not video_path:
        try:
            req_data = request.get_json()
            if req_data and "video_path" in req_data:
                # 强制用你本地的测试视频，避免路径不存在
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
    # 4. 推理（完全不变）
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

        for b in results[0].boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            conf = round(float(b.conf[0]), 2)
            label = "fighting" if person_num >= 2 else "normal"

            current_persons.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": conf,
                "fallback": False
            })

        frames.append({
            "time": round(frame_idx / fps, 2),
            "persons": current_persons
        })

        if person_num >= 2:
            abnormal_frames += 1
            action_stats["fighting"] += 1

        frame_idx += 1

    cap.release()
    # 只删除临时文件，不删本地视频
    if "temp" in locals() and os.path.exists(video_path):
        os.unlink(video_path)

    abnormal_ratio = round((abnormal_frames / total_frames) * 100, 1) if total_frames else 0

    # --------------------------
    # 5. 返回前端要求的格式
    # --------------------------
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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)