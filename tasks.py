# tasks.py
import os
import json
import time
from pathlib import Path

# NOTE: 在 worker 进程里加载模型（避免 web 进程负担）
def run_infer(job_dir: str, video_path: str):
    """
    job_dir: str path to job directory (render_jobs/<job_id>)
    video_path: path to input video inside that job_dir
    """
    job_dir = Path(job_dir)
    status_file = job_dir / "status.json"
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        # 写入 running 状态
        status_file.write_text(json.dumps({"status": "running"}), encoding="utf-8")

        # 延迟导入 heavy libs，避免导入失败影响 web process
        import cv2
        import numpy as np
        from ultralytics import YOLO

        # model path 可以通过环境变量覆盖
        MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
        model = YOLO(MODEL_PATH)
        PERSON_CLASS = 0

        # 推理开始
        t0 = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frames_out = []
        frame_idx = 0
        action_stats = {"running": 0, "fighting": 0, "falling": 0, "climbing": 0}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, classes=[PERSON_CLASS], verbose=False)
            person_num = len(results[0].boxes)
            persons = []
            for b in results[0].boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                conf = round(float(b.conf[0]), 2)
                label = "fighting" if person_num >= 2 else "normal"
                persons.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": conf})

            frames_out.append({
                "time": round(frame_idx / fps, 3),
                "frame": frame_idx,
                "persons": persons
            })

            if person_num >= 2:
                action_stats["fighting"] += 1

            frame_idx += 1

        cap.release()
        duration = time.time() - t0

        # 写输出 JSON 到 data/results/<basename>.json
        RESULTS_DIR = Path("data") / "results"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_name = Path(video_path).stem + ".json"
        out_path = str(RESULTS_DIR / out_name)

        result_payload = {
            "total_frames": frame_idx,
            "duration_s": round(duration, 3),
            "action_stats": action_stats,
            "frames": frames_out
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)

        # 写 status.json 成功信息，包含 result 文件路径
        status = {"status": "success", "duration_s": round(duration, 3), "result": out_path}
        status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    except Exception as e:
        status_file.write_text(json.dumps({"status": "failed", "error": str(e)}), encoding="utf-8")
        raise
