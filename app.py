# app.py
import os
import uuid
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file, url_for
from redis import Redis
from rq import Queue

app = Flask(__name__)

# Redis / RQ 配置（本地开发默认）
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT)
q = Queue("default", connection=redis_conn)

# 目录配置
ROOT = Path.cwd()
WORKDIR = ROOT / "render_jobs"
RESULTS_DIR = ROOT / "data" / "results"
WORKDIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# POST /infer_behavior: 上传视频或传 video_path 并入队处理
@app.route("/infer_behavior", methods=["POST"])
def infer_behavior():
    # 支持文件上传或 JSON 传 video_path
    video_path = None
    filename = None

    if "video" in request.files:
        file = request.files["video"]
        if file.filename != "":
            job_id = uuid.uuid4().hex
            job_dir = WORKDIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            filename = file.filename
            input_path = str(job_dir / filename)
            file.save(input_path)
            video_path = input_path
    else:
        # 尝试从 JSON body 读取 video_path（兼容旧前端）
        try:
            req_data = request.get_json() or {}
            if "video_path" in req_data and req_data["video_path"]:
                # 如果客户端传来绝对/相对路径，先验证文件存在
                cand = req_data["video_path"]
                if os.path.exists(cand):
                    job_id = uuid.uuid4().hex
                    job_dir = WORKDIR / job_id
                    job_dir.mkdir(parents=True, exist_ok=True)
                    # 复制到 job 目录（避免后续被外部删除）
                    import shutil
                    filename = os.path.basename(cand)
                    input_path = str(job_dir / filename)
                    shutil.copyfile(cand, input_path)
                    video_path = input_path
                else:
                    return jsonify({"code": 404, "msg": f"video_path not found: {cand}", "data": None}), 404
        except Exception:
            return jsonify({"code": 400, "msg": "invalid request body", "data": None}), 400

    if not video_path:
        return jsonify({"code": 400, "msg": "no video uploaded or video_path provided", "data": None}), 400

    # 确保 job_id, job_dir, input_path 已设置
    if 'job_id' not in locals():
        job_id = uuid.uuid4().hex
        job_dir = WORKDIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        # move/copy input into job_dir if needed
        if video_path and Path(video_path).parent != job_dir:
            import shutil
            filename = filename or os.path.basename(video_path)
            input_path = str(job_dir / filename)
            shutil.copyfile(video_path, input_path)
            video_path = input_path

    # 创建初始状态文件
    status_file = job_dir / "status.json"
    status_file.write_text(json.dumps({"status": "queued"}), encoding="utf-8")

    # 入队：tasks.run_infer(job_dir_str, video_path)
    from tasks import run_infer
    rq_job = q.enqueue(run_infer, str(job_dir), str(video_path))

    return jsonify({
        "code": 202,
        "msg": "job accepted",
        "data": {
            "job_id": job_id,
            "rq_id": rq_job.get_id(),
            "status_url": url_for("get_status", job_id=job_id, _external=True),
            "result_url": url_for("get_result", job_id=job_id, _external=True)
        }
    }), 202

@app.route("/infer_behavior/<job_id>/status", methods=["GET"])
def get_status(job_id):
    job_dir = WORKDIR / job_id
    status_file = job_dir / "status.json"
    if not job_dir.exists() or not status_file.exists():
        return jsonify({"code": 404, "msg": "job not found", "data": None}), 404
    data = json.loads(status_file.read_text(encoding="utf-8"))
    return jsonify({"code": 200, "msg": "ok", "data": data})

@app.route("/infer_behavior/<job_id>/result", methods=["GET"])
def get_result(job_id):
    job_dir = WORKDIR / job_id
    status_file = job_dir / "status.json"
    if not job_dir.exists() or not status_file.exists():
        return jsonify({"code": 404, "msg": "job not found", "data": None}), 404
    data = json.loads(status_file.read_text(encoding="utf-8"))
    if data.get("status") != "success":
        return jsonify({"code": 202, "msg": "result not ready", "data": data}), 202
    result_path = data.get("result")
    if not result_path or not os.path.exists(result_path):
        return jsonify({"code": 500, "msg": "result file missing", "data": None}), 500
    # 返回 JSON 文件下载
    return send_file(result_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
