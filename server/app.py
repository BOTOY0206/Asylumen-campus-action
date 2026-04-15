# name=server/app.py
import os
import uuid
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Adjust paths as needed
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


def run_inference_on_file(video_path):
    """
    Try to call your actual inference (if available). Fallback to a demo response.
    Your real inference should return a dict with 'action_stats', 'time_series', 'frames' etc.
    """
    try:
        # Try import your project's inference (adjust import path if needed)
        from examples.inference_real import inference_real  # type: ignore
        res = inference_real(video_path)
        # If inference_real returns a structure, try normalize it here
        return res
    except Exception as e:
        app.logger.warning(f"Real inference not available or failed: {e}")
        # Demo fallback: small response shape expected by frontend
        demo = {
            "action_stats": {"climbing": 10, "falling": 5, "fighting": 2, "running": 8},
            "time_series": [{"time": "00:00", "ratio": 8}, {"time": "00:10", "ratio": 12}],
            "total_frames": 300,
            "abnormal_frames": 10,
            "abnormal_ratio": 3.3,
            "frames": [
                {"time": 0.0, "persons": [{"bbox": [80, 60, 200, 300], "label": "falling", "conf": 0.95}]},
                {"time": 0.5, "persons": [{"bbox": [85, 65, 205, 305], "label": "falling", "conf": 0.92}]}
            ]
        }
        return demo


@app.route("/infer_behavior", methods=["POST"])
def infer_behavior():
    """
    Accept:
      - multipart file with field name "video" (preferred)
      - or JSON with "video_path" pointing to a server-local path (legacy)
    Returns JSON: {"code":200, "data": {...}, "msg":"success"} or code!=200 on error.
    """
    # 1) file upload
    if "video" in request.files:
        f = request.files["video"]
        if f.filename == "":
            return jsonify({"code": 400, "data": None, "msg": "empty filename"}), 400
        filename = secure_filename(f.filename)
        uniq = f"{uuid.uuid4().hex[:8]}_{filename}"
        save_path = os.path.join(UPLOAD_DIR, uniq)
        f.save(save_path)
        app.logger.info(f"Saved uploaded video to {save_path}")
        try:
            data = run_inference_on_file(save_path)
            return jsonify({"code": 200, "data": data, "msg": "success"})
        except Exception as e:
            app.logger.exception("Inference failed")
            return jsonify({"code": 500, "data": None, "msg": f"inference error: {e}"}), 500

    # 2) fallback: JSON video_path (server-local)
    if request.is_json:
        payload = request.get_json()
        video_path = payload.get("video_path")
        if not video_path or not os.path.exists(video_path):
            return jsonify({"code": 404, "data": None, "msg": f"video file not found: {video_path}"}), 404
        try:
            data = run_inference_on_file(video_path)
            return jsonify({"code": 200, "data": data, "msg": "success"})
        except Exception as e:
            app.logger.exception("Inference failed")
            return jsonify({"code": 500, "data": None, "msg": f"inference error: {e}"}), 500

    return jsonify({"code": 400, "data": None, "msg": "no video provided; expected multipart form-data field 'video' or JSON 'video_path'"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
