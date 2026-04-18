# app_sqlite.py
import os
import uuid
import json
import sqlite3
from pathlib import Path
from flask import Flask, request, jsonify, send_file

DB_PATH = "jobs.db"
ROOT = Path.cwd()
WORKDIR = ROOT / "render_jobs"
RESULTS_DIR = ROOT / "data" / "results"
WORKDIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            filename TEXT,
            input_path TEXT,
            result_path TEXT,
            status TEXT,
            error TEXT,
            created_at REAL,
            started_at REAL,
            finished_at REAL
        )
        """)
        conn.commit()

def insert_job(job_id, filename, input_path):
    import time
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, input_path, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, filename, input_path, "queued", time.time())
        )
        conn.commit()

@app.route("/infer_behavior", methods=["POST"])
def create_job():
    f = request.files.get("video")
    if not f:
        return jsonify({"code":400,"msg":"no file","data":None}), 400

    job_id = uuid.uuid4().hex
    job_dir = WORKDIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    filename = f.filename
    input_path = str(job_dir / filename)
    f.save(input_path)

    insert_job(job_id, filename, input_path)

    return jsonify({
        "code":202,
        "msg":"accepted",
        "data":{
            "job_id": job_id,
            "status_url": f"/infer_behavior/{job_id}/status",
            "result_url": f"/infer_behavior/{job_id}/result"
        }
    }), 202

@app.route("/infer_behavior/<job_id>/status", methods=["GET"])
def job_status(job_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, status, error, result_path, created_at, started_at, finished_at FROM jobs WHERE id=?", (job_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({"code":404,"msg":"not found","data":None}), 404
        data = {
            "id": row[0],
            "status": row[1],
            "error": row[2],
            "result_path": row[3],
            "created_at": row[4],
            "started_at": row[5],
            "finished_at": row[6]
        }
        return jsonify({"code":200,"msg":"ok","data":data})

@app.route("/infer_behavior/<job_id>/result", methods=["GET"])
def job_result(job_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT status, result_path FROM jobs WHERE id=?", (job_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({"code":404,"msg":"not found","data":None}), 404
        status, result_path = row
        if status != "success":
            return jsonify({"code":202,"msg":"not ready","data":{"status":status}}), 202
        if not result_path or not os.path.exists(result_path):
            return jsonify({"code":500,"msg":"result missing","data":None}), 500
        return send_file(result_path, as_attachment=True)

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
