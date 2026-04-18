# worker_sqlite.py
import os
import time
import json
import sqlite3
from pathlib import Path

DB_PATH = "jobs.db"
POLL_INTERVAL = 2.0  # seconds
RESULTS_DIR = Path("data") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def claim_job(conn):
    """
    Find one queued job and atomically set to running.
    Returns the job row or None.
    """
    cur = conn.cursor()
    # Use a transaction to avoid races
    cur.execute("BEGIN IMMEDIATE")
    row = cur.execute("SELECT id, input_path FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
    if not row:
        conn.commit()
        return None
    job_id, input_path = row
    import time
    started_at = time.time()
    cur.execute("UPDATE jobs SET status=?, started_at=? WHERE id=?", ("running", started_at, job_id))
    conn.commit()
    return {"id": job_id, "input_path": input_path}

def set_job_result(conn, job_id, status, result_path=None, error=None):
    import time
    finished_at = time.time()
    cur = conn.cursor()
    cur.execute("UPDATE jobs SET status=?, result_path=?, error=?, finished_at=? WHERE id=?",
                (status, result_path, error, finished_at, job_id))
    conn.commit()

def run_inference_on(video_path):
    # call your existing function from examples/inference_real
    from examples.inference_real import inference_real
    # inference_real writes JSON file to data/results/<basename>.json and returns result list
    return inference_real(video_path)

def main_loop():
    print("worker started, polling DB:", DB_PATH)
    while True:
        try:
            with sqlite3.connect(DB_PATH, timeout=30) as conn:
                job = claim_job(conn)
                if not job:
                    time.sleep(POLL_INTERVAL)
                    continue
                job_id = job["id"]
                input_path = job["input_path"]
                print(f"claimed job {job_id} -> {input_path}")
                try:
                    t0 = time.time()
                    res = run_inference_on(input_path)
                    duration = round(time.time() - t0, 3)
                    out_name = Path(input_path).stem + ".json"
                    out_path = str(Path("data") / "results" / out_name)
                    # ensure out_path exists (inference_real should write it)
                    if not Path(out_path).exists() and res:
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(res, f, ensure_ascii=False, indent=2)
                    set_job_result(conn, job_id, "success", result_path=out_path)
                    print(f"job {job_id} done in {duration}s -> {out_path}")
                except Exception as e:
                    print("job failed:", e)
                    set_job_result(conn, job_id, "failed", error=str(e))
        except Exception as e:
            print("worker DB loop error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main_loop()
