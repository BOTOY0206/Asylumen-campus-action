# tasks.py
import os
import json
import time
from pathlib import Path

def run_infer(job_dir: str, video_path: str):
    """
    RQ worker will call this function.
    - job_dir: path to render_jobs/<job_id>
    - video_path: path to the input video (inside job_dir)
    """
    job_dir = Path(job_dir)
    status_file = job_dir / "status.json"
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        # 标记为 running
        status_file.write_text(json.dumps({"status": "running"}), encoding="utf-8")

        # 延迟导入 heavy libs 和你的推理脚本，避免在 web 进程加载
        import time as _time
        # 使用你仓库里的 inference_real 函数
        from examples.inference_real import inference_real

        t0 = _time.time()
        # 调用现有函数（它会写入 data/results/<name>.json）
        result = inference_real(video_path)
        duration = _time.time() - t0

        # 推断输出文件约定：data/results/<basename>.json
        out_name = Path(video_path).stem + ".json"
        out_path = str(Path("data") / "results" / out_name)

        # 如果 inference_real 返回空但文件存在，也算成功
        if not Path(out_path).exists() and result:
            # 兜底：把内存 result 写到 out_path
            Path("data/results").mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        status = {
            "status": "success",
            "duration_s": round(duration, 3),
            "result": out_path if Path(out_path).exists() else None
        }
        status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    except Exception as e:
        # 捕获异常并写入 status.json，确保能在 Web 端看到错误
        err = {"status": "failed", "error": str(e)}
        try:
            status_file.write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        # 把异常继续抛出（RQ worker 会记录堆栈）
        raise
