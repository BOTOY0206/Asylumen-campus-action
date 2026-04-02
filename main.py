from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os
from typing import Dict

app = FastAPI(title="行为识别系统API")

# 任务存储
task_storage: Dict[str, Dict] = {}

# 导入你的推理 wrapper
from services.inference_wrapper import inference_wrapper

# 后台推理任务
def run_task(task_id, video_path):
    try:
        res = inference_wrapper(video_path)
        task_storage[task_id] = res
    except Exception as e:
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["error_message"] = str(e)

# 提交推理任务
@app.post("/infer_behavior")
async def infer_behavior(background_tasks: BackgroundTasks, video_path: str):
    task_id = str(uuid.uuid4())
    task_storage[task_id] = {
        "task_id": task_id,
        "status": "running",
        "video_path": video_path,
        "output_path": "",
        "error_message": ""
    }
    background_tasks.add_task(run_task, task_id, video_path)
    return {"task_id": task_id, "status": "running"}

# 查询任务状态
@app.get("/task/{task_id}")
async def get_task(task_id: str):
    return task_storage.get(task_id, {"status": "not_found"})

# 健康检查
@app.get("/health")
def health():
    return {"status": "ok"}
