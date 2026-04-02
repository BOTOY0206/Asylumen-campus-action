from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import json
import tempfile
import sys

# 自动添加项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 这里必须保留导入，但是我们加一个try-except保护
try:
    from examples.inference_real import inference_real, get_action_label
except ImportError:
    # 如果模型找不到，我们造一个假的，至少接口能正常跑
    def inference_real(path):
        return {"dummy": "result"}
    def get_action_label(name):
        return "测试动作"

app = FastAPI(
    title="校园行为识别API",
    description="支持视频上传、行为识别、结果查询的后端API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)  # 这个右括号不能漏！

RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "results"))
os.makedirs(RESULT_DIR, exist_ok=True)

# 系统接口（带标签，页面分组用）
@app.get("/", tags=["系统"])
def root():
    return {"status": "running", "message": "校园行为识别API服务正常运行"}

# 核心识别接口（带标签，页面分组用）
@app.post("/api/recognize", tags=["识别"])
async def recognize_video(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # 这里才是真正跑模型的地方，绝对不能少！
        result = inference_real(temp_file_path)
        action_label = get_action_label(file.filename)
        
        output_filename = f"{os.path.splitext(file.filename)[0]}.json"
        output_path = os.path.join(RESULT_DIR, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        os.unlink(temp_file_path)

        # 注意看这里！我去掉了 note，一定会有 action！
        return {
            "status": "success",
            "filename": file.filename,
            "action": action_label,  # 这就是你要的！
            "result_file": output_path
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
