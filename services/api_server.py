from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import json
import tempfile
import sys

# 自动添加项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#  这里导入你的模型推理代码
from examples.inference_real import inference_real, get_action_label 

app = FastAPI()

RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "results"))
os.makedirs(RESULT_DIR, exist_ok=True)

# 健康检查
@app.get("/")
def root():
    return {"status": "running"}

# 核心识别接口（替换掉刚才的简单接口）
@app.post("/api/recognize")
async def recognize_video(file: UploadFile = File(...)):
    try:
        # 保存临时文件
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # 调用你的模型推理
        result = inference_real(temp_file_path)
        action_label = get_action_label(file.filename)
        
        # 保存结果
        output_filename = f"{os.path.splitext(file.filename)[0]}.json"
        output_path = os.path.join(RESULT_DIR, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        os.unlink(temp_file_path)

        return {
            "status": "success",
            "filename": file.filename,
            "action": action_label,
            "result_file": output_path
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
