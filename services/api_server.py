from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# 测试接口
@app.get("/")
def home():
    return {"status": "ok"}

# 核心：视频上传接口（极简版，一定能出按钮）
@app.post("/api/recognize")
def recognize(file: UploadFile = File(...)):
    return {
        "status": "success",
        "filename": file.filename,
        "note": "按钮能出来就代表成功了"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
