from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
import os
import json
import tempfile
import sys

# 自动添加项目根目录到Python路径，解决模块导入问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.inference_real import inference_real, get_action_label

# 初始化FastAPI应用
app = FastAPI(
    title="校园行为识别API",
    description="支持视频上传、行为识别、结果查询的后端API服务",
    version="1.0.0",
    # 关闭默认docs，用自定义CDN版本
    docs_url=None,
    redoc_url=None
)

# 全局配置
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "results"))
os.makedirs(RESULT_DIR, exist_ok=True)

# 🔴 核心：自定义Swagger UI，强制从CDN加载静态资源，彻底解决空白页
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        # 强制使用CDN资源，不走本地，彻底解决加载失败
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# 健康检查接口
@app.get("/", tags=["系统"])
def root():
    return {"status": "running", "message": "校园行为识别API服务正常运行"}

# 视频识别接口
@app.post("/api/recognize", tags=["识别"])
async def recognize_video(file: UploadFile = File(...)):
    try:
        # 安全处理临时文件
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # 调用真实识别逻辑
        result = inference_real(temp_file_path)
        video_name = file.filename or "unknown_video"
        action_label = get_action_label(video_name)
        
        # 保存结果到本地
        output_filename = f"{os.path.splitext(video_name)[0]}.json"
        output_path = os.path.join(RESULT_DIR, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 删除临时文件
        os.unlink(temp_file_path)

        return {
            "status": "success",
            "video_name": video_name,
            "action": action_label,
            "result_file": output_path,
            "result": result
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# 结果查询接口
@app.get("/api/results/{video_name}", tags=["结果"])
async def get_result(video_name: str):
    result_path = os.path.join(RESULT_DIR, f"{video_name}.json")
    if not os.path.exists(result_path):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "结果文件不存在"}
        )
    
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    return {"status": "success", "result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
