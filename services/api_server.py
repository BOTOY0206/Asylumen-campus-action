from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import json
import cv2
import tempfile
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.inference_real import inference_real, get_action_label

app = FastAPI(title="Campus Action Recognition API")

RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "results"))
os.makedirs(RESULT_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/api/recognize")
async def recognize_video(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            video_path = tmp.name

        result = inference_real(video_path)
        video_name = file.filename
        action_label = get_action_label(video_name)
        
        output_filename = os.path.splitext(video_name)[0] + ".json"
        output_path = os.path.join(RESULT_DIR, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        os.unlink(video_path)

        return {
            "status": "success",
            "video": video_name,
            "action": action_label,
            "result_file": output_path,
            "result": result
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
