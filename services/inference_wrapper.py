import uuid
import os
import sys
import json
from typing import Dict, List

# ============== 关键：强制把 examples 目录加入 Python 路径 ==============
# 不管从哪里启动，都能找到 examples 里的 inference_real
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
sys.path.insert(0, EXAMPLES_DIR)  # 强制加入路径，彻底解决导入问题

# ============== 现在绝对能导入成功！！！ ==============
from inference_real import inference_real

# ====================== 核心：统一 JSON 输出格式 ======================
def convert_to_standard_json(
    raw_frames: List[Dict],
    video_path: str,
    task_id: str,
    output_path: str
) -> Dict:
    """
    把你的 inference_real 输出 → 转换成统一标准 JSON
    格式完全按要求：task_id / status / video_path / output_path / result / frames
    """
    standard_result = {
        "task_id": task_id,
        "status": "success",
        "video_path": video_path,
        "output_path": output_path,
        "error_message": "",
        "result": {
            "frames": []
        }
    }

    # 逐帧转换格式
    for frame in raw_frames:
        if not frame.get("persons"):
            continue

        person = frame["persons"][0]
        standard_frame = {
            "frame_id": frame["frame"],
            "timestamp": frame["time"],
            "bbox": person["bbox"],
            "behavior": person["label"],
            "confidence": person["conf"]
        }
        standard_result["result"]["frames"].append(standard_frame)

    return standard_result

# ====================== 推理包装器（对外调用入口） ======================
def inference_wrapper(
    video_path: str,
    output_dir: str = "data/output"
) -> Dict:
    """
    统一推理包装器：
    1. 调用你的 inference_real
    2. 自动生成标准 JSON
    3. 自动保存文件
    4. 自动异常处理
    """
    task_id = str(uuid.uuid4())

    # 初始化返回结构
    final_result = {
        "task_id": task_id,
        "status": "running",
        "video_path": video_path,
        "output_path": "",
        "error_message": "",
        "result": {"frames": []}
    }

    try:
        # 1. 检查视频是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在：{video_path}")

        # 2. 调用你原本的推理函数（完全不改动你的代码）
        print(f"\n=== 开始推理：{os.path.basename(video_path)} ===")
        raw_result = inference_real(video_path)

        if not raw_result:
            raise Exception("推理返回空结果")

        # 3. 生成输出路径
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{task_id}.json"
        output_path = os.path.join(output_dir, output_filename)
        final_result["output_path"] = output_path

        # 4. 转换成统一标准 JSON
        standard_data = convert_to_standard_json(
            raw_frames=raw_result,
            video_path=video_path,
            task_id=task_id,
            output_path=output_path
        )

        # 5. 保存标准 JSON 文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(standard_data, f, indent=2, ensure_ascii=False)

        # 6. 更新状态
        final_result.update(standard_data)
        final_result["status"] = "success"
        print(f" 推理完成！标准JSON已保存：{output_path}")

        return final_result

    except Exception as e:
        # 异常处理
        final_result["status"] = "failed"
        final_result["error_message"] = str(e)
        print(f" 推理失败：{str(e)}")
        return final_result

# ====================== 命令行直接运行 ======================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法：python services/inference_wrapper.py 视频路径")
        print("示例：python services/inference_wrapper.py data/sample_videos/test.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    result = inference_wrapper(video_path)
    print("\n=== 最终返回结果 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
