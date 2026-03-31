import json
import os
from pathlib import Path

def inference_real(video_path, output_path="data/sample_videos/demo_log.json"):
    """
    真实推理接口占位：
    输入视频路径，输出与 demo_log.json 同格式的识别结果
    """
    # 这里先输出一份与 stub 一致的占位结构（后续替换为真实模型推理）
    result = [
        {
            "time": 0.07525444030761719,
            "frame": 0,
            "persons": [
                {
                    "person_id": 1,
                    "bbox": [147, 817, 313, 853],
                    "label": "loiter",
                    "conf": 0.94
                }
            ]
        },
        {
            "time": 0.08886122703552246,
            "frame": 1,
            "persons": [
                {
                    "person_id": 1,
                    "bbox": [599, 878, 787, 983],
                    "label": "normal",
                    "conf": 0.73
                }
            ]
        }
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
   print(f"Generated placeholder result: {output_path}")
    print("提示：这是占位数据，后续请替换为真实模型推理逻辑")

if __name__ == "__main__":
    # 随便选一个你有的视频路径测试（这里先用占位逻辑演示）
    video = "data/sample_videos/hit_wall_backward_1-1.mp4"  # 换成你实际的视频路径
    inference_real(video)
