# examples/inference_stub.py
# 改造为可调用的函数，支持传入任意视频路径，生成对应日志
import cv2
import json
import time
import random
import os

# 定义可导入的函数：inference_demo
def inference_demo(video_path, out_log_path=None):
    """
    假检测函数：读取视频，生成随机bbox和标签的json日志
    :param video_path: 输入视频路径
    :param out_log_path: 输出日志路径，默认自动生成
    :return: 日志路径
    """
    # 如果没指定输出路径，自动生成
    if out_log_path is None:
        out_dir = os.path.join(os.path.dirname(video_path), "demo_logs")
        os.makedirs(out_dir, exist_ok=True)
        out_log_path = os.path.join(out_dir, f"{os.path.basename(video_path).split('.')[0]}_log.json")
    
    # 校验视频文件
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频，请检查路径：{video_path}")

    results = []
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = time.time() - start_time
        persons = []
        # 随机生成0-2个行人
        for pid in range(random.randint(0, 2)):
            h, w = frame.shape[:2]
            # 随机生成bbox
            x1 = random.randint(0, max(0, w-100))
            y1 = random.randint(0, max(0, h-100))
            x2 = min(w, x1 + random.randint(30, 200))
            y2 = min(h, y1 + random.randint(30, 200))
            # 随机标签和置信度
            label = random.choice(["normal", "loiter", "phone", "fall"])
            conf = round(random.uniform(0.5, 0.99), 2)
            persons.append({
                "person_id": pid+1, 
                "bbox": [x1, y1, x2, y2], 
                "label": label, 
                "conf": conf
            })
        results.append({"time": ts, "frame": frame_idx, "persons": persons})
        frame_idx += 1

    cap.release()
    # 保存日志
    os.makedirs(os.path.dirname(out_log_path), exist_ok=True)
    with open(out_log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已生成假检测日志：{out_log_path}")
    return out_log_path, results

# 定义get_action_label函数（根据标签返回中文说明）
def get_action_label(label):
    label_map = {
        "normal": "正常行为",
        "loiter": "逗留徘徊",
        "phone": "使用手机",
        "fall": "摔倒"
    }
    return label_map.get(label, "未知行为")

# 保留原脚本的独立运行能力（兼容原项目）
if __name__ == "__main__":
    # 原默认路径
    DEFAULT_VIDEO = "data/sample_videos/sample1.mp4"
    DEFAULT_LOG = "data/sample_videos/demo_log.json"
    try:
        inference_demo(DEFAULT_VIDEO, DEFAULT_LOG)
    except Exception as e:
        print(f"错误：{e}")
        exit(1)
