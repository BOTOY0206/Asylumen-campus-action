import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

# ========== 【完全复制你视频的真实绝对路径，一字不改】 ==========
VIDEO_DIR = r"C:\Users\33955\Desktop\Asylumen-campus-action\data\sample_videos"
DATASET_ROOT = r"C:\Users\33955\Desktop\Asylumen-campus-action\dataset"
FRAME_STEP = 10  # 每10帧抽1张图片

# ========== 完整匹配你所有视频文件名标签 ==========
LABEL_MAP = {
    "phone": "call",
    "laying": "lie",
    "hit_wall": "hit",
    "slap": "slap",
    "smoking": "smoke",
    "squating": "squat",
    "stand": "stand",
    "pointing": "point",
    "sample": "normal",
    "kick": "kick"  # ✅ 新增踢腿标签
}

# 自动创建数据集文件夹
for split in ["train", "val"]:
    os.makedirs(f"{DATASET_ROOT}/{split}", exist_ok=True)

all_frames = []
all_labels = []

# 遍历你的所有视频
video_list = os.listdir(VIDEO_DIR)
print(f"📂 扫描到文件夹内文件总数：{len(video_list)} 个")

for video_file in video_list:
    # 只处理MP4视频
    if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
        print(f"跳过非视频文件：{video_file}")
        continue

    # 专属适配hit_wall双单词文件名
    if video_file.startswith("hit_wall"):
        raw_label = "hit_wall"
    else:
        raw_label = video_file.split("_")[0]

    # 跳过无匹配标签的文件
    if raw_label not in LABEL_MAP:
        print(f"⚠️ 无对应行为标签，跳过：{video_file}")
        continue

    final_label = LABEL_MAP[raw_label]
    video_path = os.path.join(VIDEO_DIR, video_file)
    print(f"✅ 正在处理：{video_file} → 标准行为：{final_label}")

    # 打开视频抽帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_file}")
        continue

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_STEP == 0:
            img_name = f"{final_label}_{len(all_frames)}.jpg"
            temp_path = f"temp_frames/{img_name}"
            os.makedirs("temp_frames", exist_ok=True)
            cv2.imwrite(temp_path, frame)
            all_frames.append(temp_path)
            all_labels.append(final_label)
        frame_count += 1
    cap.release()

# 防空报错：没有图片直接退出
print(f"\n🎬 本次总共抽取有效训练图片：{len(all_frames)} 张")
if len(all_frames) == 0:
    print("❌ 没有读到任何视频画面！")
    exit()

# 分层划分：保证每个行为都有训练+验证图片
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    all_frames, all_labels, test_size=0.1, random_state=42, stratify=all_labels
)

# 自动归类到YOLO数据集格式
for img, label in zip(train_imgs, train_labels):
    save_folder = f"{DATASET_ROOT}/train/{label}"
    os.makedirs(save_folder, exist_ok=True)
    shutil.move(img, os.path.join(save_folder, os.path.basename(img)))

for img, label in zip(val_imgs, val_labels):
    save_folder = f"{DATASET_ROOT}/val/{label}"
    os.makedirs(save_folder, exist_ok=True)
    shutil.move(img, os.path.join(save_folder, os.path.basename(img)))

# 清理临时文件
shutil.rmtree("temp_frames", ignore_errors=True)
print("\n🎉 数据集全自动制作完成！可以直接运行训练命令")