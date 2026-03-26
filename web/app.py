# web/app.py
# Streamlit 演示：从 demo_log.json 读取并显示最后一帧的检测结果
import streamlit as st
import json
import os

LOG_PATH = "data/sample_videos/demo_log.json"
st.title("校园行为识别 - 演示页面（占位）")

if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    st.write("检测到的帧数：", len(data))
    if len(data) > 0:
        last = data[-1]
        st.write("最新时间戳/帧：", last["time"], "/", last["frame"])
        st.write("检测到的人员：")
        st.json(last["persons"])
else:
    st.write("尚未生成日志，请先运行 examples/inference_stub.py（在仓库根目录）")
