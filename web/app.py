import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import tempfile
from collections import Counter
import pandas as pd
import altair as alt
import cv2
import json

# -------------------------- 页面配置 --------------------------
st.set_page_config(
    page_title="校园行为智能识别系统",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------- CSS 美化 --------------------------
st.markdown("""
<style>
.stApp { background-color: #e0e9f5; }
.main-title {
    font-size: 2.5rem; color: #1E88E5; font-weight: bold;
    text-align: center; margin: 20px 0 30px 0;
}
.dark-card {
    background-color: #1e293b; border-radius: 16px; padding: 24px;
    color: #ffffff; margin-bottom: 20px; border: 1px solid #334155;
}
.side-badge {
    background-color: #475569; color: #ffffff; padding: 8px 16px;
    border-radius: 8px; font-size: 1.2rem; font-weight: bold;
    display: inline-block; margin-bottom: 20px;
}
.stat-item { font-size: 1.1rem; margin: 12px 0; color: #ffffff; }
.stButton>button {
    background-color: #3b82f6; color: white; border-radius: 8px;
    font-weight: 600; padding: 0.6rem 1.5rem; border: none;
}
.stButton>button:hover { background-color: #2563eb; }
h2, h3 { color: #ffffff !important; margin-bottom: 15px; }
p, li { color: #1e293b !important; font-size: 1rem; }
.stSuccess {
    background-color: #dcfce7; color: #166534; border-radius: 8px; padding: 1rem;
}
.stInfo {
    background-color: #e2e8f0; color: #0f172a; border-radius: 8px; padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- 标题 --------------------------
st.markdown('<p class="main-title">校园行为智能识别系统</p>', unsafe_allow_html=True)

# -------------------------- 【核心修复】强制使用真实推理，彻底禁用模拟数据 --------------------------
# 彻底删除try-except模拟兜底，强制调用你的真实推理模型
from examples.inference_real import inference_real, get_action_label

# -------------------------- 分栏 --------------------------
col1, col2 = st.columns([3, 7])

# -------------------------- 左侧：统计 + 上传 --------------------------
with col1:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="side-badge">视频统计</div>', unsafe_allow_html=True)
    st.subheader("数据概览")

    stat_placeholder = st.empty()
    stat_placeholder.markdown("""
    <div class="stat-item">异常占比：-</div>
    <div class="stat-item">视频时长：-</div>
    <div class="stat-item">总帧数：-</div>
    <div class="stat-item">异常次数：-</div>
    <div class="stat-item">检测人数：-</div>
    """, unsafe_allow_html=True)

    st.subheader("视频上传")
    video_file = st.file_uploader("选择视频", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    video_path = None

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_file.read())
            video_path = f.name
        st.success(f"已加载：{video_file.name}")

        cap = cv2.VideoCapture(video_path)
        st.subheader("视频信息")
        st.write(f"• 文件名：{video_file.name}")
        st.write(f"• 总帧数：{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        st.write(f"• 帧率：{cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
        st.write(f"• 分辨率：{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 右侧：图表 --------------------------
with col2:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="main-title" style="font-size: 2rem; color: #ffffff;">行为统计分析</div>', unsafe_allow_html=True)
    chart_container = st.container()
    result_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 【核心修复】识别逻辑，绑定真实推理结果 --------------------------
if st.button("开始行为识别", type="primary", disabled=not video_path):
    with st.spinner("正在分析..."):
        try:
            # 强制调用你的真实推理模型，彻底禁用模拟数据
            results = inference_real(video_path)

            all_labels = []
            time_labels = []
            conf_list = []
            abnormal = 0
            total = 0

            # 遍历真实推理结果，统计所有行为类别
            for frame in results:
                t = frame.get("time", 0)
                for p in frame.get("persons", []):
                    label = p.get("label")
                    conf = p.get("conf", 0.0)
                    all_labels.append(label)
                    time_labels.append((t, label))
                    conf_list.append(conf)
                    total += 1
                    # 统计异常行为（非normal的所有类别）
                    if label != "normal":
                        abnormal += 1

            # 【修复】左侧数据概览，真实数据填充
            if total > 0:
                video_duration = round(results[-1]["time"], 2) if len(results) > 0 else 0
                abnormal_rate = round(abnormal / total * 100, 1)
                stat_placeholder.markdown(f"""
                <div class="stat-item">异常占比：{abnormal_rate}%</div>
                <div class="stat-item">视频时长：{video_duration} 秒</div>
                <div class="stat-item">总帧数：{len(results)}</div>
                <div class="stat-item">异常次数：{abnormal}</div>
                <div class="stat-item">检测人数：{total}</div>
                """, unsafe_allow_html=True)

            # 【修复】行为分类柱状图，真实数据驱动，动态Y轴
            with chart_container:
                if all_labels:
                    count = Counter(all_labels)
                    # 把英文标签转成中文，对应你的行为类别
                    label_cn_map = {
                        "normal": "正常",
                        "hit_wall": "撞墙",
                        "kick": "踢打",
                        "smoking": "抽烟",
                        "slap_face": "扇脸",
                        "slap_table": "拍桌",
                        "phone": "玩手机",
                        "laying": "躺卧",
                        "pointing": "指认",
                        "squating": "蹲坐",
                        "standing": "站立",
                        "touch": "触摸",
                        "whole_process": "完整流程"
                    }
                    rows = [{"行为": label_cn_map.get(k, k), "次数": v} for k, v in count.items()]
                    bar_df = pd.DataFrame(rows)
                else:
                    bar_df = pd.DataFrame({"行为": [], "次数": []})

                # 动态Y轴，适配真实数据，不再固定0-2500
                bar = alt.Chart(bar_df).mark_bar(color="#4f8cff").encode(
                    x=alt.X("行为:O", title="行为类别", sort="-y"),
                    y=alt.Y("次数:Q", title="次数", scale=alt.Scale(domain=[0, bar_df["次数"].max() + 100] if not bar_df.empty else [0, 100]))
                ).properties(height=360)

                # 【修复】时间趋势折线图，真实数据驱动，不再平线
                if time_labels:
                    trend_df = pd.DataFrame(time_labels, columns=["时间", "行为"])
                    trend_df["秒"] = trend_df["时间"].astype(float).floordiv(1).astype(int)
                    trend_cnt = trend_df.groupby("秒").size().reset_index(name="次数")
                else:
                    trend_cnt = pd.DataFrame({"秒": [], "次数": []})

                line = alt.Chart(trend_cnt).mark_line(color="#fff", point=True).encode(
                    x=alt.X("秒:Q", title="秒"),
                    y=alt.Y("次数:Q", title="次数", scale=alt.Scale(domain=[0, trend_cnt["次数"].max() + 5] if not trend_cnt.empty else [0, 30]))
                ).properties(height=360)

                c1, c2 = st.columns(2)
                with c1: st.altair_chart(bar, use_container_width=True)
                with c2: st.altair_chart(line, use_container_width=True)

            # 置信度Slider
            st.divider()
            st.subheader("置信度阈值")
            if conf_list:
                min_c = min(conf_list)
                max_c = max(conf_list)
                st.slider("筛选阈值", 0.0, 1.0, min_c, key="conf")

            # BBox可视化
            st.subheader("识别框可视化")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret and results:
                for p in results[0]["persons"]:
                    bbox = p.get("bbox", [])
                    label = p.get("label")
                    conf = p.get("conf", 0)
                    if len(bbox) == 4:
                        x1,y1,x2,y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            cap.release()

            # JSON下载
            result_placeholder.success("识别完成！")
            outname = os.path.splitext(video_file.name)[0] + ".json"
            st.download_button("下载结果", json.dumps(results, indent=2, ensure_ascii=False), outname)

        except Exception as e:
            st.error(f"出错：{str(e)}")
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)

# -------------------------- 视频预览 --------------------------
st.divider()
st.markdown('<div class="dark-card">', unsafe_allow_html=True)
st.subheader("视频预览")
if video_file:
    st.video(video_file)
else:
    st.info("请上传视频")
st.markdown('</div>', unsafe_allow_html=True)
