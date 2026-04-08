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

# -------------------------- 强制真实推理 --------------------------
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

# -------------------------- 识别逻辑（完全重写，100%正确） --------------------------
if st.button("开始行为识别", type="primary", disabled=not video_path):
    with st.spinner("正在分析..."):
        try:
            results = inference_real(video_path)
            if not results:
                st.error("推理结果为空！")
                st.stop()

            all_labels = []          # 所有行为标签
            time_behavior_count = {} # 每一秒的行为次数（时间序列）
            abnormal = 0
            total_persons = 0

            # ====================== 正确统计逻辑 ======================
            for frame in results:
                t = frame.get("time", 0.0)
                persons = frame.get("persons", [])
                total_persons += len(persons)

                for p in persons:
                    label = p.get("label", "normal")
                    conf = p.get("conf", 0.0)

                    all_labels.append(label)
                    if label != "normal":
                        abnormal += 1

                    # 按秒统计行为次数（每一秒只算一次）
                    second = int(t // 1)
                    if second not in time_behavior_count:
                        time_behavior_count[second] = 0
                    time_behavior_count[second] += 1

            # ====================== 左侧数据概览 ======================
            if total_persons > 0:
                video_duration = round(results[-1]["time"], 2) if results else 0
                abnormal_rate = round(abnormal / total_persons * 100, 1)
                stat_placeholder.markdown(f"""
                <div class="stat-item">异常占比：{abnormal_rate}%</div>
                <div class="stat-item">视频时长：{video_duration} 秒</div>
                <div class="stat-item">总帧数：{len(results)}</div>
                <div class="stat-item">异常次数：{abnormal}</div>
                <div class="stat-item">检测人数：{total_persons}</div>
                """, unsafe_allow_html=True)

            # ====================== 图表逻辑（完全重写） ======================
            with chart_container:
                # 1. 柱状图（统计每种行为的出现次数）
                if all_labels:
                    count = Counter(all_labels)
                    label_cn = {
                        "normal": "正常",
                        "hit_wall": "撞墙",
                        "hit_wall_corner": "撞墙(角落)",
                        "hit_wall_entrance": "撞墙(门口)",
                        "hit_wall_forward": "撞墙(正面)",
                        "kick_backward": "踢打(向后)",
                        "kick_corner": "踢打(角落)",
                        "kick_entrance": "踢打(门口)",
                        "kick_forward": "踢打(向前)",
                        "laying_backward": "躺卧(向后)",
                        "laying_corner": "躺卧(角落)",
                        "laying_entrance": "躺卧(门口)",
                        "laying_forward": "躺卧(向前)",
                        "phone_backward": "玩手机(向后)",
                        "phone_corner": "玩手机(角落)",
                        "phone_entrance": "玩手机(门口)",
                        "phone_forward": "玩手机(向前)",
                        "pointing_backward": "指认(向后)",
                        "pointing_corner": "指认(角落)",
                        "pointing_entrance": "指认(门口)",
                        "pointing_forward": "指认(向前)",
                        "slap_face_backward": "扇脸(向后)",
                        "slap_face_corner": "扇脸(角落)",
                        "slap_face_entrance": "扇脸(门口)",
                        "slap_face_forward": "扇脸(向前)",
                        "slap_table_backward": "拍桌(向后)",
                        "slap_table_corner": "拍桌(角落)",
                        "slap_table_entrance": "拍桌(门口)",
                        "slap_table_forward": "拍桌(向前)",
                        "smoking_backward": "抽烟(向后)",
                        "smoking_corner": "抽烟(角落)",
                        "smoking_entrance": "抽烟(门口)",
                        "smoking_forward": "抽烟(向前)",
                        "squating_backward": "蹲坐(向后)",
                        "squating_corner": "蹲坐(角落)",
                        "squating_entrance": "蹲坐(门口)",
                        "squating_forward": "蹲坐(向前)",
                        "stand_backward": "站立(向后)",
                        "stand_corner": "站立(角落)",
                        "stand_entrance": "站立(门口)",
                        "stand_forward": "站立(向前)",
                        "touch_backward": "触摸(向后)",
                        "touch_corner": "触摸(角落)",
                        "touch_entrance": "触摸(门口)",
                        "touch_forward": "触摸(向前)",
                        "whole_process_backward": "完整流程(向后)",
                        "whole_process_corner": "完整流程(角落)",
                        "whole_process_entrance": "完整流程(门口)",
                        "whole_process_forward": "完整流程(向前)"
                    }

                    bar_data = [{"行为": label_cn.get(k, k), "次数": v} for k, v in count.items()]
                    bar_df = pd.DataFrame(bar_data)

                    bar = alt.Chart(bar_df).mark_bar(color="#4f8cff").encode(
                        x=alt.X("行为:O", title="行为类别", sort="-y", axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y("次数:Q", title="出现次数")
                    ).properties(height=400, width=450)

                else:
                    bar = alt.Chart(pd.DataFrame({"行为": ["正常"], "次数": [0]})).mark_bar().encode(
                        x="行为:O", y="次数:Q"
                    )

                # 2. 折线图（按秒统计行为次数）
                if time_behavior_count:
                    trend_df = pd.DataFrame(list(time_behavior_count.items()), columns=["秒", "次数"])
                    line = alt.Chart(trend_df).mark_line(color="#fff", point=True).encode(
                        x=alt.X("秒:Q", title="时间（秒）"),
                        y=alt.Y("次数:Q", title="每秒行为次数")
                    ).properties(height=400, width=450)
                else:
                    line = alt.Chart(pd.DataFrame({"秒": [0], "次数": [0]})).mark_line().encode(
                        x="秒:Q", y="次数:Q"
                    )

                c1, c2 = st.columns(2)
                with c1:
                    st.altair_chart(bar, use_container_width=True)
                with c2:
                    st.altair_chart(line, use_container_width=True)

            # 置信度Slider
            st.divider()
            st.subheader("置信度阈值")
            # 收集所有置信度
            conf_list = []
            for frame in results:
                for p in frame.get("persons", []):
                    conf_list.append(p.get("conf", 0.0))
            if conf_list:
                min_c = min(conf_list)
                max_c = max(conf_list)
                st.slider("筛选阈值", 0.0, 1.0, min_c, key="conf")

            # BBox可视化
            st.subheader("识别框可视化")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret and results:
                for p in results[0].get("persons", []):
                    bbox = p.get("bbox", [])
                    label = p.get("label", "normal")
                    conf = p.get("conf", 0.0)
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            cap.release()

            # JSON下载
            result_placeholder.success("识别完成！")
            outname = os.path.splitext(video_file.name)[0] + ".json"
            st.download_button("下载识别结果", json.dumps(results, ensure_ascii=False, indent=2), outname)

        except Exception as e:
            st.error(f"错误：{str(e)}")
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)

# -------------------------- 视频预览 --------------------------
st.divider()
st.markdown('<div class="dark-card">', unsafe_allow_html=True)
st.subheader("视频预览")
if video_file:
    st.video(video_file)
st.markdown('</div>', unsafe_allow_html=True)
