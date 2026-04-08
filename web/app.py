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

# -------------------------- 推理函数 --------------------------
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

# -------------------------- 识别逻辑（完全按你的需求重写） --------------------------
if st.button("开始行为识别", type="primary", disabled=not video_path):
    with st.spinner("正在分析..."):
        try:
            # 1. 调用真实推理模型
            results = inference_real(video_path)
            if not results:
                st.error("推理结果为空，请检查模型！")
                st.stop()

            # 2. 统计所有行为，区分正常/异常
            all_labels = []
            time_behavior = {}  # 每一秒的行为次数
            normal_count = 0
            abnormal_count = 0
            total_persons = 0

            # 定义异常行为类别（你可以根据实际情况修改）
            abnormal_labels = {
                "hit_wall", "kick", "smoking", "slap_face", "slap_table",
                "phone", "laying", "pointing", "squating", "touch"
            }

            for frame in results:
                t = frame.get("time", 0.0)
                persons = frame.get("persons", [])
                total_persons += len(persons)
                second = int(t // 1)  # 按秒统计

                for p in persons:
                    label = p.get("label", "normal")
                    all_labels.append(label)

                    # 统计正常/异常
                    if label == "normal":
                        normal_count += 1
                    else:
                        abnormal_count += 1

                    # 统计每一秒的行为次数
                    if second not in time_behavior:
                        time_behavior[second] = 0
                    time_behavior[second] += 1

            # 3. 左侧数据概览（真实数据填充）
            if total_persons > 0:
                video_duration = round(results[-1]["time"], 2) if len(results) > 0 else 0
                abnormal_rate = round(abnormal_count / total_persons * 100, 1)
                stat_placeholder.markdown(f"""
                <div class="stat-item">异常占比：{abnormal_rate}%</div>
                <div class="stat-item">视频时长：{video_duration} 秒</div>
                <div class="stat-item">总帧数：{len(results)}</div>
                <div class="stat-item">异常次数：{abnormal_count}</div>
                <div class="stat-item">检测人数：{total_persons}</div>
                """, unsafe_allow_html=True)

            # 4. 柱状图（正常 vs 异常 直观对比）
            with chart_container:
                # 统计正常/异常总次数
                bar_data = [
                    {"行为类别": "正常行为", "次数": normal_count},
                    {"行为类别": "异常行为", "次数": abnormal_count}
                ]
                bar_df = pd.DataFrame(bar_data)

                # 柱状图：正常蓝色，异常红色，直观对比
                bar = alt.Chart(bar_df).mark_bar().encode(
                    x=alt.X("行为类别:O", title="行为类别", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("次数:Q", title="出现次数"),
                    color=alt.Color("行为类别:O", scale=alt.Scale(
                        domain=["正常行为", "异常行为"],
                        range=["#4f8cff", "#ef4444"]
                    ))
                ).properties(height=400, width=450, title="正常行为 vs 异常行为 次数对比")

                # 5. 折线图（每一秒的行为总次数，直观展示时间分布）
                trend_df = pd.DataFrame(list(time_behavior.items()), columns=["时间(秒)", "次数"])
                line = alt.Chart(trend_df).mark_line(color="#4f8cff", point=True).encode(
                    x=alt.X("时间(秒):Q", title="时间（秒）"),
                    y=alt.Y("次数:Q", title="每秒行为次数")
                ).properties(height=400, width=450, title="每秒行为次数时间分布")

                # 并排显示图表
                c1, c2 = st.columns(2)
                with c1:
                    st.altair_chart(bar, use_container_width=True)
                with c2:
                    st.altair_chart(line, use_container_width=True)

            # 6. 置信度Slider
            st.divider()
            st.subheader("置信度阈值")
            conf_list = []
            for frame in results:
                for p in frame.get("persons", []):
                    conf_list.append(p.get("conf", 0.0))
            if conf_list:
                min_c = min(conf_list)
                max_c = max(conf_list)
                st.slider("筛选阈值", 0.0, 1.0, min_c, key="conf")

            # 7. BBox可视化
            st.subheader("识别框可视化")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret and results:
                for p in results[0]["persons"]:
                    bbox = p.get("bbox", [])
                    label = p.get("label", "normal")
                    conf = p.get("conf", 0.0)
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        # 异常行为用红色框，正常用绿色框
                        color = (0, 255, 0) if label == "normal" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            cap.release()

            # 8. JSON下载
            result_placeholder.success("识别完成！")
            outname = os.path.splitext(video_file.name)[0] + ".json"
            st.download_button("下载识别结果", json.dumps(results, indent=2, ensure_ascii=False), outname)

        except Exception as e:
            st.error(f"识别出错：{str(e)}")
            import traceback
            st.error(traceback.format_exc())
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
