import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import tempfile
from collections import Counter
import pandas as pd
import altair as alt

# -------------------------- 页面配置 --------------------------
st.set_page_config(
    page_title="校园行为智能识别系统",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------- CSS --------------------------
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
st.markdown('<p class="main-title">📹 校园行为智能识别系统</p>', unsafe_allow_html=True)

# -------------------------- 导入推理函数 --------------------------
try:
    from examples.inference_stub import inference_demo, get_action_label
except ImportError:
    st.warning("使用模拟数据演示")
    def inference_demo(video_path, out_log_path=None):
        return "demo.json", [
            {"time": 0.0, "persons": [{"label": "phone"}]},
            {"time": 0.5, "persons": [{"label": "phone"}]},
            {"time": 1.0, "persons": [{"label": "normal"}]},
            {"time": 1.5, "persons": [{"label": "phone"}]},
            {"time": 2.0, "persons": [{"label": "phone"}]},
            {"time": 2.5, "persons": [{"label": "fall"}]},
            {"time": 3.2, "persons": [{"label": "loiter"}]},
            {"time": 4.8, "persons": [{"label": "normal"}]},
        ]
    def get_action_label(label):
        return {"normal":"正常", "loiter":"逗留", "phone":"使用手机", "fall":"摔倒"}.get(label, "未知")

# -------------------------- 分栏 --------------------------
col1, col2 = st.columns([3, 7])

# -------------------------- 左侧：统计 + 上传 --------------------------
with col1:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="side-badge">视频统计</div>', unsafe_allow_html=True)
    st.subheader("数据概览")

    st.markdown('<div class="stat-item">异常占比：3.5%</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-item">视频时长：50.00 秒</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-item">总帧数：1600</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-item">异常次数：2</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-item">检测人数：25</div>', unsafe_allow_html=True)

    st.subheader("视频上传")
    video_file = st.file_uploader("选择视频", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    video_path = None

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_file.read())
            video_path = f.name
        st.success(f"已加载：{video_file.name}")

        st.subheader("视频信息")
        st.write(f"• 文件名：{video_file.name}")
        st.write(f"• 总帧数：1137（模拟）")
        st.write(f"• 帧率：30 FPS（模拟）")
        st.write(f"• 分辨率：1920×1080（模拟）")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 右侧：图表 --------------------------
with col2:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="main-title" style="font-size: 2rem; color: #ffffff;">行为统计分析</div>', unsafe_allow_html=True)
    chart_container = st.container()
    result_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 识别逻辑 --------------------------
if st.button("开始行为识别", type="primary", disabled=not video_path):
    with st.spinner("正在分析..."):
        try:
            log_path, results = inference_demo(video_path)

            # 收集标签与时间
            all_labels = []
            time_labels = []
            for frame in results:
                t = frame.get("time", 0)
                for p in frame.get("persons", []):
                    label = p.get("label")
                    all_labels.append(label)
                    time_labels.append((t, label))

            # ========================= 图表：左柱状 + 右折线 =========================
            with chart_container:
                chart_container.empty()

                # --- 左：行为分类柱状图 ---
                if all_labels:
                    count = Counter(all_labels)
                    order = ["normal", "loiter", "fall", "phone"]
                    rows = []
                    for k in order:
                        if k in count:
                            rows.append({"行为": get_action_label(k), "次数": count[k]})
                    for k, v in count.items():
                        if k not in order:
                            rows.append({"行为": get_action_label(k), "次数": v})
                    bar_df = pd.DataFrame(rows)
                else:
                    bar_df = pd.DataFrame({"行为": [], "次数": []})

                bar = alt.Chart(bar_df).mark_bar(color="#4f8cff").encode(
                    x=alt.X("行为:O", sort=None, title="行为类别"),
                    y=alt.Y("次数:Q", title="出现次数")
                ).properties(height=360)

                # --- 右：时间趋势折线图（按 1 秒分桶） ---
                if time_labels:
                    trend_df = pd.DataFrame(time_labels, columns=["时间", "行为"])
                    trend_df["秒"] = trend_df["时间"].astype(float).floordiv(1).astype(int)
                    trend_cnt = trend_df.groupby("秒").size().reset_index(name="次数")
                else:
                    trend_cnt = pd.DataFrame({"秒": [], "次数": []})

                line = alt.Chart(trend_cnt).mark_line(color="#ffffff", point=True).encode(
                    x=alt.X("秒:Q", title="时间（秒）"),
                    y=alt.Y("次数:Q", title="出现次数")
                ).properties(height=360)

                c1, c2 = st.columns(2)
                with c1:
                    st.altair_chart(bar, use_container_width=True)
                with c2:
                    st.altair_chart(line, use_container_width=True)

            # ========================= 行为详情 =========================
            result_placeholder.success("识别完成！")
            if all_labels:
                st.subheader("行为详情")
                for label, cnt in Counter(all_labels).items():
                    st.write(f"- {get_action_label(label)}：{cnt} 次")

                st.subheader("时间戳")
                st.write(f"起始时间：{results[0].get('time', 0):.2f}s")
            else:
                st.info("未检测到目标行为")

            # -------------------------- 底部功能按钮 --------------------------
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            with col_btn1:
                st.button("分析", key="btn1")
            with col_btn2:
                st.button("数神", key="btn2")
            with col_btn3:
                st.button("数频", key="btn3")
            with col_btn4:
                st.button("导出", key="btn4")

        except Exception as e:
            st.error(f" 出错：{str(e)}")
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
