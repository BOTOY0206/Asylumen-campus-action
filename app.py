"""
web/app.py

Campus behavior recognition Streamlit front-end (demo-mode first).
Enhanced: remote API integration, robust label extraction, charts fallback,
bbox visualization fixes, frames/key normalization and label aliasing.

MODIFICATION:
- Swapped the positions of the right-side charts (bar + line) and the bbox+slider video visualization.
  Charts are now rendered below the visualization section; visualization is rendered inside the right column
  (chart container) to match the requested layout change.
- Added AXIS_LABEL_MAP so the chart x-axis labels map to backend labels as requested.
- No other logic changed. Only layout placement and chart-label mapping.
"""

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
import numpy as np
import requests
import re

# --------------------------
# Safe import of local inference (do not crash UI if torch missing)
# --------------------------
try:
    from examples.inference_real import inference_real, get_action_label  # type: ignore
    MODEL_AVAILABLE = True
    _MODEL_IMPORT_ERROR = None
except Exception as _e:
    MODEL_AVAILABLE = False
    _MODEL_IMPORT_ERROR = _e

    def inference_real(video_path):
        raise RuntimeError("本地推理不可用：导入 inference 模块失败，原因：" + str(_MODEL_IMPORT_ERROR))

    def get_action_label(x):
        return str(x)

# --------------------------
# Defaults and API URL
# --------------------------
DEFAULT_API_URL = "https://gills-expediter-dreary.ngrok-free.dev/infer_behavior"
if 'api_url' not in st.session_state:
    st.session_state['api_url'] = DEFAULT_API_URL

# --------------------------
# Persisted session state keys
# --------------------------
for key, default in [
    ('results', None), ('results_json', None), ('video_tmp', None), ('use_demo_local', False),
    ('conf', 0.0), ('last_uploaded_name', None), ('use_demo_mode', False), ('api_result', None),
    ('api_display_counts', None), ('api_time_series', None), ('api_stats', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------- Page config & CSS + centered light hero ----------
st.set_page_config(page_title="校园行为智能识别系统", page_icon="🎥", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root{
  --bg: #fbfdfe;
  --panel-bg: #ffffff;
  --title-color: #111827;
  --subtitle-color: #6b7280;
  --primary: #2563eb;
  --muted: #6b7280;
  --radius: 12px;
}

[data-testid="stApp"] * { font-family: "Inter", "Noto Sans SC", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
[data-testid="stApp"] > section {
  background: var(--bg) !important;
  padding: 20px 24px !important;
}

.hero {
  width: 100%;
  background: transparent;
  padding: 36px 12px 12px 12px;
  margin-bottom: 10px;
  display: flex;
  justify-content: center;
}
.hero-inner {
  max-width: 1100px;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  text-align: center;
  padding: 8px 12px;
}

.hero-row {
  display: flex;
  align-items: center;
  gap: 18px;
  justify-content: center;
  flex-wrap: nowrap;
}

.hero-icon {
  width: 72px;
  height: 72px;
  flex: 0 0 72px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: rgba(16,24,40,0.04);
}
.hero-icon svg { width: 56px; height: 56px; }

.hero-title {
  font-weight: 700;
  font-size: 48px;
  margin: 0;
  color: var(--title-color);
  line-height: 1.02;
}
.hero-subtitle {
  margin: 0;
  color: var(--subtitle-color);
  font-size: 15px;
  font-weight: 500;
  margin-top: 6px;
}

.dark-card {
  background: linear-gradient(180deg, rgba(8,13,20,0.98), rgba(13,18,26,0.98));
  border-radius: var(--radius);
  padding: 18px;
  color: #f8fafc;
  margin-bottom: 18px;
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: 0 10px 30px rgba(2,6,23,0.16);
}

.light-card {
  background: var(--panel-bg);
  border-radius: 12px;
  padding: 14px;
  color: var(--title-color);
  border: 1px solid rgba(16,24,40,0.06);
  box-shadow: 0 6px 16px rgba(15,23,42,0.06);
  margin-bottom: 12px;
}

.side-badge {
  display:inline-block;
  background: linear-gradient(90deg, var(--primary), #4f46e5);
  color: #fff;
  padding: 8px 14px;
  border-radius: 12px;
  font-weight: 700;
  margin-bottom: 14px;
}

.stat-item {
  font-size: 1.02rem;
  margin: 8px 0;
  color: var(--title-color);
}

.dark-card .stat-item { color: #e6eefc; }

.stButton>button, .stDownloadButton>button {
  background: linear-gradient(90deg, var(--primary), #1e40af);
  color: #ffffff;
  border-radius: 10px;
  font-weight: 700;
  padding: 8px 16px;
  border: none;
  box-shadow: 0 8px 18px rgba(37,99,235,0.10);
}

.stFileUploader { border-radius: 10px; padding: 10px; }
.stVideoContainer, .stVideo { border-radius: 10px; overflow: hidden; }

.stCheckbox > label { font-weight: 600; color: var(--title-color); }
.stSlider > div { margin-top: 6px; margin-bottom: 6px; }

@media (max-width: 900px) {
  .hero-title { font-size: 28px; }
  .hero-icon { width:56px; height:56px; }
  .hero-icon svg { width:44px; height:44px; }
  .hero-inner { padding: 6px; gap:6px; }
  .hero-row { gap: 10px; padding: 0 6px; }
}
</style>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-row">
      <div class="hero-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <rect x="1.5" y="5" width="13" height="12" rx="2" fill="#111827" opacity="0.06"/>
          <path d="M15 7l6-4v18l-6-4" fill="#111827" opacity="0.9"/>
          <circle cx="7.5" cy="11" r="2.8" fill="#111827" opacity="0.95"/>
        </svg>
      </div>
      <h1 class="hero-title">校园行为智能识别系统</h1>
    </div>
    <div class="hero-subtitle">基于AI的校园监控视频行为分析平台</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------- Header & Demo toggle --------------------------
top_col1, top_col2 = st.columns([1, 4])
with top_col1:
    st.session_state['use_demo_mode'] = st.checkbox(
        "Use demo JSON instead of live inference",
        value=st.session_state.get('use_demo_mode', False),
        help="勾选后仅加载本地 data/sample_videos/demo_log.json（不触发推理）"
    )
with top_col2:
    api_url_input = st.text_input("Remote API URL", value=st.session_state.get('api_url', DEFAULT_API_URL), help="后端 infer_behavior 接口 URL")
    st.session_state['api_url'] = api_url_input
    if st.session_state.get('use_demo_mode', False):
        st.info("Demo 模式已启用：前端仅加载 demo_log.json 并展示统计/可视化，不会触发推理。")

# -------------------------- Layout columns --------------------------
col1, col2 = st.columns([3, 7])

# -------------------------- Left: upload & stats --------------------------
with col1:
    st.markdown('<div class="light-card">', unsafe_allow_html=True)
    st.markdown('<div class="side-badge">视频统计</div>', unsafe_allow_html=True)
    st.subheader("数据概览")
    stat_placeholder = st.empty()

    def _render_stat_html(abnormal_rate, video_duration, total_frames, abnormal_count, total_persons, fallback_count):
        return f"""
        <div style="display:flex;flex-direction:column;gap:10px;padding:6px 4px;">
          <div style="display:flex;align-items:center;gap:12px;font-size:15px;color:var(--title-color);">
            <span style="width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;font-size:18px;border-radius:6px;background:rgba(16,24,40,0.04)">🏷️</span>
            <span>异常占比：</span><span style="color:var(--muted);font-weight:600;margin-left:6px;">{abnormal_rate}</span>
          </div>
          <div style="display:flex;align-items:center;gap:12px;font-size:15px;color:var(--title-color);">
            <span style="width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;font-size:18px;border-radius:6px;background:rgba(16,24,40,0.04)">⏱️</span>
            <span>视频时长：</span><span style="color:var(--muted);font-weight:600;margin-left:6px;">{video_duration} 秒</span>
          </div>
          <div style="display:flex;align-items:center;gap:12px;font-size:15px;color:var(--title-color);">
            <span style="width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;font-size:18px;border-radius:6px;background:rgba(16,24,40,0.04)">🎞️</span>
            <span>总帧数：</span><span style="color:var(--muted);font-weight:600;margin-left:6px;">{total_frames}</span>
          </div>
          <div style="display:flex;align-items:center;gap:12px;font-size:15px;color:var(--title-color);">
            <span style="width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;font-size:18px;border-radius:6px;background:rgba(16,24,40,0.04)">⚠️</span>
            <span>异常次数：</span><span style="color:var(--muted);font-weight:600;margin-left:6px;">{abnormal_count}</span>
          </div>
          <div style="display:flex;align-items:center;gap:12px;font-size:15px;color:var(--title-color);">
            <span style="width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;font-size:18px;border-radius:6px;background:rgba(16,24,40,0.04)">👥</span>
            <span>检测人数：</span><span style="color:var(--muted);font-weight:600;margin-left:6px;">{total_persons}</span>
          </div>
          <div style="display:flex;align-items:center;gap:12px;font-size:15px;color:var(--title-color);">
            <span style="width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;font-size:18px;border-radius:6px;background:rgba(16,24,40,0.04)">🔁</span>
            <span>回退(fallback)帧数：</span><span style="color:var(--muted);font-weight:600;margin-left:6px;">{fallback_count}</span>
          </div>
        </div>
        """

    stat_placeholder.markdown(_render_stat_html("-", "-", "-", "-", "-", "-"), unsafe_allow_html=True)

    st.subheader("视频上传")
    video_file = st.file_uploader("选择视频", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_file.read())
            video_path = f.name
        st.success(f"已加载：{video_file.name}")
        prev_name = st.session_state.get('last_uploaded_name')
        if prev_name != video_file.name:
            st.session_state['results'] = None
            st.session_state['results_json'] = None
            st.session_state['use_demo_local'] = False
            st.session_state['conf'] = 0.0
            st.session_state['api_result'] = None
            st.session_state['api_display_counts'] = None
            st.session_state['api_time_series'] = None
            st.session_state['api_stats'] = None
        st.session_state['video_tmp'] = video_path
        st.session_state['last_uploaded_name'] = video_file.name
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            st.subheader("视频信息")
            st.write(f"• 文件名：{video_file.name}")
            st.write(f"• 总帧数：{total_frames}")
            st.write(f"• 帧率：{fps:.1f} FPS")
            st.write(f"• 分辨率：{width}×{height}")
        except Exception:
            pass
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- Right: charts container --------------------------
with col2:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="main-title" style="font-size: 1.6rem; color: #ffffff;">行为统计分析</div>', unsafe_allow_html=True)
    chart_container = st.container()
    result_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- Helpers --------------------------
def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    return obj

def norm_key(k: str) -> str:
    if not isinstance(k, str):
        return str(k)
    return re.sub(r'[^0-9a-zA-Z]+', '_', k.strip().lower())

def extract_label_from_person(p):
    if p is None:
        return "unknown"
    if isinstance(p, str):
        return p
    if isinstance(p, (int, np.integer)):
        try:
            return get_action_label(int(p))
        except Exception:
            return str(p)
    if isinstance(p, dict):
        for k in ("label", "action", "pred", "class", "class_name", "label_name"):
            if k in p and p[k] is not None:
                lbl = p[k]
                try:
                    if isinstance(lbl, (int, np.integer, float, np.floating)):
                        return get_action_label(int(lbl))
                    if isinstance(lbl, (bytes, bytearray)):
                        return lbl.decode("utf-8", errors="ignore")
                    return str(lbl)
                except Exception:
                    return str(lbl)
        for k in ("label_id", "label_idx", "class_id"):
            if k in p and p[k] is not None:
                try:
                    return get_action_label(int(p[k]))
                except Exception:
                    return str(p[k])
    try:
        return str(p)
    except Exception:
        return "unknown"

LABEL_TRANSLATE = {
    "normal": "正常",
    "climbing": "攀爬",
    "falling": "摔倒",
    "fighting": "打斗",
    "running": "奔跑",
    "phone": "玩手机",
    "call": "打电话",
    "pointing": "指认",
    "slap": "拍打",
    "smoke": "抽烟",
    "squat": "下蹲",
    "stand": "站立",
    "touch": "触摸",
    "lie": "躺倒",
    "hit": "撞墙",
    "kick": "踢腿"
}

def translate_label(k: str) -> str:
    nk = norm_key(k)
    if nk in LABEL_TRANSLATE:
        return LABEL_TRANSLATE[nk]
    return k.replace('_', ' ').title() if isinstance(k, str) else str(k)

AXIS_LABEL_MAP_RAW = {
    "phone": "call",
    "laying": "lie",
    "hit_wall": "hit",
    "slap": "slap",
    "smoking": "smoke",
    "squating": "squat",
    "stand": "stand",
    "pointing": "point",
    "sample": "normal",
    "kick": "kick"
}
AXIS_LABEL_MAP = {norm_key(k): v for k, v in AXIS_LABEL_MAP_RAW.items()}

def find_frames_in_data(d):
    if not isinstance(d, dict):
        return None
    candidates = ['frames', 'results', 'detailed_frames', 'detections', 'items', 'frames_list']
    for k in candidates:
        if k in d and isinstance(d[k], list):
            return d[k]
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            sample = v[0]
            if any(x in sample for x in ('time', 'persons', 'bbox', 'frame_idx', 'frame')):
                return v
    return None

def find_action_stats_in_data(d):
    if not isinstance(d, dict):
        return None
    if 'action_stats' in d and isinstance(d['action_stats'], dict):
        return d['action_stats']
    for k, v in d.items():
        if isinstance(v, dict):
            if all(isinstance(x, (int, np.integer, float, np.floating)) for x in v.values()):
                return v
    return None

def parse_time_to_seconds(t):
    if t is None:
        return 0.0
    try:
        if isinstance(t, (int, float, np.integer, np.floating)):
            return float(t)
    except Exception:
        pass
    if isinstance(t, str):
        s = t.strip()
        if ':' in s:
            parts = s.split(':')
            if all(part.isdigit() for part in parts):
                parts = [int(p) for p in parts]
                if len(parts) == 3:
                    return parts[0] * 3600 + parts[1] * 60 + parts[2]
                if len(parts) == 2:
                    return parts[0] * 60 + parts[1]
                if len(parts) == 1:
                    return float(parts[0])
        try:
            return float(s)
        except Exception:
            return 0.0
    try:
        return float(t)
    except Exception:
        return 0.0

# -------------------------- Recognition control --------------------------
btn_label = "加载 demo 数据" if st.session_state.get('use_demo_mode') else "开始行为识别"
button_enabled = st.session_state.get('use_demo_mode') or (st.session_state.get('video_tmp') is not None)

if st.button(btn_label, type="primary", disabled=not button_enabled):
    with st.spinner("正在分析..."):
        try:
            if st.session_state.get('use_demo_mode'):
                demo_path = os.path.join("data", "sample_videos", "demo_log.json")
                if not os.path.exists(demo_path):
                    st.error("未找到 demo JSON：data/sample_videos/demo_log.json。请先生成或取消 demo 模式。")
                    st.stop()
                _demo_obj = json.load(open(demo_path, 'r', encoding='utf-8'))
                try:
                    st.session_state['results_json'] = json.dumps(_demo_obj, ensure_ascii=False)
                except Exception:
                    st.session_state['results_json'] = json.dumps(to_serializable(_demo_obj), ensure_ascii=False)
                st.session_state['results'] = _demo_obj
                st.session_state['use_demo_local'] = True
                st.session_state['api_result'] = None
                st.session_state['api_display_counts'] = None
                st.session_state['api_time_series'] = None
                st.session_state['api_stats'] = None

                if not st.session_state.get('video_tmp'):
                    candidate_paths = [
                        os.path.join("data", "sample_videos", "demo.mp4"),
                        os.path.join("data", "sample_videos", "demo_video.mp4"),
                        os.path.join("data", "sample_videos", "demo_sample.mp4"),
                    ]
                    found = None
                    for p in candidate_paths:
                        if os.path.exists(p):
                            found = p
                            break
                    if found:
                        st.session_state['video_tmp'] = found
                        st.info(f"Demo 模式：使用示例视频作为帧源：{found}")
                    else:
                        st.warning("已加载 demo 数据，但未检测到本地视频用于帧预览。")
                else:
                    st.info("已加载 demo 数据；检测到上传视频，保留用于帧可视化。")
                result_placeholder.success("已加载 demo 数据（未触发推理）。")

            else:
                tmp_vid = st.session_state.get('video_tmp')
                if not tmp_vid or not os.path.exists(tmp_vid):
                    st.error("请先上传视频或启 demo 模式。")
                    st.stop()

                api_url = st.session_state.get('api_url', DEFAULT_API_URL)
                remote_success = False
                try:
                    with open(tmp_vid, "rb") as vf:
                        files = {"video": (os.path.basename(tmp_vid), vf, "video/mp4")}
                        resp = requests.post(api_url, files=files, timeout=600)
                    resp.raise_for_status()
                    j = resp.json()
                    if isinstance(j, dict) and j.get("code", 0) == 200 and "data" in j:
                        data = j["data"]
                        st.success("远端 API 上传并返回成功（结果已载入）")
                        st.session_state['api_result'] = data
                        action_stats = find_action_stats_in_data(data) or {}
                        st.session_state['api_display_counts'] = action_stats
                        st.session_state['api_time_series'] = data.get("time_series", []) or []
                        st.session_state['api_stats'] = {
                            "abnormal_ratio": data.get("abnormal_ratio"),
                            "abnormal_frames": data.get("abnormal_frames"),
                            "total_frames": data.get("total_frames")
                        }
                        frames = find_frames_in_data(data)
                        if frames:
                            st.session_state['results'] = frames
                            try:
                                st.session_state['results_json'] = json.dumps({"frames": frames}, ensure_ascii=False)
                            except Exception:
                                st.session_state['results_json'] = json.dumps(to_serializable({"frames": frames}), ensure_ascii=False)
                        else:
                            try:
                                st.session_state['results_json'] = json.dumps(data, ensure_ascii=False)
                            except Exception:
                                st.session_state['results_json'] = json.dumps(to_serializable(data), ensure_ascii=False)
                        remote_success = True
                    else:
                        st.error(f"远端返回异常: {j.get('msg', str(j))}")
                except requests.exceptions.RequestException as e:
                    st.warning(f"远端上传失败：{e}")
                except Exception as e:
                    st.warning(f"远端上传失败：{e}")

                if not remote_success and MODEL_AVAILABLE:
                    try:
                        _res = inference_real(tmp_vid)
                        st.session_state['results_json'] = json.dumps(_res, ensure_ascii=False)
                        st.session_state['results'] = _res
                        st.session_state['use_demo_local'] = False
                        result_placeholder.success("本地推理已完成（远端不可用）。")
                    except Exception as e:
                        st.error(f"本地推理失败：{e}")
                        st.session_state['results'] = None
                elif not remote_success:
                    st.error("远端不可用且本地模型不可用。")
        except Exception as e:
            st.error(f"识别出错：{str(e)}")

if st.button("清除识别结果 / 删除临时视频"):
    tmp = st.session_state.get('video_tmp')
    if tmp and os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass
    for k in ['results', 'results_json', 'video_tmp', 'use_demo_local', 'last_uploaded_name', 'conf',
              'api_result', 'api_display_counts', 'api_time_series', 'api_stats']:
        st.session_state[k] = None if k != 'conf' else 0.0
    st.success("已清除结果和临时视频！")

# -------------------------- Result Render --------------------------
if st.session_state.get('results_json'):
    try:
        loaded = json.loads(st.session_state['results_json'])
        if isinstance(loaded, dict) and 'frames' in loaded and isinstance(loaded['frames'], list):
            frames_list = loaded['frames']
            st.session_state['results'] = frames_list
        else:
            frames_list = loaded if isinstance(loaded, list) else []
    except Exception:
        frames_list = []
else:
    frames_list = []

api_counts = st.session_state.get('api_display_counts') or {}
api_time_series = st.session_state.get('api_time_series') or []
api_stats = st.session_state.get('api_stats') or {}

if not frames_list and not api_counts:
    st.info("请上传视频并点击“开始行为识别”")
else:
    # 行为统计
    all_labels = []
    total_persons = 0
    abnormal_count = 0
    ABNORMAL = {"call", "smoke", "squat", "lie", "hit", "kick", "slap", "climbing", "fighting", "running", "point"}

    for frame in frames_list:
        persons = frame.get("persons", [])
        if persons:
            p = persons[0]
            lbl = extract_label_from_person(p)
            all_labels.append(lbl)
            total_persons = 1
            if lbl in ABNORMAL:
                abnormal_count += 1

    # 统计数据
    total_frames = api_stats.get("total_frames", len(frames_list))
    abnormal_ratio = f"{api_stats.get('abnormal_ratio', 0)}%"
    video_duration = round(total_frames / 25, 1) if total_frames else 0

    # 刷新统计面板
    stat_placeholder.markdown(
        _render_stat_html(abnormal_ratio, video_duration, total_frames, abnormal_count, total_persons, 0),
        unsafe_allow_html=True
    )

    # 统计图
    with chart_container:
        if api_counts:
            df = pd.DataFrame(list(api_counts.items()), columns=["行为", "次数"])
        else:
            df = pd.DataFrame(Counter(all_labels).items(), columns=["行为", "次数"])

        df["行为"] = df["行为"].apply(translate_label)
        bar = alt.Chart(df).mark_bar(color='#3b82f6').encode(
            x=alt.X('行为', sort='-y'), y='次数'
        ).properties(height=250, title="行为频次统计")
        st.altair_chart(bar, use_container_width=True)

        st.markdown("---")
        st.subheader("实时行为预览")
        cap = cv2.VideoCapture(st.session_state.get('video_tmp', ''))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = st.slider("帧进度", 0, max(total-1, 0), 0) if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret and frame_idx < len(frames_list):
            data = frames_list[frame_idx]
            persons = data.get("persons", [])
            if persons:
                p = persons[0]
                lbl = translate_label(extract_label_from_person(p))
                conf = p.get("conf", 1.0)
                cv2.putText(frame, f"{lbl} {conf}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB", use_column_width=True)
        cap.release()
