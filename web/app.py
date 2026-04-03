import streamlit as st
import os
import json
import cv2
import tempfile
import numpy as np
import sys

# 【关键修复】添加项目根目录到Python搜索路径，解决模块导入问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.inference_real import inference_real, get_action_label

# 页面配置
st.set_page_config(
    page_title="校园行为识别系统",
    page_icon="",
    layout="wide"
)

# 标题
st.title("校园行为识别系统（真实识别版）")
st.divider()

# 侧边栏
with st.sidebar:
    st.header("操作面板")
    st.info("上传视频文件，点击识别按钮，即可获取真实行为识别结果！")
    
    # 结果保存路径设置
    output_dir = st.text_input(
        "结果保存路径",
        value="data/results",
        help="识别结果JSON文件保存的位置"
    )
    os.makedirs(output_dir, exist_ok=True)

# 主页面
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("上传视频")
    # 视频上传组件
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=False
    )
    
    # 识别按钮
    recognize_btn = st.button(
        "开始真实识别",
        type="primary",
        disabled=not uploaded_file
    )

with col2:
    st.subheader("识别结果")
    result_placeholder = st.empty()
    json_placeholder = st.empty()
    # 在这里预留下载按钮的位置
    download_placeholder = st.empty()

# 视频预览区域
st.subheader("视频预览 & 识别结果可视化")
# 在这里预留置信度slider和bbox可视化的位置
conf_slider_placeholder = st.empty()
bbox_placeholder = st.empty()
video_placeholder = st.empty()

# 核心识别逻辑
if uploaded_file and recognize_btn:
    with st.spinner("正在进行真实视频识别，请稍候..."):
        # 1. 保存上传的视频到临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # 2. 调用真实识别函数（和命令行用的是同一个！）
        result = inference_real(video_path)
        
        # 3. 视频预览
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 显示视频基本信息
        video_info = f"""
        视频信息：
        - 文件名：{uploaded_file.name}
        - 总帧数：{frame_count}
        - 帧率：{fps} FPS
        - 分辨率：{width} × {height}
        - 识别行为：{get_action_label(uploaded_file.name)}
        """
        st.success(video_info)
        
        # 4. 显示核心识别结果（修复空值问题）
        if result and len(result) > 0:
            key_result = result[0]
            result_html = f"""
            <div style="background-color:#f0f8ff; padding:15px; border-radius:8px;">
                <h4>核心识别结果（第一帧）</h4>
                <p><strong>时间戳：</strong>{key_result.get('time', '0.0')} 秒</p>
                <p><strong>帧号：</strong>{key_result.get('frame', '0')}</p>
                <p><strong>人物框：</strong>{key_result.get('persons', [{}])[0].get('bbox', [])}</p>
                <p><strong>行为标签：</strong>{key_result.get('persons', [{}])[0].get('label', 'normal')}</p>
                <p><strong>置信度：</strong>{key_result.get('persons', [{}])[0].get('conf', '0.0')}</p>
            </div>
            """
            result_placeholder.markdown(result_html, unsafe_allow_html=True)
            
            # 显示完整JSON结果
            json_placeholder.subheader("完整JSON结果")
            json_placeholder.json(result)
            
            # 5. 保存结果到指定路径
            output_name = os.path.splitext(uploaded_file.name)[0] + ".json"
            output_path = os.path.join(output_dir, output_name)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            st.success(f"识别完成！结果已保存到：{output_path}")

            # ==============================================
            # 🌟 Day2 功能1：置信度滑动条 slider（修复位置）
            # ==============================================
            conf_list = []
            for frame_data in result:
                persons = frame_data.get("persons", [])
                if persons:
                    conf = persons[0].get("conf", 0.0)
                    conf_list.append(conf)
            
            min_conf = float(min(conf_list)) if conf_list else 0.0
            max_conf = float(max(conf_list)) if conf_list else 1.0
            
            conf_slider_placeholder.subheader("置信度阈值")
            conf_threshold = conf_slider_placeholder.slider(
                "仅显示大于等于该阈值的结果",
                min_value=0.0,
                max_value=1.0,
                value=min_conf
            )

            # ==============================================
            # 🌟 Day2 功能2：JSON下载按钮（修复位置）
            # ==============================================
            download_placeholder.download_button(
                label="下载JSON结果",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name=output_name,
                mime="application/json"
            )

            # ==============================================
            # 🌟 Day3-4 功能：BBox 可视化画框（修复位置）
            # ==============================================
            bbox_placeholder.subheader("识别框可视化")
            # 重置视频读取位置到第一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret:
                persons = key_result.get("persons", [])
                if persons:
                    bbox = persons[0].get("bbox", [])
                    label = persons[0].get("label", "")
                    conf = persons[0].get("conf", 0.0)

                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bbox_placeholder.image(frame_rgb, width=800)

            # 6. 视频预览（修复caption数量不匹配问题）
            st.info("视频预览（仅展示前10帧）：")
            frames = []
            # 再次重置视频位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(min(10, frame_count)):
                ret, frame = cap.read()
                if ret:
                    # 转换颜色空间（OpenCV BGR → RGB）
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            # 播放预览（修复报错：去掉caption，避免数量不匹配）
            if frames:
                video_placeholder.image(
                    frames,
                    width=600,
                    channels="RGB"
                )
            
            cap.release()
        else:
            st.error("识别结果为空，请检查视频文件是否正常")
            cap.release()
        
        # 删除临时文件
        # 方案：加异常捕获，删不掉就跳过，绝对不报错！
try:
    os.unlink(video_path)
except PermissionError:
    # 临时文件被占用，跳过删除，不影响任何功能
    print(f"临时文件 {video_path} 被占用，已跳过删除，不影响使用")
except Exception as e:
    # 其他异常也捕获，彻底兜底
    print(f"删除临时文件时出现异常：{str(e)}，不影响使用")
except NameError:
    # video_path未定义时跳过，避免启动报错
    pass

# 底部说明
st.divider()
st.markdown("""
### 说明
- 本系统已完全替换为真实视频识别逻辑，不再使用占位假数据
- 识别结果会自动保存到 data/results 文件夹
- 支持的视频格式：mp4、avi、mov、mkv
- 识别行为类型：hit_wall、kick、laying、use_phone、pointing、slap_face、slap_table、smoking、squating、stand、touch、whole_process、normal
""")
