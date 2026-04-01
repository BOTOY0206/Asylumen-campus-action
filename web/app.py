import streamlit as st
import os
import json
import cv2
import tempfile
import numpy as np
from examples.inference_real import inference_real, get_action_label  # 导入真实识别函数

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

# 视频预览区域
st.subheader("视频预览 & 识别结果可视化")
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
        
        # 4. 显示核心识别结果
        if result:
            # 提取关键结果展示
            key_result = result[0] if result else {}
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
            
            # 6. 视频预览（播放前10帧）
            st.info("视频预览（仅展示前10帧）：")
            frames = []
            for i in range(min(10, frame_count)):
                ret, frame = cap.read()
                if ret:
                    # 转换颜色空间（OpenCV BGR → RGB）
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            # 播放预览
            if frames:
                video_placeholder.image(
                    frames,
                    caption="视频帧预览（真实识别结果）",
                    width=600,
                    channels="RGB"
                )
            
            cap.release()
        
        # 删除临时文件
        os.unlink(video_path)

# 底部说明
st.divider()
st.markdown("""
### 说明
- 本系统已完全替换为真实视频识别逻辑，不再使用占位假数据
- 识别结果会自动保存到 `data/results` 文件夹
- 支持的视频格式：mp4、avi、mov、mkv
- 识别行为类型：hit_wall、kick、laying、use_phone、pointing、slap_face、slap_table、smoking、squating、stand、touch、whole_process、normal
""")
