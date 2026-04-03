# 校园行为识别系统 - F1 向 F2 交接报告
**交接人**：F1（何睿祺）
**接收人**：F2（同学）
**项目状态**：F1 核心功能已 100% 完成，可直接进行 UI 美化与演示准备

## 📊 当前进度一句话总结
后端 FastAPI 接口 `http://localhost:8000/infer_behavior` 已稳定运行并支持真实视频推理；前端 Streamlit 页面已实现视频上传、任务状态轮询、JSON 结果展示与下载；CI 自动化测试通过。最近完成的真实 Demo 推理：`phone_backward_3-1.mp4` → 生成结果 `data/results/phone_backward_3-1.json`。

## 🚀 本地运行指南（F2 启动服务用）
### 1. 激活虚拟环境
```cmd
venv\Scripts\activate
