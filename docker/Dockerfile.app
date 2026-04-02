# 基础镜像：官方 Python 3.13 最新版
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（避免 Python 3.13 缺少库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["sh", "docker/start.sh"]
