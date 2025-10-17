# 1. 基础镜像：使用官方的 Python 3.11 轻量级 (Slim) 镜像
# 这比完整版镜像更小，启动更快。
FROM python:3.11-slim

# 2. 设定工作目录：所有后续操作都在这个目录中进行
WORKDIR /app

# 3. 复制依赖文件并安装
# 先复制依赖清单，利用 Docker 的缓存机制，如果依赖没变，就不用重复安装
COPY requirements.txt .

# 安装 Python 依赖
# --no-cache-dir 减少镜像大小；-r 指定从文件安装
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制应用代码和 AI 模型
# 复制 FastAPI 应用文件和模型文件
COPY app.py .
COPY anomaly_detector.pkl .

# 5. 暴露端口：FastAPI 默认使用 8000 端口
EXPOSE 8000

# 6. 容器启动命令：使用 Uvicorn 启动 FastAPI 应用
# --host 0.0.0.0 允许外部访问
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]