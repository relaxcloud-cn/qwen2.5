FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY pyproject.toml poetry.lock ./
COPY . .

# 安装依赖
RUN poetry install --no-interaction --no-ansi --no-root

# 创建模型目录
RUN mkdir -p /app/models

# 暴露端口
EXPOSE 8002

# 启动命令
CMD ["poetry", "run", "python3", "server.py"]
