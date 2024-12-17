# qwen2.5
通过 vLLM 部署 Qwen2.5 模型的服务端。

## 特性
- 使用 vLLM 进行高性能推理
- 支持多 GPU 部署
- 离线模型加载，避免网络问题
- FastAPI 后端服务

## 快速开始

### 1. 下载模型
在启动服务之前，需要先下载模型文件到本地。你可以选择以下任一方式：

#### 方式一：使用 huggingface-cli（推荐）
首先安装 huggingface-cli：
```bash
pip install --upgrade huggingface_hub
```

然后下载模型：
```bash
# 设置镜像（如果需要）
export HF_ENDPOINT=https://hf-mirror.com
export HF_MIRROR=https://hf-mirror.com

# 下载模型到指定目录
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-AWQ --local-dir ./models/Qwen2.5-3B-Instruct-AWQ
```

#### 方式二：手动下载
1. 访问模型页面：https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ
2. 下载所有必要的模型文件
3. 将文件放置在 `./models/Qwen2.5-3B-Instruct-AWQ` 目录下

### 2. 启动服务
```bash
# 使用 poetry 安装依赖
poetry install

# 启动服务
poetry run python server.py
```

### 3. API 使用
服务启动后，可以通过 HTTP API 进行调用：

```bash
curl -X POST "http://localhost:8002/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "你好"}
       ]
     }'
```

## 配置说明
主要配置项在 `config.py` 中：

- `MODEL_PATH`：模型本地路径
- `MAX_GPU_COUNT`：最大使用的 GPU 数量
- `GPU_MEMORY_UTILIZATION`：GPU 显存使用率
- 其他参数说明请参考 `config.py` 的注释

## Docker 部署
```bash
# 构建镜像
docker build -t qwen2.5-server .

# 运行容器
docker run -d \
  --gpus all \
  -v $(pwd)/models:/app/models \
  -p 8002:8002 \
  qwen2.5-server
