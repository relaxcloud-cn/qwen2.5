import os
import signal
import time
import multiprocessing
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import List, Optional, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from config import settings

# 定义安全中间件
class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # 检查内容类型
        content_type = request.headers.get("content-type", "").lower()
        if not any(allowed_type in content_type for allowed_type in settings.ALLOWED_CONTENT_TYPES):
            return JSONResponse(
                status_code=415,
                content={"detail": f"Unsupported media type. Allowed types: {settings.ALLOWED_CONTENT_TYPES}"}
            )

        # 检查内容长度
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request too large. Maximum size is {settings.MAX_REQUEST_SIZE} bytes"}
            )

        response = await call_next(request)
        return response

# 初始化模型和tokenizer
gpu_count = torch.cuda.device_count()
tensor_parallel_size = min(gpu_count, settings.MAX_GPU_COUNT)
print(f"Found {gpu_count} GPUs, using {tensor_parallel_size} for tensor parallel")

models = []
tokenizers = []

def cleanup():
    """清理资源的函数"""
    print("\nCleaning up resources...")
    try:
        # 在这里添加任何需要的清理代码
        for model in models:
            try:
                del model
            except:
                pass
        models.clear()
        torch.cuda.empty_cache()
        print("Cleanup completed.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def signal_handler(sig, frame):
    """处理信号的函数"""
    print("\nReceived signal to shutdown...")
    try:
        cleanup()
    finally:
        # 确保无论如何都会退出
        os._exit(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    print("Loading models...")
    try:
        model = LLM(
            model=settings.MODEL_PATH,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
            max_num_batched_tokens=settings.MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=settings.MAX_NUM_SEQS,
            trust_remote_code=settings.TRUST_REMOTE_CODE
        )
        models.append(model)
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH, trust_remote_code=True)
        tokenizers.append(tokenizer)
    except Exception as e:
        print(f"Error during startup: {e}")
        cleanup()
        raise e
    yield
    # 关闭时清理资源
    print("\nShutting down application...")
    cleanup()

app = FastAPI(title="Qwen2.5 vLLM Server", lifespan=lifespan)

# 添加中间件
app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# 定义请求模型
class Message(BaseModel):
    role: str  # system, user, assistant
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model_index: Optional[int] = 0
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    repetition_penalty: Optional[float] = 1.05
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

# 定义响应模型
class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.model_index >= len(models):
        raise HTTPException(status_code=400, detail="Invalid model index")
    
    # 使用tokenizer处理聊天模板
    tokenizer = tokenizers[request.model_index]
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        max_tokens=request.max_tokens
    )
    
    # 使用指定的模型生成响应
    model = models[request.model_index]
    outputs = model.generate([text], sampling_params)
    
    if not outputs:
        raise HTTPException(status_code=500, detail="Failed to generate response")

    # 构建 OpenAI 风格的响应
    response_message = Message(
        role="assistant",
        content=outputs[0].outputs[0].text
    )
    
    choice = Choice(
        index=0,
        message=response_message,
        finish_reason="stop"
    )
    
    # 获取模型名称
    model_name = settings.MODEL_PATH
    
    # 计算token使用量（这里是一个简单的估计）
    prompt_tokens = len(tokenizer.encode(text))
    completion_tokens = len(tokenizer.encode(response_message.content))
    
    return ChatResponse(
        id=f"chatcmpl-{id(outputs)}",
        created=int(time.time()),
        model=model_name,
        choices=[choice],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    )

@app.get("/models")
async def list_models():
    return {"models": [settings.MODEL_PATH]}

def main():
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动服务器
    import uvicorn
    config = uvicorn.Config(
        app,
        host=settings.HOST,
        port=settings.PORT,
        workers=1,
        loop="asyncio",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,  # 10秒后强制关闭
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()
