import os
import signal
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams
import asyncio
from config import settings

app = FastAPI(title="Qwen2.5 vLLM Server")

# 初始化多个模型实例
models = []
for model_name in settings.MODEL_NAMES:
    model = LLM(
        model=model_name,
        tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
        max_num_batched_tokens=settings.MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=settings.MAX_NUM_SEQS,
        trust_remote_code=settings.TRUST_REMOTE_CODE
    )
    models.append(model)

# 定义请求模型
class ChatRequest(BaseModel):
    prompt: str
    model_index: Optional[int] = 0  # 默认使用第一个模型
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048

# 定义响应模型
class ChatResponse(BaseModel):
    response: str
    model_index: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.model_index >= len(models):
        raise HTTPException(status_code=400, detail="Invalid model index")
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )
    
    # 使用指定的模型生成响应
    model = models[request.model_index]
    outputs = model.generate(request.prompt, sampling_params)
    
    if not outputs:
        raise HTTPException(status_code=500, detail="Failed to generate response")
    
    return ChatResponse(
        response=outputs[0].outputs[0].text,
        model_index=request.model_index
    )

@app.get("/models")
async def get_models():
    return {"models": settings.MODEL_NAMES}

if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    import uvicorn
    config = uvicorn.Config(
        "server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        reload_delay=0.25,
        log_level="info",
        access_log=True,
        workers=1
    )
    server = uvicorn.Server(config)
    server.run()
