from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Qwen模型配置
    MODEL_NAMES: List[str] = ["Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-7B-Chat"]  # 默认加载两个相同的模型
    TENSOR_PARALLEL_SIZE: int = 1  # 每个模型的张量并行大小
    GPU_MEMORY_UTILIZATION: float = 0.9  # GPU显存利用率
    MAX_NUM_BATCHED_TOKENS: int = 4096  # 最大批处理token数
    MAX_NUM_SEQS: int = 256  # 最大序列数
    TRUST_REMOTE_CODE: bool = True  # 信任远程代码
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
