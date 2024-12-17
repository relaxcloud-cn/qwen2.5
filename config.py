import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Qwen模型配置
    MODEL_NAMES: List[str] = ["Qwen/Qwen1.5-7B-Chat"]
    MAX_GPU_COUNT: int = 1  # 默认最多使用1个GPU
    TENSOR_PARALLEL_SIZE: int = 1
    GPU_MEMORY_UTILIZATION: float = 0.9
    MAX_NUM_BATCHED_TOKENS: int = 32768  # 与模型的 max_model_len 保持一致
    MAX_NUM_SEQS: int = 64
    TRUST_REMOTE_CODE: bool = True
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # HuggingFace配置
    HF_ENDPOINT: str = "https://hf-mirror.com"
    HF_MIRROR: str = "https://hf-mirror.com"
    HF_HOME: str = "~/.cache/huggingface/hub"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保 HF_HOME 路径被正确展开
        self.HF_HOME = os.path.expanduser(self.HF_HOME)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
