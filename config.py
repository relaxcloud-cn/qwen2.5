from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8002

    # 模型配置
    MODEL_PATH: str = "./models/Qwen2.5-3B-Instruct-AWQ"  # 本地模型路径
    MAX_GPU_COUNT: int = 2
    TENSOR_PARALLEL_SIZE: int = 1
    GPU_MEMORY_UTILIZATION: float = 0.9
    MAX_NUM_BATCHED_TOKENS: int = 32768
    MAX_NUM_SEQS: int = 64
    TRUST_REMOTE_CODE: bool = True

    # 安全配置
    MAX_REQUEST_SIZE: int = 1 * 1024 * 1024  # 1MB
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    ALLOWED_CONTENT_TYPES: List[str] = ["application/json"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保模型路径是绝对路径
        self.MODEL_PATH = os.path.abspath(os.path.expanduser(self.MODEL_PATH))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
