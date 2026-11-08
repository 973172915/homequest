"""全局配置：统一从环境变量 / .env 读取，避免密钥散落在代码里。"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """应用配置。所有字段都可通过同名大写环境变量覆盖。"""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- DeepSeek ----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    # 官方模型 id 是 deepseek-flash（DeepSeek-V4.1-Flash-1017）
    deepseek_model: str = "deepseek-flash"
    # 思考模式强度：none（关闭，RAG 生成首选）/ low / high / max
    deepseek_reasoning_effort: str = "none"
    deepseek_temperature: float = 0.3

    # ---- 向量化 ----
    hf_endpoint: str = "https://hf-mirror.com"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # ---- 检索 ----
    retrieve_top_k: int = 10
    retrieve_fetch_k: int = 50
    kb_chunk_size: int = 400
    kb_chunk_overlap: int = 60
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # ---- 对话记忆 ----
    # 最近多少轮完整保留（1 轮 = 用户 + 助手）。超出部分压成滚动摘要
    max_history_turns: int = 15

    # ---- 路径 ----
    @property
    def data_raw(self) -> Path:
        """优先用完整数据；仓库里只带 2000 条样本，取到哪个用哪个。"""
        full = ROOT / "data" / "sh_data.csv"
        if full.exists():
            return full
        return ROOT / "data" / "sample_2000.csv"

    @property
    def data_kb(self) -> Path:
        return ROOT / "data" / "kb"

    @property
    def artifacts(self) -> Path:
        return ROOT / "artifacts"

    @property
    def processed(self) -> Path:
        return self.artifacts / "processed"

    @property
    def index_dir(self) -> Path:
        """向量索引目录（FAISS）。"""
        return self.artifacts / "index"

    @property
    def models_dir(self) -> Path:
        return self.artifacts / "models"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts / "reports"

    def ensure_dirs(self) -> None:
        """创建所有输出目录，幂等。"""
        for p in (self.processed, self.index_dir, self.models_dir, self.reports_dir,
                  self.artifacts / "hf"):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()

# HuggingFace 官方站在本机不通，提前指向镜像（必须在 import sentence_transformers 之前生效）；
# 缓存目录也放到项目内，避免写到用户目录 %USERPROFILE%\.cache。
os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
os.environ.setdefault("HF_HOME", str(settings.artifacts / "hf"))
os.environ.setdefault("HF_HUB_CACHE", str(settings.artifacts / "hf" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(settings.artifacts / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# 受限环境里不能建符号链接（hf 会探测），改为直接复制文件
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
