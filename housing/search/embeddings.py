"""向量化封装。

bge 系列中文检索模型有一个官方技巧：query 侧加指令前缀能显著提升召回，
文档侧不加。这里把差异封装掉，调用方不用关心。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from housing.config import settings

# bge 中文模型的推荐 query 前缀
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@lru_cache(maxsize=4)
def _build(model_name: str, device: str) -> HuggingFaceEmbeddings:
    """模型加载一次就好：sentence-transformers 初始化要几百毫秒到数秒。"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings(model_name: str | None = None, device: str | None = None) -> HuggingFaceEmbeddings:
    """构造（或复用）embedding 模型。余弦相似度要求向量归一化。"""
    return _build(model_name or settings.embedding_model, device or settings.embedding_device)


def with_query_prefix(query: str) -> str:
    """给检索 query 加 bge 指令前缀（已加过则不重复加）。"""
    if query.startswith(BGE_ZH_QUERY_PREFIX):
        return query
    return BGE_ZH_QUERY_PREFIX + query
